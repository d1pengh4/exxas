"""
Redis 기반 Rate Limiting 미들웨어 (Token Bucket)
- 분석 엔드포인트: 분당 10회 (IP당)
- 인증 엔드포인트: 분당 5회 (IP당, 브루트포스 방지)
- 일반 엔드포인트: 분당 60회
"""
import time
import json
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


# 경로별 설정: (max_requests, window_seconds)
RATE_LIMITS = {
    "/api/v1/analyze": (10, 60),        # 분당 10회 (분석)
    "/api/v1/analyze/batch": (3, 60),   # 분당 3회 (배치)
    "/api/v1/auth/token": (5, 60),      # 분당 5회 (로그인 브루트포스 방지)
    "/api/v1/auth/register": (3, 300),  # 5분당 3회 (가입 스팸 방지)
    "default": (60, 60),                # 분당 60회
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_url: str):
        super().__init__(app)
        self._redis_url = redis_url
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    async def dispatch(self, request: Request, call_next) -> Response:
        # 헬스체크, 스태틱, 스트리밍은 제외
        path = request.url.path
        if path in ("/health", "/docs", "/openapi.json", "/redoc") or \
           "/stream" in path or path.startswith("/static"):
            return await call_next(request)

        # IP 추출 (프록시 헤더 지원)
        ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
        if "," in ip:
            ip = ip.split(",")[0].strip()

        # 경로 매칭 (긴 prefix 우선 — 더 구체적인 규칙이 이김)
        max_req, window = RATE_LIMITS["default"]
        for prefix in sorted((k for k in RATE_LIMITS if k != "default"), key=len, reverse=True):
            if path.startswith(prefix):
                max_req, window = RATE_LIMITS[prefix]
                break

        # Redis 슬라이딩 윈도우 카운터
        try:
            redis = await self._get_redis()
            key = f"rate:{ip}:{path.split('/')[3] if path.count('/') >= 3 else path}"
            now = int(time.time())
            window_start = now - window

            pipe = redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {str(now) + f":{id(request)}": now})
            pipe.zcard(key)
            pipe.expire(key, window)
            results = await pipe.execute()
            count = results[2]

            if count > max_req:
                retry_after = window
                logger.warning(f"Rate limit exceeded: {ip} {path} ({count}/{max_req})")
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"요청 한도 초과 ({max_req}회/{window}초). {retry_after}초 후 재시도.",
                        "retry_after": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )
        except Exception as e:
            # Redis 장애 시 통과 (graceful degradation)
            logger.debug(f"Rate limit check failed (passing): {e}")

        response = await call_next(request)

        # Rate limit 헤더 추가
        try:
            response.headers["X-RateLimit-Limit"] = str(max_req)
            response.headers["X-RateLimit-Window"] = str(window)
        except Exception:
            pass

        return response


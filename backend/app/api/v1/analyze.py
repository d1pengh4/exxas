"""
분석 API v2
- POST /analyze: 이미지 업로드 → Celery 큐
- GET  /analyze/{job_id}: 결과 조회
- GET  /analyze/{job_id}/stream: SSE 실시간 스트림
- GET  /analyze/history: 과거 분석 목록
- POST /analyze/{job_id}/feedback: 결과 피드백
"""
import uuid
import json
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from loguru import logger

from ...core.database import get_db, get_redis, get_raw_redis
from ...models.analysis import Analysis, AnalysisStatus
from ...models.user import User
from ...api.v1.auth import get_current_user, get_optional_user
from ...tasks.analysis import run_analysis_task

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic"}
MAX_SIZE = 50 * 1024 * 1024


class StartResponse(BaseModel):
    job_id: str
    status: str
    message: str


class AnalysisOut(BaseModel):
    job_id: str
    status: str
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    confidence: Optional[float] = None
    confidence_label: Optional[str] = None
    exploration_mode: Optional[str] = None
    total_steps: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    evidence_chain: Optional[list] = None
    hypothesis_tree: Optional[dict] = None
    final_reasoning: Optional[str] = None
    hallucination_check_passed: Optional[bool] = None
    image_manipulation_suspected: Optional[bool] = None
    created_at: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


class FeedbackIn(BaseModel):
    is_correct: bool
    actual_location: str = ""


@router.post("/analyze", response_model=StartResponse)
async def start_analysis(
    file: UploadFile = File(...),
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
    raw_redis=Depends(get_raw_redis),
):
    # 파일 검증
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"지원 형식: JPEG, PNG, WEBP, HEIC")
    image_bytes = await file.read()
    if len(image_bytes) > MAX_SIZE:
        raise HTTPException(400, "파일 50MB 초과")
    if len(image_bytes) < 500:
        raise HTTPException(400, "파일이 너무 작습니다")

    # 로그인 사용자 — 월간 사용량 리셋 & 한도 확인
    if user:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        reset_at = user.usage_reset_at
        if reset_at:
            if not reset_at.tzinfo:
                reset_at = reset_at.replace(tzinfo=timezone.utc)
            this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if reset_at < this_month_start:
                user.monthly_usage = 0
                user.usage_reset_at = now
        if not user.can_analyze:
            raise HTTPException(429, f"월 사용 한도 초과 ({user.plan} 플랜: {user.monthly_limit}회/월)")
        user.monthly_usage += 1
        await db.commit()

    media_type = file.content_type or "image/jpeg"
    job_id = str(uuid.uuid4())

    # 플랜별 기능 플래그를 Redis에 함께 저장 (Celery 태스크에서 참조)
    plan_flags = {
        "osint_enabled": user.has_feature("osint_reverse_search") if user else False,
        "manipulation_enabled": user.has_feature("manipulation_detection") if user else False,
        "full_pipeline": user.has_feature("full_pipeline") if user else False,
    }

    # DB에 분석 레코드 생성
    analysis = Analysis(
        job_id=job_id,
        user_id=user.id if user else None,
        status=AnalysisStatus.queued,
        image_size_bytes=len(image_bytes),
        image_media_type=media_type,
    )
    db.add(analysis)
    await db.commit()

    # Redis에 이미지 임시 저장 (1시간) — 바이너리 전용 커넥션 사용
    await raw_redis.setex(f"job:{job_id}:image", 3600, image_bytes)
    await raw_redis.setex(f"job:{job_id}:media_type", 3600, media_type.encode())
    await redis.setex(f"job:{job_id}:plan_flags", 3600, json.dumps(plan_flags))
    await redis.setex(
        f"job:{job_id}:status",
        86400,
        json.dumps({"job_id": job_id, "status": "queued", "message": "수사 대기 중"}),
    )

    # Celery 태스크
    run_analysis_task.delay(job_id)
    logger.info(f"Analysis queued: {job_id} ({len(image_bytes)//1024}KB)")

    return StartResponse(job_id=job_id, status="queued", message="수사가 시작되었습니다")


@router.get("/analyze/{job_id}", response_model=AnalysisOut)
async def get_result(
    job_id: str,
    redis=Depends(get_redis),
    db: AsyncSession = Depends(get_db),
):
    # Redis 먼저 (빠름)
    raw = await redis.get(f"job:{job_id}:status")
    if raw:
        data = json.loads(raw)
        return AnalysisOut(**{k: v for k, v in data.items() if k in AnalysisOut.model_fields})

    # DB 조회
    result = await db.execute(select(Analysis).where(Analysis.job_id == job_id))
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(404, "분석 결과를 찾을 수 없습니다")

    return AnalysisOut(
        job_id=analysis.job_id,
        status=analysis.status,
        location=analysis.location,
        latitude=analysis.latitude,
        longitude=analysis.longitude,
        confidence=analysis.confidence,
        confidence_label=analysis.confidence_label,
        exploration_mode=str(analysis.exploration_mode) if analysis.exploration_mode else None,
        total_steps=analysis.total_steps,
        elapsed_seconds=analysis.elapsed_seconds,
        evidence_chain=analysis.evidence_chain,
        hypothesis_tree=analysis.hypothesis_tree,
        final_reasoning=analysis.final_reasoning,
        hallucination_check_passed=analysis.hallucination_check_passed,
        image_manipulation_suspected=analysis.manipulation_suspected,
        error=analysis.error_message,
    )


@router.get("/analyze/{job_id}/stream")
async def stream_result(job_id: str, redis=Depends(get_redis)):
    """SSE 실시간 스트리밍"""

    async def gen():
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"job:{job_id}:updates")
        try:
            while True:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
                if msg:
                    yield f"data: {msg['data']}\n\n"
                    parsed = json.loads(msg["data"])
                    if parsed.get("status") in ("completed", "failed"):
                        break
        finally:
            await pubsub.unsubscribe(f"job:{job_id}:updates")
            await pubsub.aclose()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/analyses/history", response_model=list[AnalysisOut])
async def get_history(
    limit: int = Query(default=10, le=50),
    offset: int = Query(default=0),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Analysis)
        .where(Analysis.user_id == user.id)
        .order_by(desc(Analysis.created_at))
        .limit(limit)
        .offset(offset)
    )
    rows = result.scalars().all()
    return [
        AnalysisOut(
            job_id=a.job_id,
            status=a.status,
            location=a.location,
            address=a.address,
            latitude=a.latitude,
            longitude=a.longitude,
            confidence=a.confidence,
            confidence_label=a.confidence_label,
            total_steps=a.total_steps,
            elapsed_seconds=a.elapsed_seconds,
            exploration_mode=str(a.exploration_mode) if a.exploration_mode else None,
            image_manipulation_suspected=a.manipulation_suspected,
            created_at=a.created_at.isoformat() if a.created_at else None,
        )
        for a in rows
    ]


@router.post("/analyze/{job_id}/feedback")
async def submit_feedback(
    job_id: str,
    body: FeedbackIn,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Analysis).where(Analysis.job_id == job_id))
    analysis = result.scalar_one_or_none()

    if analysis:
        from datetime import datetime
        analysis.user_feedback_correct = body.is_correct
        analysis.user_feedback_actual_location = body.actual_location
        analysis.feedback_at = datetime.utcnow()
        await db.commit()

    from ...services.selflearn import get_selflearn_service
    svc = get_selflearn_service()
    await svc.record_feedback(
        job_id=job_id,
        predicted_location=analysis.location if analysis else "",
        confidence=analysis.confidence or 0.0 if analysis else 0.0,
        is_correct=body.is_correct,
        actual_location=body.actual_location,
    )

    return {"status": "ok", "job_id": job_id}


# ── 배치 분석 ────────────────────────────────────────────────────────────────

class BatchStartResponse(BaseModel):
    batch_id: str
    job_ids: list[str]
    total: int
    status: str


@router.post("/analyze/batch", response_model=BatchStartResponse)
async def start_batch_analysis(
    files: list[UploadFile] = File(...),
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
    raw_redis=Depends(get_raw_redis),
):
    """여러 이미지를 한번에 분석 (Pro: 10장, Expert: 50장)"""
    max_batch = user.max_batch_size if user else 1
    if not (user and user.has_feature("batch_upload")):
        raise HTTPException(403, "배치 업로드는 Pro 플랜 이상에서 사용 가능합니다")

    if len(files) > max_batch:
        raise HTTPException(400, f"{max_batch}장 초과 — 현재 플랜 한도 초과")
    if len(files) == 0:
        raise HTTPException(400, "파일이 없습니다")

    # 사용량 체크 (배치 전체)
    if user and user.monthly_usage + len(files) > user.monthly_limit:
        raise HTTPException(429, f"월 한도 초과 예정 — 남은 횟수: {user.monthly_limit - user.monthly_usage}")

    import uuid
    batch_id = str(uuid.uuid4())
    job_ids = []

    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            continue
        image_bytes = await file.read()
        if len(image_bytes) > MAX_SIZE or len(image_bytes) < 500:
            continue

        media_type = file.content_type or "image/jpeg"
        job_id = str(uuid.uuid4())

        analysis = Analysis(
            job_id=job_id,
            user_id=user.id if user else None,
            status=AnalysisStatus.queued,
            image_size_bytes=len(image_bytes),
            image_media_type=media_type,
        )
        db.add(analysis)

        # 배치에서도 플랜 기능 플래그 저장 (Celery 태스크가 OSINT 여부 확인)
        batch_plan_flags = {
            "osint_enabled": user.has_feature("osint_reverse_search") if user else False,
            "manipulation_enabled": user.has_feature("manipulation_detection") if user else False,
            "full_pipeline": user.has_feature("full_pipeline") if user else False,
        }
        await raw_redis.setex(f"job:{job_id}:image", 3600, image_bytes)
        await raw_redis.setex(f"job:{job_id}:media_type", 3600, media_type.encode())
        await redis.setex(f"job:{job_id}:plan_flags", 3600, json.dumps(batch_plan_flags))
        await redis.setex(
            f"job:{job_id}:status",
            86400,
            json.dumps({"job_id": job_id, "status": "queued", "batch_id": batch_id}),
        )

        run_analysis_task.delay(job_id)
        job_ids.append(job_id)

    if user and job_ids:
        user.monthly_usage += len(job_ids)

    await redis.setex(
        f"batch:{batch_id}:jobs",
        86400,
        json.dumps(job_ids),
    )

    await db.commit()
    logger.info(f"Batch analysis queued: {batch_id} ({len(job_ids)} jobs)")

    return BatchStartResponse(
        batch_id=batch_id,
        job_ids=job_ids,
        total=len(job_ids),
        status="queued",
    )


@router.get("/analyze/batch/{batch_id}")
async def get_batch_status(
    batch_id: str,
    redis=Depends(get_redis),
):
    """배치 분석 전체 상태 조회"""
    raw = await redis.get(f"batch:{batch_id}:jobs")
    if not raw:
        raise HTTPException(404, "배치를 찾을 수 없습니다")

    job_ids = json.loads(raw)
    statuses = []
    for job_id in job_ids:
        job_raw = await redis.get(f"job:{job_id}:status")
        if job_raw:
            statuses.append(json.loads(job_raw))
        else:
            statuses.append({"job_id": job_id, "status": "unknown"})

    completed = sum(1 for s in statuses if s.get("status") == "completed")
    failed = sum(1 for s in statuses if s.get("status") == "failed")

    return {
        "batch_id": batch_id,
        "total": len(job_ids),
        "completed": completed,
        "failed": failed,
        "pending": len(job_ids) - completed - failed,
        "jobs": statuses,
    }


# ── 보고서 다운로드 ──────────────────────────────────────────────────────────

@router.get("/analyze/{job_id}/report")
async def download_report(
    job_id: str,
    format: str = Query(default="markdown", pattern="^(markdown|json)$"),
    user: Optional[User] = Depends(get_optional_user),
    redis=Depends(get_redis),
    db: AsyncSession = Depends(get_db),
):
    """분석 리포트 다운로드 (Markdown/JSON) — Pro 이상"""
    if user and not user.has_feature("report_download"):
        raise HTTPException(403, "리포트 다운로드는 Pro 플랜 이상에서 사용 가능합니다")

    # 결과 조회
    raw = await redis.get(f"job:{job_id}:status")
    if raw:
        analysis_data = json.loads(raw)
    else:
        result = await db.execute(select(Analysis).where(Analysis.job_id == job_id))
        analysis = result.scalar_one_or_none()
        if not analysis:
            raise HTTPException(404, "분석 결과를 찾을 수 없습니다")
        analysis_data = {
            "job_id": analysis.job_id,
            "status": analysis.status,
            "location": analysis.location,
            "latitude": analysis.latitude,
            "longitude": analysis.longitude,
            "confidence": analysis.confidence,
            "confidence_label": analysis.confidence_label,
            "total_steps": analysis.total_steps,
            "elapsed_seconds": analysis.elapsed_seconds,
            "exploration_mode": str(analysis.exploration_mode) if analysis.exploration_mode else "",
            "evidence_chain": analysis.evidence_chain or [],
            "final_reasoning": analysis.final_reasoning or "",
            "hallucination_check_passed": analysis.hallucination_check_passed,
        }

    if analysis_data.get("status") != "completed":
        raise HTTPException(400, "아직 분석이 완료되지 않았습니다")

    from ...services.report import build_report, report_to_markdown
    report = build_report(analysis_data)

    if format == "json":
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content=report.__dict__,
            headers={"Content-Disposition": f'attachment; filename="exxas_report_{job_id[:8]}.json"'},
        )
    else:
        md = report_to_markdown(report)
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            content=md,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="exxas_report_{job_id[:8]}.md"'},
        )


# ── 플랜 정보 ────────────────────────────────────────────────────────────────

@router.get("/plan/features")
async def get_plan_features(user: Optional[User] = Depends(get_optional_user)):
    """현재 플랜 기능 정보"""
    from ...models.user import PLAN_FEATURES, PlanType
    plan = user.plan if user else PlanType.free
    features = PLAN_FEATURES.get(plan, PLAN_FEATURES[PlanType.free])
    return {
        "plan": plan,
        "monthly_limit": user.monthly_limit if user else 5,
        "monthly_usage": user.monthly_usage if user else 0,
        "features": features,
    }


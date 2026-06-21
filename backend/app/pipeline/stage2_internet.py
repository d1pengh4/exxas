"""
Stage 2: 인터넷 역추적 — 한국 전용 v2
- 역방향 이미지 검색 8종 병렬
  1. Naver Vision Landmark API ★ (한국 랜드마크 직접 감지)
  2. Naver Vision OCR ★ (한국 간판/텍스트 직접 인식)
  3. Kakao Vision ★ (한국 특화)
  4. Google Lens (SerpAPI 또는 URL 경유)
  5. Yandex (업로드 → URL 경유 폴백)
  6. TinEye
  7. Bing Visual Search
  8. Naver SmartLens (웹 스크래핑)
- 이미지 호스팅 멀티 폴백 (catbox → tmpfiles → freeimage)
- 발견된 URL 딥 크롤 (네이버 블로그/플레이스, 카카오맵 좌표 추출)
"""
import asyncio
import base64
import hashlib
import tempfile
import os
from dataclasses import dataclass, field
from typing import Optional
import httpx
from loguru import logger
from ..core.config import settings


@dataclass
class ReverseSearchResult:
    source: str
    url: str = ""
    title: str = ""
    location_hint: str = ""
    date_found: str = ""
    confidence: float = 0.0


@dataclass
class InternetResult:
    reverse_search_results: list[ReverseSearchResult] = field(default_factory=list)
    wayback_first_seen: str = ""
    wayback_url: str = ""
    best_match: Optional[ReverseSearchResult] = None
    location_hints: list[str] = field(default_factory=list)
    naver_landmark: str = ""       # Naver Vision API 랜드마크 직접 감지 결과
    naver_landmark_lat: float = 0.0
    naver_landmark_lon: float = 0.0


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


async def run(image_bytes: bytes, image_hash: str = "") -> InternetResult:
    result = InternetResult()

    # ── Redis 캐시 체크 (24시간 TTL) ──────────────────────────────────────────
    cache_key = f"stage2:cache:{image_hash}" if image_hash else ""
    if cache_key:
        try:
            from ..core.database import get_redis
            import json as _json
            _r = await get_redis()
            cached_raw = await _r.get(cache_key)
            if cached_raw:
                cached = _json.loads(cached_raw)
                result.location_hints = cached.get("location_hints", [])
                result.naver_landmark = cached.get("naver_landmark", "")
                result.naver_landmark_lat = cached.get("naver_landmark_lat", 0.0)
                result.naver_landmark_lon = cached.get("naver_landmark_lon", 0.0)
                result.wayback_first_seen = cached.get("wayback_first_seen", "")
                result.wayback_url = cached.get("wayback_url", "")
                result.reverse_search_results = [
                    ReverseSearchResult(**r) for r in cached.get("results", [])
                ]
                if result.reverse_search_results:
                    result.best_match = max(result.reverse_search_results, key=lambda x: x.confidence)
                logger.info(f"[Stage2] 캐시 히트: {image_hash[:16]}... hints={result.location_hints[:2]}")
                return result
        except Exception as _ce:
            logger.debug(f"Stage2 cache read failed: {_ce}")

    # 이미지 공개 URL 미리 확보 (여러 검색엔진이 공유)
    image_url = await _upload_to_temp_host(image_bytes)

    tasks = [
        _search_naver_vision_landmark(image_bytes),   # ★ 한국 랜드마크 직접 감지
        _search_naver_vision_ocr(image_bytes),        # ★ 한국 간판/텍스트 OCR
        _search_kakao_vision(image_bytes),             # ★ 카카오 비전 (한국 특화)
        _search_google_lens(image_bytes, image_url),
        _search_yandex(image_bytes, image_url),
        _search_tineye(image_bytes),
        _search_bing_visual(image_bytes),
        _search_naver_smartlens(image_bytes, image_url),
    ]

    search_results = await asyncio.gather(*tasks, return_exceptions=True)
    all_results: list[ReverseSearchResult] = []

    for i, r in enumerate(search_results):
        if isinstance(r, Exception):
            logger.debug(f"Search task {i} failed: {r}")
        elif isinstance(r, list):
            all_results.extend(r)
        elif isinstance(r, dict) and "landmark" in r:
            # Naver Vision Landmark 특수 처리
            if r.get("landmark"):
                result.naver_landmark = r["landmark"]
                result.naver_landmark_lat = r.get("lat", 0.0)
                result.naver_landmark_lon = r.get("lon", 0.0)
                all_results.append(ReverseSearchResult(
                    source="naver_vision_landmark",
                    title=r["landmark"],
                    location_hint=r["landmark"],
                    confidence=r.get("confidence", 0.90),
                ))

    result.reverse_search_results = all_results

    # URL 크롤링 체인 (발견된 URL에서 위치 심층 추출)
    crawl_results = await _crawl_result_urls(all_results[:6])
    all_results.extend(crawl_results)

    # ── 결과 Redis 캐시 저장 ──────────────────────────────────────────────────
    if cache_key:
        try:
            from ..core.database import get_redis
            import json as _json
            _r = await get_redis()
            cache_data = {
                "location_hints": list({r.location_hint for r in all_results if r.location_hint}),
                "naver_landmark": result.naver_landmark,
                "naver_landmark_lat": result.naver_landmark_lat,
                "naver_landmark_lon": result.naver_landmark_lon,
                "wayback_first_seen": result.wayback_first_seen,
                "wayback_url": result.wayback_url,
                "results": [
                    {"source": r.source, "url": r.url, "title": r.title,
                     "location_hint": r.location_hint, "confidence": r.confidence,
                     "date_found": r.date_found}
                    for r in all_results[:20]
                ],
            }
            # 빈 결과(힌트 없음)는 1시간만 캐시, 유효 결과는 24시간
            _has_useful = bool(cache_data["location_hints"] or cache_data["naver_landmark"] or cache_data["results"])
            _ttl = 86400 if _has_useful else 3600
            await _r.setex(cache_key, _ttl, _json.dumps(cache_data, ensure_ascii=False))
            logger.debug(f"[Stage2] 결과 캐시 저장: {cache_key}")
        except Exception as _ce:
            logger.debug(f"Stage2 cache write failed: {_ce}")

    # Wayback Machine — catbox URL로 실제 이미지 검색
    if image_url:
        wayback = await _check_wayback(image_url)
        if wayback:
            result.wayback_first_seen = wayback.get("first_seen", "")
            result.wayback_url = wayback.get("url", "")

    hints = {r.location_hint for r in all_results if r.location_hint}
    result.location_hints = list(hints)

    if all_results:
        result.best_match = max(all_results, key=lambda x: x.confidence)

    logger.info(
        f"[Stage2] 역방향검색 완료: {len(all_results)}건, "
        f"hints={result.location_hints[:3]}, "
        f"landmark={result.naver_landmark or '없음'}"
    )
    return result


# ── 이미지 호스팅 (멀티 폴백) ──────────────────────────────────────────────────

async def _upload_to_temp_host(image_bytes: bytes) -> str:
    """
    무료 임시 이미지 호스팅 — 3개 서버 폴백
    1순위: catbox.moe (72시간)
    2순위: tmpfiles.org (60분)
    3순위: freeimage.host
    """
    import certifi
    ssl_ctx_opts = {"verify": certifi.where()}

    # 1순위: catbox.moe
    try:
        async with httpx.AsyncClient(timeout=15.0, **ssl_ctx_opts) as client:
            resp = await client.post(
                "https://catbox.moe/user/api.php",
                data={"reqtype": "fileupload"},
                files={"fileToUpload": ("image.jpg", image_bytes, "image/jpeg")},
            )
            if resp.status_code == 200:
                url = resp.text.strip()
                if url.startswith("https://"):
                    logger.debug(f"Upload: catbox.moe OK → {url[:60]}")
                    return url
    except Exception as e:
        logger.debug(f"catbox.moe failed: {e}")

    # 2순위: tmpfiles.org
    try:
        async with httpx.AsyncClient(timeout=12.0, **ssl_ctx_opts) as client:
            resp = await client.post(
                "https://tmpfiles.org/api/v1/upload",
                files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            )
            if resp.status_code == 200:
                data = resp.json()
                url = data.get("data", {}).get("url", "")
                # tmpfiles URL을 직접 접근 URL로 변환
                if url:
                    # https://tmpfiles.org/12345/image.jpg → https://tmpfiles.org/dl/12345/image.jpg
                    url = url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
                    logger.debug(f"Upload: tmpfiles.org OK → {url[:60]}")
                    return url
    except Exception as e:
        logger.debug(f"tmpfiles.org failed: {e}")

    # 3순위: freeimage.host (base64)
    try:
        b64 = base64.b64encode(image_bytes).decode()
        async with httpx.AsyncClient(timeout=12.0, **ssl_ctx_opts) as client:
            resp = await client.post(
                "https://freeimage.host/api/1/upload",
                data={"key": "6d207e02198a847aa98d0a2a901485a5", "action": "upload", "source": b64},
            )
            if resp.status_code == 200:
                data = resp.json()
                url = data.get("image", {}).get("url", "")
                if url:
                    logger.debug(f"Upload: freeimage.host OK → {url[:60]}")
                    return url
    except Exception as e:
        logger.debug(f"freeimage.host failed: {e}")

    logger.warning("모든 임시 이미지 호스팅 실패")
    return ""


# ── Naver Vision Landmark API ★ ────────────────────────────────────────────────

async def _search_naver_vision_landmark(image_bytes: bytes) -> dict:
    """
    네이버 Vision API — 랜드마크 감지
    한국 주요 랜드마크 직접 인식 (좌표 포함)
    API: https://openapi.naver.com/v1/vision/landmark
    """
    if not (settings.NAVER_CLIENT_ID and settings.NAVER_CLIENT_SECRET):
        return {}

    import certifi
    try:
        async with httpx.AsyncClient(
            timeout=15.0, verify=certifi.where()
        ) as client:
            resp = await client.post(
                "https://openapi.naver.com/v1/vision/landmark",
                headers={
                    "X-Naver-Client-Id": settings.NAVER_CLIENT_ID,
                    "X-Naver-Client-Secret": settings.NAVER_CLIENT_SECRET,
                    "Content-Type": "application/octet-stream",
                },
                content=image_bytes,
            )

            if resp.status_code != 200:
                logger.debug(f"Naver Vision Landmark: status {resp.status_code}")
                # multipart 방식도 시도
                resp2 = await client.post(
                    "https://openapi.naver.com/v1/vision/landmark",
                    headers={
                        "X-Naver-Client-Id": settings.NAVER_CLIENT_ID,
                        "X-Naver-Client-Secret": settings.NAVER_CLIENT_SECRET,
                    },
                    files={"image": ("image.jpg", image_bytes, "image/jpeg")},
                )
                if resp2.status_code != 200:
                    return {}
                resp = resp2

            data = resp.json()
            landmarks = data.get("landmarks", []) or data.get("result", {}).get("landmarks", [])

            if not landmarks:
                return {}

            best = max(landmarks, key=lambda x: x.get("confidence", 0))
            name = best.get("name", "")
            confidence = float(best.get("confidence", 0))

            # 좌표 추출 (일부 응답에 포함)
            lat = float(best.get("lat", 0) or best.get("latitude", 0) or 0)
            lon = float(best.get("lng", 0) or best.get("longitude", 0) or 0)

            logger.info(f"[Naver Vision Landmark] 감지: {name} (신뢰도 {confidence:.2%})")
            return {
                "landmark": name,
                "confidence": min(confidence, 0.95),
                "lat": lat,
                "lon": lon,
            }

    except Exception as e:
        logger.debug(f"Naver Vision Landmark failed: {e}")
        return {}


# ── Naver Vision OCR API ★ ─────────────────────────────────────────────────────

async def _search_naver_vision_ocr(image_bytes: bytes) -> list[ReverseSearchResult]:
    """
    네이버 Vision API — OCR (간판·현수막·번호판 한글 텍스트 직접 인식)
    API: https://openapi.naver.com/v1/vision/ocr
    Stage 3 OCR과 독립적으로 실행 → 교차 검증
    """
    if not (settings.NAVER_CLIENT_ID and settings.NAVER_CLIENT_SECRET):
        return []

    import certifi
    results: list[ReverseSearchResult] = []
    try:
        async with httpx.AsyncClient(timeout=12.0, verify=certifi.where()) as client:
            resp = await client.post(
                "https://openapi.naver.com/v1/vision/ocr",
                headers={
                    "X-Naver-Client-Id": settings.NAVER_CLIENT_ID,
                    "X-Naver-Client-Secret": settings.NAVER_CLIENT_SECRET,
                },
                files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            # 응답 구조: {"result": {"text": "...", "fields": [{"inferText": "..."}]}}
            raw_text = data.get("result", {}).get("text", "") or ""
            fields = data.get("result", {}).get("fields", []) or []

            # 필드별 텍스트 수집
            all_texts: list[str] = []
            if raw_text:
                all_texts.append(raw_text)
            for f in fields:
                t = f.get("inferText", "").strip()
                if t and len(t) >= 2:
                    all_texts.append(t)

            if not all_texts:
                return []

            combined = " ".join(all_texts[:30])
            hint = _extract_location_hint(combined)
            logger.info(f"[Naver Vision OCR] 텍스트 {len(all_texts)}개 감지, hint={hint}")

            if hint:
                results.append(ReverseSearchResult(
                    source="naver_vision_ocr",
                    title=combined[:120],
                    location_hint=hint,
                    confidence=0.78,
                ))

            # 텍스트 조각 중 위치 키워드 탐색
            for t in all_texts:
                t_hint = _extract_location_hint(t)
                if t_hint and t_hint != hint:
                    results.append(ReverseSearchResult(
                        source="naver_vision_ocr",
                        title=t[:80],
                        location_hint=t_hint,
                        confidence=0.72,
                    ))
                    hint = t_hint  # 중복 방지

    except Exception as e:
        logger.debug(f"Naver Vision OCR failed: {e}")
    return results[:5]


# ── Kakao Vision API ★ ─────────────────────────────────────────────────────────

async def _search_kakao_vision(image_bytes: bytes) -> list[ReverseSearchResult]:
    """
    카카오 Vision API — 장면 이해 + OCR (한국 이미지 특화)
    - product: 상품/간판 인식
    - face: (사용 안함)
    - ocr: 한글 텍스트 추출
    API: https://dapi.kakao.com/v2/vision/
    """
    from ..core.config import settings as _s
    if not _s.KAKAO_API_KEY:
        return []

    import certifi
    results: list[ReverseSearchResult] = []

    headers_base = {
        "Authorization": f"KakaoAK {_s.KAKAO_API_KEY}",
    }

    # 1. Kakao OCR
    try:
        async with httpx.AsyncClient(timeout=12.0, verify=certifi.where()) as client:
            resp = await client.post(
                "https://dapi.kakao.com/v2/vision/text/ocr",
                headers=headers_base,
                files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            )
            if resp.status_code == 200:
                data = resp.json()
                words = [r.get("recognition_words", []) for r in data.get("result", [])]
                texts = [w for group in words for w in group if isinstance(w, str) and len(w) >= 2]
                combined = " ".join(texts[:30])
                hint = _extract_location_hint(combined)
                if hint:
                    results.append(ReverseSearchResult(
                        source="kakao_vision_ocr",
                        title=combined[:120],
                        location_hint=hint,
                        confidence=0.76,
                    ))
                    logger.info(f"[Kakao Vision OCR] hint={hint}")
            elif resp.status_code == 401:
                # Bearer 토큰 폴백
                if _s.KAKAO_ACCESS_TOKEN:
                    resp2 = await client.post(
                        "https://dapi.kakao.com/v2/vision/text/ocr",
                        headers={"Authorization": f"Bearer {_s.KAKAO_ACCESS_TOKEN}"},
                        files={"image": ("image.jpg", image_bytes, "image/jpeg")},
                    )
                    if resp2.status_code == 200:
                        data2 = resp2.json()
                        words2 = [r.get("recognition_words", []) for r in data2.get("result", [])]
                        texts2 = [w for group in words2 for w in group if isinstance(w, str) and len(w) >= 2]
                        combined2 = " ".join(texts2[:30])
                        hint2 = _extract_location_hint(combined2)
                        if hint2:
                            results.append(ReverseSearchResult(
                                source="kakao_vision_ocr",
                                title=combined2[:120],
                                location_hint=hint2,
                                confidence=0.76,
                            ))
    except Exception as e:
        logger.debug(f"Kakao Vision OCR failed: {e}")

    # 2. Kakao Scene 이해 (product/scene multi-tag)
    try:
        async with httpx.AsyncClient(timeout=12.0, verify=certifi.where()) as client:
            resp = await client.post(
                "https://dapi.kakao.com/v2/vision/multitag/generate",
                headers=headers_base,
                files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            )
            if resp.status_code == 200:
                data = resp.json()
                tags = [t.get("label", "") for t in data.get("result", {}).get("label", [])
                        if t.get("confidence", 0) > 0.3]
                combined_tags = " ".join(tags)
                hint = _extract_location_hint(combined_tags)
                if hint:
                    results.append(ReverseSearchResult(
                        source="kakao_vision_scene",
                        title=f"장면 태그: {combined_tags[:80]}",
                        location_hint=hint,
                        confidence=0.55,
                    ))
    except Exception as e:
        logger.debug(f"Kakao Vision Scene failed: {e}")

    return results[:4]


# ── Naver SmartLens ────────────────────────────────────────────────────────────

async def _search_naver_smartlens(image_bytes: bytes, image_url: str = "") -> list[ReverseSearchResult]:
    """
    네이버 SmartLens 역이미지 검색 — 한국 이미지에 특화
    웹 스크래핑 방식 (API 키 불필요)
    """
    import certifi
    results: list[ReverseSearchResult] = []

    # 방법 1: Naver Image Search with URL
    if image_url:
        try:
            async with httpx.AsyncClient(
                timeout=20.0,
                verify=certifi.where(),
                headers={**_HEADERS, "Referer": "https://www.naver.com/"},
                follow_redirects=True,
            ) as client:
                resp = await client.get(
                    "https://search.naver.com/search.naver",
                    params={
                        "where": "image",
                        "query": "",
                        "sm": "tab_smr.top",
                        "actype": "image",
                        "url": image_url,
                    },
                )
                if resp.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for tag in soup.find_all(["strong", "span", "a", "p"], limit=50):
                        text = tag.get_text(strip=True)
                        if not text or len(text) < 3:
                            continue
                        hint = _extract_location_hint(text)
                        if hint:
                            results.append(ReverseSearchResult(
                                source="naver_image",
                                url=tag.get("href", ""),
                                title=text[:100],
                                location_hint=hint,
                                confidence=0.68,
                            ))
                    if results:
                        logger.debug(f"Naver SmartLens URL: {len(results)}건")
                        return results[:6]
        except Exception as e:
            logger.debug(f"Naver SmartLens URL failed: {e}")

    # 방법 2: Naver Smart Lens 직접 업로드
    try:
        async with httpx.AsyncClient(
            timeout=20.0,
            verify=certifi.where(),
            headers={
                **_HEADERS,
                "Referer": "https://www.naver.com/",
                "Origin": "https://www.naver.com",
            },
            follow_redirects=True,
        ) as client:
            # SmartLens 업로드 엔드포인트
            resp = await client.post(
                "https://s.search.naver.com/imagesearch/api/upload",
                files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            )
            if resp.status_code == 200:
                data = resp.json()
                search_url = data.get("url") or data.get("imageUrl", "")
                if search_url:
                    # 검색 결과 페이지 가져오기
                    resp2 = await client.get(search_url)
                    if resp2.status_code == 200:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(resp2.text, "html.parser")
                        for tag in soup.find_all(["a", "strong", "h3"], limit=40):
                            text = tag.get_text(strip=True)
                            hint = _extract_location_hint(text)
                            if hint or len(text) > 8:
                                results.append(ReverseSearchResult(
                                    source="naver_smartlens",
                                    url=tag.get("href", ""),
                                    title=text[:100],
                                    location_hint=hint,
                                    confidence=0.70,
                                ))
                        logger.debug(f"Naver SmartLens upload: {len(results)}건")
                        return results[:6]
    except Exception as e:
        logger.debug(f"Naver SmartLens upload failed: {e}")

    return results


# ── Google Lens ────────────────────────────────────────────────────────────────

async def _search_google_lens(image_bytes: bytes, image_url: str = "") -> list[ReverseSearchResult]:
    """Google Lens — SerpAPI 우선, 없으면 직접 검색"""
    if settings.SERP_API_KEY:
        try:
            result = await _google_lens_via_serp(image_bytes, image_url)
            if result:
                return result
        except Exception as e:
            logger.debug(f"Google Lens SerpAPI failed: {e}")

    return await _google_lens_direct(image_url)


async def _google_lens_direct(image_url: str) -> list[ReverseSearchResult]:
    """Google Images by URL (직접 업로드 대신 URL 경유)"""
    if not image_url:
        return []
    import certifi
    try:
        async with httpx.AsyncClient(
            timeout=20.0, headers=_HEADERS,
            verify=certifi.where(), follow_redirects=True,
        ) as client:
            resp = await client.get(
                "https://www.google.com/searchbyimage",
                params={"image_url": image_url, "hl": "ko", "gl": "kr"},
            )
            if resp.status_code not in (200, 301, 302):
                return []
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            seen: set[str] = set()
            for tag in soup.find_all(["h3", "span", "a"], limit=50):
                text = tag.get_text(strip=True)
                href = tag.get("href", "")
                if not text or len(text) < 4 or href in seen:
                    continue
                seen.add(href)
                hint = _extract_location_hint(text)
                if hint or len(text) > 10:
                    results.append(ReverseSearchResult(
                        source="google_images",
                        url=href if href.startswith("http") else "",
                        title=text[:120],
                        location_hint=hint,
                        confidence=0.62,
                    ))
            return results[:8]
    except Exception as e:
        logger.debug(f"Google Images direct failed: {e}")
        return []


async def _google_lens_via_serp(image_bytes: bytes, image_url: str = "") -> list[ReverseSearchResult]:
    """SerpAPI Google Lens"""
    import certifi
    url = image_url or await _upload_to_temp_host(image_bytes)
    if not url:
        return []

    try:
        async with httpx.AsyncClient(timeout=20.0, verify=certifi.where()) as client:
            resp = await client.get(
                "https://serpapi.com/search",
                params={
                    "engine": "google_lens",
                    "url": url,
                    "api_key": settings.SERP_API_KEY,
                    "hl": "ko",
                    "gl": "kr",
                },
            )
            data = resp.json()
            if "error" in data and "visual_matches" not in data:
                return []

            results = []
            # knowledge_graph
            kg = data.get("knowledge_graph", {})
            if kg.get("title"):
                kg_text = kg["title"]
                if kg.get("subtitle"): kg_text += " " + kg["subtitle"]
                if kg.get("description"): kg_text += " " + kg["description"][:80]
                results.append(ReverseSearchResult(
                    source="google_lens_kg",
                    url=kg.get("website", ""),
                    title=kg_text[:150],
                    location_hint=_extract_location_hint(kg_text),
                    confidence=0.88,
                ))
            # text_results
            for tr in data.get("text_results", [])[:3]:
                text = tr.get("title", "") or tr.get("snippet", "")
                if text:
                    results.append(ReverseSearchResult(
                        source="google_lens_text",
                        url=tr.get("link", ""),
                        title=text[:120],
                        location_hint=_extract_location_hint(text),
                        confidence=0.80,
                    ))
            # visual_matches — title + snippet 모두 파싱
            for item in data.get("visual_matches", [])[:8]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                text = (title + " " + snippet).strip()
                if text:
                    results.append(ReverseSearchResult(
                        source="google_lens",
                        url=item.get("link", ""),
                        title=text[:150],
                        location_hint=_extract_location_hint(text),
                        confidence=0.72,
                    ))
            # entities
            for ent in data.get("entities", [])[:3]:
                name = ent.get("name", "")
                if name:
                    results.append(ReverseSearchResult(
                        source="google_lens_entity",
                        title=name,
                        location_hint=_extract_location_hint(name),
                        confidence=0.70,
                    ))

            logger.info(f"SerpAPI Google Lens: {len(results)}건")
            return results
    except Exception as e:
        logger.debug(f"SerpAPI Lens failed: {e}")
        return []


# ── Yandex ────────────────────────────────────────────────────────────────────

async def _search_yandex(image_bytes: bytes, image_url: str = "") -> list[ReverseSearchResult]:
    """Yandex 역이미지 — 직접 업로드 → URL 경유 → Playwright 순"""
    try:
        result = await _yandex_via_upload(image_bytes)
        if result: return result
    except Exception as e:
        logger.debug(f"Yandex upload failed: {e}")

    try:
        url = image_url or await _upload_to_temp_host(image_bytes)
        result = await _yandex_via_url(url)
        if result: return result
    except Exception as e:
        logger.debug(f"Yandex URL failed: {e}")

    return await _yandex_via_playwright(image_bytes)


async def _yandex_via_url(image_url: str) -> list[ReverseSearchResult]:
    if not image_url:
        return []
    async with httpx.AsyncClient(
        timeout=20.0, headers=_HEADERS, follow_redirects=True
    ) as client:
        resp = await client.get(
            "https://yandex.com/images/search",
            params={"url": image_url, "rpt": "imageview", "hl": "ko"},
        )
        if resp.status_code != 200:
            return []
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        results = _parse_yandex_results(soup)

        # Sites 탭 추가 파싱
        for a in soup.find_all("a", href=True):
            if "Sites" in a.get_text() or "Сайты" in a.get_text():
                href = a["href"]
                if href.startswith("/"):
                    try:
                        sites_resp = await client.get("https://yandex.com" + href)
                        if sites_resp.status_code == 200:
                            soup2 = BeautifulSoup(sites_resp.text, "html.parser")
                            results.extend(_parse_yandex_results(soup2))
                    except Exception:
                        pass
                break

    return results[:8]


async def _yandex_via_upload(image_bytes: bytes) -> list[ReverseSearchResult]:
    async with httpx.AsyncClient(
        timeout=20.0, headers=_HEADERS, follow_redirects=True
    ) as client:
        upload = await client.post(
            "https://yandex.com/images-apphost/image-download",
            params={"cbird": "111", "images_avatars_size": "preview", "images_avatars_namespace": "mtime"},
            files={"upfile": ("image.jpg", image_bytes, "image/jpeg")},
            headers={**_HEADERS, "Referer": "https://yandex.com/images/", "Origin": "https://yandex.com"},
        )
        if upload.status_code != 200:
            return []
        data = upload.json()
        image_url = data.get("image_url") or data.get("url", "")
        if not image_url:
            return []
        search = await client.get(
            "https://yandex.com/images/search",
            params={"url": image_url, "rpt": "imageview", "hl": "ko"},
        )
        if search.status_code != 200:
            return []
        from bs4 import BeautifulSoup
        return _parse_yandex_results(BeautifulSoup(search.text, "html.parser"))


async def _yandex_via_playwright(image_bytes: bytes) -> list[ReverseSearchResult]:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return []
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
            ctx = await browser.new_context(user_agent=_HEADERS["User-Agent"], locale="ko-KR")
            page = await ctx.new_page()
            await page.goto("https://yandex.com/images/", wait_until="domcontentloaded", timeout=20000)
            file_input = page.locator('input[type="file"]')
            if await file_input.count() > 0:
                await file_input.set_input_files(tmp_path)
            else:
                async with page.expect_file_chooser(timeout=8000) as fc_info:
                    await page.locator('[data-bem*="camera"], .input__upload, button[aria-label*="image"]').first.click(timeout=5000)
                fc = await fc_info.value
                await fc.set_files(tmp_path)
            await page.wait_for_load_state("networkidle", timeout=20000)
            from bs4 import BeautifulSoup
            results = _parse_yandex_results(BeautifulSoup(await page.content(), "html.parser"))
            await browser.close()
            return results
    except Exception as e:
        logger.debug(f"Yandex Playwright failed: {e}")
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _parse_yandex_results(soup) -> list[ReverseSearchResult]:
    results = []
    seen: set[str] = set()

    for tag in soup.find_all("a", href=True, limit=60):
        href = tag["href"]
        text = tag.get_text(strip=True)
        if not href.startswith("http") or "yandex." in href or href in seen:
            continue
        seen.add(href)
        hint = _extract_location_hint(text)
        results.append(ReverseSearchResult(
            source="yandex", url=href, title=text[:100],
            location_hint=hint, confidence=0.58,
        ))

    for tag in soup.find_all("a", href=True):
        href = tag.get("href", "")
        text = tag.get_text(strip=True)
        if "images/search?text=" in href and text and len(text) > 2:
            hint = _extract_location_hint(text)
            t = text.lower()
            if any(k in t for k in ("корейск", "korean", "korea", "한국", "seoul", "busan")):
                hint = hint or "한국"
            elif any(k in t for k in ("japan", "tokyo", "japон")):
                hint = hint or "일본"
            if hint:
                results.append(ReverseSearchResult(
                    source="yandex_tag", title=f"Yandex 태그: {text}",
                    location_hint=hint, confidence=0.48,
                ))

    return results[:8]


# ── TinEye ────────────────────────────────────────────────────────────────────

async def _search_tineye(image_bytes: bytes) -> list[ReverseSearchResult]:
    try:
        async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS, follow_redirects=True) as client:
            resp = await client.post(
                "https://tineye.com/search/",
                data={"url": ""},
                files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            )
            if resp.status_code not in (200, 302):
                return []
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for item in soup.find_all("div", class_=lambda c: c and "match" in c.lower())[:5]:
                link = item.find("a", href=True)
                title_el = item.find(["h3", "p", "span"])
                if link:
                    results.append(ReverseSearchResult(
                        source="tineye",
                        url=link["href"],
                        title=(title_el.get_text(strip=True) if title_el else "")[:100],
                        confidence=0.68,
                    ))
            return results
    except Exception as e:
        logger.debug(f"TinEye failed: {e}")
        return []


# ── Bing Visual Search ─────────────────────────────────────────────────────────

async def _search_bing_visual(image_bytes: bytes) -> list[ReverseSearchResult]:
    if not settings.BING_SEARCH_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.bing.microsoft.com/v7.0/images/visualsearch",
                headers={"Ocp-Apim-Subscription-Key": settings.BING_SEARCH_API_KEY},
                files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            results = []
            for tag in data.get("tags", [])[:3]:
                for action in tag.get("actions", []):
                    if action.get("actionType") == "PagesIncluding":
                        for item in action.get("data", {}).get("value", [])[:3]:
                            title = item.get("name", "")
                            results.append(ReverseSearchResult(
                                source="bing_visual",
                                url=item.get("contentUrl", ""),
                                title=title,
                                location_hint=_extract_location_hint(title),
                                confidence=0.62,
                            ))
            return results
    except Exception as e:
        logger.debug(f"Bing Visual failed: {e}")
        return []


# ── Wayback Machine ────────────────────────────────────────────────────────────

async def _check_wayback(image_url: str) -> dict:
    """Wayback Machine CDX API로 이미지 URL의 최초 아카이브 날짜 조회.
    catbox.moe 등 임시 호스팅 URL을 직접 검색해 원본 이미지가 인터넷에 언제 처음 올라왔는지 확인."""
    if not image_url or not image_url.startswith("http"):
        return {}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                "http://web.archive.org/cdx/search/cdx",
                params={
                    "output": "json",
                    "limit": 1,
                    "fl": "timestamp,original",
                    "filter": "statuscode:200",
                    "url": image_url,
                    "matchType": "exact",
                    "from": "20000101",
                    "to": "20991231",
                    "collapse": "digest",
                },
            )
            if resp.status_code != 200:
                return {}
            data = resp.json()
            if len(data) > 1:
                ts, original_url = data[1]
                return {
                    "first_seen": ts,
                    "url": f"https://web.archive.org/web/{ts}/{original_url}",
                }
    except Exception as e:
        logger.debug(f"Wayback failed: {e}")
    return {}


# ── URL 크롤링 체인 ────────────────────────────────────────────────────────────

async def _crawl_result_urls(results: list[ReverseSearchResult]) -> list[ReverseSearchResult]:
    crawl_tasks = [
        _crawl_single_url(r.url, r.source)
        for r in results if r.url and r.url.startswith("http")
    ]
    if not crawl_tasks:
        return []
    crawled = await asyncio.gather(*crawl_tasks, return_exceptions=True)
    return [c for c in crawled if isinstance(c, ReverseSearchResult) and c.location_hint]


_NON_GEO_DOMAINS = {
    "namu.wiki", "wikipedia.org", "wikimedia.org",
    "kream.co.kr", "musinsa.com", "coupang.com", "gmarket.co.kr",
    "youtube.com", "youtu.be", "tiktok.com",
    "twitter.com", "x.com", "reddit.com",
    "naver.com/music", "melon.com", "bugs.co.kr",
    "imdb.com", "allmusic.com", "discogs.com",
}

async def _crawl_single_url(url: str, source: str) -> Optional[ReverseSearchResult]:
    try:
        import re
        # 위치 정보와 무관한 사이트는 크롤 스킵
        from urllib.parse import urlparse
        _host = urlparse(url).netloc.lstrip("www.")
        if any(_host == d or _host.endswith("." + d) for d in _NON_GEO_DOMAINS):
            return None
        async with httpx.AsyncClient(
            timeout=8.0, headers=_HEADERS, follow_redirects=True
        ) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text[:60000], "html.parser")

            # 1. meta geo.position (GPS 좌표)
            geo_meta = soup.find("meta", attrs={"name": "geo.position"})
            if geo_meta:
                coords = geo_meta.get("content", "")
                if ";" in coords:
                    lat_v, lon_v = coords.split(";", 1)
                    return ReverseSearchResult(
                        source=f"{source}_crawl", url=url,
                        title=f"메타 GPS: {coords}",
                        location_hint=f"좌표: {lat_v.strip()}, {lon_v.strip()}",
                        confidence=0.82,
                    )

            # 2. 네이버 플레이스 URL에서 좌표 추출
            if "place.naver.com" in url or "map.naver.com" in url:
                coords_m = re.search(r"lat[=%]([0-9.]+).*?[&,lng|lon][=%]([0-9.]+)", url)
                if coords_m:
                    return ReverseSearchResult(
                        source="naver_place_crawl", url=url,
                        title="네이버 플레이스 좌표",
                        location_hint=f"좌표: {coords_m.group(1)}, {coords_m.group(2)}",
                        confidence=0.88,
                    )
                # 네이버 플레이스 페이지에서 장소명 추출
                place_name = soup.find("span", class_=lambda c: c and "place" in str(c).lower())
                if place_name:
                    name = place_name.get_text(strip=True)
                    hint = _extract_location_hint(name) or name[:50]
                    return ReverseSearchResult(
                        source="naver_place_crawl", url=url,
                        title=name, location_hint=hint, confidence=0.80,
                    )

            # 3. 카카오맵 URL에서 좌표 추출
            if "kakaomap.com" in url or "map.kakao.com" in url:
                coords_m = re.search(r"[?&]map_type=\w+&lat=([0-9.]+)&lng=([0-9.]+)", url)
                if not coords_m:
                    coords_m = re.search(r"@([0-9.]+),([0-9.]+)", url)
                if coords_m:
                    return ReverseSearchResult(
                        source="kakao_map_crawl", url=url,
                        title="카카오맵 좌표",
                        location_hint=f"좌표: {coords_m.group(1)}, {coords_m.group(2)}",
                        confidence=0.88,
                    )

            # 4. Open Graph locality
            og_loc = soup.find("meta", property="og:locality") or soup.find("meta", attrs={"name": "locality"})
            if og_loc:
                loc = og_loc.get("content", "")
                if loc:
                    hint = _extract_location_hint(loc) or loc[:50]
                    return ReverseSearchResult(
                        source=f"{source}_crawl", url=url,
                        title=f"OG locality: {loc}",
                        location_hint=hint, confidence=0.75,
                    )

            # 5. 네이버 블로그 지도 링크
            if "blog.naver.com" in url or "m.blog.naver.com" in url:
                map_link = soup.find("a", href=lambda h: h and "maps.naver.com" in h)
                if map_link:
                    map_url = map_link.get("href", "")
                    coords_m = re.search(r"lat=([0-9.]+).*lng=([0-9.]+)", map_url)
                    if coords_m:
                        return ReverseSearchResult(
                            source="naver_blog_map", url=url,
                            title="네이버 블로그 지도 링크",
                            location_hint=f"좌표: {coords_m.group(1)}, {coords_m.group(2)}",
                            confidence=0.85,
                        )

            # 6. 본문 텍스트 키워드 탐색
            og_title_tag = soup.find("meta", property="og:title") or soup.find("title")
            og_title = og_title_tag.get("content", og_title_tag.get_text("")) if og_title_tag else ""
            og_title = og_title[:100].strip()

            text = soup.get_text(" ", strip=True)[:4000]
            hint = _extract_location_hint(text)
            if hint:
                return ReverseSearchResult(
                    source=f"{source}_crawl", url=url,
                    title=og_title or text[:80],
                    location_hint=hint, confidence=0.55,
                )

            # 7. OG 타이틀에 POI 키워드
            _POI_KW = ["샤브", "스테이크", "레스토랑", "맛집", "카페", "컨벤션", "호텔",
                       "롯데", "현대", "신세계", "센트럴", "강남", "홍대", "해운대", "잠실"]
            if og_title and any(k in og_title for k in _POI_KW):
                hint = _extract_location_hint(og_title) or og_title[:50]
                return ReverseSearchResult(
                    source=f"{source}_crawl", url=url,
                    title=og_title, location_hint=hint, confidence=0.52,
                )

    except Exception as e:
        logger.debug(f"URL crawl failed ({url[:50]}): {e}")
    return None


# ── 위치 키워드 사전 ───────────────────────────────────────────────────────────

_LOC_KEYWORDS = {
    # 서울 주요 지역
    "서울": "한국 서울", "Seoul": "한국 서울",
    "강남": "한국 서울 강남구", "Gangnam": "한국 서울 강남구",
    "역삼": "한국 서울 강남구 역삼동", "삼성동": "한국 서울 강남구 삼성동",
    "청담": "한국 서울 강남구 청담동", "압구정": "한국 서울 강남구 압구정동",
    "서초": "한국 서울 서초구", "반포": "한국 서울 서초구 반포",
    "센트럴시티": "한국 서울 서초구 반포 센트럴시티",
    "홍대": "한국 서울 마포구 홍대", "Hongdae": "한국 서울 마포구 홍대",
    "합정": "한국 서울 마포구 합정", "상수": "한국 서울 마포구 상수",
    "마포": "한국 서울 마포구",
    "명동": "한국 서울 중구 명동", "Myeongdong": "한국 서울 중구 명동",
    "종로": "한국 서울 종로구", "인사동": "한국 서울 종로구 인사동",
    "이태원": "한국 서울 용산구 이태원", "Itaewon": "한국 서울 용산구 이태원",
    "용산": "한국 서울 용산구",
    "여의도": "한국 서울 영등포구 여의도", "Yeouido": "한국 서울 영등포구 여의도",
    "영등포": "한국 서울 영등포구",
    "잠실": "한국 서울 송파구 잠실", "Jamsil": "한국 서울 송파구 잠실",
    "송파": "한국 서울 송파구",
    "동대문": "한국 서울 동대문구 동대문", "Dongdaemun": "한국 서울 동대문구",
    "성수": "한국 서울 성동구 성수동",
    "왕십리": "한국 서울 성동구 왕십리",
    "신촌": "한국 서울 서대문구 신촌",
    "은평": "한국 서울 은평구",
    "노원": "한국 서울 노원구",
    "강북": "한국 서울 강북구",
    "도봉": "한국 서울 도봉구",
    "강동": "한국 서울 강동구",
    "관악": "한국 서울 관악구",
    "한강": "한국 서울 한강공원", "Hangang": "한국 서울 한강공원",
    # 부산
    "부산": "한국 부산", "Busan": "한국 부산", "Pusan": "한국 부산",
    "해운대": "한국 부산 해운대", "Haeundae": "한국 부산 해운대",
    "광안리": "한국 부산 광안리", "Gwangalli": "한국 부산 광안리",
    "남포동": "한국 부산 남포동",
    "서면": "한국 부산 서면",
    "기장": "한국 부산 기장",
    "감천": "한국 부산 감천문화마을",
    # 기타 주요 도시
    "인천": "한국 인천", "Incheon": "한국 인천",
    "송도": "한국 인천 송도",
    "대구": "한국 대구", "Daegu": "한국 대구",
    "동성로": "한국 대구 동성로",
    "대전": "한국 대전", "Daejeon": "한국 대전",
    "광주": "한국 광주", "Gwangju": "한국 광주",
    "울산": "한국 울산", "Ulsan": "한국 울산",
    "수원": "한국 수원", "Suwon": "한국 수원",
    "성남": "한국 성남", "판교": "한국 성남 판교",
    "분당": "한국 성남 분당",
    "고양": "한국 경기 고양",
    "일산": "한국 경기 고양 일산",
    "제주": "한국 제주", "Jeju": "한국 제주",
    # 한국 일반
    "한국": "한국", "Korean": "한국", "Korea": "한국",
    "корейск": "한국",   # 러시아어 "Korean"
    "한국인": "한국",
    # 랜드마크
    "경복궁": "한국 서울 종로구 경복궁",
    "남산": "한국 서울 용산구 남산",
    "롯데월드": "한국 서울 송파구 잠실 롯데월드",
    "코엑스": "한국 서울 강남구 코엑스",
    "에버랜드": "한국 경기 용인 에버랜드",
    "카이스트": "한국 대전 KAIST",
    "ICC컨벤션": "한국 대전 ICC컨벤션홀",
    # 일본
    "도쿄": "일본 도쿄", "Tokyo": "일본 도쿄", "東京": "일본 도쿄",
    "오사카": "일본 오사카", "Osaka": "일본 오사카", "大阪": "일본 오사카",
    "교토": "일본 교토", "Kyoto": "일본 교토", "京都": "일본 교토",
    "Japan": "일본", "日本": "일본",
    # 중국
    "北京": "중국 베이징", "Beijing": "중국 베이징",
    "上海": "중국 상하이", "Shanghai": "중국 상하이",
    "China": "중국", "中国": "중국",
}


def _extract_location_hint(text: str) -> str:
    if not text:
        return ""
    # 1. 키워드 사전 매칭
    for kw, loc in _LOC_KEYWORDS.items():
        if kw in text:
            return loc
    # 2. 한국어 주소 패턴 직접 추출 (시/군/구 + 동/읍/면/로/길)
    import re as _re
    _addr_pat = _re.search(
        r'([가-힣]+(?:특별시|광역시|특별자치시|도|특별자치도)?)\s*'
        r'([가-힣]+(?:시|군|구))\s*'
        r'([가-힣]+(?:동|읍|면|로|길|대로))',
        text
    )
    if _addr_pat:
        return _addr_pat.group(0).strip()
    # 3. "시 구" 수준 패턴
    _city_pat = _re.search(r'([가-힣]{2,5}(?:시|군))\s+([가-힣]{2,5}구)', text)
    if _city_pat:
        return _city_pat.group(0).strip()
    return ""

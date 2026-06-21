"""
Stage 3: OCR → GIS 매핑
- 다국어 텍스트 추출 (PaddleOCR + EasyOCR)
- 언어 감지 → 국가 필터링
- 도로명/POI/번호판/우편번호 → GIS API 교차검증
- 네이버/카카오/Google Places 매칭
"""
import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional
import httpx
from loguru import logger
from ..core.config import settings


@dataclass
class OCRText:
    text: str
    confidence: float
    bbox: list = field(default_factory=list)
    language: str = ""


@dataclass
class POIMatch:
    name: str
    address: str
    latitude: float
    longitude: float
    source: str  # "naver" | "kakao" | "google"
    confidence: float = 0.0


@dataclass
class OCRGISResult:
    # OCR 결과
    all_texts: list[OCRText] = field(default_factory=list)
    detected_languages: list[str] = field(default_factory=list)
    country_hints: list[str] = field(default_factory=list)

    # 번호판
    license_plates: list[str] = field(default_factory=list)
    plate_country: str = ""

    # 우편번호
    postal_codes: list[str] = field(default_factory=list)
    postal_country: str = ""

    # GIS 매칭 결과
    poi_matches: list[POIMatch] = field(default_factory=list)
    best_match: Optional[POIMatch] = None

    # 주소 구성요소
    road_names: list[str] = field(default_factory=list)
    building_names: list[str] = field(default_factory=list)

    # ── 실내/소형 객체 특화 ──
    phone_numbers: list[str] = field(default_factory=list)
    phone_regions: list[str] = field(default_factory=list)
    business_reg_numbers: list[str] = field(default_factory=list)
    brand_names: list[tuple] = field(default_factory=list)       # [(brand, location)]
    currency_hints: list[str] = field(default_factory=list)
    address_fragments: list[str] = field(default_factory=list)
    document_type: str = ""
    barcodes: list[str] = field(default_factory=list)

    # ── 한국 전용 분석 ──
    korea_analysis: Optional[dict] = None


async def run(image_bytes: bytes) -> OCRGISResult:
    from ..services.indoor_analyzer import (
        extract_receipt_signals, brand_quick_lookup,
        detect_document_type,
    )

    result = OCRGISResult()

    # OCR 실행
    texts = await _run_ocr(image_bytes)
    result.all_texts = texts

    if not texts:
        return result

    # 언어 감지
    result.detected_languages = _detect_languages(texts)
    result.country_hints = _languages_to_countries(result.detected_languages)

    text_list = [t.text for t in texts]
    all_text_str = " ".join(text_list)

    # 번호판 탐지
    plates = _detect_license_plates(all_text_str)
    result.license_plates = plates
    if plates:
        result.plate_country = _plate_to_country(plates[0])

    # 우편번호 탐지
    postal = _detect_postal_codes(all_text_str, result.detected_languages)
    result.postal_codes = postal

    # ── 실내/문서 신호 추출 ────────────────────────────────
    signals = extract_receipt_signals(text_list)
    result.phone_numbers = signals["phone_numbers"]
    result.phone_regions = signals["phone_regions"]
    result.business_reg_numbers = signals["business_reg_numbers"]
    result.address_fragments = signals["addresses"]
    result.currency_hints = signals["currency_hints"]

    # 브랜드 감지
    result.brand_names = brand_quick_lookup(text_list)

    # 문서 유형 감지
    result.document_type = detect_document_type(text_list)

    # 바코드/QR (pyzbar 사용 가능 시)
    result.barcodes = await _detect_barcodes(image_bytes)

    # GIS API 교차검증
    poi_matches = await _search_pois(texts, result.detected_languages)
    result.poi_matches = poi_matches
    if poi_matches:
        result.best_match = max(poi_matches, key=lambda p: p.confidence)

    # 전화번호 지역으로 best_match가 없으면 보완
    if not result.best_match and result.phone_regions:
        region = result.phone_regions[0]
        # 전화번호 지역 → POI 검색 시도
        for name in text_list[:3]:
            if len(name) >= 2:
                extra = await _naver_place_search(name)
                if extra:
                    result.poi_matches.extend(extra)
                    result.best_match = max(extra, key=lambda p: p.confidence)
                    break

    # ── 한국 전용 심층 분석 ────────────────────────────────
    if "ko" in result.detected_languages:
        try:
            from ..services.korea_specializer import analyze_korea_location
            kr_lat = result.best_match.latitude if result.best_match else None
            kr_lon = result.best_match.longitude if result.best_match else None
            korea = await analyze_korea_location(text_list, kr_lat, kr_lon)
            result.korea_analysis = korea
            # 한국 분석으로 best_match 보강
            if korea.get("lat") and (not result.best_match or korea.get("confidence", 0) > 0.85):
                result.best_match = POIMatch(
                    name=korea.get("best_location", "한국"),
                    address=korea.get("address", korea.get("best_location", "")),
                    latitude=korea["lat"],
                    longitude=korea["lon"],
                    source="korea_specializer",
                    confidence=korea.get("confidence", 0.85),
                )
                if korea.get("best_location"):
                    result.poi_matches.insert(0, result.best_match)
            # city_hint도 country_hints에 반영
            if korea.get("city_hint") and korea["city_hint"] not in result.country_hints:
                result.country_hints.append(korea["city_hint"])
        except Exception as e:
            logger.debug(f"Korea specializer skipped: {e}")

    return result


async def _run_ocr(image_bytes: bytes) -> list[OCRText]:
    texts = []

    # PaddleOCR (주력) — 미설치 시 EasyOCR 폴백
    try:
        paddle_texts = await _run_paddleocr(image_bytes)
        texts.extend(paddle_texts)
    except Exception as e:
        logger.debug(f"PaddleOCR 스킵 (미설치 또는 오류): {type(e).__name__}")

    # EasyOCR (보조, 특히 라틴 문자) — 20초 타임아웃 (preload 후 실제 OCR만 수행)
    if len(texts) < 3:
        import asyncio as _asyncio
        try:
            easy_texts = await _asyncio.wait_for(_run_easyocr(image_bytes), timeout=20.0)
            for t in easy_texts:
                if t.text not in [x.text for x in texts]:
                    texts.append(t)
        except _asyncio.TimeoutError:
            logger.warning("EasyOCR timed out (20s), skipping")
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")

    # 신뢰도 필터링 (0.5 이상)
    return [t for t in texts if t.confidence >= 0.5]


_paddle_ocr_instance = None


async def _run_paddleocr(image_bytes: bytes) -> list[OCRText]:
    import asyncio
    from paddleocr import PaddleOCR
    import numpy as np
    from PIL import Image
    import io

    def _sync():
        global _paddle_ocr_instance
        # PaddleOCR 신버전: show_log 미지원, predict() 사용
        import os, logging
        os.environ.setdefault("PADDLEOCR_LOG_LEVEL", "ERROR")
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        # 인스턴스 캐싱 — 모델 로딩은 최초 1회만
        if _paddle_ocr_instance is None:
            _paddle_ocr_instance = PaddleOCR(use_angle_cls=True, lang="korean")
        ocr = _paddle_ocr_instance
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)
        # 신버전: ocr.predict(), 구버전: ocr.ocr()
        try:
            raw = ocr.predict(img_array)
            # predict()는 list[dict] 반환
            texts = []
            if isinstance(raw, list) and raw:
                for item in raw:
                    if isinstance(item, dict):
                        for rec in item.get("rec_res", item.get("result", [])):
                            text = rec.get("text", rec[0] if isinstance(rec, (list,tuple)) else "")
                            conf = rec.get("score", rec[1] if isinstance(rec, (list,tuple)) and len(rec)>1 else 0.5)
                            if text:
                                texts.append(OCRText(text=str(text).strip(), confidence=float(conf), bbox=[], language=""))
            return texts
        except Exception:
            # 폴백: 구버전 API
            result = ocr.ocr(img_array, cls=True)
            texts = []
            if result and result[0]:
                for line in result[0]:
                    bbox, (text, conf) = line
                    texts.append(OCRText(text=text.strip(), confidence=float(conf), bbox=bbox, language=""))
            return texts

    return await asyncio.to_thread(_sync)


_easyocr_instance = None


def preload_easyocr() -> None:
    """Celery worker 시작 시 EasyOCR Reader 사전 로딩 (첫 작업 타임아웃 방지)"""
    global _easyocr_instance
    try:
        import easyocr
        if _easyocr_instance is None:
            _easyocr_instance = easyocr.Reader(["ko", "en"], gpu=False)
    except Exception as e:
        import logging
        logging.getLogger("celery.preload").warning(f"[preload] EasyOCR load failed: {e}")


async def _run_easyocr(image_bytes: bytes) -> list[OCRText]:
    import asyncio
    import easyocr
    import numpy as np
    from PIL import Image
    import io

    def _sync():
        global _easyocr_instance
        if _easyocr_instance is None:
            _easyocr_instance = easyocr.Reader(["ko", "en"], gpu=False)
        reader = _easyocr_instance

        # CPU 속도 최적화: 긴 변 1280px 이하로 리사이즈 (간판 텍스트는 충분히 읽힘)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        max_side = 1280
        if max(w, h) > max_side:
            ratio = max_side / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        result = reader.readtext(np.array(img))
        # 신뢰도 0.4 이상 + 2글자 이상만 채택 (노이즈 제거)
        return [
            OCRText(text=text.strip(), confidence=float(conf), bbox=bbox)
            for bbox, text, conf in result
            if conf >= 0.4 and len(text.strip()) >= 2
        ]

    return await asyncio.to_thread(_sync)


def _detect_languages(texts: list[OCRText]) -> list[str]:
    languages = set()
    for t in texts:
        text = t.text
        # 한국어
        if re.search(r"[\uAC00-\uD7A3]", text):
            languages.add("ko")
        # 일본어
        if re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", text):
            languages.add("ja")
        # 중국어
        if re.search(r"[\u4E00-\u9FFF]", text) and "ja" not in languages:
            languages.add("zh")
        # 영어/라틴
        if re.search(r"[a-zA-Z]{2,}", text):
            languages.add("en")
        # 아랍어
        if re.search(r"[\u0600-\u06FF]", text):
            languages.add("ar")
        # 태국어
        if re.search(r"[\u0E00-\u0E7F]", text):
            languages.add("th")
    return list(languages)


def _languages_to_countries(languages: list[str]) -> list[str]:
    mapping = {
        "ko": "한국",
        "ja": "일본",
        "zh": "중국/대만",
        "ar": "중동/아랍권",
        "th": "태국",
    }
    return [mapping[lang] for lang in languages if lang in mapping]


def _detect_license_plates(text: str) -> list[str]:
    patterns = [
        r"\d{2,3}[가-힣]\d{4}",               # 한국 (12가3456, 123가4567)
        r"[가-힣]{2}\d{2}[가-힣]\d{4}",        # 한국 구형
        r"[A-Z]{2,3}\d{2,4}[A-Z]{0,2}",        # 유럽
        r"\d{3}[-\s]?\d{4}",                   # 일본 (간략)
    ]
    plates = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        plates.extend(matches)
    return list(set(plates))


def _plate_to_country(plate: str) -> str:
    if re.search(r"[가-힣]", plate):
        return "한국"
    if re.match(r"\d{3}-\d{4}", plate):
        return "일본"
    if re.match(r"[A-Z]{2,3}\d{2,4}", plate):
        return "유럽"
    return ""


def _detect_postal_codes(text: str, languages: list[str]) -> list[str]:
    codes = []
    if "ko" in languages:
        # 한국 (5자리)
        codes.extend(re.findall(r"\b\d{5}\b", text))
    if "en" in languages:
        # 미국 ZIP
        codes.extend(re.findall(r"\b\d{5}(?:-\d{4})?\b", text))
    return list(set(codes))


async def _search_pois(texts: list[OCRText], languages: list[str]) -> list[POIMatch]:
    matches = []
    search_queries = _extract_search_queries(texts)

    tasks = []
    for query in search_queries[:8]:  # Fix J: 5→8개 쿼리
        if "ko" in languages or not languages:
            tasks.append(_naver_place_search(query))
            tasks.append(_kakao_place_search(query))  # Fix H: Kakao 교차 검증
        if settings.GOOGLE_MAPS_API_KEY:
            tasks.append(_google_place_search(query))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, list):
            matches.extend(r)

    return matches



async def _kakao_place_search(query: str) -> list[POIMatch]:
    if not settings.KAKAO_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://dapi.kakao.com/v2/local/search/keyword.json",
                params={"query": query, "size": 5},
                headers={"Authorization": f"KakaoAK {settings.KAKAO_API_KEY}"},
            )
            data = resp.json()
            matches = []
            for item in data.get("documents", []):
                lat = float(item.get("y", 0))
                lon = float(item.get("x", 0))
                if 33 <= lat <= 38 and 124 <= lon <= 132:
                    matches.append(POIMatch(
                        name=item.get("place_name", ""),
                        address=item.get("road_address_name") or item.get("address_name", ""),
                        latitude=lat,
                        longitude=lon,
                        source="kakao",
                        confidence=0.82,
                    ))
            return matches
    except Exception as e:
        logger.warning(f"Kakao place search failed: {e}")
        return []

async def _detect_barcodes(image_bytes: bytes) -> list[str]:
    """pyzbar로 바코드/QR 코드 감지 (설치된 경우)"""
    def _sync():
        try:
            from pyzbar.pyzbar import decode
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            decoded = decode(img)
            return [d.data.decode("utf-8", errors="ignore") for d in decoded if d.data]
        except ImportError:
            return []
        except Exception as e:
            logger.debug(f"barcode detect: {e}")
            return []
    return await asyncio.to_thread(_sync)


def _extract_search_queries(texts: list[OCRText]) -> list[str]:
    queries = []
    for t in sorted(texts, key=lambda x: x.confidence, reverse=True):
        text = t.text.strip()
        if len(text) >= 2 and t.confidence >= 0.7:
            queries.append(text)
    return queries[:10]


async def _naver_place_search(query: str) -> list[POIMatch]:
    if not settings.NAVER_CLIENT_ID:
        return []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://openapi.naver.com/v1/search/local.json",
                params={"query": query, "display": 5},
                headers={
                    "X-Naver-Client-Id": settings.NAVER_CLIENT_ID,
                    "X-Naver-Client-Secret": settings.NAVER_CLIENT_SECRET,
                },
            )
            data = resp.json()
            matches = []
            for item in data.get("items", []):
                # 네이버 좌표는 KATECH → WGS84 변환 필요
                x = float(item.get("mapx", 0)) / 1e7
                y = float(item.get("mapy", 0)) / 1e7
                matches.append(POIMatch(
                    name=re.sub(r"<[^>]+>", "", item.get("title", "")),
                    address=item.get("address", ""),
                    latitude=y,
                    longitude=x,
                    source="naver",
                    confidence=0.8,
                ))
            return matches
    except Exception as e:
        logger.warning(f"Naver place search failed: {e}")
        return []




async def _google_place_search(query: str) -> list[POIMatch]:
    if not settings.GOOGLE_MAPS_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://maps.googleapis.com/maps/api/place/textsearch/json",
                params={"query": query, "key": settings.GOOGLE_MAPS_API_KEY},
            )
            data = resp.json()
            matches = []
            for place in data.get("results", [])[:5]:
                loc = place.get("geometry", {}).get("location", {})
                matches.append(POIMatch(
                    name=place.get("name", ""),
                    address=place.get("formatted_address", ""),
                    latitude=loc.get("lat", 0),
                    longitude=loc.get("lng", 0),
                    source="google",
                    confidence=0.9,
                ))
            return matches
    except Exception as e:
        logger.warning(f"Google place search failed: {e}")
        return []

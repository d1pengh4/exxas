"""
실내 씬 분석 — 영수증/문서/라벨 파싱, 브랜드 조회, 바코드 조회, 실내 OSINT
"""
import re
import asyncio
from typing import Optional
from loguru import logger


# ── 한국 전화번호 지역코드 ──────────────────────────────────
_PHONE_REGION: dict[str, str] = {
    "02": "서울",
    "031": "경기",
    "032": "인천",
    "033": "강원",
    "041": "충남",
    "042": "대전",
    "043": "충북",
    "044": "세종",
    "051": "부산",
    "052": "울산",
    "053": "대구",
    "054": "경북",
    "055": "경남",
    "061": "전남",
    "062": "광주",
    "063": "전북",
    "064": "제주",
}

# ── 한국 주요 브랜드 DB ──────────────────────────────────────
_BRAND_DB: dict[str, dict] = {
    "스타벅스": {"country": "KR", "city_hint": None},
    "이디야": {"country": "KR", "city_hint": None},
    "투썸플레이스": {"country": "KR", "city_hint": None},
    "맥도날드": {"country": "KR", "city_hint": None},
    "롯데리아": {"country": "KR", "city_hint": None},
    "파리바게뜨": {"country": "KR", "city_hint": None},
    "뚜레쥬르": {"country": "KR", "city_hint": None},
    "CU": {"country": "KR", "city_hint": None},
    "GS25": {"country": "KR", "city_hint": None},
    "이마트": {"country": "KR", "city_hint": None},
    "롯데마트": {"country": "KR", "city_hint": None},
    "홈플러스": {"country": "KR", "city_hint": None},
    "올리브영": {"country": "KR", "city_hint": None},
    "다이소": {"country": "KR", "city_hint": None},
    "BBQ": {"country": "KR", "city_hint": None},
    "교촌치킨": {"country": "KR", "city_hint": None},
    "bhc치킨": {"country": "KR", "city_hint": None},
}

# ── 영수증 신호 추출 ─────────────────────────────────────────
_PHONE_RE = re.compile(r'(0\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4})')
_BIZ_REG_RE = re.compile(r'\d{3}-\d{2}-\d{5}')
_ADDR_KEYWORDS = ["시", "구", "동", "로", "길", "번지", "호"]
_CURRENCY_PATTERNS = {
    "KRW": [r'₩', r'원', r'KRW'],
    "USD": [r'\$', r'USD', r'달러'],
    "JPY": [r'¥', r'JPY', r'엔'],
    "CNY": [r'¥', r'CNY', r'위안'],
    "EUR": [r'€', r'EUR', r'유로'],
}


def extract_receipt_signals(texts: list[str]) -> dict:
    """영수증/문서에서 위치 신호 추출"""
    combined = " ".join(texts)

    # 전화번호 → 지역
    phone_numbers = []
    phone_regions = []
    for m in _PHONE_RE.finditer(combined):
        phone_raw = re.sub(r'[-.\s]', '', m.group(0))
        phone_numbers.append(m.group(0))
        for code, region in _PHONE_REGION.items():
            if phone_raw.startswith(code):
                if region not in phone_regions:
                    phone_regions.append(region)
                break

    # 사업자등록번호
    biz_regs = list(set(_BIZ_REG_RE.findall(combined)))

    # 주소 패턴
    addresses = []
    for text in texts:
        has_addr = sum(1 for kw in _ADDR_KEYWORDS if kw in text)
        if has_addr >= 2 and len(text) > 5:
            addresses.append(text)

    # 상호명 (짧은 한글 텍스트, 가게 이름으로 추정)
    store_names = []
    for text in texts:
        stripped = text.strip()
        if 2 <= len(stripped) <= 20 and any('\uAC00' <= c <= '\uD7A3' for c in stripped):
            if not any(c.isdigit() for c in stripped[:3]):
                store_names.append(stripped)

    # 통화 힌트
    currency_hints = []
    for currency, patterns in _CURRENCY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, combined):
                if currency not in currency_hints:
                    currency_hints.append(currency)
                break

    return {
        "phone_numbers": phone_numbers,
        "phone_regions": phone_regions,
        "business_reg_numbers": biz_regs,
        "store_names": store_names[:10],
        "addresses": addresses[:5],
        "currency_hints": currency_hints,
    }


def brand_quick_lookup(texts: list[str]) -> list[tuple[str, str]]:
    """텍스트에서 알려진 브랜드 탐지 → [(브랜드명, 위치힌트)] 반환"""
    combined = " ".join(texts)
    results = []
    for brand, info in _BRAND_DB.items():
        if brand in combined:
            location = info.get("city_hint") or info.get("country", "KR")
            results.append((brand, location))
    return results


def detect_document_type(texts: list[str]) -> str:
    """문서 유형 감지"""
    combined = " ".join(texts).lower()
    if any(kw in combined for kw in ["영수증", "receipt", "합계", "부가세", "vat"]):
        return "영수증"
    if any(kw in combined for kw in ["메뉴", "menu", "주문", "order"]):
        return "메뉴판"
    if any(kw in combined for kw in ["간판", "open", "영업시간", "hours"]):
        return "간판"
    if any(kw in combined for kw in ["버스", "지하철", "노선", "승강장"]):
        return "교통안내판"
    if any(kw in combined for kw in ["도로명", "번지", "우편"]):
        return "주소표지"
    return "일반문서"


async def brand_web_search(brand: str) -> dict:
    """브랜드명으로 웹 검색하여 위치 정보 추출"""
    try:
        from .osint_chain import web_search as _web_search
        results = await _web_search(f"{brand} 위치 주소 한국")
        locations = []
        for r in results[:5]:
            snippet = r.get("snippet", "") + r.get("title", "")
            # 주소 패턴 검색
            for kw in ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "경기"]:
                if kw in snippet:
                    locations.append(kw)
        location_hints = list(dict.fromkeys(locations))[:3]
        return {
            "brand": brand,
            "search_results": results[:3],
            "location_hints": location_hints,
        }
    except Exception as e:
        logger.warning(f"brand_web_search failed: {e}")
        # Check local DB
        if brand in _BRAND_DB:
            info = _BRAND_DB[brand]
            return {
                "brand": brand,
                "country": info.get("country"),
                "location_hints": [info["city_hint"]] if info.get("city_hint") else [],
            }
        return {"brand": brand, "location_hints": []}


async def barcode_lookup(barcode: str) -> dict:
    """바코드 → 제품/제조국 조회"""
    # 바코드 국가코드 (GS1 prefix)
    gs1_prefixes = {
        "880": "대한민국",
        "450": "일본", "451": "일본", "459": "일본",
        "690": "중국", "691": "중국", "692": "중국",
        "400": "독일", "401": "독일",
        "300": "프랑스", "301": "프랑스",
        "000": "미국", "001": "미국",
    }

    result = {"barcode": barcode, "country": None, "product": None}

    # 국가 코드 추출
    if len(barcode) >= 3:
        prefix3 = barcode[:3]
        prefix = gs1_prefixes.get(prefix3)
        if prefix:
            result["country"] = prefix
        elif len(barcode) >= 2:
            prefix2 = barcode[:2]
            result["country"] = gs1_prefixes.get(prefix2 + "0")

    # Open Food Facts API 시도 (무료)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json")
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == 1:
                    product = data.get("product", {})
                    result["product"] = product.get("product_name", "")
                    result["brand"] = product.get("brands", "")
                    result["countries"] = product.get("countries", "")
    except Exception:
        pass

    return result


async def run_interior_osint(
    texts: list[str],
    doc_type: str = "",
    brands: list[str] = None,
    phone_regions: list[str] = None,
    currency_hints: list[str] = None,
    barcodes: list[str] = None,
) -> dict:
    """실내/문서 통합 OSINT"""
    brands = brands or []
    phone_regions = phone_regions or []
    currency_hints = currency_hints or []
    barcodes = barcodes or []

    result: dict = {
        "doc_type": doc_type,
        "location_candidates": [],
        "naver_places": [],
        "best_guess": None,
    }

    # 브랜드 조회 병렬
    brand_tasks = [brand_web_search(b) for b in brands[:3]]
    if brand_tasks:
        brand_results = await asyncio.gather(*brand_tasks, return_exceptions=True)
        for br in brand_results:
            if isinstance(br, dict) and br.get("location_hints"):
                for loc in br["location_hints"]:
                    result["location_candidates"].append({"location": loc, "score": 0.6, "source": "brand"})

    # 전화번호 지역
    for region in phone_regions[:3]:
        result["location_candidates"].append({"location": region, "score": 0.8, "source": "phone_region"})

    # 최고 후보
    if result["location_candidates"]:
        best = max(result["location_candidates"], key=lambda x: x["score"])
        result["best_guess"] = best["location"]

    # 네이버 POI 검색 (상호명으로)
    if texts:
        try:
            from ..pipeline.stage3_ocr_gis import _naver_place_search
            store_signals = extract_receipt_signals(texts)
            for store in store_signals.get("store_names", [])[:2]:
                places = await _naver_place_search(store)
                for p in places[:2]:
                    result["naver_places"].append({
                        "name": p.name,
                        "address": p.address,
                        "lat": p.latitude,
                        "lon": p.longitude,
                    })
        except Exception as e:
            logger.debug(f"Naver POI search failed: {e}")

    return result

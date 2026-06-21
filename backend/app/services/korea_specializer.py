"""
한국 전용 위치 특화 분석 엔진 v1
- 편의점/카페/아파트/지하철역 인식 → 구/동 수준 위치 확정
- 한국 우편번호 → 좌표 매핑
- 한국 번호판 지역코드 → 시/군/구
- 행정안전부 도로명주소 API
- 네이버 로드뷰 좌표 탐색
- 한국 랜드마크 DB
"""
import asyncio
import re
import ssl
import aiohttp
import certifi
from loguru import logger
from typing import Optional

_SSL = ssl.create_default_context(cafile=certifi.where())


def _connector():
    return aiohttp.TCPConnector(ssl=_SSL)


# ═══════════════════════════════════════════════════════════
# ① 한국 브랜드/체인 DB — 간판 하나로 도시/구 특정
# ═══════════════════════════════════════════════════════════

# 편의점 체인 (전국)
KR_CONVENIENCE_STORES = {
    "GS25": {"region": "전국", "confidence": 0.7},
    "CU": {"region": "전국", "confidence": 0.7},
    "세븐일레브": {"region": "전국", "confidence": 0.7},
    "7-ELEVEN": {"region": "전국", "confidence": 0.7},
    "이마트24": {"region": "전국", "confidence": 0.7},
    "미니스톱": {"region": "전국", "confidence": 0.7},
}

# 카페 체인 (도시 분포 특성 포함)
KR_CAFE_CHAINS = {
    "스타벅스": {"primary": "서울/수도권/광역시", "confidence": 0.6},
    "이디야": {"primary": "전국", "confidence": 0.6},
    "투썸플레이스": {"primary": "전국", "confidence": 0.6},
    "메가커피": {"primary": "전국", "confidence": 0.6},
    "메가MGC": {"primary": "전국", "confidence": 0.6},
    "빽다방": {"primary": "전국", "confidence": 0.6},
    "컴포즈커피": {"primary": "전국", "confidence": 0.6},
    "할리스": {"primary": "서울/수도권", "confidence": 0.65},
    "탐앤탐스": {"primary": "전국", "confidence": 0.6},
    "폴바셋": {"primary": "서울/수도권/부산", "confidence": 0.7},
    "엔제리너스": {"primary": "서울/수도권", "confidence": 0.65},
    "커피빈": {"primary": "서울/수도권", "confidence": 0.65},
    "파스쿠찌": {"primary": "전국", "confidence": 0.6},
    "카페베네": {"primary": "전국", "confidence": 0.6},
    "드롭탑": {"primary": "서울/수도권", "confidence": 0.65},
    "달콤커피": {"primary": "전국", "confidence": 0.6},
    "더벤티": {"primary": "전국", "confidence": 0.6},
    "청년다방": {"primary": "전국", "confidence": 0.6},
}

# 패스트푸드 (전국)
KR_FAST_FOOD = {
    "롯데리아": {"region": "전국", "confidence": 0.65},
    "맥도날드": {"region": "전국", "confidence": 0.65},
    "버거킹": {"region": "전국", "confidence": 0.65},
    "KFC": {"region": "전국", "confidence": 0.65},
    "파파이스": {"region": "전국", "confidence": 0.65},
    "맘스터치": {"region": "전국", "confidence": 0.65},
    "노브랜드버거": {"region": "전국", "confidence": 0.65},
    "서브웨이": {"region": "전국", "confidence": 0.65},
    "쉐이크쉑": {"region": "서울/수도권/부산", "confidence": 0.7},
    "파이브가이즈": {"region": "서울/수도권", "confidence": 0.8},
    "교촌치킨": {"region": "전국", "confidence": 0.65},
    "BBQ": {"region": "전국", "confidence": 0.65},
    "BHC": {"region": "전국", "confidence": 0.65},
    "네네치킨": {"region": "전국", "confidence": 0.65},
    "처갓집양념치킨": {"region": "전국", "confidence": 0.65},
    "굽네치킨": {"region": "전국", "confidence": 0.65},
    "호식이두마리치킨": {"region": "전국", "confidence": 0.65},
}

# 아파트 브랜드 (단지명+브랜드 = 매우 정밀한 위치 특정 가능)
KR_APT_BRANDS = [
    "래미안",       # 삼성물산
    "자이",         # GS건설
    "힐스테이트",   # 현대엔지니어링
    "e편한세상",    # DL이앤씨
    "아이파크",     # HDC현대산업개발
    "푸르지오",     # 대우건설
    "롯데캐슬",     # 롯데건설
    "더샵",         # 포스코이앤씨
    "SK뷰",         # SK에코플랜트
    "에코",         # 제일건설 등
    "리슈빌",       # 진흥기업
    "한화포레나",   # 한화건설
    "두산위브",     # 두산건설
    "호반써밋",     # 호반건설
    "코오롱하늘채", # 코오롱글로벌
    "금호어울림",   # 금호건설
    "중흥S클래스",  # 중흥건설
    "우미린",       # 우미건설
    "동문굿모닝힐", # 동문건설
    "신안인스빌",   # 신안건설
    "대림e편한세상",
    "현대힐스테이트",
    "한신더휴",
    "대우푸르지오",
    "삼성래미안",
]

# 마트/쇼핑
KR_RETAIL = {
    "이마트": {"region": "전국", "confidence": 0.7},
    "홈플러스": {"region": "전국", "confidence": 0.7},
    "롯데마트": {"region": "전국", "confidence": 0.7},
    "코스트코": {"region": "서울/수도권/부산/대구/광주", "confidence": 0.75},
    "다이소": {"region": "전국", "confidence": 0.65},
    "올리브영": {"region": "전국", "confidence": 0.65},
    "다이소": {"region": "전국", "confidence": 0.65},
    "W컨셉": {"region": "온라인", "confidence": 0.5},
    "무신사": {"region": "온라인", "confidence": 0.5},
    "교보문고": {"region": "서울/수도권/광역시", "confidence": 0.7},
    "YES24": {"region": "전국", "confidence": 0.65},
}

# 은행
KR_BANKS = {
    "KB국민은행": {"region": "전국", "confidence": 0.65},
    "국민은행": {"region": "전국", "confidence": 0.65},
    "신한은행": {"region": "전국", "confidence": 0.65},
    "우리은행": {"region": "전국", "confidence": 0.65},
    "하나은행": {"region": "전국", "confidence": 0.65},
    "KEB하나": {"region": "전국", "confidence": 0.65},
    "NH농협": {"region": "전국", "confidence": 0.65},
    "농협은행": {"region": "전국", "confidence": 0.65},
    "IBK기업은행": {"region": "전국", "confidence": 0.65},
    "부산은행": {"region": "부산/경남", "confidence": 0.85},
    "대구은행": {"region": "대구/경북", "confidence": 0.85},
    "전북은행": {"region": "전북", "confidence": 0.85},
    "광주은행": {"region": "광주/전남", "confidence": 0.85},
    "제주은행": {"region": "제주", "confidence": 0.9},
    "경남은행": {"region": "경남", "confidence": 0.85},
}

# 전국 브랜드 통합 딕셔너리
ALL_KR_BRANDS: dict[str, dict] = {}
ALL_KR_BRANDS.update(KR_CONVENIENCE_STORES)
ALL_KR_BRANDS.update(KR_CAFE_CHAINS)
ALL_KR_BRANDS.update(KR_FAST_FOOD)
ALL_KR_BRANDS.update(KR_RETAIL)
ALL_KR_BRANDS.update(KR_BANKS)


# ═══════════════════════════════════════════════════════════

# ② 지하철역 DB — 역명 보이면 GPS 즉시 확정 (전국 600+역)
# ═══════════════════════════════════════════════════════════
from ..data.korea_stations_db import KR_SUBWAY_STATIONS  # noqa: F401


# 역명 접미사 제거용 패턴
_STATION_SUFFIXES = re.compile(r"(역|station)$", re.IGNORECASE)


# ═══════════════════════════════════════════════════════════
# ③ 한국 번호판 지역코드 → 시/군/구
# ═══════════════════════════════════════════════════════════

KR_PLATE_REGION_CODES: dict[str, str] = {
    # 서울
    "가": "서울", "나": "서울", "다": "서울", "라": "서울", "마": "서울",
    "거": "서울", "너": "서울", "더": "서울", "러": "서울", "머": "서울",
    "고": "서울", "노": "서울", "도": "서울", "로": "서울", "모": "서울",
    # 경기
    "경기": "경기도",
    # 번호판 지역 코드 (2자리 숫자로 시작)
    "11": "서울", "12": "인천", "13": "경기",
    "21": "부산", "22": "경남", "23": "울산",
    "31": "경북", "32": "대구",
    "41": "충남", "42": "대전", "43": "충북", "44": "세종",
    "51": "전북", "52": "광주", "53": "전남",
    "61": "강원", "71": "제주",
}

# ═══════════════════════════════════════════════════════════
# ④ 한국 우편번호 → 지역 (5자리)
# ═══════════════════════════════════════════════════════════

KR_POSTAL_RANGES: list[tuple[int, int, str, float, float]] = [
    # (start, end, city_name, lat, lon)
    (1000, 9999,   "서울",     37.5665, 126.9780),
    (10000, 18999, "경기도",   37.2636, 127.0286),
    (21000, 23999, "인천",     37.4563, 126.7052),
    (24000, 26999, "강원도",   37.8813, 127.7298),
    (27000, 29999, "충청북도", 36.6357, 127.4915),
    (30000, 31999, "세종/충남", 36.4800, 127.2890),
    (32000, 35999, "충청남도", 36.5184, 126.8000),
    (34000, 35999, "대전",     36.3504, 127.3845),
    (36000, 40999, "경상북도", 36.5760, 128.5056),
    (41000, 43999, "대구",     35.8714, 128.6014),
    (44000, 45999, "울산",     35.5384, 129.3114),
    (46000, 49999, "부산",     35.1796, 129.0756),
    (50000, 53999, "경상남도", 35.4606, 128.2132),
    (54000, 57999, "전라북도", 35.7175, 127.1530),
    (57000, 59999, "전라남도", 34.8161, 126.4630),
    (61000, 62999, "광주",     35.1595, 126.8526),
    (63000, 63999, "제주도",   33.4890, 126.4983),
]


def postal_code_to_region(postal_code: str) -> Optional[tuple[str, float, float]]:
    """한국 우편번호 → (지역명, lat, lon)"""
    code = re.sub(r"\D", "", postal_code)
    if len(code) != 5:
        return None
    n = int(code)
    for start, end, city, lat, lon in KR_POSTAL_RANGES:
        if start <= n <= end:
            return city, lat, lon
    return None


# ═══════════════════════════════════════════════════════════
# ⑤ OCR 텍스트에서 한국 특화 단서 추출
# ═══════════════════════════════════════════════════════════

def extract_korea_clues(texts: list[str]) -> dict:
    """
    OCR 텍스트에서 한국 위치 단서 추출.
    반환: {
        brands, subway_stations, apt_brands, postal_codes,
        address_fragments, phone_regions, confidence_boost
    }
    """
    full_text = " ".join(texts)
    clues: dict = {
        "brands": [],
        "subway_stations": [],
        "apt_brands": [],
        "postal_codes": [],
        "address_fragments": [],
        "phone_regions": [],
        "jibeon_addresses": [],
        "road_addresses": [],
        "confidence_boost": 0.0,
        "city_hint": "",
    }

    # ── 브랜드 인식 ──────────────────────────────────────
    for brand, info in ALL_KR_BRANDS.items():
        if brand in full_text:
            clues["brands"].append({"brand": brand, "region": info.get("region", "전국"),
                                    "confidence": info.get("confidence", 0.6)})
            clues["confidence_boost"] += 0.05

    # ── 아파트 브랜드 인식 ────────────────────────────────
    for apt in KR_APT_BRANDS:
        if apt in full_text:
            clues["apt_brands"].append(apt)
            clues["confidence_boost"] += 0.10  # 아파트 단지명은 매우 정밀
            # 아파트 단지명 + 동/호수 패턴
            m = re.search(rf"{re.escape(apt)}\s*([가-힣0-9]+)", full_text)
            if m:
                clues["address_fragments"].append(f"{apt} {m.group(1)}")

    # ── 지하철역 인식 ─────────────────────────────────────
    for station_name, info in KR_SUBWAY_STATIONS.items():
        # "강남역", "강남역 방면" 등
        if station_name + "역" in full_text or station_name in full_text:
            clues["subway_stations"].append({
                "name": station_name,
                "lat": info["lat"],
                "lon": info["lon"],
                "line": info["line"],
                "city": info["city"],
            })
            clues["confidence_boost"] += 0.20  # 지하철역 = 매우 높은 정확도
            if not clues["city_hint"]:
                clues["city_hint"] = info["city"]

    # ── 우편번호 ──────────────────────────────────────────
    postal_matches = re.findall(r"\b(\d{5})\b", full_text)
    for code in postal_matches:
        result = postal_code_to_region(code)
        if result:
            city, lat, lon = result
            clues["postal_codes"].append({"code": code, "city": city, "lat": lat, "lon": lon})
            clues["confidence_boost"] += 0.15
            if not clues["city_hint"]:
                clues["city_hint"] = city

    # ── 도로명주소 패턴 ───────────────────────────────────
    road_patterns = [
        r"[가-힣]{2,5}(?:특별시|광역시|특별자치시|도)\s+[가-힣]{2,5}(?:시|군|구)\s+[가-힣0-9\s]{2,20}(?:로|길)\s*\d+",
        r"[가-힣]{2,5}(?:구|군)\s+[가-힣0-9\s]{2,20}(?:로|길)\s*\d+",
        r"[가-힣]{1,10}(?:로|길)\s*\d+(?:-\d+)?(?:\s*[가-힣동번로길]+)?",
    ]
    for pat in road_patterns:
        for m in re.findall(pat, full_text):
            if len(m) >= 5:
                clues["road_addresses"].append(m.strip())
                clues["confidence_boost"] += 0.25

    # ── 지번주소 패턴 ─────────────────────────────────────
    jibeon = re.findall(r"[가-힣]{2,5}(?:동|읍|면)\s*\d+(?:-\d+)?번지?", full_text)
    clues["jibeon_addresses"].extend(jibeon)
    if jibeon:
        clues["confidence_boost"] += 0.20

    # ── 전화번호 지역코드 ─────────────────────────────────
    KR_AREA_CODES = {
        "02": "서울", "031": "경기", "032": "인천", "033": "강원",
        "041": "충남", "042": "대전", "043": "충북", "044": "세종",
        "051": "부산", "052": "울산", "053": "대구", "054": "경북",
        "055": "경남", "061": "전남", "062": "광주", "063": "전북", "064": "제주",
    }
    phone_nums = re.findall(r"\b(0\d{1,2})-?\d{3,4}-?\d{4}\b", full_text)
    for code in phone_nums:
        area = KR_AREA_CODES.get(code.replace("-", ""), "")
        if area and area not in clues["phone_regions"]:
            clues["phone_regions"].append(area)
            clues["confidence_boost"] += 0.12

    # ── 시/도 명칭 직접 감지 ─────────────────────────────
    KR_CITY_NAMES = {
        "서울": (37.5665, 126.9780), "부산": (35.1796, 129.0756),
        "인천": (37.4563, 126.7052), "대구": (35.8714, 128.6014),
        "대전": (36.3504, 127.3845), "광주": (35.1595, 126.8526),
        "울산": (35.5384, 129.3114), "세종": (36.4800, 127.2890),
        "경기": (37.4138, 127.5183), "강원": (37.8228, 128.1555),
        "충북": (36.6357, 127.4917), "충남": (36.5184, 126.8000),
        "전북": (35.7175, 127.1530), "전남": (34.8679, 126.9910),
        "경북": (36.4919, 128.8889), "경남": (35.4606, 128.2132),
        "제주": (33.4996, 126.5312),
    }
    for city_name, (clat, clon) in KR_CITY_NAMES.items():
        if city_name in full_text and city_name not in clues["city_hint"]:
            if not clues["city_hint"]:
                clues["city_hint"] = city_name
            clues["confidence_boost"] += 0.05

    # 상한 적용
    clues["confidence_boost"] = min(clues["confidence_boost"], 0.55)
    return clues


# ═══════════════════════════════════════════════════════════
# ⑥ 행정안전부 도로명주소 API
# ═══════════════════════════════════════════════════════════

async def search_juso_api(query: str, api_key: str = "") -> list[dict]:
    """
    행정안전부 도로명주소 API — 텍스트 주소 → 정확한 좌표.
    무료 API 키: https://www.juso.go.kr/addrlink/devAddrLinkRequestGuide.do
    """
    if not api_key:
        from ..core.config import settings
        api_key = getattr(settings, "JUSO_API_KEY", "")

    if not api_key:
        return []

    url = "https://www.juso.go.kr/addrlink/addrLinkApi.do"
    params = {
        "confmKey": api_key,
        "currentPage": 1,
        "countPerPage": 5,
        "keyword": query,
        "resultType": "json",
        "hstryYn": "N",
    }
    try:
        async with aiohttp.ClientSession(connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                data = await resp.json(content_type=None)

        results = []
        if not isinstance(data, dict):
            return []
        for item in (data.get("results") or {}).get("juso") or []:
            results.append({
                "road_address": item.get("roadAddr", ""),
                "jibeon_address": item.get("jibunAddr", ""),
                "zipcode": item.get("zipNo", ""),
                "building_name": item.get("bdNm", ""),
                "sigungu": item.get("siNm", "") + " " + item.get("sggNm", ""),
                "dong": item.get("legalDong", ""),
            })
        logger.debug(f"[juso_api] '{query}' → {len(results)}건")
        return results
    except Exception as e:
        logger.debug(f"[juso_api] 실패: {e}")
        return []


# ═══════════════════════════════════════════════════════════
# ⑦ 네이버 지오코딩 API
# ═══════════════════════════════════════════════════════════

async def naver_geocode(address: str) -> Optional[dict]:
    """
    네이버 클라우드 지오코딩 API → 정확한 WGS84 좌표.
    반환: {"lat": float, "lon": float, "address": str}
    """
    from ..core.config import settings
    client_id = getattr(settings, "NAVER_CLIENT_ID", "")
    client_secret = getattr(settings, "NAVER_CLIENT_SECRET", "")
    if not client_id:
        return None

    url = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
    }
    params = {"query": address, "count": 1}
    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                data = await resp.json()

        addresses = data.get("addresses", [])
        if not addresses:
            return None
        a = addresses[0]
        return {
            "lat": float(a.get("y", 0)),
            "lon": float(a.get("x", 0)),
            "address": a.get("roadAddress") or a.get("jibunAddress", ""),
        }
    except Exception as e:
        logger.debug(f"[naver_geocode] {e}")
        return None


async def kakao_geocode(address: str) -> Optional[dict]:
    """카카오 지오코딩 폴백 → 네이버 지오코딩으로 대체."""
    return await naver_geocode(address)


# ═══════════════════════════════════════════════════════════
# ⑧ 네이버 로드뷰 메타데이터 검색 (좌표 → 로드뷰 가용 여부)
# ═══════════════════════════════════════════════════════════

async def naver_roadview_check(lat: float, lon: float, radius_m: int = 100) -> dict:
    """
    네이버 로드뷰 파노라마 API로 주변 로드뷰 이미지 메타데이터 조회.
    반환: {"available": bool, "pano_id": str, "lat": float, "lon": float, "date": str}
    """
    from ..core.config import settings
    client_id = getattr(settings, "NAVER_CLIENT_ID", "")
    if not client_id:
        return {"available": False}

    url = "https://naveropenapi.apigw.ntruss.com/map-streetview/v2/pano"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": getattr(settings, "NAVER_CLIENT_SECRET", ""),
    }
    params = {"lat": lat, "lon": lon, "output": "json"}
    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return {"available": False}
                data = await resp.json()

        pano = data.get("result", {})
        if pano.get("id"):
            return {
                "available": True,
                "pano_id": pano.get("id"),
                "lat": float(pano.get("lat", lat)),
                "lon": float(pano.get("lon", lon)),
                "date": pano.get("sdate", ""),
                "description": pano.get("description", ""),
            }
        return {"available": False}
    except Exception as e:
        logger.debug(f"[naver_roadview] {e}")
        return {"available": False}


# ═══════════════════════════════════════════════════════════

# ⑨ 한국 랜드마크 DB (300+개)
# ═══════════════════════════════════════════════════════════
from ..data.korea_landmarks_db import KR_LANDMARKS, KR_DISTRICTS, KR_UNIVERSITIES, KR_HOSPITALS  # noqa: F401



def match_landmark(text: str) -> Optional[dict]:
    """텍스트에서 한국 랜드마크/대학/병원 감지 (aliases 포함)"""
    # 1순위: 랜드마크 이름 직접 매칭
    for name, info in KR_LANDMARKS.items():
        if name in text:
            return {"name": name, **info}
    # aliases 매칭
    for name, info in KR_LANDMARKS.items():
        for alias in info.get("aliases", []):
            if alias and alias in text:
                return {"name": name, **info}
    # 2순위: 대학교
    try:
        from ..data.korea_landmarks_db import KR_UNIVERSITIES
        for name, info in KR_UNIVERSITIES.items():
            if name in text:
                return {"name": name, "lat": info["lat"], "lon": info["lon"],
                        "city": info["city"], "category": "대학교"}
            for alias in info.get("aliases", []):
                if alias and alias in text:
                    return {"name": name, "lat": info["lat"], "lon": info["lon"],
                            "city": info["city"], "category": "대학교"}
    except Exception:
        pass
    # 3순위: 병원
    try:
        from ..data.korea_landmarks_db import KR_HOSPITALS
        for name, info in KR_HOSPITALS.items():
            if name in text:
                return {"name": name, "lat": info["lat"], "lon": info["lon"],
                        "city": info["city"], "category": "병원"}
            for alias in info.get("aliases", []):
                if alias and alias in text:
                    return {"name": name, "lat": info["lat"], "lon": info["lon"],
                            "city": info["city"], "category": "병원"}
    except Exception:
        pass
    # 4순위: 행정구역 중심
    try:
        from ..data.korea_landmarks_db import KR_DISTRICTS
        for name, info in KR_DISTRICTS.items():
            if name in text:
                return {"name": name, "lat": info["lat"], "lon": info["lon"],
                        "city": f"{info['city']} {name}", "category": "행정구역"}
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════
# ⑩ Redis 캐시 빠른 조회 헬퍼
# ═══════════════════════════════════════════════════════════

async def _redis_quick_lookup(texts: list[str], full_text: str) -> Optional[dict]:
    """
    시딩된 Redis 캐시에서 역명/랜드마크/도시를 O(1)로 조회.
    Returns dict with best_location, lat, lon, confidence, city_hint or None.
    """
    try:
        import redis.asyncio as aioredis
        r = await aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
    except Exception:
        return None

    try:
        import json

        # 조회 우선순위: 지하철역 > 랜드마크 > 도시
        # 각 텍스트 토큰 + 전체 텍스트에서 연속 2~6글자 슬라이딩 윈도우 검색
        candidates = set()
        for t in texts:
            t = t.strip()
            if not t:
                continue
            candidates.add(t)
            # 슬라이딩 윈도우 (2~8글자 서브스트링)
            for length in range(2, min(len(t) + 1, 9)):
                for start in range(len(t) - length + 1):
                    candidates.add(t[start:start + length])

        best: Optional[dict] = None
        best_priority = -1  # 높을수록 우선

        for token in candidates:
            if not token:
                continue

            # 1. 지하철역 조회 (최우선, priority=3)
            for key in [f"kr:station:{token}", f"kr:station:{token}역"]:
                val = await r.get(key)
                if val and best_priority < 3:
                    data = json.loads(val)
                    station_name = token.rstrip("역")
                    best = {
                        "best_location": f"{station_name}역 ({data.get('city', '')})",
                        "lat": data["lat"],
                        "lon": data["lon"],
                        "confidence": 0.92,
                        "city_hint": data.get("city", ""),
                        "subway_station": {"name": station_name, "lat": data["lat"], "lon": data["lon"],
                                           "line": data.get("line", ""), "city": data.get("city", "")},
                    }
                    best_priority = 3
                    break

            if best_priority == 3:
                break

            # 2. 랜드마크 조회 (priority=2)
            val = await r.get(f"kr:landmark:{token}")
            if val and best_priority < 2:
                data = json.loads(val)
                best = {
                    "best_location": f"{token} ({data.get('city', '')})",
                    "lat": data["lat"],
                    "lon": data["lon"],
                    "confidence": 0.88,
                    "city_hint": data.get("city", ""),
                }
                best_priority = 2

            # 3. 도시 조회 (priority=1, 낮은 신뢰도)
            val = await r.get(f"kr:city:{token}")
            if val and best_priority < 1:
                data = json.loads(val)
                best = {
                    "best_location": data.get("city", token),
                    "lat": data["lat"],
                    "lon": data["lon"],
                    "confidence": 0.60,
                    "city_hint": data.get("city", token),
                }
                best_priority = 1

        return best
    except Exception as e:
        logger.debug(f"[korea] Redis 조회 실패: {e}")
        return None
    finally:
        await r.aclose()


# ═══════════════════════════════════════════════════════════
# ⑪ 통합 한국 위치 분석 함수
# ═══════════════════════════════════════════════════════════

async def analyze_korea_location(
    texts: list[str],
    initial_lat: Optional[float] = None,
    initial_lon: Optional[float] = None,
) -> dict:
    """
    모든 한국 위치 단서를 통합 분석.
    반환: {
        clues, best_location, lat, lon, confidence,
        address, city_hint, landmark, subway_station
    }
    """
    result: dict = {
        "clues": {},
        "best_location": "",
        "lat": initial_lat,
        "lon": initial_lon,
        "confidence": 0.0,
        "address": "",
        "city_hint": "",
        "landmark": None,
        "subway_station": None,
        "geocode_results": [],
    }

    full_text = " ".join(texts)

    # ① Redis 캐시 빠른 조회 (시딩된 데이터 즉시 사용)
    redis_hit = await _redis_quick_lookup(texts, full_text)
    if redis_hit and redis_hit.get("confidence", 0) >= 0.80:
        result.update(redis_hit)
        logger.info(f"[korea] Redis 캐시 히트: {redis_hit.get('best_location')} conf={redis_hit.get('confidence'):.0%}")

    # ① 랜드마크 매칭 (즉각적, 최고 신뢰도)
    lm = match_landmark(full_text)
    if lm:
        result["landmark"] = lm
        result["best_location"] = lm["city"]
        result["lat"] = lm["lat"]
        result["lon"] = lm["lon"]
        result["confidence"] = 0.92
        logger.info(f"[korea] 랜드마크 확정: {lm['name']} @ {lm['city']}")

    # ② 한국 단서 추출
    clues = extract_korea_clues(texts)
    result["clues"] = clues

    # ③ 지하철역 (정밀 좌표 즉시 확정)
    if clues["subway_stations"]:
        station = clues["subway_stations"][0]
        result["subway_station"] = station
        result["lat"] = station["lat"]
        result["lon"] = station["lon"]
        result["city_hint"] = station["city"]
        if result["confidence"] < 0.88:
            result["best_location"] = station["city"]
            result["confidence"] = 0.88
        logger.info(f"[korea] 지하철역 확정: {station['name']}역 @ {station['city']}")

    # ④ 우편번호 좌표
    if clues["postal_codes"] and not result["lat"]:
        pc = clues["postal_codes"][0]
        result["lat"] = pc["lat"]
        result["lon"] = pc["lon"]
        result["city_hint"] = pc["city"]
        if result["confidence"] < 0.75:
            result["confidence"] = 0.75
        logger.info(f"[korea] 우편번호 확정: {pc['code']} → {pc['city']}")

    # ⑤ 도로명주소 → 지오코딩
    all_addresses = clues["road_addresses"] + clues["jibeon_addresses"] + clues["address_fragments"]
    for addr in all_addresses[:3]:
        if len(addr) < 5:
            continue
        # Naver 지오코딩 시도
        geo = await naver_geocode(addr)
        if not geo:
            geo = await kakao_geocode(addr)
        if geo and geo.get("lat"):
            result["geocode_results"].append(geo)
            result["lat"] = geo["lat"]
            result["lon"] = geo["lon"]
            result["address"] = geo.get("address", addr)
            result["best_location"] = geo.get("address", addr)
            if result["confidence"] < 0.90:
                result["confidence"] = 0.90
            logger.info(f"[korea] 주소 지오코딩: {addr} → ({geo['lat']:.4f},{geo['lon']:.4f})")
            break

    # ⑥ 도시 힌트 설정
    if not result["city_hint"]:
        if clues["phone_regions"]:
            result["city_hint"] = clues["phone_regions"][0]
        elif clues["brands"]:
            # 지역 한정 브랜드에서 추출
            for b in clues["brands"]:
                region = b.get("region", "전국")
                if region != "전국" and "전국" not in region:
                    result["city_hint"] = region
                    break

    if not result["best_location"] and result["city_hint"]:
        result["best_location"] = result["city_hint"]

    result["confidence"] = min(result["confidence"] + clues["confidence_boost"], 0.95)

    return result

"""
한국 차량번호판 OCR → 지역 코딩
신형(2006~): 12가3456, 서울12가3456
구형: 서울 12 가 3456
영업용: 서울 12 가 3456
"""
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlateResult:
    raw_text: str
    region_code: str = ""
    region_detail: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_km: float = 30.0
    confidence: float = 0.0
    plate_type: str = ""  # "신형", "구형", "영업용"


# 신형 번호판 지역코드 (앞 2자리 숫자)
_REGION_BY_CODE: dict[str, tuple[str, float, float]] = {
    "서울": ("서울특별시", 37.5665, 126.9780),
    "부산": ("부산광역시", 35.1796, 129.0756),
    "대구": ("대구광역시", 35.8714, 128.6014),
    "인천": ("인천광역시", 37.4563, 126.7052),
    "광주": ("광주광역시", 35.1595, 126.8526),
    "대전": ("대전광역시", 36.3504, 127.3845),
    "울산": ("울산광역시", 35.5384, 129.3114),
    "세종": ("세종특별자치시", 36.4800, 127.2890),
    "경기": ("경기도", 37.4138, 127.5183),
    "강원": ("강원도", 37.8228, 128.1555),
    "충북": ("충청북도", 36.6357, 127.4917),
    "충남": ("충청남도", 36.5184, 126.8000),
    "전북": ("전라북도", 35.7175, 127.1530),
    "전남": ("전라남도", 34.8160, 126.4629),
    "경북": ("경상북도", 36.5760, 128.5056),
    "경남": ("경상남도", 35.4606, 128.2132),
    "제주": ("제주특별자치도", 33.4996, 126.5312),
}

# 서울 자치구 → 좌표
_SEOUL_DISTRICTS: dict[str, tuple[float, float]] = {
    "종로": (37.5735, 126.9790),
    "중구": (37.5638, 126.9970),
    "용산": (37.5384, 126.9654),
    "성동": (37.5634, 127.0369),
    "광진": (37.5385, 127.0823),
    "동대문": (37.5744, 127.0396),
    "중랑": (37.6063, 127.0927),
    "성북": (37.5894, 127.0167),
    "강북": (37.6396, 127.0255),
    "도봉": (37.6688, 127.0471),
    "노원": (37.6543, 127.0568),
    "은평": (37.6176, 126.9227),
    "서대문": (37.5791, 126.9368),
    "마포": (37.5663, 126.9015),
    "양천": (37.5170, 126.8665),
    "강서": (37.5509, 126.8496),
    "구로": (37.4954, 126.8874),
    "금천": (37.4569, 126.8956),
    "영등포": (37.5264, 126.8962),
    "동작": (37.5124, 126.9393),
    "관악": (37.4784, 126.9516),
    "서초": (37.4837, 127.0324),
    "강남": (37.5172, 127.0473),
    "송파": (37.5145, 127.1059),
    "강동": (37.5301, 127.1238),
}

# 신형 번호판 패턴: 12가3456, 서울12가3456
_NEW_PLATE_RE = re.compile(
    r'([가-힣]{2})?(\d{2,3})\s*([가-힣])\s*(\d{4})'
)
# 구형/영업용: 서울 12 가 3456
_OLD_PLATE_RE = re.compile(
    r'([가-힣]{2})\s+(\d{2,3})\s+([가-힣])\s+(\d{4})'
)

# 가나다... 문자 범위
_HANGUL_CHARS = set("가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후그느드르므브스으즈츠크트프흐기니디리미비시이지치키티피히")


def parse_license_plates_from_texts(texts: list[str]) -> list[PlateResult]:
    """텍스트 목록에서 한국 번호판 패턴 추출 → 지역 정보 반환"""
    results: list[PlateResult] = []

    for raw in texts:
        text = raw.strip().replace(" ", "")

        # 신형 시도
        m = _NEW_PLATE_RE.search(text)
        if m:
            region_str, num1, letter, num2 = m.groups()
            raw_text = m.group(0)

            result = PlateResult(raw_text=raw_text, plate_type="신형")

            if region_str and region_str in _REGION_BY_CODE:
                detail, lat, lon = _REGION_BY_CODE[region_str]
                result.region_code = region_str
                result.region_detail = detail
                result.latitude = lat
                result.longitude = lon
                result.confidence = 0.85
                result.radius_km = 20.0
            else:
                # 번호판 숫자만으로 지역 추정 불가 — 서울 추정
                result.region_code = "알수없음"
                result.region_detail = "지역 미상"
                result.confidence = 0.30

            results.append(result)
            continue

        # 구형 시도 (원본 텍스트, 공백 있음)
        m2 = _OLD_PLATE_RE.search(raw)
        if m2:
            region_str, num1, letter, num2 = m2.groups()
            raw_text = m2.group(0)

            result = PlateResult(raw_text=raw_text, plate_type="구형")
            if region_str in _REGION_BY_CODE:
                detail, lat, lon = _REGION_BY_CODE[region_str]
                result.region_code = region_str
                result.region_detail = detail
                result.latitude = lat
                result.longitude = lon
                result.confidence = 0.80
                result.radius_km = 25.0
            else:
                result.region_code = region_str
                result.confidence = 0.40

            results.append(result)

    # 신뢰도 내림차순 정렬
    results.sort(key=lambda r: r.confidence, reverse=True)
    return results

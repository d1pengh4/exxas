"""
A. 한국 대중교통 DB 직접 매칭
- 버스 정류장 번호/이름 → 운행 도시 즉시 확정
- 지하철역 이름/호선 → 좌표 확정
- OCR 텍스트에서 자동 탐지
공공데이터포털 API(PUBLIC_DATA_API_KEY) + 하드코딩 대용량 내장 DB
"""
import re
import asyncio
import httpx
from loguru import logger
from ..core.config import settings

# ─────────────────────────────────────────────────────────────────────────────
# 내장 지하철역 DB (주요 역, 확장 가능)
# {역이름: {line, city, lat, lon}}
# ─────────────────────────────────────────────────────────────────────────────
SUBWAY_DB: dict[str, dict] = {
    # 서울 1호선
    "소요산": {"line": "1호선", "city": "경기 동두천", "lat": 37.9380, "lon": 127.0597},
    "동두천": {"line": "1호선", "city": "경기 동두천", "lat": 37.9023, "lon": 127.0590},
    "양주": {"line": "1호선", "city": "경기 양주", "lat": 37.7858, "lon": 127.0456},
    "도봉산": {"line": "1호선", "city": "서울 도봉", "lat": 37.6888, "lon": 127.0458},
    "도봉": {"line": "1호선", "city": "서울 도봉", "lat": 37.6686, "lon": 127.0468},
    "방학": {"line": "1호선", "city": "서울 도봉", "lat": 37.6539, "lon": 127.0447},
    "창동": {"line": "1·4호선", "city": "서울 도봉", "lat": 37.6527, "lon": 127.0476},
    "노원": {"line": "1호선", "city": "서울 노원", "lat": 37.6554, "lon": 127.0563},
    "석계": {"line": "1호선", "city": "서울 노원", "lat": 37.6354, "lon": 127.0619},
    "광운대": {"line": "1호선", "city": "서울 노원", "lat": 37.6210, "lon": 127.0595},
    "월계": {"line": "1호선", "city": "서울 노원", "lat": 37.6150, "lon": 127.0583},
    "녹천": {"line": "1호선", "city": "서울 노원", "lat": 37.6074, "lon": 127.0567},
    "성북": {"line": "1호선", "city": "서울 강북", "lat": 37.5929, "lon": 127.0168},
    "청량리": {"line": "1호선", "city": "서울 동대문", "lat": 37.5808, "lon": 127.0455},
    "회기": {"line": "1호선", "city": "서울 동대문", "lat": 37.5901, "lon": 127.0574},
    "외대앞": {"line": "1호선", "city": "서울 동대문", "lat": 37.5960, "lon": 127.0609},
    "서울역": {"line": "1·4호선", "city": "서울 중구", "lat": 37.5545, "lon": 126.9707},
    "시청": {"line": "1·2호선", "city": "서울 중구", "lat": 37.5653, "lon": 126.9774},
    "종각": {"line": "1호선", "city": "서울 종로", "lat": 37.5701, "lon": 126.9828},
    "종로3가": {"line": "1·3·5호선", "city": "서울 종로", "lat": 37.5713, "lon": 126.9918},
    "종로5가": {"line": "1호선", "city": "서울 종로", "lat": 37.5711, "lon": 126.9999},
    "동대문": {"line": "1·4호선", "city": "서울 종로", "lat": 37.5715, "lon": 127.0094},
    "동묘앞": {"line": "1호선", "city": "서울 동대문", "lat": 37.5707, "lon": 127.0161},
    "신설동": {"line": "1·2호선", "city": "서울 동대문", "lat": 37.5743, "lon": 127.0251},
    "제기동": {"line": "1호선", "city": "서울 동대문", "lat": 37.5808, "lon": 127.0369},
    "영등포": {"line": "1호선", "city": "서울 영등포", "lat": 37.5157, "lon": 126.9072},
    "신도림": {"line": "1·2호선", "city": "서울 구로", "lat": 37.5087, "lon": 126.8912},
    "구로": {"line": "1호선", "city": "서울 구로", "lat": 37.5016, "lon": 126.8817},
    "독산": {"line": "1호선", "city": "서울 금천", "lat": 37.4847, "lon": 126.8939},
    "가산디지털단지": {"line": "1·7호선", "city": "서울 금천", "lat": 37.4810, "lon": 126.8827},
    "금천구청": {"line": "1호선", "city": "서울 금천", "lat": 37.4578, "lon": 126.8952},
    "석수": {"line": "1호선", "city": "경기 안양", "lat": 37.4372, "lon": 126.9004},
    "관악": {"line": "1호선", "city": "경기 안양", "lat": 37.4295, "lon": 126.9038},
    "안양": {"line": "1호선", "city": "경기 안양", "lat": 37.3950, "lon": 126.9282},
    "군포": {"line": "1호선", "city": "경기 군포", "lat": 37.3622, "lon": 126.9336},
    "의왕": {"line": "1호선", "city": "경기 의왕", "lat": 37.3444, "lon": 126.9689},
    "수원": {"line": "1호선", "city": "경기 수원", "lat": 37.2665, "lon": 127.0004},
    "천안": {"line": "1호선", "city": "충남 천안", "lat": 36.8090, "lon": 127.1495},

    # 서울 2호선
    "강남": {"line": "2호선", "city": "서울 강남", "lat": 37.4979, "lon": 127.0276},
    "역삼": {"line": "2호선", "city": "서울 강남", "lat": 37.5005, "lon": 127.0365},
    "선릉": {"line": "2호선", "city": "서울 강남", "lat": 37.5043, "lon": 127.0491},
    "삼성": {"line": "2호선", "city": "서울 강남", "lat": 37.5088, "lon": 127.0627},
    "잠실": {"line": "2호선", "city": "서울 송파", "lat": 37.5133, "lon": 127.1001},
    "잠실나루": {"line": "2호선", "city": "서울 송파", "lat": 37.5130, "lon": 127.0871},
    "구의": {"line": "2호선", "city": "서울 광진", "lat": 37.5343, "lon": 127.0943},
    "건대입구": {"line": "2·7호선", "city": "서울 광진", "lat": 37.5402, "lon": 127.0699},
    "홍대입구": {"line": "2호선", "city": "서울 마포", "lat": 37.5573, "lon": 126.9247},
    "합정": {"line": "2·6호선", "city": "서울 마포", "lat": 37.5499, "lon": 126.9148},
    "당산": {"line": "2·9호선", "city": "서울 영등포", "lat": 37.5340, "lon": 126.9007},
    "여의도": {"line": "5·9호선", "city": "서울 영등포", "lat": 37.5219, "lon": 126.9244},
    "신촌": {"line": "2호선", "city": "서울 마포", "lat": 37.5553, "lon": 126.9366},
    "이대": {"line": "2호선", "city": "서울 서대문", "lat": 37.5572, "lon": 126.9464},
    "아현": {"line": "2호선", "city": "서울 마포", "lat": 37.5552, "lon": 126.9561},
    "충정로": {"line": "2·5호선", "city": "서울 서대문", "lat": 37.5567, "lon": 126.9627},

    # 서울 3호선
    "대화": {"line": "3호선", "city": "경기 고양", "lat": 37.6732, "lon": 126.7605},
    "주엽": {"line": "3호선", "city": "경기 고양", "lat": 37.6716, "lon": 126.7754},
    "정발산": {"line": "3호선", "city": "경기 고양", "lat": 37.6662, "lon": 126.7887},
    "마두": {"line": "3호선", "city": "경기 고양", "lat": 37.6568, "lon": 126.7820},
    "백석": {"line": "3호선", "city": "경기 고양", "lat": 37.6461, "lon": 126.8048},
    "대곡": {"line": "3호선", "city": "경기 고양", "lat": 37.6277, "lon": 126.8227},
    "화정": {"line": "3호선", "city": "경기 고양", "lat": 37.6295, "lon": 126.8361},
    "원당": {"line": "3호선", "city": "경기 고양", "lat": 37.6396, "lon": 126.8618},
    "원흥": {"line": "3호선", "city": "경기 고양", "lat": 37.6430, "lon": 126.8790},
    "삼송": {"line": "3호선", "city": "경기 고양", "lat": 37.6460, "lon": 126.9013},
    "지축": {"line": "3호선", "city": "서울 은평", "lat": 37.6370, "lon": 126.9190},
    "구파발": {"line": "3호선", "city": "서울 은평", "lat": 37.6290, "lon": 126.9302},
    "연신내": {"line": "3·6호선", "city": "서울 은평", "lat": 37.6193, "lon": 126.9200},
    "불광": {"line": "3호선", "city": "서울 은평", "lat": 37.6105, "lon": 126.9294},
    "녹번": {"line": "3호선", "city": "서울 은평", "lat": 37.6013, "lon": 126.9360},
    "홍제": {"line": "3호선", "city": "서울 서대문", "lat": 37.5940, "lon": 126.9413},
    "무악재": {"line": "3호선", "city": "서울 종로", "lat": 37.5826, "lon": 126.9428},
    "독립문": {"line": "3호선", "city": "서울 종로", "lat": 37.5776, "lon": 126.9596},
    "경복궁": {"line": "3호선", "city": "서울 종로", "lat": 37.5765, "lon": 126.9748},
    "안국": {"line": "3호선", "city": "서울 종로", "lat": 37.5782, "lon": 126.9852},
    "교대": {"line": "2·3호선", "city": "서울 서초", "lat": 37.4934, "lon": 127.0141},
    "양재": {"line": "3호선", "city": "서울 서초", "lat": 37.4846, "lon": 127.0344},
    "도곡": {"line": "3호선", "city": "서울 강남", "lat": 37.4889, "lon": 127.0445},
    "대치": {"line": "3호선", "city": "서울 강남", "lat": 37.4933, "lon": 127.0572},
    "학여울": {"line": "3호선", "city": "서울 강남", "lat": 37.4993, "lon": 127.0648},

    # 부산 지하철
    "서면": {"line": "부산1·2호선", "city": "부산 부산진", "lat": 35.1577, "lon": 129.0589},
    "부산역": {"line": "부산1호선", "city": "부산 동구", "lat": 35.1147, "lon": 129.0416},
    "해운대": {"line": "부산2호선", "city": "부산 해운대", "lat": 35.1636, "lon": 129.1680},
    "수영": {"line": "부산2·3호선", "city": "부산 수영", "lat": 35.1447, "lon": 129.1132},
    "광안": {"line": "부산2호선", "city": "부산 수영", "lat": 35.1529, "lon": 129.1230},
    "남포": {"line": "부산1호선", "city": "부산 중구", "lat": 35.0978, "lon": 129.0268},
    "자갈치": {"line": "부산1호선", "city": "부산 중구", "lat": 35.0966, "lon": 129.0289},
    "덕천": {"line": "부산2·3호선", "city": "부산 북구", "lat": 35.1968, "lon": 128.9877},
    "사직": {"line": "부산3호선", "city": "부산 동래", "lat": 35.1921, "lon": 129.0616},
    "연산": {"line": "부산1·3호선", "city": "부산 연제", "lat": 35.1818, "lon": 129.0818},
    "센텀시티": {"line": "부산2호선", "city": "부산 해운대", "lat": 35.1699, "lon": 129.1337},
    "벡스코": {"line": "부산2호선", "city": "부산 해운대", "lat": 35.1698, "lon": 129.1457},
    "장산": {"line": "부산2호선", "city": "부산 해운대", "lat": 35.1734, "lon": 129.2005},

    # 대구 지하철
    "동대구": {"line": "대구1호선", "city": "대구 동구", "lat": 35.8793, "lon": 128.6277},
    "반월당": {"line": "대구1·2호선", "city": "대구 중구", "lat": 35.8665, "lon": 128.5938},
    "대구역": {"line": "대구1호선", "city": "대구 북구", "lat": 35.8779, "lon": 128.5943},

    # 인천 지하철
    "인천터미널": {"line": "인천1호선", "city": "인천 남구", "lat": 37.4521, "lon": 126.7048},
    "부평": {"line": "수도권1호선", "city": "인천 부평", "lat": 37.4890, "lon": 126.7221},
    "계양": {"line": "인천1호선", "city": "인천 계양", "lat": 37.5381, "lon": 126.7385},

    # 광주 지하철
    "광주송정": {"line": "광주1호선", "city": "광주 광산", "lat": 35.1382, "lon": 126.7941},
    "농성": {"line": "광주1호선", "city": "광주 서구", "lat": 35.1484, "lon": 126.8869},
    "금남로4가": {"line": "광주1호선", "city": "광주 동구", "lat": 35.1501, "lon": 126.9134},

    # 대전 지하철
    "대전": {"line": "대전1호선", "city": "대전 동구", "lat": 36.3323, "lon": 127.4345},
    "시청": {"line": "대전1호선", "city": "대전 서구", "lat": 36.3504, "lon": 127.3847},
}

# ─────────────────────────────────────────────────────────────────────────────
# 버스 노선 지역 DB (버스 번호 → 운행 도시)
# 서울/부산/대구/인천/광주/대전 주요 간선/지선 버스
# ─────────────────────────────────────────────────────────────────────────────
BUS_ROUTE_DB: dict[str, dict] = {
    # 서울 간선버스 (파란색, 100번대~700번대)
    "100": {"city": "서울", "area": "종로/청계천", "color": "파란색"},
    "101": {"city": "서울", "area": "강북/도봉", "color": "파란색"},
    "102": {"city": "서울", "area": "성북/종로", "color": "파란색"},
    "103": {"city": "서울", "area": "도봉/종로", "color": "파란색"},
    "104": {"city": "서울", "area": "노원/강남", "color": "파란색"},
    "105": {"city": "서울", "area": "강북/마포", "color": "파란색"},
    "107": {"city": "서울", "area": "동대문/강남", "color": "파란색"},
    "108": {"city": "서울", "area": "도봉/강남", "color": "파란색"},
    "109": {"city": "서울", "area": "노원/중구", "color": "파란색"},
    "110": {"city": "서울", "area": "강북/중구", "color": "파란색"},
    "140": {"city": "서울", "area": "강북/중구", "color": "파란색"},
    "143": {"city": "서울", "area": "성북/강남", "color": "파란색"},
    "144": {"city": "서울", "area": "성북/강남", "color": "파란색"},
    "145": {"city": "서울", "area": "도봉/강남", "color": "파란색"},
    "146": {"city": "서울", "area": "성북/강남", "color": "파란색"},
    "147": {"city": "서울", "area": "노원/강남", "color": "파란색"},
    "148": {"city": "서울", "area": "노원/강남", "color": "파란색"},
    "150": {"city": "서울", "area": "강북/영등포", "color": "파란색"},
    "151": {"city": "서울", "area": "강북/마포", "color": "파란색"},
    "152": {"city": "서울", "area": "강북/마포", "color": "파란색"},
    "153": {"city": "서울", "area": "성북/마포", "color": "파란색"},
    "162": {"city": "서울", "area": "성동/강서", "color": "파란색"},
    "260": {"city": "서울", "area": "은평/강남", "color": "파란색"},
    "261": {"city": "서울", "area": "은평/강남", "color": "파란색"},
    "262": {"city": "서울", "area": "은평/강남", "color": "파란색"},
    "271": {"city": "서울", "area": "노원/강남", "color": "파란색"},
    "272": {"city": "서울", "area": "노원/강남", "color": "파란색"},
    "370": {"city": "서울", "area": "강서/강남", "color": "파란색"},
    "371": {"city": "서울", "area": "강서/도심", "color": "파란색"},
    "373": {"city": "서울", "area": "강서/강남", "color": "파란색"},
    "470": {"city": "서울", "area": "양천/강남", "color": "파란색"},
    "472": {"city": "서울", "area": "강서/도심", "color": "파란색"},
    "571": {"city": "서울", "area": "구로/도심", "color": "파란색"},
    "N13": {"city": "서울", "area": "서울 심야", "color": "파란색"},
    "N26": {"city": "서울", "area": "서울 심야", "color": "파란색"},
    # 부산 버스
    "1": {"city": "부산", "area": "부산 도심", "color": "일반"},
    "1-1": {"city": "부산", "area": "부산 도심", "color": "일반"},
    "11": {"city": "부산", "area": "부산 도심/해운대", "color": "일반"},
    "15": {"city": "부산", "area": "부산 도심", "color": "일반"},
    "40": {"city": "부산", "area": "해운대/기장", "color": "일반"},
    "41": {"city": "부산", "area": "해운대", "color": "일반"},
    "100": {"city": "부산", "area": "사상/해운대", "color": "급행"},
    "1001": {"city": "부산", "area": "부산 광역", "color": "광역"},
    # 대구 버스
    "309": {"city": "대구", "area": "대구 동구", "color": "일반"},
    "349": {"city": "대구", "area": "대구 북구", "color": "일반"},
    "401": {"city": "대구", "area": "대구 수성", "color": "일반"},
    "503": {"city": "대구", "area": "대구 달서", "color": "일반"},
}

# ─────────────────────────────────────────────────────────────────────────────
# 버스 정류장 번호 패턴 (번호판형)
# ─────────────────────────────────────────────────────────────────────────────
_BUS_STOP_PATTERN = re.compile(r'\b(\d{5,6})\b')       # 5~6자리 정류장 번호
_BUS_ROUTE_PATTERN = re.compile(r'\b(\d{1,4}[가-힣]?)\s*번\b')  # n번 버스
_SUBWAY_STATION_PATTERN = re.compile(r'([가-힣]{2,8}역)\b')      # XX역
_LINE_PATTERN = re.compile(r'(\d호선|부산\d호선|인천\d호선|대구\d호선|광주\d호선|대전\d호선)')


def match_transit(texts: list[str]) -> dict:
    """
    OCR 텍스트 목록에서 대중교통 단서 탐지
    반환: {type, name, city, lat, lon, confidence, details}
    """
    best: dict | None = None
    full_text = " ".join(texts)

    # 1) 지하철역 이름 매칭
    for text in texts:
        # "XX역" 패턴
        m = _SUBWAY_STATION_PATTERN.search(text)
        if m:
            station_raw = m.group(1)
            station_name = station_raw.rstrip("역")
            if station_name in SUBWAY_DB:
                info = SUBWAY_DB[station_name]
                return {
                    "type": "subway",
                    "name": f"{station_name}역",
                    "line": info["line"],
                    "city": info["city"],
                    "lat": info["lat"],
                    "lon": info["lon"],
                    "confidence": 0.93,
                    "details": f"지하철 {info['line']} {station_name}역",
                }
        # 역 이름 직접 포함 (XX역 없이 이름만)
        for station_name, info in SUBWAY_DB.items():
            if station_name in text and len(station_name) >= 2:
                if not best or info.get("confidence", 0) > best.get("confidence", 0):
                    best = {
                        "type": "subway",
                        "name": f"{station_name}역",
                        "line": info["line"],
                        "city": info["city"],
                        "lat": info["lat"],
                        "lon": info["lon"],
                        "confidence": 0.88,
                        "details": f"지하철 {info['line']} {station_name}역 (텍스트 매칭)",
                    }

    # 2) 버스 번호 매칭
    for text in texts:
        m = _BUS_ROUTE_PATTERN.search(text)
        if m:
            route_num = m.group(1)
            if route_num in BUS_ROUTE_DB:
                info = BUS_ROUTE_DB[route_num]
                # 버스는 정확한 좌표 없음 → 도시 수준만 제공
                city_coords = _CITY_COORDS.get(info["city"], {})
                result = {
                    "type": "bus",
                    "name": f"{route_num}번 버스",
                    "city": info["city"],
                    "area": info["area"],
                    "confidence": 0.78,
                    "details": f"{info['color']} {route_num}번 버스 ({info['area']})",
                }
                result.update(city_coords)
                if not best:
                    best = result

    if best:
        return best
    return {}


async def transit_search_api(query: str) -> list[dict]:
    """
    공공데이터포털 버스정류장 API 조회 (API 키 있을 때)
    없으면 내장 DB 텍스트 검색으로 폴백
    """
    api_key = settings.PUBLIC_DATA_API_KEY
    if api_key and api_key.strip():
        try:
            url = "http://apis.data.go.kr/1613000/BusSttnInfoInqireService/getSttnNoList"
            params = {
                "serviceKey": api_key,
                "pageNo": 1,
                "numOfRows": 5,
                "nodeNm": query,
                "_type": "json",
            }
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(url, params=params)
                data = resp.json()
                items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
                if isinstance(items, dict):
                    items = [items]
                results = []
                for item in items:
                    results.append({
                        "type": "bus_stop",
                        "name": item.get("nodenm", ""),
                        "stop_id": item.get("nodeid", ""),
                        "city": item.get("cityname", ""),
                        "lat": float(item.get("gpslati", 0)),
                        "lon": float(item.get("gpslong", 0)),
                        "confidence": 0.90,
                    })
                if results:
                    logger.debug(f"[transit_api] '{query}' → {len(results)}건")
                    return results
        except Exception as e:
            logger.debug(f"[transit_api] 실패: {e}")

    # 내장 DB 폴백 — 지하철역 이름 검색
    results = []
    q_lower = query.lower()
    for name, info in SUBWAY_DB.items():
        if query in name or q_lower in name.lower():
            results.append({
                "type": "subway",
                "name": f"{name}역",
                "line": info["line"],
                "city": info["city"],
                "lat": info["lat"],
                "lon": info["lon"],
                "confidence": 0.88,
            })
    return results[:5]


# 도시별 중심 좌표 (버스 노선 결과에 사용)
_CITY_COORDS: dict[str, dict] = {
    "서울": {"lat": 37.5665, "lon": 126.9780},
    "부산": {"lat": 35.1796, "lon": 129.0756},
    "대구": {"lat": 35.8714, "lon": 128.6014},
    "인천": {"lat": 37.4563, "lon": 126.7052},
    "광주": {"lat": 35.1595, "lon": 126.8526},
    "대전": {"lat": 36.3504, "lon": 127.3845},
    "울산": {"lat": 35.5384, "lon": 129.3114},
    "수원": {"lat": 37.2636, "lon": 127.0286},
    "고양": {"lat": 37.6584, "lon": 126.8320},
}

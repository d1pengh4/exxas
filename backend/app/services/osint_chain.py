"""
OSINT 체인 서비스 v2
웹 검색 / 네이버 블로그 / OSM Overpass / Mapillary / URL 딥 크롤
"""
import asyncio
import json
import re
import ssl
import urllib.parse
from typing import Optional
from loguru import logger

import aiohttp
import certifi

_SSL = ssl.create_default_context(cafile=certifi.where())


def _connector() -> aiohttp.TCPConnector:
    return aiohttp.TCPConnector(ssl=_SSL)


# ── DuckDuckGo 웹 검색 ──────────────────────────────────────────────────────

async def web_search(query: str, max_results: int = 6) -> list[dict]:
    """
    웹 검색 — 우선순위: SerpAPI > DuckDuckGo > Brave Search
    반환: [{"title": ..., "url": ..., "snippet": ...}]
    """
    from ..core.config import settings

    serp_key = getattr(settings, "SERP_API_KEY", "") or getattr(settings, "SERPAPI_KEY", "")
    if serp_key:
        return await _serpapi_search(query, max_results, serp_key)

    results = await _ddg_search(query, max_results)
    if results:
        return results

    # DuckDuckGo 차단/실패 시 Brave Search API 폴백
    brave_key = getattr(settings, "BRAVE_SEARCH_API_KEY", "")
    if brave_key:
        return await _brave_search(query, max_results, brave_key)

    return []


async def _brave_search(query: str, max_results: int, api_key: str) -> list[dict]:
    """Brave Search API — DDG IP 차단 시 폴백 (API 키 필요)."""
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {"q": query, "count": min(max_results, 20), "country": "KR", "search_lang": "ko"}
    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        results = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            })
        logger.debug(f"[web_search] Brave '{query}' → {len(results)}건")
        return results
    except Exception as e:
        logger.warning(f"[web_search] Brave 실패: {e}")
        return []


async def _ddg_search(query: str, max_results: int) -> list[dict]:
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    data = {"q": query, "kl": "ko-kr"}

    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                html = await resp.text()

        results = []
        # DuckDuckGo HTML 결과 파싱
        # <a class="result__a" href="...">title</a>
        # <a class="result__snippet">snippet</a>
        links = re.findall(
            r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )
        snippets = re.findall(
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )

        for i, (href, title) in enumerate(links[:max_results]):
            # DuckDuckGo redirect URL 해제
            real_url = href
            if "//duckduckgo.com/l/" in href or href.startswith("/"):
                m = re.search(r"uddg=([^&]+)", href)
                if m:
                    real_url = urllib.parse.unquote(m.group(1))

            snippet = re.sub(r"<[^>]+>", "", snippets[i]) if i < len(snippets) else ""
            results.append({
                "title": re.sub(r"<[^>]+>", "", title).strip(),
                "url": real_url,
                "snippet": snippet.strip(),
            })

        logger.debug(f"[web_search] DDG '{query}' → {len(results)}건")
        return results

    except Exception as e:
        logger.warning(f"[web_search] DDG 실패: {e}")
        return []


async def _serpapi_search(query: str, max_results: int, api_key: str = "") -> list[dict]:
    from ..core.config import settings
    if not api_key:
        api_key = getattr(settings, "SERP_API_KEY", "") or getattr(settings, "SERPAPI_KEY", "")
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": api_key,
        "hl": "ko",
        "gl": "kr",
        "num": max_results,
    }
    try:
        async with aiohttp.ClientSession(connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()

        return [
            {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
            for r in data.get("organic_results", [])[:max_results]
        ]
    except Exception as e:
        logger.warning(f"[web_search] SerpAPI 실패: {e}, DDG로 폴백")
        return await _ddg_search(query, max_results)


# ── 네이버 블로그 검색 ──────────────────────────────────────────────────────

async def search_naver_blog(query: str, max_results: int = 5) -> list[dict]:
    """
    네이버 블로그 검색 API (NAVER_CLIENT_ID/SECRET 설정 시).
    미설정 시 네이버 검색 HTML 스크래핑.
    반환: [{"title": ..., "url": ..., "description": ..., "location_hint": ...}]
    """
    from ..core.config import settings

    client_id = getattr(settings, "NAVER_CLIENT_ID", "")
    client_secret = getattr(settings, "NAVER_CLIENT_SECRET", "")

    if client_id and client_secret:
        return await _naver_blog_api(query, client_id, client_secret, max_results)

    return await _naver_blog_scrape(query, max_results)


async def _naver_blog_api(query: str, client_id: str, client_secret: str, max_results: int) -> list[dict]:
    url = "https://openapi.naver.com/v1/search/blog"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    params = {"query": query, "display": max_results, "sort": "sim"}

    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                data = await resp.json()

        results = []
        for item in data.get("items", []):
            title = re.sub(r"<[^>]+>", "", item.get("title", ""))
            desc = re.sub(r"<[^>]+>", "", item.get("description", ""))
            location_hint = _extract_location_from_text(desc)
            results.append({
                "title": title,
                "url": item.get("link", ""),
                "description": desc[:200],
                "location_hint": location_hint,
            })

        logger.debug(f"[naver_blog] API '{query}' → {len(results)}건")
        return results

    except Exception as e:
        logger.warning(f"[naver_blog] API 실패: {e}")
        return []


async def _naver_blog_scrape(query: str, max_results: int) -> list[dict]:
    """네이버 블로그 검색 HTML 스크래핑 (API 키 없을 때 폴백)"""
    url = "https://search.naver.com/search.naver"
    params = {"query": query, "where": "blog", "sm": "tab_jum"}
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"}

    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                html = await resp.text()

        # 블로그 제목/링크 파싱
        items = re.findall(
            r'<a[^>]+class="[^"]*title[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )
        descs = re.findall(
            r'<div[^>]+class="[^"]*dsc[^"]*"[^>]*>(.*?)</div>',
            html, re.DOTALL
        )

        results = []
        for i, (link, title) in enumerate(items[:max_results]):
            desc_raw = descs[i] if i < len(descs) else ""
            desc = re.sub(r"<[^>]+>", "", desc_raw).strip()[:200]
            location_hint = _extract_location_from_text(desc)
            results.append({
                "title": re.sub(r"<[^>]+>", "", title).strip(),
                "url": link,
                "description": desc,
                "location_hint": location_hint,
            })

        logger.debug(f"[naver_blog] scrape '{query}' → {len(results)}건")
        return results

    except Exception as e:
        logger.warning(f"[naver_blog] 스크래핑 실패: {e}")
        return []


# ── OSM Overpass API (POI 검색) ─────────────────────────────────────────────

async def osm_poi_search(
    query: str,
    lat: float,
    lon: float,
    radius_m: int = 500,
    max_results: int = 8,
) -> list[dict]:
    """
    OpenStreetMap Overpass API로 반경 내 POI 검색.
    반환: [{"name": ..., "type": ..., "lat": ..., "lon": ..., "address": ...}]
    """
    # Overpass QL — name 포함 또는 타입 매칭
    overpass_url = "https://overpass-api.de/api/interpreter"

    ql = f"""
[out:json][timeout:10];
(
  node["name"~"{query}",i](around:{radius_m},{lat},{lon});
  way["name"~"{query}",i](around:{radius_m},{lat},{lon});
  node["amenity"](around:{radius_m},{lat},{lon});
  node["shop"](around:{radius_m},{lat},{lon});
);
out center {max_results};
"""

    try:
        async with aiohttp.ClientSession(connector=_connector()) as session:
            async with session.post(
                overpass_url,
                data={"data": ql},
                timeout=aiohttp.ClientTimeout(total=12),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

        results = []
        for elem in data.get("elements", [])[:max_results]:
            tags = elem.get("tags", {})
            name = tags.get("name", tags.get("name:ko", ""))
            if not name:
                continue

            elem_lat = elem.get("lat") or elem.get("center", {}).get("lat", lat)
            elem_lon = elem.get("lon") or elem.get("center", {}).get("lon", lon)

            addr_parts = [
                tags.get("addr:country", ""),
                tags.get("addr:city", ""),
                tags.get("addr:street", ""),
                tags.get("addr:housenumber", ""),
            ]
            address = " ".join(p for p in addr_parts if p)

            results.append({
                "name": name,
                "type": tags.get("amenity") or tags.get("shop") or tags.get("tourism") or "poi",
                "lat": elem_lat,
                "lon": elem_lon,
                "address": address,
                "tags": {k: v for k, v in tags.items() if k in ("phone", "website", "opening_hours")},
            })

        logger.debug(f"[osm_poi] '{query}' @ ({lat:.4f},{lon:.4f}) r={radius_m}m → {len(results)}건")
        return results

    except Exception as e:
        logger.warning(f"[osm_poi] 실패: {e}")
        return []


# ── Mapillary 근처 이미지 (street view 대체) ─────────────────────────────────

async def mapillary_nearby(
    lat: float,
    lon: float,
    radius_m: int = 100,
    max_results: int = 5,
) -> list[dict]:
    """
    Mapillary API로 근처 street-level 이미지 검색.
    MAPILLARY_TOKEN 설정 시 실제 API 호출.
    반환: [{"id": ..., "lat": ..., "lon": ..., "captured_at": ..., "thumb_url": ...}]
    """
    from ..core.config import settings
    token = getattr(settings, "MAPILLARY_TOKEN", "")

    if not token:
        logger.debug("[mapillary] 토큰 없음 — 스킵")
        return []

    url = "https://graph.mapillary.com/images"
    params = {
        "access_token": token,
        "fields": "id,geometry,captured_at,thumb_256_url,compass_angle",
        "bbox": _bbox(lat, lon, radius_m),
        "limit": max_results,
    }

    try:
        async with aiohttp.ClientSession(connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

        results = []
        for feat in data.get("data", []):
            coords = feat.get("geometry", {}).get("coordinates", [0, 0])
            results.append({
                "id": feat["id"],
                "lat": coords[1],
                "lon": coords[0],
                "captured_at": feat.get("captured_at", ""),
                "thumb_url": feat.get("thumb_256_url", ""),
                "compass_angle": feat.get("compass_angle"),
            })

        logger.debug(f"[mapillary] ({lat:.4f},{lon:.4f}) r={radius_m}m → {len(results)}건")
        return results

    except Exception as e:
        logger.warning(f"[mapillary] 실패: {e}")
        return []


# ── URL 딥 크롤 ──────────────────────────────────────────────────────────────

async def deep_crawl_url(url: str) -> dict:
    """
    단일 URL 딥 크롤 — 지리 메타태그, OG 태그, 주소 키워드, 지도 링크 파싱.
    반환: {"url": ..., "title": ..., "location_hints": [...], "lat": ..., "lon": ..., "text_excerpt": ...}
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    }

    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=10),
                allow_redirects=True,
                max_redirects=3,
            ) as resp:
                if resp.status != 200:
                    return {"url": url, "error": f"HTTP {resp.status}"}
                # 최대 500KB만 읽기
                raw = await resp.content.read(512 * 1024)
                html = raw.decode("utf-8", errors="ignore")

    except Exception as e:
        logger.warning(f"[deep_crawl] 접속 실패 {url}: {e}")
        return {"url": url, "error": str(e)}

    result: dict = {"url": url, "location_hints": [], "lat": None, "lon": None}

    # 제목
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
    result["title"] = re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else ""

    # geo.position 메타 태그
    m = re.search(r'<meta[^>]+name=["\']geo\.position["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if m:
        coords = m.group(1).split(";")
        if len(coords) >= 2:
            try:
                result["lat"] = float(coords[0].strip())
                result["lon"] = float(coords[1].strip())
                result["location_hints"].append(f"geo.position: {result['lat']},{result['lon']}")
            except ValueError:
                pass

    # OG 지역 메타 태그
    for prop in ("og:locality", "og:region", "og:country-name", "place:location:latitude", "place:location:longitude"):
        prop_escaped = re.escape(prop)
        pat = r'<meta[^>]+property=["\']' + prop_escaped + r'["\'][^>]+content=["\']([^"\']+)["\']'
        m = re.search(pat, html, re.IGNORECASE)
        if m:
            result["location_hints"].append(f"{prop}: {m.group(1)}")

    # Naver 지도 링크에서 좌표 추출
    for pat in [
        r"maps\.naver\.com[^\s\"']*[?&]lat=([0-9.\-]+)[^\s\"']*[?&]lng=([0-9.\-]+)",
        r"maps\.naver\.com[^\s\"']*lng=([0-9.\-]+)[^\s\"']*lat=([0-9.\-]+)",
        r"map\.kakao\.com[^\s\"']*[?&]q=([0-9.\-]+),([0-9.\-]+)",
    ]:
        m = re.search(pat, html)
        if m and result["lat"] is None:
            try:
                a, b = float(m.group(1)), float(m.group(2))
                # lat은 보통 30~40, lon은 110~140 (한국)
                if 30 <= a <= 45:
                    result["lat"], result["lon"] = a, b
                else:
                    result["lat"], result["lon"] = b, a
                result["location_hints"].append(f"지도링크: {result['lat']},{result['lon']}")
            except ValueError:
                pass

    # Google Maps 링크
    m = re.search(r"google\.com/maps[^\s\"']*@([0-9.\-]+),([0-9.\-]+)", html)
    if m and result["lat"] is None:
        try:
            result["lat"] = float(m.group(1))
            result["lon"] = float(m.group(2))
            result["location_hints"].append(f"Google Maps: {result['lat']},{result['lon']}")
        except ValueError:
            pass

    # 본문 주소/위치 키워드 추출
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    location_hint = _extract_location_from_text(text[:3000])
    if location_hint:
        result["location_hints"].append(f"본문키워드: {location_hint}")

    result["text_excerpt"] = text[:500].strip()

    logger.debug(f"[deep_crawl] {url} → hints={result['location_hints']}")
    return result


# ── 내부 유틸 ────────────────────────────────────────────────────────────────

def _extract_location_from_text(text: str) -> str:
    """텍스트에서 한국/해외 주소·위치 키워드 추출"""
    found = []

    # 한국 주요 지명 (빠른 매칭)
    major_kr = [
        "해운대", "광안리", "송정", "기장", "남포동", "서면", "동래",  # 부산
        "강남", "강북", "홍대", "신촌", "명동", "이태원", "압구정", "강동",  # 서울
        "한강공원", "반포", "여의도", "잠실", "성수", "건대", "혜화",  # 서울 2
        "롯데월드", "롯데타워", "코엑스", "에버랜드", "남이섬", "스카이워크",  # 랜드마크
        "제주도", "성산일출봉", "한라산", "협재", "중문", "서귀포",  # 제주
        "경복궁", "광화문", "북촌", "인사동", "남산",  # 서울 관광
        "속초", "강릉", "춘천", "원주",  # 강원
        "전주", "여수", "순천", "광양",  # 전라
        "경주", "포항", "안동", "구미", "울산",  # 경상
        "인천공항", "김포공항", "수원", "판교", "분당",  # 수도권
    ]
    for kw in major_kr:
        if kw in text and kw not in found:
            found.append(kw)
            if len(found) >= 3:
                break

    # 한국 주소 패턴
    kr_patterns = [
        r"([가-힣]{1,4}(?:특별시|광역시|특별자치시|특별자치도|도))\s*[가-힣]{1,4}(?:시|군|구)",
        r"([가-힣]{1,6}(?:구|군))\s*([가-힣]{1,6}(?:동|읍|면))",
        r"([가-힣]{1,6}(?:로|길))\s*\d+",  # 도로명주소
        r"(\d{5})\s*([가-힣])",  # 한국 우편번호
    ]
    for pat in kr_patterns:
        for m in re.findall(pat, text)[:2]:
            loc = m if isinstance(m, str) else " ".join(p for p in m if p)
            if loc and loc not in found:
                found.append(loc)
            if len(found) >= 4:
                break

    # 해외 도시/국가 패턴 (영어)
    intl_patterns = [
        r"\b(Tokyo|Osaka|Kyoto|Yokohama|Nagoya|Sapporo)\b",
        r"\b(Beijing|Shanghai|Guangzhou|Shenzhen|Chengdu|Hangzhou)\b",
        r"\b(New York|Los Angeles|Chicago|San Francisco|Seattle|Boston|Miami)\b",
        r"\b(London|Paris|Berlin|Rome|Madrid|Amsterdam|Vienna|Zurich)\b",
        r"\b(Bangkok|Singapore|Kuala Lumpur|Jakarta|Manila|Ho Chi Minh|Hanoi)\b",
        r"\b(Sydney|Melbourne|Auckland|Toronto|Vancouver|Montreal)\b",
        r"\b(Dubai|Riyadh|Istanbul|Cairo|Mumbai|Delhi|Bangalore)\b",
        r"\b(Taiwan|Taipei|Hong Kong|Macau)\b",
        # 일본 도시 (한자/히라가나)
        r"(東京|大阪|京都|横浜|名古屋|札幌|福岡|神戸)",
        # 중국 도시
        r"(北京|上海|广州|深圳|成都|杭州|武汉|西安)",
        # 우편번호 패턴 (〒일본, 미국 5자리)
        r"〒\d{3}-?\d{4}",
        r"\b\d{5}(?:-\d{4})?\b",  # 미국 ZIP
    ]
    for pat in intl_patterns:
        for m in re.findall(pat, text, re.IGNORECASE)[:1]:
            loc = m if isinstance(m, str) else m[0]
            if loc and loc not in found:
                found.append(loc)
                break  # 카테고리당 1개

    return ", ".join(found[:4])


def _bbox(lat: float, lon: float, radius_m: int) -> str:
    """위도/경도 반경을 bbox 문자열로 변환"""
    deg_per_m_lat = 1 / 111_320
    deg_per_m_lon = 1 / (111_320 * abs(max(0.01, __import__("math").cos(__import__("math").radians(lat)))))
    dlat = radius_m * deg_per_m_lat
    dlon = radius_m * deg_per_m_lon
    return f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"


# ═══════════════════════════════════════════════════════════════════════════════
# OSINT 체인 엔진 v3 — 단서 연쇄 자동화
# ═══════════════════════════════════════════════════════════════════════════════

# ── 사업자등록번호 조회 ────────────────────────────────────────────────────────

async def biz_reg_lookup(reg_number: str) -> dict:
    """
    한국 사업자등록번호 → 상호명/주소 조회.
    공공데이터포털 API → 웹 검색 폴백.
    reg_number: "000-00-00000" 형식
    """
    reg_clean = re.sub(r"[^0-9]", "", reg_number)
    if len(reg_clean) != 10:
        return {"error": "사업자등록번호는 10자리여야 합니다"}

    formatted = f"{reg_clean[:3]}-{reg_clean[3:5]}-{reg_clean[5:]}"

    # 공공데이터포털 사업자 진위확인 API 폴백: 웹 검색
    queries = [
        f"사업자등록번호 {formatted} 상호 주소",
        f'"{formatted}" 업체',
        f"사업자 {reg_clean} 회사명",
    ]
    for q in queries:
        results = await web_search(q, max_results=4)
        for r in results:
            text = r.get("title", "") + " " + r.get("snippet", "")
            loc = _extract_location_from_text(text)
            # 상호명 패턴 추출
            store = ""
            for pat in [
                r'["\'【\[]([가-힣a-zA-Z0-9\s]{2,30})["\'】\]]',
                r'상호[:\s]+([가-힣a-zA-Z0-9\s]{2,20})',
                r'업체명[:\s]+([가-힣a-zA-Z0-9\s]{2,20})',
            ]:
                m = re.search(pat, text)
                if m:
                    store = m.group(1).strip()
                    break
            if loc or store:
                return {
                    "reg_number": formatted,
                    "store_name": store,
                    "location_hint": loc,
                    "source_url": r.get("url", ""),
                    "raw_snippet": text[:200],
                }
    return {"reg_number": formatted, "error": "검색 결과 없음"}


# ── 전화번호 → 상호명/주소 체인 ───────────────────────────────────────────────

async def phone_lookup(phone: str) -> dict:
    """
    전화번호 → 상호명/주소 조회 (웹 검색 체인).
    지역코드로 도시 확정 + 상호 검색.
    """
    phone_clean = re.sub(r"[^\d+]", "", phone)

    # 지역코드 → 도시 (한국)
    KR_CODES = {
        "02": "서울", "031": "경기", "032": "인천", "033": "강원",
        "041": "충남", "042": "대전", "043": "충북", "044": "세종",
        "051": "부산", "052": "울산", "053": "대구", "054": "경북",
        "055": "경남", "061": "전남", "062": "광주", "063": "전북", "064": "제주",
    }
    city = ""
    for code, name in sorted(KR_CODES.items(), key=lambda x: -len(x[0])):
        if phone_clean.startswith(code) or phone_clean.startswith("82" + code.lstrip("0")):
            city = name
            break

    queries = [
        f'전화 "{phone}" 상호 주소',
        f"{phone} 업체 위치",
    ]
    if city:
        queries.insert(0, f"{city} {phone} 상호명")

    for q in queries:
        results = await web_search(q, max_results=4)
        for r in results:
            text = r.get("title", "") + " " + r.get("snippet", "")
            loc = _extract_location_from_text(text)
            store = ""
            for pat in [r'([가-힣a-zA-Z0-9]{2,20}(?:점|식당|카페|마트|병원|약국|학원|센터|빌딩))',
                        r'상호[:\s]+([가-힣a-zA-Z0-9\s]{2,20})']:
                m = re.search(pat, text)
                if m:
                    store = m.group(1).strip()
                    break
            if loc or store:
                return {
                    "phone": phone,
                    "city_from_area_code": city,
                    "store_name": store,
                    "location_hint": loc,
                    "source_url": r.get("url", ""),
                }

    return {"phone": phone, "city_from_area_code": city, "error": "상호 검색 결과 없음"}


# ── SNS/블로그 URL 위치 추출 ─────────────────────────────────────────────────

async def crawl_social_location(url: str) -> dict:
    """
    SNS/블로그 URL에서 위치 정보 심층 추출.
    Instagram, Naver Blog, Twitter, YouTube, 일반 웹페이지 지원.
    반환: {"url", "location_hints", "lat", "lon", "place_name", "address"}
    """
    result: dict = {
        "url": url,
        "location_hints": [],
        "lat": None,
        "lon": None,
        "place_name": "",
        "address": "",
    }

    # 기본 크롤
    base = await deep_crawl_url(url)
    result["location_hints"].extend(base.get("location_hints", []))
    if base.get("lat"):
        result["lat"] = base["lat"]
        result["lon"] = base["lon"]

    text = base.get("text_excerpt", "")

    # Naver 블로그: 내장된 지도 링크 추출
    if "blog.naver.com" in url or "m.blog.naver.com" in url:
        # 네이버 지도 장소 링크 패턴
        map_ids = re.findall(r'place\.map\.naver\.com/place/(\d+)', text + base.get("title", ""))
        for place_id in map_ids[:2]:
            place_info = await _naver_place_by_id(place_id)
            if place_info:
                result["place_name"] = place_info.get("name", "")
                result["address"] = place_info.get("address", "")
                if place_info.get("lat"):
                    result["lat"] = place_info["lat"]
                    result["lon"] = place_info["lon"]
                result["location_hints"].append(
                    f"네이버 블로그 내 장소: {place_info.get('name')} @ {place_info.get('address')}"
                )
                break

        # 블로그 본문 주소 추출
        loc = _extract_location_from_text(text)
        if loc:
            result["location_hints"].append(f"블로그 본문: {loc}")

    # Instagram: og:description / 위치 태그
    elif "instagram.com" in url:
        headers = {
            "User-Agent": "facebookexternalhit/1.1",  # Instagram OG 메타 접근
            "Accept-Language": "ko-KR,ko;q=0.9",
        }
        try:
            async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    html = await resp.text()
            # 위치 태그: "location" 또는 "에서" 패턴
            m = re.search(r'"location":\s*\{[^}]*"name"\s*:\s*"([^"]+)"', html)
            if m:
                result["place_name"] = m.group(1)
                result["location_hints"].append(f"Instagram 위치태그: {m.group(1)}")
            # OG description에서 위치 추출
            m = re.search(r'<meta[^>]+property="og:description"[^>]+content="([^"]+)"', html, re.IGNORECASE)
            if m:
                loc = _extract_location_from_text(m.group(1))
                if loc:
                    result["location_hints"].append(f"Instagram 설명: {loc}")
        except Exception as e:
            logger.debug(f"Instagram crawl: {e}")

    # YouTube: 영상 위치 메타데이터
    elif "youtube.com" in url or "youtu.be" in url:
        vid_id = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
        if vid_id:
            try:
                api_url = f"https://www.youtube.com/oembed?url={urllib.parse.quote(url)}&format=json"
                async with aiohttp.ClientSession(connector=_connector()) as session:
                    async with session.get(api_url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            loc = _extract_location_from_text(
                                data.get("title", "") + " " + data.get("author_name", "")
                            )
                            if loc:
                                result["location_hints"].append(f"YouTube: {loc}")
            except Exception as e:
                logger.debug(f"YouTube crawl: {e}")

    # 카카오맵 공유 링크
    elif "kakaomap.com" in url or "map.kakao.com" in url:
        m = re.search(r"q=([0-9.\-]+),([0-9.\-]+)", url)
        if m:
            result["lat"] = float(m.group(1))
            result["lon"] = float(m.group(2))
            result["location_hints"].append(f"카카오맵 좌표: {result['lat']},{result['lon']}")

    logger.debug(f"[crawl_social] {url} → hints={len(result['location_hints'])}, lat={result['lat']}")
    return result


async def _naver_place_by_id(place_id: str) -> dict:
    """Naver Place ID → 상세 정보"""
    url = f"https://place.map.naver.com/place/{place_id}/home"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/html",
        "Referer": "https://map.naver.com/",
    }
    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                html = await resp.text()
        name = ""
        m = re.search(r'"name"\s*:\s*"([^"]+)"', html)
        if m:
            name = m.group(1)
        addr = ""
        m = re.search(r'"address"\s*:\s*"([^"]+)"', html)
        if m:
            addr = m.group(1)
        lat, lon = None, None
        m = re.search(r'"y"\s*:\s*"([0-9.]+)".*?"x"\s*:\s*"([0-9.]+)"', html, re.DOTALL)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
        return {"name": name, "address": addr, "lat": lat, "lon": lon}
    except Exception:
        return {}


# ── 검색 결과 URL 자동 체인 크롤 ──────────────────────────────────────────────

async def chain_crawl_urls(search_results: list[dict], max_urls: int = 4) -> list[dict]:
    """
    웹/블로그 검색 결과 URL을 자동으로 딥 크롤 → 위치 정보 체인 추출.
    SNS URL은 crawl_social_location으로 전문 처리.
    """
    crawled = []
    seen = set()
    for r in search_results:
        url = r.get("url", "")
        if not url or url in seen or not url.startswith("http"):
            continue
        seen.add(url)
        if len(crawled) >= max_urls:
            break

        try:
            if any(d in url for d in ("instagram.com", "blog.naver.com", "youtube.com", "youtu.be", "kakaomap")):
                result = await crawl_social_location(url)
            else:
                result = await deep_crawl_url(url)
            crawled.append(result)
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.debug(f"chain_crawl {url}: {e}")

    return crawled


# ── 자동 단서 체인 (핵심 엔진) ────────────────────────────────────────────────

_CLUE_TYPE_MAP = {
    "phone": "phone",
    "전화": "phone",
    "biz_reg": "biz_reg",
    "사업자": "biz_reg",
    "store": "store",
    "상호": "store",
    "brand": "store",
    "브랜드": "store",
    "가게": "store",
    "가게이름": "store",
    "address": "address",
    "주소": "address",
    "menu": "menu",
    "메뉴": "menu",
    "url": "url",
    "barcode": "barcode",
    "바코드": "barcode",
    "image_url": "image_url",
    "building": "building",
    "건물": "building",
    "건물명": "building",
    "product": "product",
    "제품": "product",
    "제품명": "product",
    "상품": "product",
}


async def auto_chain_from_clue(clue_type: str, value: str, region_hint: str = "") -> dict:
    """
    단서 타입+값을 받아 자동으로 최적 OSINT 체인 실행.
    - phone → 지역코드 → 상호 웹 검색 → Naver Place
    - biz_reg → 사업자 조회 → 상호명 → Naver Place
    - store/brand → Naver Place + 웹 검색 + 블로그 병렬
    - address → Naver Place + OSM POI
    - menu → 맛집 검색 → Naver Place
    - url → crawl_social_location → 딥 크롤 체인
    반환: {"clue_type", "value", "chain_steps", "location_candidates", "best_guess", "confidence"}
    """
    from .indoor_analyzer import naver_place_scrape

    clue_type = _CLUE_TYPE_MAP.get(clue_type.lower(), clue_type.lower())
    chain_steps = []
    location_scores: dict[str, float] = {}
    best_place: dict = {}

    async def _score_place(place: dict, weight: float):
        if place.get("address") and place.get("lat"):
            key = place["address"]
            location_scores[key] = location_scores.get(key, 0) + weight
            nonlocal best_place
            if not best_place or weight > location_scores.get(best_place.get("address", ""), 0):
                best_place = place

    async def _naver_chain(query: str, weight: float = 3.0):
        places = await naver_place_scrape(query + (" " + region_hint if region_hint else ""), max_results=3)
        chain_steps.append({"step": "naver_place", "query": query, "results": len(places)})
        for p in places:
            await _score_place(p, weight)
        return places

    async def _web_chain(query: str, weight: float = 1.5):
        results = await web_search(query, max_results=5)
        chain_steps.append({"step": "web_search", "query": query, "results": len(results)})
        all_text = " ".join(r.get("title", "") + " " + r.get("snippet", "") for r in results)
        hint = _extract_location_from_text(all_text)
        if hint:
            location_scores[hint] = location_scores.get(hint, 0) + weight
        # URL 체인: 상위 2개 URL 크롤
        crawled = await chain_crawl_urls(results, max_urls=2)
        for c in crawled:
            if c.get("lat"):
                location_scores[f"({c['lat']:.4f},{c['lon']:.4f})"] = \
                    location_scores.get(f"({c['lat']:.4f},{c['lon']:.4f})", 0) + weight * 1.5
        return results, hint

    # ── phone ──────────────────────────────────────────────────────────────────
    if clue_type == "phone":
        phone_result = await phone_lookup(value)
        chain_steps.append({"step": "phone_lookup", "result": phone_result})
        city = phone_result.get("city_from_area_code", "")
        store = phone_result.get("store_name", "")
        if city:
            location_scores[city] = location_scores.get(city, 0) + 2.5
        if store:
            await _naver_chain(store + (" " + city if city else ""), weight=4.0)
            await _web_chain(f"{store} {city} 위치 주소", weight=2.0)

    # ── biz_reg ────────────────────────────────────────────────────────────────
    elif clue_type == "biz_reg":
        biz = await biz_reg_lookup(value)
        chain_steps.append({"step": "biz_reg_lookup", "result": biz})
        store = biz.get("store_name", "")
        loc = biz.get("location_hint", "")
        if loc:
            location_scores[loc] = location_scores.get(loc, 0) + 3.0
        if store:
            await _naver_chain(store, weight=4.5)
            await _web_chain(f"{store} 주소 위치", weight=2.0)

    # ── store / brand ──────────────────────────────────────────────────────────
    elif clue_type in ("store", "brand"):
        # 병렬: Naver Place + 웹 검색 + 블로그
        naver_task = _naver_chain(value, weight=4.0)
        web_task = _web_chain(f"{value} 위치 주소 지점", weight=2.0)
        blog_results = await search_naver_blog(f"{value} 방문 후기 위치", max_results=3)
        chain_steps.append({"step": "naver_blog", "query": value, "results": len(blog_results)})
        for b in blog_results:
            hint = b.get("location_hint", "")
            if hint:
                location_scores[hint] = location_scores.get(hint, 0) + 1.5
        await asyncio.gather(naver_task, web_task)

        # 2차 체인: 지역힌트 + 상호명 재검색
        if location_scores and not best_place.get("lat"):
            top_region = max(location_scores, key=lambda k: location_scores[k])
            await _naver_chain(f"{value} {top_region}", weight=3.0)

    # ── address ────────────────────────────────────────────────────────────────
    elif clue_type == "address":
        await _naver_chain(value, weight=4.0)
        # 주소에서 도로명만 추출해 재검색
        road_match = re.search(r"([가-힣]{1,6}로\s*\d+)", value)
        if road_match:
            await _naver_chain(road_match.group(1), weight=2.5)
        loc = _extract_location_from_text(value)
        if loc:
            location_scores[loc] = location_scores.get(loc, 0) + 2.0

    # ── menu ───────────────────────────────────────────────────────────────────
    elif clue_type == "menu":
        region = region_hint or "서울"
        queries = [
            f"{value} 맛집 {region}",
            f"{value} 음식점 위치",
            f"{value} 식당 주소",
        ]
        tasks = [_naver_chain(q, weight=3.0) for q in queries[:2]]
        tasks.append(_web_chain(queries[2], weight=1.5))
        await asyncio.gather(*tasks)

    # ── url ────────────────────────────────────────────────────────────────────
    elif clue_type == "url":
        social = await crawl_social_location(value)
        chain_steps.append({"step": "crawl_social", "url": value, "hints": social.get("location_hints", [])})
        if social.get("lat"):
            coord_key = f"({social['lat']:.4f},{social['lon']:.4f})"
            location_scores[coord_key] = location_scores.get(coord_key, 0) + 5.0
            best_place = {"name": social.get("place_name", ""), "address": social.get("address", ""),
                          "lat": social["lat"], "lon": social["lon"]}
        for hint in social.get("location_hints", []):
            location_scores[hint] = location_scores.get(hint, 0) + 2.0
        # 장소명 발견 시 Naver Place 체인
        if social.get("place_name"):
            await _naver_chain(social["place_name"], weight=3.5)

    # ── building (건물명) ───────────────────────────────────────────────────────
    elif clue_type == "building":
        # 건물명 → Naver Place + 웹 검색 + 블로그
        naver_task = _naver_chain(value, weight=4.5)
        web_task = _web_chain(f"{value} 건물 주소 위치", weight=2.0)
        blog_results = await search_naver_blog(f"{value} 건물 위치 방문", max_results=3)
        chain_steps.append({"step": "naver_blog_building", "query": value, "results": len(blog_results)})
        for b in blog_results:
            hint = b.get("location_hint", "")
            if hint:
                location_scores[hint] = location_scores.get(hint, 0) + 1.5
        await asyncio.gather(naver_task, web_task)
        # 지역힌트 확보 후 2차 Naver Place
        if location_scores and not best_place.get("lat"):
            top_region = max(location_scores, key=lambda k: location_scores[k])
            await _naver_chain(f"{value} {top_region}", weight=3.0)
        # 부가: "XXX 빌딩", "XXX 타워" 변형도 검색
        for suffix in ("빌딩", "타워", "센터", "플라자"):
            if suffix not in value:
                await _web_chain(f"{value}{suffix} 주소", weight=1.0)
                break

    # ── product (제품/브랜드명) ────────────────────────────────────────────────
    elif clue_type == "product":
        # 제품명 → 제조국/원산지 + 판매처 + 배포 지역
        queries = [
            f"{value} 제조국 원산지",
            f"{value} 판매 지역 어디",
            f"{value} 브랜드 본사 위치",
            f'"{value}" where sold country',
        ]
        tasks = [_web_chain(q, weight=2.0) for q in queries[:3]]
        await asyncio.gather(*tasks)
        # 제품이 한국 특산품/지역 한정인지 확인
        blog_results = await search_naver_blog(f"{value} 어디서 파나요 구매", max_results=3)
        chain_steps.append({"step": "naver_blog_product", "query": value, "results": len(blog_results)})
        for b in blog_results:
            hint = b.get("location_hint", "")
            if hint:
                location_scores[hint] = location_scores.get(hint, 0) + 2.0
        # Naver Place에도 상품명으로 검색 (특산품 가게)
        await _naver_chain(f"{value} 판매점", weight=2.5)

    # ── 결과 정렬 ─────────────────────────────────────────────────────────────
    candidates = sorted(location_scores.items(), key=lambda x: -x[1])
    best_guess = ""
    if best_place.get("address") and best_place.get("lat"):
        best_guess = f"{best_place.get('name', '')} {best_place['address']} ({best_place['lat']:.4f},{best_place['lon']:.4f})"
    elif candidates:
        best_guess = candidates[0][0]

    total_score = sum(v for _, v in candidates[:3])
    confidence = min(0.95, total_score / 15.0) if total_score > 0 else 0.0

    logger.info(f"[auto_chain] {clue_type}='{value}' → best={best_guess} conf={confidence:.2f}")
    return {
        "clue_type": clue_type,
        "value": value,
        "chain_steps": chain_steps,
        "location_candidates": [{"location": loc, "score": sc} for loc, sc in candidates[:5]],
        "best_guess": best_guess,
        "best_place": best_place,
        "confidence": round(confidence, 3),
    }


# ── 역방향 이미지 검색 결과 URL 체인 ──────────────────────────────────────────

async def chain_from_reverse_search(reverse_results: list[dict]) -> dict:
    """
    역방향 이미지 검색 결과(URL 목록)를 자동으로 체인 크롤.
    SNS/블로그/뉴스 URL을 우선 처리하여 위치 정보 추출.
    """
    # 우선순위: SNS → 블로그 → 뉴스 → 일반
    def _priority(url: str) -> int:
        if any(d in url for d in ("instagram.com", "blog.naver.com")):
            return 0
        if any(d in url for d in ("youtube.com", "tiktok.com", "twitter.com", "x.com")):
            return 1
        if any(d in url for d in ("news.", ".news.", "media.", "press.")):
            return 2
        return 3

    sorted_results = sorted(reverse_results, key=lambda r: _priority(r.get("url", "")))
    crawled = await chain_crawl_urls(sorted_results, max_urls=5)

    location_scores: dict[str, float] = {}
    best_with_coords: dict = {}

    for c in crawled:
        url = c.get("url", "")
        # 좌표 획득 시 최고 점수
        if c.get("lat") and c.get("lon"):
            key = f"({c['lat']:.4f},{c['lon']:.4f})"
            w = 5.0 if "instagram" in url or "blog.naver" in url else 3.0
            location_scores[key] = location_scores.get(key, 0) + w
            if not best_with_coords:
                best_with_coords = c
        # 위치 힌트
        for hint in c.get("location_hints", []):
            clean = re.sub(r"^[^:]+: ", "", hint)  # "본문키워드: " 제거
            location_scores[clean] = location_scores.get(clean, 0) + 2.0
        # place_name
        if c.get("place_name"):
            location_scores[c["place_name"]] = location_scores.get(c["place_name"], 0) + 3.0

    candidates = sorted(location_scores.items(), key=lambda x: -x[1])
    best_guess = ""
    if best_with_coords.get("lat"):
        best_guess = (f"{best_with_coords.get('place_name', '')} "
                      f"({best_with_coords['lat']:.4f},{best_with_coords['lon']:.4f})").strip()
    elif candidates:
        best_guess = candidates[0][0]

    return {
        "crawled_count": len(crawled),
        "location_candidates": [{"location": loc, "score": sc} for loc, sc in candidates[:5]],
        "best_guess": best_guess,
        "best_coords": {"lat": best_with_coords.get("lat"), "lon": best_with_coords.get("lon")}
            if best_with_coords else {},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# A. 네이버 통합 API 확장
# ═══════════════════════════════════════════════════════════════════════════════

async def naver_news_search(query: str, max_results: int = 5) -> list[dict]:
    """
    네이버 뉴스 검색 API — 없으면 HTML 스크래핑 폴백.
    반환: [{"title", "url", "description", "pub_date", "location_hint"}]
    """
    from ..core.config import settings
    client_id = getattr(settings, "NAVER_CLIENT_ID", "")
    client_secret = getattr(settings, "NAVER_CLIENT_SECRET", "")

    if client_id and client_secret:
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        }
        params = {"query": query, "display": max_results, "sort": "date"}
        try:
            async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = []
                        for item in data.get("items", []):
                            title = re.sub(r"<[^>]+>", "", item.get("title", ""))
                            desc = re.sub(r"<[^>]+>", "", item.get("description", ""))
                            results.append({
                                "title": title,
                                "url": item.get("originallink", item.get("link", "")),
                                "description": desc[:200],
                                "pub_date": item.get("pubDate", ""),
                                "location_hint": _extract_location_from_text(title + " " + desc),
                            })
                        logger.debug(f"[naver_news] API '{query}' → {len(results)}건")
                        return results
        except Exception as e:
            logger.debug(f"[naver_news] API 실패, 스크래핑 폴백: {e}")

    return await _naver_news_scrape(query, max_results)


async def _naver_news_scrape(query: str, max_results: int) -> list[dict]:
    """네이버 뉴스 HTML 스크래핑 폴백"""
    url = "https://search.naver.com/search.naver"
    params = {"query": query, "where": "news", "sm": "tab_jum", "sort": "1"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9",
        "Referer": "https://www.naver.com/",
    }
    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                html = await resp.text()

        results = []
        # 뉴스 기사 제목+링크
        articles = re.findall(
            r'<a[^>]+class="[^"]*news_tit[^"]*"[^>]+href="([^"]+)"[^>]*title="([^"]+)"',
            html
        )
        # 설명 텍스트
        descs = re.findall(r'<div[^>]+class="[^"]*dsc_wrap[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL)
        # 날짜
        dates = re.findall(r'<span[^>]+class="[^"]*info[^"]*"[^>]*>\s*([0-9]{4}\.[0-9]{2}\.[0-9]{2})', html)

        for i, (link, title) in enumerate(articles[:max_results]):
            desc_raw = descs[i] if i < len(descs) else ""
            desc = re.sub(r"<[^>]+>", "", desc_raw).strip()[:200]
            pub_date = dates[i] if i < len(dates) else ""
            results.append({
                "title": title,
                "url": link,
                "description": desc,
                "pub_date": pub_date,
                "location_hint": _extract_location_from_text(title + " " + desc),
            })

        logger.debug(f"[naver_news] scrape '{query}' → {len(results)}건")
        return results
    except Exception as e:
        logger.warning(f"[naver_news] 스크래핑 실패: {e}")
        return []


async def naver_local_search(query: str, max_results: int = 5) -> list[dict]:
    """
    네이버 로컬(장소) 검색 API — 없으면 Naver Place 스크래핑 폴백.
    반환: [{"title", "address", "roadAddress", "lat", "lon", "category", "telephone"}]
    """
    from ..core.config import settings
    client_id = getattr(settings, "NAVER_CLIENT_ID", "")
    client_secret = getattr(settings, "NAVER_CLIENT_SECRET", "")

    if client_id and client_secret:
        url = "https://openapi.naver.com/v1/search/local.json"
        headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        }
        params = {"query": query, "display": max_results, "sort": "comment"}
        try:
            async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = []
                        for item in data.get("items", []):
                            title = re.sub(r"<[^>]+>", "", item.get("title", ""))
                            mapx = item.get("mapx", "")
                            mapy = item.get("mapy", "")
                            lat, lon = None, None
                            if mapx and mapy:
                                try:
                                    lat = float(mapy) / 1e7
                                    lon = float(mapx) / 1e7
                                    if not (33 <= lat <= 38 and 124 <= lon <= 132):
                                        lat, lon = None, None
                                except ValueError:
                                    pass
                            results.append({
                                "title": title,
                                "address": item.get("address", ""),
                                "roadAddress": item.get("roadAddress", ""),
                                "lat": lat, "lon": lon,
                                "category": item.get("category", ""),
                                "telephone": item.get("telephone", ""),
                            })
                        logger.debug(f"[naver_local] API '{query}' → {len(results)}건")
                        return results
        except Exception as e:
            logger.debug(f"[naver_local] API 실패, 스크래핑 폴백: {e}")

    # 폴백: naver_place_scrape (indoor_analyzer) → 동일 포맷으로 변환
    return await _naver_local_scrape(query, max_results)


async def _naver_local_scrape(query: str, max_results: int) -> list[dict]:
    """
    Naver 장소 검색 HTML 스크래핑 — API 없을 때 폴백.
    indoor_analyzer의 naver_place_scrape (HTML 스크래핑) 재사용.
    """
    try:
        from .indoor_analyzer import naver_place_scrape
        places = await naver_place_scrape(query, max_results)
        results = [
            {
                "title": p.get("name", ""),
                "address": p.get("address", ""),
                "roadAddress": p.get("address", ""),
                "lat": p.get("lat"),
                "lon": p.get("lon"),
                "category": p.get("category", ""),
                "telephone": p.get("phone", ""),
            }
            for p in places
        ]
        logger.debug(f"[naver_local] scrape '{query}' → {len(results)}건")
        return results
    except Exception as e:
        logger.warning(f"[naver_local] scrape 폴백 실패: {e}")
        return []


async def naver_knowledge_in(query: str, max_results: int = 4) -> list[dict]:
    """
    네이버 지식iN 검색 API — 없으면 HTML 스크래핑 폴백.
    반환: [{"title", "url", "description", "location_hint"}]
    """
    from ..core.config import settings
    client_id = getattr(settings, "NAVER_CLIENT_ID", "")
    client_secret = getattr(settings, "NAVER_CLIENT_SECRET", "")

    if client_id and client_secret:
        url = "https://openapi.naver.com/v1/search/kin.json"
        headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        }
        params = {"query": query, "display": max_results, "sort": "point"}
        try:
            async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = []
                        for item in data.get("items", []):
                            title = re.sub(r"<[^>]+>", "", item.get("title", ""))
                            desc = re.sub(r"<[^>]+>", "", item.get("description", ""))
                            results.append({
                                "title": title,
                                "url": item.get("link", ""),
                                "description": desc[:200],
                                "location_hint": _extract_location_from_text(title + " " + desc),
                            })
                        logger.debug(f"[naver_knowledge_in] API '{query}' → {len(results)}건")
                        return results
        except Exception as e:
            logger.debug(f"[naver_knowledge_in] API 실패, 스크래핑 폴백: {e}")

    # 폴백: 지식iN 검색 HTML 스크래핑
    url = "https://search.naver.com/search.naver"
    params = {"query": query, "where": "kin", "sm": "tab_jum"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9",
        "Referer": "https://www.naver.com/",
    }
    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                html = await resp.text()

        results = []
        # 지식iN 제목+링크
        items = re.findall(
            r'<a[^>]+class="[^"]*question[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )
        descs = re.findall(
            r'<div[^>]+class="[^"]*answer[^"]*"[^>]*>(.*?)</div>',
            html, re.DOTALL
        )
        for i, (link, title) in enumerate(items[:max_results]):
            title_clean = re.sub(r"<[^>]+>", "", title).strip()
            desc_raw = descs[i] if i < len(descs) else ""
            desc = re.sub(r"<[^>]+>", "", desc_raw).strip()[:200]
            results.append({
                "title": title_clean,
                "url": f"https://kin.naver.com{link}" if link.startswith("/") else link,
                "description": desc,
                "location_hint": _extract_location_from_text(title_clean + " " + desc),
            })

        logger.debug(f"[naver_knowledge_in] scrape '{query}' → {len(results)}건")
        return results
    except Exception as e:
        logger.warning(f"[naver_knowledge_in] 스크래핑 실패: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# B. 카카오 로컬 검색 확장
# ═══════════════════════════════════════════════════════════════════════════════

async def kakao_local_search(query: str, lat: float = 0.0, lon: float = 0.0, max_results: int = 5) -> list[dict]:
    """
    카카오 키워드 장소 검색 API — 없으면 Daum 지도 검색 스크래핑 폴백.
    반환: [{"name", "address", "road_address", "lat", "lon", "category", "phone", "place_url"}]
    """
    from ..core.config import settings
    api_key = getattr(settings, "KAKAO_API_KEY", "") or getattr(settings, "KAKAO_REST_API_KEY", "")

    if api_key:
        url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        headers = {"Authorization": f"KakaoAK {api_key}"}
        params: dict = {"query": query, "size": max_results}
        if lat and lon:
            params["y"] = lat
            params["x"] = lon
            params["sort"] = "distance"
        try:
            async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = []
                        for doc in data.get("documents", []):
                            try:
                                lat_r = float(doc.get("y", 0))
                                lon_r = float(doc.get("x", 0))
                            except (ValueError, TypeError):
                                lat_r, lon_r = 0.0, 0.0
                            results.append({
                                "name": doc.get("place_name", ""),
                                "address": doc.get("address_name", ""),
                                "road_address": doc.get("road_address_name", ""),
                                "lat": lat_r if 33 <= lat_r <= 38 else None,
                                "lon": lon_r if 124 <= lon_r <= 132 else None,
                                "category": doc.get("category_name", ""),
                                "phone": doc.get("phone", ""),
                                "place_url": doc.get("place_url", ""),
                            })
                        logger.debug(f"[kakao_local] API '{query}' → {len(results)}건")
                        return results
        except Exception as e:
            logger.debug(f"[kakao_local] API 실패, 스크래핑 폴백: {e}")

    return await _kakao_local_scrape(query, lat, lon, max_results)


async def _kakao_local_scrape(query: str, lat: float, lon: float, max_results: int) -> list[dict]:
    """
    카카오 로컬 API 없을 때 폴백 — Naver Local API or 스크래핑 사용.
    """
    # 1차: Naver Local API로 교차 검색 (네이버 API 키가 있으면 더 정확)
    naver_results = await naver_local_search(query, max_results)
    if naver_results:
        logger.debug(f"[kakao_local] naver_local fallback '{query}' → {len(naver_results)}건")
        return [
            {
                "name": r.get("title", ""),
                "address": r.get("address", ""),
                "road_address": r.get("roadAddress", ""),
                "lat": r.get("lat"),
                "lon": r.get("lon"),
                "category": r.get("category", ""),
                "phone": r.get("telephone", ""),
                "place_url": "",
            }
            for r in naver_results
        ]

    # 2차: Daum 지도 검색 HTML 스크래핑
    url = "https://search.daum.net/search"
    params = {"w": "local", "q": query, "DA": "YZR"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://map.kakao.com/",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    results = []
                    # Daum 지도 검색 결과 파싱
                    items = re.findall(
                        r'<div[^>]+class="[^"]*cont_place[^"]*"[^>]*>(.*?)</div>\s*</li>',
                        html, re.DOTALL
                    )
                    for item in items[:max_results]:
                        name_m = re.search(r'<strong[^>]*>(.*?)</strong>', item)
                        addr_m = re.search(r'<span[^>]+class="[^"]*addr[^"]*"[^>]*>(.*?)</span>', item)
                        name = re.sub(r"<[^>]+>", "", name_m.group(1)).strip() if name_m else ""
                        addr = re.sub(r"<[^>]+>", "", addr_m.group(1)).strip() if addr_m else ""
                        if name:
                            results.append({
                                "name": name,
                                "address": addr,
                                "road_address": addr,
                                "lat": None, "lon": None,
                                "category": "", "phone": "", "place_url": "",
                            })
                    if results:
                        logger.debug(f"[kakao_local] Daum scrape '{query}' → {len(results)}건")
                        return results
    except Exception as e:
        logger.debug(f"[kakao_local] Daum scrape 실패: {e}")

    logger.warning(f"[kakao_local] 모든 폴백 실패: {query}")
    return []


async def kakao_category_search(category_code: str, lat: float, lon: float, radius_m: int = 500, max_results: int = 8) -> list[dict]:
    """
    카카오 반경 내 카테고리별 POI 검색 — 상권 특성 파악.
    category_code: FD6(음식점), CE7(카페), CS2(편의점), MT1(대형마트), CT1(문화시설), SW8(지하철역)
    API 없으면 OSM Overpass로 폴백.
    반환: [{"name", "address", "lat", "lon", "category", "distance"}]
    """
    from ..core.config import settings
    api_key = getattr(settings, "KAKAO_API_KEY", "") or getattr(settings, "KAKAO_REST_API_KEY", "")
    if not api_key:
        # 폴백: OSM Overpass로 동일한 카테고리 검색
        CAT_TO_OSM = {
            "FD6": "restaurant", "CE7": "cafe", "CS2": "convenience",
            "MT1": "supermarket", "CT1": "arts_centre", "SW8": "subway",
        }
        osm_type = CAT_TO_OSM.get(category_code, "amenity")
        osm_results = await osm_poi_search(osm_type, lat, lon, radius_m=radius_m, max_results=max_results)
        return [
            {"name": r["name"], "address": r.get("address", ""),
             "lat": r["lat"], "lon": r["lon"],
             "category": r.get("type", osm_type), "distance": ""}
            for r in osm_results
        ]

    url = "https://dapi.kakao.com/v2/local/search/category.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {
        "category_group_code": category_code,
        "x": lon, "y": lat,
        "radius": min(radius_m, 20000),
        "size": max_results,
        "sort": "distance",
    }

    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

        results = []
        for doc in data.get("documents", []):
            try:
                lat_r = float(doc.get("y", 0))
                lon_r = float(doc.get("x", 0))
            except (ValueError, TypeError):
                lat_r, lon_r = 0.0, 0.0
            results.append({
                "name": doc.get("place_name", ""),
                "address": doc.get("address_name", ""),
                "lat": lat_r if 33 <= lat_r <= 38 else None,
                "lon": lon_r if 124 <= lon_r <= 132 else None,
                "category": doc.get("category_name", ""),
                "distance": doc.get("distance", ""),
            })

        logger.debug(f"[kakao_category] {category_code} @ ({lat:.4f},{lon:.4f}) r={radius_m}m → {len(results)}건")
        return results

    except Exception as e:
        logger.warning(f"[kakao_category] 실패: {e}")
        return []


async def kakao_address_search(address: str) -> dict:
    """
    카카오 주소 → 정규화된 법정동 주소 + 정밀 좌표.
    API 없으면 Juso API 폴백.
    반환: {"address", "road_address", "lat", "lon", "region_1depth", "region_2depth", "region_3depth"}
    """
    from ..core.config import settings
    api_key = getattr(settings, "KAKAO_API_KEY", "") or getattr(settings, "KAKAO_REST_API_KEY", "")
    if not api_key:
        # 폴백: 행안부 Juso API (이미 구현된 korea_specializer 활용)
        try:
            from ..core.config import settings as s
            juso_key = getattr(s, "JUSO_API_KEY", "")
            if juso_key:
                from .korea_specializer import search_juso_api
                results = await search_juso_api(address, juso_key)
                if results:
                    r = results[0]
                    return {
                        "address": r.get("jibunAddr", ""),
                        "road_address": r.get("roadAddr", ""),
                        "lat": r.get("lat"),
                        "lon": r.get("lon"),
                        "region_1depth": r.get("siNm", ""),
                        "region_2depth": r.get("sggNm", ""),
                        "region_3depth": r.get("emdNm", ""),
                    }
        except Exception as e:
            logger.debug(f"[kakao_address] Juso 폴백 실패: {e}")
        return {}

    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": address, "analyze_type": "similar"}

    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json()

        docs = data.get("documents", [])
        if not docs:
            return {}
        doc = docs[0]
        addr_info = doc.get("address") or {}
        road_info = doc.get("road_address") or {}
        try:
            lat = float(doc.get("y", 0))
            lon = float(doc.get("x", 0))
        except (ValueError, TypeError):
            lat, lon = 0.0, 0.0

        return {
            "address": addr_info.get("address_name", doc.get("address_name", "")),
            "road_address": road_info.get("address_name", ""),
            "lat": lat if 33 <= lat <= 38 else None,
            "lon": lon if 124 <= lon <= 132 else None,
            "region_1depth": addr_info.get("region_1depth_name", ""),
            "region_2depth": addr_info.get("region_2depth_name", ""),
            "region_3depth": addr_info.get("region_3depth_name", ""),
        }

    except Exception as e:
        logger.warning(f"[kakao_address] 실패: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# C. Flickr 지오태그 이미지 검색
# ═══════════════════════════════════════════════════════════════════════════════

async def flickr_geo_search(lat: float, lon: float, radius_km: float = 5.0, max_results: int = 8) -> list[dict]:
    """
    Flickr API — 없으면 Flickr 공개 지도 페이지 스크래핑 폴백.
    반환: [{"id", "title", "tags", "lat", "lon", "url", "location_hint"}]
    """
    from ..core.config import settings
    api_key = getattr(settings, "FLICKR_API_KEY", "")

    if api_key:
        url = "https://api.flickr.com/services/rest/"
        params = {
            "method": "flickr.photos.search",
            "api_key": api_key,
            "lat": lat, "lon": lon,
            "radius": min(radius_km, 32),
            "radius_units": "km",
            "extras": "geo,tags,url_sq,title",
            "per_page": max_results,
            "format": "json",
            "nojsoncallback": 1,
            "privacy_filter": 1,
            "has_geo": 1,
        }
        try:
            async with aiohttp.ClientSession(connector=_connector()) as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        photos = data.get("photos", {}).get("photo", [])
                        results = []
                        for p in photos:
                            try:
                                p_lat = float(p.get("latitude", 0))
                                p_lon = float(p.get("longitude", 0))
                            except (ValueError, TypeError):
                                p_lat, p_lon = 0.0, 0.0
                            tags = p.get("tags", "")
                            results.append({
                                "id": p.get("id", ""),
                                "title": p.get("title", ""),
                                "tags": tags[:200],
                                "lat": p_lat if 33 <= p_lat <= 38 else None,
                                "lon": p_lon if 124 <= p_lon <= 132 else None,
                                "url": p.get("url_sq", ""),
                                "location_hint": _extract_location_from_text(p.get("title", "") + " " + tags),
                            })
                        logger.debug(f"[flickr_geo] API ({lat:.4f},{lon:.4f}) → {len(results)}건")
                        return results
        except Exception as e:
            logger.debug(f"[flickr_geo] API 실패: {e}")

    # 폴백: Flickr 공개 좌표 검색 (API 키 없는 버전)
    # Flickr는 로그인 없이도 map 페이지 URL로 지역 사진 접근 가능
    try:
        # bbox: lon_min,lat_min,lon_max,lat_max
        deg = radius_km / 111.0
        bbox = f"{lon-deg},{lat-deg},{lon+deg},{lat+deg}"
        url = f"https://www.flickr.com/map/feeds/geo/?bbox={bbox}&lang=ko&format=json&nojsoncallback=1"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        }
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    items = data.get("items", [])
                    results = []
                    for item in items[:max_results]:
                        media = item.get("media", {})
                        title = item.get("title", "")
                        tags_raw = item.get("tags", "")
                        loc_hint = _extract_location_from_text(title + " " + tags_raw)
                        results.append({
                            "id": item.get("link", "").split("/")[-2] if item.get("link") else "",
                            "title": title,
                            "tags": tags_raw[:200],
                            "lat": None, "lon": None,
                            "url": media.get("m", ""),
                            "location_hint": loc_hint,
                        })
                    if results:
                        logger.debug(f"[flickr_geo] public feed ({lat:.4f},{lon:.4f}) → {len(results)}건")
                        return results
    except Exception as e:
        logger.debug(f"[flickr_geo] public feed 실패: {e}")

    return []


async def flickr_text_search(query: str, max_results: int = 6) -> list[dict]:
    """
    Flickr 사진 텍스트 검색 — API 없으면 Flickr 공개 검색 스크래핑 폴백.
    반환: [{"id", "title", "tags", "lat", "lon", "location_hint"}]
    """
    from ..core.config import settings
    api_key = getattr(settings, "FLICKR_API_KEY", "")

    if api_key:
        url = "https://api.flickr.com/services/rest/"
        params = {
            "method": "flickr.photos.search",
            "api_key": api_key,
            "text": query,
            "extras": "geo,tags,title",
            "per_page": max_results,
            "format": "json",
            "nojsoncallback": 1,
            "privacy_filter": 1,
            "has_geo": 1,
            "sort": "relevance",
        }
        try:
            async with aiohttp.ClientSession(connector=_connector()) as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        photos = data.get("photos", {}).get("photo", [])
                        results = []
                        for p in photos:
                            try:
                                p_lat = float(p.get("latitude", 0))
                                p_lon = float(p.get("longitude", 0))
                            except (ValueError, TypeError):
                                p_lat, p_lon = 0.0, 0.0
                            tags = p.get("tags", "")
                            results.append({
                                "id": p.get("id", ""),
                                "title": p.get("title", ""),
                                "tags": tags[:200],
                                "lat": p_lat if 33 <= p_lat <= 38 else None,
                                "lon": p_lon if 124 <= p_lon <= 132 else None,
                                "location_hint": _extract_location_from_text(p.get("title", "") + " " + tags),
                            })
                        logger.debug(f"[flickr_text] API '{query}' → {len(results)}건")
                        return results
        except Exception as e:
            logger.debug(f"[flickr_text] API 실패: {e}")

    # 폴백: Flickr 공개 텍스트 검색 피드
    try:
        url = "https://api.flickr.com/services/feeds/photos_public.gne"
        params = {"tags": query.replace(" ", ","), "lang": "ko-kr", "format": "json", "nojsoncallback": 1}
        headers = {"User-Agent": "Mozilla/5.0 (compatible)"}
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    items = data.get("items", [])
                    results = []
                    for item in items[:max_results]:
                        title = item.get("title", "")
                        tags_raw = item.get("tags", "")
                        results.append({
                            "id": item.get("link", "").split("/")[-2] if item.get("link") else "",
                            "title": title,
                            "tags": tags_raw[:200],
                            "lat": None, "lon": None,
                            "location_hint": _extract_location_from_text(title + " " + tags_raw),
                        })
                    logger.debug(f"[flickr_text] public feed '{query}' → {len(results)}건")
                    return results
    except Exception as e:
        logger.debug(f"[flickr_text] public feed 실패: {e}")

    return []


# ═══════════════════════════════════════════════════════════════════════════════
# D. 공공데이터 포털 API 연동
# ═══════════════════════════════════════════════════════════════════════════════

async def public_data_biz_reg(reg_number: str) -> dict:
    """
    공공데이터포털 국세청 사업자등록 상태 조회 API.
    reg_number: "000-00-00000" 형식 (10자리 숫자)
    반환: {"reg_number", "status", "store_name", "address", "business_type", "closed"}
    """
    from ..core.config import settings
    api_key = getattr(settings, "PUBLIC_DATA_API_KEY", "")

    reg_clean = re.sub(r"[^0-9]", "", reg_number)
    if len(reg_clean) != 10:
        return {"error": "사업자등록번호는 10자리여야 합니다"}

    formatted = f"{reg_clean[:3]}-{reg_clean[3:5]}-{reg_clean[5:]}"

    # 공공데이터 API (키 있을 때)
    if api_key:
        url = "https://api.odcloud.kr/api/nts-businessman/v1/status"
        params = {"serviceKey": api_key}
        body = {"b_no": [reg_clean]}
        try:
            async with aiohttp.ClientSession(connector=_connector()) as session:
                async with session.post(
                    url, params=params, json=body,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("data", [])
                        if items:
                            item = items[0]
                            return {
                                "reg_number": formatted,
                                "status": item.get("b_stt", ""),
                                "closed": item.get("b_stt_cd", "") in ("03", "04"),
                                "store_name": item.get("b_nm", ""),
                                "address": "",
                                "business_type": item.get("b_type", ""),
                                "source": "nts_api",
                            }
        except Exception as e:
            logger.debug(f"[public_data_biz] API 실패: {e}")

    # 폴백: 기존 web_search 기반 (biz_reg_lookup 재사용)
    result = await biz_reg_lookup(reg_number)
    result["source"] = "web_fallback"
    return result


async def cultural_heritage_lookup(query: str, lat: float = 0.0, lon: float = 0.0) -> list[dict]:
    """
    문화재청 문화재 정보 API — 문화재/사적지 사진 매칭.
    반환: [{"name", "address", "lat", "lon", "heritage_type", "image_url"}]
    """
    from ..core.config import settings
    api_key = getattr(settings, "PUBLIC_DATA_API_KEY", "")
    if not api_key:
        # 폴백: 웹 검색으로 문화재 정보 조회
        results = await web_search(f"{query} 문화재 사적지 위치 주소", max_results=4)
        hints = []
        for r in results:
            loc = _extract_location_from_text(r.get("title", "") + " " + r.get("snippet", ""))
            if loc:
                hints.append({"name": query, "address": loc, "lat": None, "lon": None,
                               "heritage_type": "unknown", "source": "web"})
        return hints[:3]

    url = "http://www.cha.go.kr/cha/SearchKindOpenapiList.do"
    params = {
        "ServiceKey": api_key,
        "ccbaMnm1": query,
        "pageUnit": "5",
        "pageIndex": "1",
    }

    try:
        async with aiohttp.ClientSession(connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                text = await resp.text()

        results = []
        # XML 파싱 (간단 regex)
        items = re.findall(r"<item>(.*?)</item>", text, re.DOTALL)
        for item in items[:5]:
            name = re.search(r"<ccbaMnm1>(.*?)</ccbaMnm1>", item)
            addr = re.search(r"<ccbaLcad>(.*?)</ccbaLcad>", item)
            htype = re.search(r"<ccbaKdcd>(.*?)</ccbaKdcd>", item)
            # 위도/경도
            lat_m = re.search(r"<lat>(.*?)</lat>", item)
            lon_m = re.search(r"<lng>(.*?)</lng>", item)
            try:
                h_lat = float(lat_m.group(1)) if lat_m else None
                h_lon = float(lon_m.group(1)) if lon_m else None
            except ValueError:
                h_lat, h_lon = None, None

            results.append({
                "name": name.group(1) if name else query,
                "address": addr.group(1) if addr else "",
                "lat": h_lat,
                "lon": h_lon,
                "heritage_type": htype.group(1) if htype else "",
                "source": "cha_api",
            })

        logger.debug(f"[cultural_heritage] '{query}' → {len(results)}건")
        return results

    except Exception as e:
        logger.warning(f"[cultural_heritage] 실패: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# E. OSINT 결과 융합 엔진 (DBSCAN 클러스터링 + 교차검증)
# ═══════════════════════════════════════════════════════════════════════════════

class OsintFusionEngine:
    """
    복수 OSINT 결과를 융합해 최적 위치 후보를 도출.
    - DBSCAN 좌표 클러스터링 (가장 조밀한 클러스터 → 최종 후보)
    - 동일 POI명 3개 이상 독립 소스 → 신뢰도 승격
    - Levenshtein 기반 중복 지명 병합
    """

    def fuse(self, osint_results: list[dict]) -> dict:
        """
        osint_results: [{"source", "name"?, "address"?, "lat"?, "lon"?, "confidence"?}, ...]
        반환: {"best_location", "best_lat", "best_lon", "confidence", "cluster_count", "sources"}
        """
        coord_points = []  # (lat, lon, source, name, address)
        name_counts: dict[str, list[str]] = {}  # name → [source, ...]

        for r in osint_results:
            lat = r.get("lat") or r.get("latitude")
            lon = r.get("lon") or r.get("longitude")
            name = r.get("name", "") or r.get("address", "") or r.get("location_hint", "")
            source = r.get("source", "unknown")

            if lat and lon and isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                if 33 <= lat <= 38 and 124 <= lon <= 132:
                    coord_points.append((float(lat), float(lon), source, name, r.get("address", "")))

            if name:
                # 유사 이름 병합
                canonical = self._find_canonical(name, name_counts)
                if canonical:
                    if source not in name_counts[canonical]:
                        name_counts[canonical].append(source)
                else:
                    name_counts[name] = [source]

        best_location = ""
        best_lat, best_lon = None, None
        confidence = 0.0
        cluster_count = 0

        # 좌표 클러스터링 (간단한 반경 0.005도 ≈ 500m 그리드)
        if coord_points:
            clusters = self._dbscan_simple(coord_points, eps=0.005)
            if clusters:
                largest = max(clusters, key=lambda c: len(c))
                cluster_count = len(largest)
                lats = [p[0] for p in largest]
                lons = [p[1] for p in largest]
                best_lat = sum(lats) / len(lats)
                best_lon = sum(lons) / len(lons)
                # 클러스터 내 address 우선
                addrs = [p[4] for p in largest if p[4]]
                best_location = addrs[0] if addrs else largest[0][3]
                # 클러스터 크기 기반 신뢰도
                confidence = min(0.92, 0.40 + cluster_count * 0.12)

        # 이름 교차검증 — 3개 이상 독립 소스 → 신뢰도 승격
        cross_validated = {
            name: srcs for name, srcs in name_counts.items()
            if len(set(srcs)) >= 3
        }
        if cross_validated:
            top_name = max(cross_validated, key=lambda k: len(cross_validated[k]))
            confidence = max(confidence, 0.88)
            if not best_location:
                best_location = top_name

        sources = list({p[2] for p in coord_points})

        return {
            "best_location": best_location,
            "best_lat": round(best_lat, 6) if best_lat else None,
            "best_lon": round(best_lon, 6) if best_lon else None,
            "confidence": round(confidence, 3),
            "cluster_count": cluster_count,
            "cross_validated_names": list(cross_validated.keys()) if cross_validated else [],
            "sources": sources,
        }

    def _dbscan_simple(self, points: list, eps: float) -> list[list]:
        """간단한 DBSCAN (scipy 없이 구현)"""
        n = len(points)
        visited = [False] * n
        clusters = []

        def _neighbors(idx: int) -> list[int]:
            lat0, lon0 = points[idx][0], points[idx][1]
            return [
                j for j in range(n)
                if abs(points[j][0] - lat0) <= eps and abs(points[j][1] - lon0) <= eps
            ]

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            nbrs = _neighbors(i)
            if len(nbrs) < 1:
                clusters.append([points[i]])
                continue
            cluster = [points[i]]
            queue = list(nbrs)
            while queue:
                j = queue.pop()
                if not visited[j]:
                    visited[j] = True
                    new_nbrs = _neighbors(j)
                    if len(new_nbrs) >= 1:
                        queue.extend(new_nbrs)
                cluster.append(points[j])
            clusters.append(cluster)

        return [c for c in clusters if c]

    def _find_canonical(self, name: str, name_counts: dict) -> str | None:
        """Levenshtein 거리 ≤ 2인 기존 이름 찾기 (중복 병합)"""
        for existing in name_counts:
            if self._levenshtein(name, existing) <= 2:
                return existing
        return None

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]


_fusion_engine = OsintFusionEngine()


async def osint_fuse(osint_results: list[dict]) -> dict:
    """
    복수 OSINT 결과를 융합해 최적 위치 후보 반환.
    각 결과는 {"source", "name"?, "address"?, "lat"?, "lon"?, "confidence"?} 형태.
    """
    return _fusion_engine.fuse(osint_results)


# ═══════════════════════════════════════════════════════════════════════════════
# F. SNS 위치 메타데이터 강화
# ═══════════════════════════════════════════════════════════════════════════════

async def youtube_geotag_extract(video_url: str) -> dict:
    """
    YouTube Data API v3 — 영상 recordingDetails.location 필드 추출.
    반환: {"video_id", "title", "lat", "lon", "location_description", "published_at"}
    """
    from ..core.config import settings
    google_key = getattr(settings, "GOOGLE_MAPS_API_KEY", "")

    # video_id 추출
    vid_id = None
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", video_url)
    if m:
        vid_id = m.group(1)
    if not vid_id:
        return {"error": "YouTube video ID 추출 실패"}

    result: dict = {"video_id": vid_id, "lat": None, "lon": None}

    # YouTube Data API (Google API 키 있을 때)
    if google_key:
        api_url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "id": vid_id,
            "key": google_key,
            "part": "snippet,recordingDetails",
        }
        try:
            async with aiohttp.ClientSession(connector=_connector()) as session:
                async with session.get(api_url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("items", [])
                        if items:
                            item = items[0]
                            snippet = item.get("snippet", {})
                            rec = item.get("recordingDetails", {})
                            result["title"] = snippet.get("title", "")
                            result["published_at"] = snippet.get("publishedAt", "")
                            loc = rec.get("location", {})
                            if loc.get("latitude") and loc.get("longitude"):
                                result["lat"] = loc["latitude"]
                                result["lon"] = loc["longitude"]
                                result["location_description"] = rec.get("locationDescription", "")
                            # 태그에서 위치 힌트
                            tags = snippet.get("tags", [])
                            tag_hint = _extract_location_from_text(" ".join(tags[:20]))
                            result["tag_location_hint"] = tag_hint
                            return result
        except Exception as e:
            logger.debug(f"[youtube_geotag] API 실패: {e}")

    # 폴백: oembed title 파싱
    try:
        oembed_url = f"https://www.youtube.com/oembed?url={urllib.parse.quote(video_url)}&format=json"
        async with aiohttp.ClientSession(connector=_connector()) as session:
            async with session.get(oembed_url, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result["title"] = data.get("title", "")
                    loc_hint = _extract_location_from_text(
                        data.get("title", "") + " " + data.get("author_name", "")
                    )
                    result["tag_location_hint"] = loc_hint
    except Exception:
        pass

    return result


async def naver_cafe_search(query: str, max_results: int = 4) -> list[dict]:
    """
    네이버 카페 검색 API — 없으면 네이버 카페 HTML 스크래핑 폴백.
    반환: [{"title", "url", "description", "cafe_name", "location_hint"}]
    """
    from ..core.config import settings
    client_id = getattr(settings, "NAVER_CLIENT_ID", "")
    client_secret = getattr(settings, "NAVER_CLIENT_SECRET", "")

    if client_id and client_secret:
        url = "https://openapi.naver.com/v1/search/cafearticle.json"
        headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        }
        params = {"query": query, "display": max_results, "sort": "sim"}
        try:
            async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = []
                        for item in data.get("items", []):
                            title = re.sub(r"<[^>]+>", "", item.get("title", ""))
                            desc = re.sub(r"<[^>]+>", "", item.get("description", ""))
                            results.append({
                                "title": title,
                                "url": item.get("link", ""),
                                "description": desc[:200],
                                "cafe_name": item.get("cafename", ""),
                                "location_hint": _extract_location_from_text(title + " " + desc),
                            })
                        logger.debug(f"[naver_cafe] API '{query}' → {len(results)}건")
                        return results
        except Exception as e:
            logger.debug(f"[naver_cafe] API 실패, 스크래핑 폴백: {e}")

    # 폴백: 네이버 카페 검색 HTML 스크래핑
    url = "https://search.naver.com/search.naver"
    params = {"query": query, "where": "article", "sm": "tab_jum"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9",
        "Referer": "https://www.naver.com/",
    }
    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                html = await resp.text()

        results = []
        items = re.findall(
            r'<a[^>]+class="[^"]*title[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )
        cafe_names = re.findall(r'<a[^>]+class="[^"]*cafe_name[^"]*"[^>]*>(.*?)</a>', html, re.DOTALL)
        descs = re.findall(r'<div[^>]+class="[^"]*dsc[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL)

        for i, (link, title) in enumerate(items[:max_results]):
            title_clean = re.sub(r"<[^>]+>", "", title).strip()
            desc_raw = descs[i] if i < len(descs) else ""
            desc = re.sub(r"<[^>]+>", "", desc_raw).strip()[:200]
            cafe = re.sub(r"<[^>]+>", "", cafe_names[i]).strip() if i < len(cafe_names) else ""
            results.append({
                "title": title_clean,
                "url": link,
                "description": desc,
                "cafe_name": cafe,
                "location_hint": _extract_location_from_text(title_clean + " " + desc),
            })

        logger.debug(f"[naver_cafe] scrape '{query}' → {len(results)}건")
        return results
    except Exception as e:
        logger.warning(f"[naver_cafe] 스크래핑 실패: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# G. 실시간 뉴스 이미지 역방향 추적
# ═══════════════════════════════════════════════════════════════════════════════

async def gdelt_image_search(query: str, max_results: int = 5) -> list[dict]:
    """
    GDELT 2.0 뉴스 인덱스 쿼리 — 429 rate limit 시 web_search 폴백.
    반환: [{"title", "url", "date", "location", "domain", "location_hint"}]
    """
    gdelt_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max_results,
        "format": "json",
        "timespan": "30d",
        "sourcelang": "Korean",
    }

    try:
        async with aiohttp.ClientSession(connector=_connector()) as session:
            async with session.get(gdelt_url, params=params, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                if resp.status == 429:
                    logger.debug("[gdelt] 429 rate limit → web_search 폴백")
                    return await _gdelt_web_fallback(query, max_results)
                if resp.status != 200:
                    return await _gdelt_web_fallback(query, max_results)
                data = await resp.json()

        articles = data.get("articles", [])
        results = []
        for art in articles[:max_results]:
            title = art.get("title", "")
            loc_hint = _extract_location_from_text(title + " " + art.get("seendate", ""))
            results.append({
                "title": title,
                "url": art.get("url", ""),
                "date": art.get("seendate", ""),
                "location": art.get("socialimage", ""),
                "domain": art.get("domain", ""),
                "location_hint": loc_hint,
            })

        logger.debug(f"[gdelt] '{query}' → {len(results)}건")
        return results if results else await _gdelt_web_fallback(query, max_results)

    except Exception as e:
        logger.debug(f"[gdelt] 실패 → 폴백: {e}")
        return await _gdelt_web_fallback(query, max_results)


async def _gdelt_web_fallback(query: str, max_results: int) -> list[dict]:
    """GDELT 미작동 시 네이버 뉴스 검색으로 대체"""
    news = await naver_news_search(f"{query} 위치 장소", max_results=max_results)
    return [
        {
            "title": r["title"],
            "url": r["url"],
            "date": r.get("pub_date", ""),
            "location": "",
            "domain": "naver_news",
            "location_hint": r.get("location_hint", ""),
        }
        for r in news
        if r.get("title")
    ]


async def naver_news_image_search(query: str, max_results: int = 5) -> list[dict]:
    """
    네이버 뉴스 이미지 검색 — 관련 보도 URL → 장소 추출.
    반환: [{"title", "url", "description", "pub_date", "location_hint", "image_url"}]
    """
    from ..core.config import settings
    client_id = getattr(settings, "NAVER_CLIENT_ID", "")
    client_secret = getattr(settings, "NAVER_CLIENT_SECRET", "")
    if not (client_id and client_secret):
        # 폴백: GDELT
        return await gdelt_image_search(query, max_results)

    url = "https://openapi.naver.com/v1/search/image"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    params = {"query": query, "display": max_results, "sort": "sim", "filter": "large"}

    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return await gdelt_image_search(query, max_results)
                data = await resp.json()

        results = []
        for item in data.get("items", []):
            title = re.sub(r"<[^>]+>", "", item.get("title", ""))
            loc_hint = _extract_location_from_text(title + " " + item.get("link", ""))
            results.append({
                "title": title,
                "url": item.get("link", ""),
                "image_url": item.get("link", ""),
                "location_hint": loc_hint,
                "pub_date": "",
                "description": "",
            })

        logger.debug(f"[naver_news_image] '{query}' → {len(results)}건")
        return results

    except Exception as e:
        logger.warning(f"[naver_news_image] 실패: {e}")
        return await gdelt_image_search(query, max_results)


# ══════════════════════════════════════════════════════════════════════════════
# E. CLOVA OCR + NER 엔진
# 네이버 CLOVA OCR API (한국어 특화) + 개체명 인식 후처리
# API 키 없으면 PaddleOCR 결과를 NER 후처리만 적용
# ══════════════════════════════════════════════════════════════════════════════

# 한국어 NER 패턴 DB
_NER_PATTERNS = {
    "phone": re.compile(
        r'(?:전화|tel|T\.?|☎)?\s*'
        r'(\d{2,4}[-\.\s]?\d{3,4}[-\.\s]?\d{4})'
    ),
    "address_road": re.compile(
        r'([가-힣]+(?:특별시|광역시|특별자치시|특별자치도|시|도)\s*'
        r'[가-힣]+(?:구|시|군)\s*'
        r'[가-힣0-9]+(?:로|대로|길)[\s\d가-힣]*)'
    ),
    "postal_code": re.compile(r'\b(\d{5})\b'),
    "business_reg": re.compile(r'\b(\d{3}-\d{2}-\d{5})\b'),
    "brand_kr": re.compile(
        r'\b(스타벅스|이디야|투썸플레이스|빽다방|메가커피|할리스|파스쿠찌|탐앤탐스|'
        r'맥도날드|버거킹|롯데리아|맘스터치|서브웨이|KFC|파파이스|'
        r'GS25|CU편의점|세븐일레븐|이마트24|미니스톱|'
        r'올리브영|다이소|이케아|이마트|홈플러스|롯데마트|'
        r'파리바게뜨|뚜레쥬르|성심당|'
        r'롯데호텔|신라호텔|조선호텔|힐튼|인터컨티넨탈|'
        r'CGV|메가박스|롯데시네마)\b'
    ),
    "landmark_kr": re.compile(
        r'\b(경복궁|창덕궁|덕수궁|창경궁|'
        r'남산타워|N서울타워|롯데월드타워|63빌딩|'
        r'해운대|광안리|송도|여의도|잠실|강남|홍대|이태원|명동|인사동|북촌|'
        r'한강|낙동강|금강|영산강|섬진강|'
        r'설악산|지리산|한라산|북한산|소백산|태백산)\b'
    ),
    "subway_kr": re.compile(r'([가-힣]{2,6}역)\s*(?:\d호선|호선)?'),
    "floor": re.compile(r'(\d+)층'),
}

# 한국 주요 브랜드 → 본사/주요 지역 (NER 결과 활용)
_BRAND_REGION: dict[str, dict] = {
    "성심당": {"city": "대전", "lat": 36.3271, "lon": 127.4275, "note": "대전 본점"},
    "크라운베이커리": {"city": "서울", "lat": 37.5665, "lon": 126.9780},
    "신전떡볶이": {"city": "서울", "lat": 37.5665, "lon": 126.9780},
    "국제시장": {"city": "부산", "lat": 35.0997, "lon": 129.0266, "note": "부산 국제시장"},
    "광장시장": {"city": "서울", "lat": 37.5697, "lon": 127.0082, "note": "서울 광장시장"},
    "남대문시장": {"city": "서울", "lat": 37.5573, "lon": 126.9757},
    "동문시장": {"city": "제주", "lat": 33.5124, "lon": 126.5283, "note": "제주 동문재래시장"},
}


async def clova_ocr(
    image_bytes: bytes,
    image_url: str = "",
) -> dict:
    """
    네이버 CLOVA OCR API 호출 (한국어 특화)
    API 키 없으면 PaddleOCR 기반 NER 폴백
    반환: {texts, entities, brands, addresses, phones, landmarks, confidence}
    """
    from ..core.config import settings
    api_key = getattr(settings, "CLOVA_OCR_API_KEY", "")
    api_url = getattr(settings, "CLOVA_OCR_API_URL", "")

    raw_texts = []

    # ── CLOVA OCR API 호출 ──
    if api_key and api_url:
        try:
            import uuid, time
            payload = {
                "version": "V2",
                "requestId": str(uuid.uuid4()),
                "timestamp": int(time.time() * 1000),
                "lang": "ko",
                "images": [],
                "enableTableDetect": False,
            }
            if image_url:
                payload["images"].append({
                    "format": "jpg",
                    "url": image_url,
                    "name": "image",
                })
            else:
                import base64
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                payload["images"].append({
                    "format": "jpg",
                    "data": b64,
                    "name": "image",
                })

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url,
                    json=payload,
                    headers={"X-OCR-SECRET": api_key},
                    timeout=aiohttp.ClientTimeout(total=12),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for img_result in data.get("images", []):
                            for field in img_result.get("fields", []):
                                text = field.get("inferText", "").strip()
                                if text:
                                    raw_texts.append(text)
                        logger.debug(f"[clova_ocr] API 성공: {len(raw_texts)}개 텍스트")
                    else:
                        logger.debug(f"[clova_ocr] API 상태 {resp.status}, 폴백")
        except Exception as e:
            logger.debug(f"[clova_ocr] API 실패: {e}, PaddleOCR 폴백")

    # ── PaddleOCR 폴백 ──
    if not raw_texts:
        try:
            from paddleocr import PaddleOCR
            import asyncio as _asyncio

            def _paddle_run():
                ocr = PaddleOCR(use_angle_cls=True, lang="korean")
                import io as _io
                import numpy as _np
                from PIL import Image as _Image
                img = _Image.open(_io.BytesIO(image_bytes)).convert("RGB")
                arr = _np.array(img)
                result = ocr.ocr(arr, cls=True)
                texts = []
                if result and result[0]:
                    for line in result[0]:
                        if line and len(line) >= 2:
                            texts.append(line[1][0])
                return texts

            raw_texts = await _asyncio.to_thread(_paddle_run)
            logger.debug(f"[clova_ocr] PaddleOCR 폴백: {len(raw_texts)}개")
        except Exception as e:
            logger.warning(f"[clova_ocr] PaddleOCR도 실패: {e}")
            return {"texts": [], "entities": {}, "error": str(e)}

    # ── NER 후처리 ──
    entities = _apply_ner(raw_texts)

    return {
        "texts": raw_texts,
        "entities": entities,
        "brands": entities.get("brands", []),
        "addresses": entities.get("addresses", []),
        "phones": entities.get("phones", []),
        "landmarks": entities.get("landmarks", []),
        "subway_stations": entities.get("subway_stations", []),
        "business_reg": entities.get("business_reg", []),
        "confidence": 0.94 if api_key else 0.82,
    }


def _apply_ner(texts: list[str]) -> dict:
    """OCR 텍스트 목록에 NER 패턴 적용"""
    full_text = " ".join(texts)
    entities: dict[str, list] = {
        "phones": [],
        "addresses": [],
        "postal_codes": [],
        "business_reg": [],
        "brands": [],
        "landmarks": [],
        "subway_stations": [],
        "floors": [],
    }

    for text in texts:
        # 전화번호
        for m in _NER_PATTERNS["phone"].finditer(text):
            phone = m.group(1)
            if phone not in entities["phones"]:
                entities["phones"].append(phone)

        # 도로명 주소
        for m in _NER_PATTERNS["address_road"].finditer(text):
            addr = m.group(1).strip()
            if addr and addr not in entities["addresses"]:
                entities["addresses"].append(addr)

        # 우편번호
        for m in _NER_PATTERNS["postal_code"].finditer(text):
            code = m.group(1)
            if code not in entities["postal_codes"]:
                entities["postal_codes"].append(code)

        # 사업자등록번호
        for m in _NER_PATTERNS["business_reg"].finditer(text):
            reg = m.group(1)
            if reg not in entities["business_reg"]:
                entities["business_reg"].append(reg)

        # 브랜드
        for m in _NER_PATTERNS["brand_kr"].finditer(text):
            brand = m.group(1)
            if brand not in entities["brands"]:
                entities["brands"].append(brand)

        # 랜드마크
        for m in _NER_PATTERNS["landmark_kr"].finditer(text):
            lm = m.group(1)
            if lm not in entities["landmarks"]:
                entities["landmarks"].append(lm)

        # 지하철역
        for m in _NER_PATTERNS["subway_kr"].finditer(text):
            station = m.group(1)
            if station not in entities["subway_stations"]:
                entities["subway_stations"].append(station)

        # 층수
        for m in _NER_PATTERNS["floor"].finditer(text):
            floor = m.group(1)
            if floor not in entities["floors"]:
                entities["floors"].append(floor)

    # 브랜드 기반 지역 힌트
    brand_hints = []
    for brand in entities["brands"]:
        if brand in _BRAND_REGION:
            brand_hints.append(_BRAND_REGION[brand])
    if brand_hints:
        entities["brand_region_hints"] = brand_hints

    return entities

"""
역방향 지오코딩 서비스
Nominatim (OpenStreetMap, 무료) 기반
"""
import asyncio
import ssl
import certifi
import aiohttp
from loguru import logger
from typing import Optional


NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
HEADERS = {"User-Agent": "EXXAS-OSINT/2.0 (contact@exxas.app)"}
_SSL = ssl.create_default_context(cafile=certifi.where())


async def reverse_geocode(lat: float, lon: float, language: str = "ko") -> Optional[str]:
    """
    위도/경도 → 주소 문자열 (Nominatim)
    language: "ko" = 한국어, "en" = 영어
    """
    if not lat or not lon:
        return None
    if abs(lat) > 90 or abs(lon) > 180:
        return None

    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "addressdetails": 1,
        "accept-language": language,
        "zoom": 18,  # 건물/거리 수준
    }

    try:
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            async with session.get(NOMINATIM_URL, params=params, timeout=aiohttp.ClientTimeout(total=8), ssl=_SSL) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

        display = data.get("display_name", "")
        if not display:
            return None

        # 한국 주소를 간결하게 정리
        addr = data.get("address", {})
        if addr.get("country_code") == "kr":
            parts = []
            for key in ("province", "city", "city_district", "suburb", "neighbourhood", "road", "house_number"):
                if val := addr.get(key):
                    parts.append(val)
            if parts:
                return " ".join(parts)

        # 그 외 국가: display_name 앞 3~4 파트만 사용
        parts = [p.strip() for p in display.split(",")]
        return ", ".join(parts[:4])

    except asyncio.TimeoutError:
        logger.debug(f"Reverse geocode timeout: {lat},{lon}")
        return None
    except Exception as e:
        logger.debug(f"Reverse geocode error: {e}")
        return None


async def forward_geocode(query: str, language: str = "ko") -> Optional[dict]:
    """
    주소/장소명 → 좌표 (Nominatim)
    Returns: {"lat": float, "lon": float, "display_name": str} or None
    """
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "accept-language": language,
    }
    try:
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            async with session.get(
                "https://nominatim.openstreetmap.org/search",
                params=params,
                timeout=aiohttp.ClientTimeout(total=8),
                ssl=_SSL,
            ) as resp:
                if resp.status != 200:
                    return None
                results = await resp.json()

        if not results:
            return None
        r = results[0]
        return {
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "display_name": r.get("display_name", ""),
        }
    except Exception as e:
        logger.debug(f"Forward geocode error: {e}")
        return None

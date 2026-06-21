"""
네이버 로드뷰 격자 샘플링 수집기
- 한국 전역을 50~200m 격자로 샘플링
- 각 지점에서 로드뷰 파노라마 이미지 다운로드
- 메타데이터: lat, lon, panoId, address (역지오코딩)
- 로드뷰는 실제 도로 상황 → 건물 외관 학습에 최적
"""
import asyncio
import json
import math
import sys
from pathlib import Path
import aiohttp
import aiofiles
from loguru import logger

# 한국 주요 도시 샘플링 영역 (도시마다 밀도 다르게)
SAMPLING_REGIONS = [
    # (이름, min_lat, max_lat, min_lon, max_lon, step_m)
    ("서울",      37.42, 37.70, 126.80, 127.18, 100),
    ("부산",      35.05, 35.32, 128.95, 129.30, 100),
    ("인천",      37.30, 37.60, 126.55, 126.85, 150),
    ("대구",      35.78, 36.00, 128.50, 128.80, 150),
    ("대전",      36.24, 36.46, 127.30, 127.55, 150),
    ("광주",      35.07, 35.26, 126.83, 127.05, 150),
    ("울산",      35.46, 35.65, 129.20, 129.45, 150),
    ("수원",      37.23, 37.33, 126.97, 127.06, 150),
    ("성남",      37.38, 37.50, 127.08, 127.18, 150),
    ("고양",      37.61, 37.73, 126.82, 126.99, 150),
    ("제주",      33.20, 33.55, 126.15, 126.95, 200),
    ("경기전체",  37.00, 37.85, 126.60, 127.80, 300),
    ("강원",      37.00, 38.20, 127.50, 129.50, 500),
    ("충청",      36.00, 37.20, 127.00, 128.00, 400),
    ("전라",      34.50, 35.80, 126.30, 127.80, 400),
    ("경상",      35.00, 36.70, 128.00, 129.50, 400),
]

NAVER_ROADVIEW_API = "https://naveropenapi.apigw.nhn.com/map-reversegeocode/v2/gc"
NAVER_STATIC_MAP   = "https://naveropenapi.apigw.nhn.com/map-static/v2/raster"

# 파노라마 이미지 URL (공개 엔드포인트)
PANO_THUMB_URL = "https://panorama.map.naver.com/basic/pano/{pano_id}/thumbnail/760/380/1"
PANO_SEARCH_URL = "https://map.naver.com/p/api/roadview/panorama/nearby"


def _meters_to_deg(meters: float, lat: float) -> tuple[float, float]:
    """미터 → 위경도 변화량 (근사)"""
    lat_deg = meters / 111_320
    lon_deg = meters / (111_320 * math.cos(math.radians(lat)))
    return lat_deg, lon_deg


def _gen_grid(region: tuple, jitter: bool = True) -> list[tuple[float, float]]:
    """격자 좌표 생성 (지터링으로 완전 균등 격자 탈피)"""
    import random
    name, min_lat, max_lat, min_lon, max_lon, step_m = region
    points = []
    lat = min_lat
    while lat <= max_lat:
        mid_lat = (lat + max_lat) / 2
        lat_step, lon_step = _meters_to_deg(step_m, mid_lat)
        lon = min_lon
        while lon <= max_lon:
            if jitter:
                jlat = lat + random.uniform(-lat_step * 0.3, lat_step * 0.3)
                jlon = lon + random.uniform(-lon_step * 0.3, lon_step * 0.3)
            else:
                jlat, jlon = lat, lon
            points.append((round(jlat, 6), round(jlon, 6)))
            lon += lon_step
        lat += lat_step
    return points


async def _fetch_pano_id(
    session: aiohttp.ClientSession,
    lat: float,
    lon: float,
) -> str | None:
    """해당 좌표 근처 로드뷰 파노라마 ID 조회"""
    try:
        params = {"lat": lat, "lng": lon, "limit": 1}
        async with session.get(
            PANO_SEARCH_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            items = data.get("result", {}).get("panoramas", [])
            if items:
                return items[0].get("id")
    except Exception:
        pass
    return None


async def _reverse_geocode(
    session: aiohttp.ClientSession,
    client_id: str,
    client_secret: str,
    lat: float,
    lon: float,
) -> str:
    """네이버 역지오코딩 → 주소 문자열"""
    try:
        params = {
            "coords": f"{lon},{lat}",
            "output": "json",
            "orders": "legalcode,addr",
        }
        headers = {
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
        }
        async with session.get(
            NAVER_ROADVIEW_API,
            params=params,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return ""
            data = await resp.json()
            results = data.get("results", [])
            if results:
                region = results[0].get("region", {})
                parts = [
                    region.get("area1", {}).get("name", ""),
                    region.get("area2", {}).get("name", ""),
                    region.get("area3", {}).get("name", ""),
                    region.get("area4", {}).get("name", ""),
                ]
                return " ".join(p for p in parts if p)
    except Exception:
        pass
    return ""


async def _download_pano(
    session: aiohttp.ClientSession,
    pano_id: str,
    dest: Path,
) -> bool:
    url = PANO_THUMB_URL.format(pano_id=pano_id)
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                return False
            content = await resp.read()
            if len(content) < 10_000:  # 10KB 미만 스킵
                return False
            async with aiofiles.open(dest, "wb") as f:
                await f.write(content)
            return True
    except Exception:
        return False


async def crawl_roadview(
    client_id: str,
    client_secret: str,
    output_dir: Path,
    max_images: int = 200_000,
    regions: list | None = None,
) -> int:
    output_dir = Path(output_dir)
    img_dir = output_dir / "roadview" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "roadview" / "metadata.jsonl"

    seen_panos: set[str] = set()
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                try:
                    seen_panos.add(json.loads(line)["pano_id"])
                except Exception:
                    pass
    logger.info(f"로드뷰 이미 수집됨: {len(seen_panos)}개, 목표: {max_images}개")

    target_regions = regions or SAMPLING_REGIONS
    collected = len(seen_panos)

    connector = aiohttp.TCPConnector(limit=8)
    async with aiohttp.ClientSession(connector=connector) as session:
        async with aiofiles.open(meta_path, "a") as meta_f:
            for region in target_regions:
                if collected >= max_images:
                    break
                region_name = region[0]
                logger.info(f"[로드뷰] 지역 시작: {region_name}")
                grid = _gen_grid(region)
                logger.info(f"  격자점 {len(grid)}개")

                for i, (lat, lon) in enumerate(grid):
                    if collected >= max_images:
                        break

                    # 파노라마 ID 조회
                    pano_id = await _fetch_pano_id(session, lat, lon)
                    if not pano_id or pano_id in seen_panos:
                        continue

                    seen_panos.add(pano_id)
                    img_path = img_dir / f"rv_{pano_id}.jpg"

                    # 이미지 다운로드
                    ok = await _download_pano(session, pano_id, img_path)
                    if not ok:
                        continue

                    # 역지오코딩 (10개마다 rate limit 고려)
                    address = ""
                    if client_id and client_secret:
                        address = await _reverse_geocode(session, client_id, client_secret, lat, lon)
                        await asyncio.sleep(0.12)  # Naver API: ~10 req/sec

                    meta = {
                        "id": f"rv_{pano_id}",
                        "source": "naver_roadview",
                        "pano_id": pano_id,
                        "latitude": lat,
                        "longitude": lon,
                        "address": address,
                        "region": region_name,
                        "image_path": f"roadview/images/rv_{pano_id}.jpg",
                    }
                    await meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    collected += 1

                    if collected % 500 == 0:
                        logger.info(f"  로드뷰 {collected}/{max_images} 수집")
                        await meta_f.flush()

    logger.info(f"로드뷰 수집 완료: {collected}개")
    return collected


if __name__ == "__main__":
    import os
    cid = os.environ.get("NAVER_CLIENT_ID", "")
    csec = os.environ.get("NAVER_CLIENT_SECRET", "")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./ml_data")
    asyncio.run(crawl_roadview(cid, csec, out, max_images=50_000))

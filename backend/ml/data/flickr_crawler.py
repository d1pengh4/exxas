"""
Flickr GPS 태그 한국 사진 수집기
- 한국 지리 경계 내 GPS 태그 공개 사진 다운로드
- 메타데이터: lat, lon, title, tags, license
- 출력: JSONL + 이미지 파일
"""
import asyncio
import json
import time
import hashlib
import sys
from pathlib import Path
from typing import Iterator
import aiohttp
import aiofiles
from loguru import logger

# 한국 bounding box
KOREA_BBOX = {
    "min_lon": 124.0, "max_lon": 132.0,
    "min_lat": 33.0,  "max_lat": 38.5,
}

# Flickr API
FLICKR_API_URL = "https://api.flickr.com/services/rest/"

# 허용 라이선스 (0=All rights, 1~6=CC, 7=No known copyright restrictions, 10=Public Domain)
ALLOWED_LICENSES = "1,2,4,5,6,7,8,9,10"


def _korea_grid(step_deg: float = 0.1) -> Iterator[tuple[float, float]]:
    """한국 전역을 격자 샘플링 — step_deg 간격으로 중심 좌표 생성"""
    lat = KOREA_BBOX["min_lat"]
    while lat <= KOREA_BBOX["max_lat"]:
        lon = KOREA_BBOX["min_lon"]
        while lon <= KOREA_BBOX["max_lon"]:
            yield round(lat, 4), round(lon, 4)
            lon += step_deg
        lat += step_deg


async def _search_flickr(
    session: aiohttp.ClientSession,
    api_key: str,
    lat: float,
    lon: float,
    radius_km: float = 8.0,
    per_page: int = 100,
    page: int = 1,
) -> list[dict]:
    params = {
        "method": "flickr.photos.search",
        "api_key": api_key,
        "format": "json",
        "nojsoncallback": 1,
        "has_geo": 1,
        "lat": lat,
        "lon": lon,
        "radius": radius_km,
        "radius_units": "km",
        "license": ALLOWED_LICENSES,
        "content_type": 1,  # photos only
        "media": "photos",
        "extras": "geo,url_m,url_l,url_o,tags,license,date_taken",
        "per_page": per_page,
        "page": page,
        "sort": "interestingness-desc",
    }
    try:
        async with session.get(FLICKR_API_URL, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            if data.get("stat") != "ok":
                return []
            return data.get("photos", {}).get("photo", [])
    except Exception as e:
        logger.debug(f"Flickr search error ({lat},{lon}): {e}")
        return []


async def _download_image(
    session: aiohttp.ClientSession,
    url: str,
    dest: Path,
) -> bool:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                return False
            content = await resp.read()
            if len(content) < 5000:  # 5KB 미만 스킵
                return False
            async with aiofiles.open(dest, "wb") as f:
                await f.write(content)
            return True
    except Exception:
        return False


def _best_url(photo: dict) -> str | None:
    """사용 가능한 최대 해상도 URL 반환"""
    for size in ("url_l", "url_m"):
        if photo.get(size):
            return photo[size]
    return None


async def crawl_flickr(
    api_key: str,
    output_dir: Path,
    max_photos: int = 500_000,
    grid_step: float = 0.15,
    radius_km: float = 8.0,
) -> int:
    """
    한국 전역 Flickr GPS 사진 수집.
    output_dir/images/ 에 이미지, output_dir/metadata.jsonl 에 메타데이터
    """
    output_dir = Path(output_dir)
    img_dir = output_dir / "flickr" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "flickr" / "metadata.jsonl"

    # 이미 수집된 항목 로드 (재시작 지원)
    seen_ids: set[str] = set()
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                try:
                    seen_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    logger.info(f"Flickr 이미 수집됨: {len(seen_ids)}개, 목표: {max_photos}개")

    collected = len(seen_ids)
    grid_points = list(_korea_grid(grid_step))

    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        async with aiofiles.open(meta_path, "a") as meta_f:
            for i, (lat, lon) in enumerate(grid_points):
                if collected >= max_photos:
                    break

                photos = await _search_flickr(session, api_key, lat, lon, radius_km)
                if not photos:
                    await asyncio.sleep(0.5)
                    continue

                tasks = []
                batch_meta = []
                for photo in photos:
                    pid = photo.get("id", "")
                    if not pid or pid in seen_ids:
                        continue
                    url = _best_url(photo)
                    if not url:
                        continue

                    photo_lat = float(photo.get("latitude", 0))
                    photo_lon = float(photo.get("longitude", 0))
                    if not (33 <= photo_lat <= 38.5 and 124 <= photo_lon <= 132):
                        continue

                    img_path = img_dir / f"flickr_{pid}.jpg"
                    if not img_path.exists():
                        tasks.append(_download_image(session, url, img_path))
                    else:
                        tasks.append(asyncio.coroutine(lambda: True)())

                    meta = {
                        "id": pid,
                        "source": "flickr",
                        "latitude": photo_lat,
                        "longitude": photo_lon,
                        "image_path": str(img_path.relative_to(output_dir)),
                        "title": photo.get("title", ""),
                        "tags": photo.get("tags", ""),
                        "license": photo.get("license", ""),
                        "date_taken": photo.get("datetaken", ""),
                    }
                    batch_meta.append((pid, meta))
                    seen_ids.add(pid)

                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for (pid, meta), ok in zip(batch_meta, results):
                        if ok is True:
                            await meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                            collected += 1

                if i % 50 == 0:
                    logger.info(f"Flickr 진행: {i}/{len(grid_points)} 격자, {collected}/{max_photos} 수집")
                    await meta_f.flush()

                # Flickr API rate limit: 3600 req/hr = 1 req/sec
                await asyncio.sleep(1.1)

    logger.info(f"Flickr 수집 완료: {collected}개")
    return collected


if __name__ == "__main__":
    import os
    key = os.environ.get("FLICKR_API_KEY", "")
    if not key:
        print("FLICKR_API_KEY 환경변수 필요")
        sys.exit(1)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./ml_data")
    asyncio.run(crawl_flickr(key, out, max_photos=int(sys.argv[2]) if len(sys.argv) > 2 else 100_000))

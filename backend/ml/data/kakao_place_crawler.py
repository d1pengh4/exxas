"""
카카오 Place API 기반 장소 사진 수집기
- 카테고리별 전국 검색 → 장소명 + 주소 + 사진 세트
- 아파트/건물 사진 + 정확한 지번주소 → OCR 학습 + CLIP 학습 동시에 활용
"""
import asyncio
import json
import sys
from pathlib import Path
import aiohttp
import aiofiles
from loguru import logger

KAKAO_LOCAL_API = "https://dapi.kakao.com/v2/local/search/keyword.json"
KAKAO_PLACE_PHOTO = "https://place.map.kakao.com/main/v/{place_id}"

# 수집할 카테고리 코드 (카카오 분류)
CATEGORIES = [
    ("AT4", "관광명소"),
    ("AD5", "숙박"),
    ("FD6", "음식점"),
    ("CE7", "카페"),
    ("CS2", "편의점"),
    ("MT1", "대형마트"),
    ("PK6", "주차장"),
    ("SW8", "지하철역"),
    ("BK9", "은행"),
    ("CT1", "문화시설"),
    ("AG2", "중개업소"),  # 부동산 → 건물 사진 풍부
    ("HP8", "병원"),
    ("PM9", "약국"),
    ("SC4", "학교"),
    ("PO3", "공공기관"),
]

# 전국 격자 (0.05도 ≈ 5km 간격)
def _korea_grid_coarse():
    for lat in range(330, 386, 5):  # 33.0 ~ 38.5
        for lon in range(1240, 1321, 5):  # 124.0 ~ 132.0
            yield lat / 10, lon / 10


async def _search_kakao(
    session: aiohttp.ClientSession,
    api_key: str,
    query: str,
    category_code: str,
    x: float,
    y: float,
    radius: int = 5000,
    page: int = 1,
) -> dict:
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {
        "query": query,
        "category_group_code": category_code,
        "x": x,
        "y": y,
        "radius": radius,
        "size": 15,
        "page": page,
        "sort": "accuracy",
    }
    try:
        async with session.get(
            KAKAO_LOCAL_API,
            headers=headers,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return {}
            return await resp.json()
    except Exception as e:
        logger.debug(f"Kakao search error: {e}")
        return {}


async def _fetch_place_photo_url(
    session: aiohttp.ClientSession,
    place_id: str,
) -> str | None:
    """카카오 Place 상세 → 대표 사진 URL"""
    try:
        async with session.get(
            KAKAO_PLACE_PHOTO.format(place_id=place_id),
            timeout=aiohttp.ClientTimeout(total=10),
            headers={"Referer": "https://map.kakao.com/"},
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            photos = data.get("basicInfo", {}).get("mainphotourl", "")
            if photos:
                return photos
            # fallback: photo list
            photo_list = data.get("photo", {}).get("photoList", [])
            if photo_list:
                return photo_list[0].get("orgurl", "")
    except Exception:
        pass
    return None


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
            if len(content) < 5000:
                return False
            async with aiofiles.open(dest, "wb") as f:
                await f.write(content)
            return True
    except Exception:
        return False


async def crawl_kakao_places(
    api_key: str,
    output_dir: Path,
    max_places: int = 300_000,
) -> int:
    output_dir = Path(output_dir)
    img_dir = output_dir / "kakao" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "kakao" / "metadata.jsonl"

    seen_ids: set[str] = set()
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                try:
                    seen_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    logger.info(f"카카오 장소 이미 수집됨: {len(seen_ids)}개")

    collected = len(seen_ids)
    connector = aiohttp.TCPConnector(limit=5)

    async with aiohttp.ClientSession(connector=connector) as session:
        async with aiofiles.open(meta_path, "a") as meta_f:
            for cat_code, cat_name in CATEGORIES:
                if collected >= max_places:
                    break
                logger.info(f"[카카오] 카테고리: {cat_name} ({cat_code})")

                for lat, lon in _korea_grid_coarse():
                    if collected >= max_places:
                        break

                    result = await _search_kakao(session, api_key, cat_name, cat_code, lon, lat)
                    documents = result.get("documents", [])

                    for doc in documents:
                        place_id = doc.get("id", "")
                        if not place_id or place_id in seen_ids:
                            continue

                        place_lat = float(doc.get("y", 0))
                        place_lon = float(doc.get("x", 0))
                        if not (33 <= place_lat <= 38.5 and 124 <= place_lon <= 132):
                            continue

                        address = doc.get("road_address_name") or doc.get("address_name", "")
                        place_name = doc.get("place_name", "")

                        # 사진 URL 조회 (rate limit 때문에 일부만)
                        photo_url = None
                        if collected % 3 == 0:  # 33%만 사진 수집 시도
                            photo_url = await _fetch_place_photo_url(session, place_id)
                            await asyncio.sleep(0.2)

                        img_path = None
                        if photo_url:
                            img_path = img_dir / f"kakao_{place_id}.jpg"
                            ok = await _download_image(session, photo_url, img_path)
                            if not ok:
                                img_path = None

                        seen_ids.add(place_id)
                        meta = {
                            "id": place_id,
                            "source": "kakao_place",
                            "latitude": place_lat,
                            "longitude": place_lon,
                            "address": address,
                            "place_name": place_name,
                            "category": cat_name,
                            "image_path": f"kakao/images/kakao_{place_id}.jpg" if img_path else None,
                            "phone": doc.get("phone", ""),
                        }
                        await meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                        collected += 1

                    await asyncio.sleep(0.11)  # Kakao API: ~10 req/sec

                    if collected % 1000 == 0 and collected > 0:
                        logger.info(f"  카카오 장소 {collected}/{max_places}")
                        await meta_f.flush()

    logger.info(f"카카오 장소 수집 완료: {collected}개")
    return collected


if __name__ == "__main__":
    import os
    key = os.environ.get("KAKAO_API_KEY", "")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./ml_data")
    asyncio.run(crawl_kakao_places(key, out, max_places=100_000))

"""
Milvus VPR 참조 DB 시딩 스크립트
- Wikimedia Commons geo search로 주요 도시 이미지 다운로드
- CosPlace 임베딩 추출
- Milvus에 저장

실행:
    cd backend
    .venv/bin/python scripts/seed_vpr_db.py
    .venv/bin/python scripts/seed_vpr_db.py --only-korea
    .venv/bin/python scripts/seed_vpr_db.py --max-per-city 20
"""
import argparse
import asyncio
import hashlib
import io
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ssl
import aiohttp
import certifi
import numpy as np
from PIL import Image
from loguru import logger

_SSL = ssl.create_default_context(cafile=certifi.where())

def _connector():
    return aiohttp.TCPConnector(ssl=_SSL)

# ── 참조 도시 목록 (lat, lon, 도시명, 반경km) ─────────────────────────────────
REFERENCE_CITIES = [
    # 한국
    (37.5665, 126.9780, "한국 서울",   3.0),
    (37.4979, 127.0276, "한국 서울 강남", 2.0),
    (37.5700, 126.9847, "한국 서울 종로", 1.5),
    (35.1796, 129.0756, "한국 부산",   3.0),
    (35.1587, 129.1603, "한국 부산 해운대", 2.0),
    (35.8714, 128.6014, "한국 대구",   2.0),
    (37.4563, 126.7052, "한국 인천",   2.0),
    (35.1595, 126.8526, "한국 광주",   2.0),
    (36.3504, 127.3845, "한국 대전",   2.0),
    (35.5384, 129.3114, "한국 울산",   2.0),
    (33.4890, 126.4983, "한국 제주",   2.0),
    # 일본
    (35.6762, 139.6503, "일본 도쿄",   3.0),
    (35.6586, 139.7454, "일본 도쿄 신주쿠", 1.5),
    (34.6937, 135.5022, "일본 오사카", 3.0),
    (35.0116, 135.7681, "일본 교토",   2.0),
    (43.0618, 141.3545, "일본 삿포로", 2.0),
    (33.5904, 130.4017, "일본 후쿠오카", 2.0),
    # 중국
    (39.9042, 116.4074, "중국 베이징", 3.0),
    (31.2304, 121.4737, "중국 상하이", 3.0),
    (23.1291, 113.2644, "중국 광저우", 2.0),
    (22.5431, 114.0579, "중국 선전",   2.0),
    (22.3193, 114.1694, "홍콩",        2.0),
    # 동남아
    (13.7563, 100.5018, "태국 방콕",   3.0),
    (1.3521,  103.8198, "싱가포르",    2.0),
    (10.8231, 106.6297, "베트남 호치민", 2.0),
    (3.1390,  101.6869, "말레이시아 쿠알라룸푸르", 2.0),
    # 미국
    (40.7128, -74.0060, "미국 뉴욕",  3.0),
    (34.0522, -118.2437,"미국 로스앤젤레스", 3.0),
    (37.7749, -122.4194,"미국 샌프란시스코", 2.0),
    (41.8781, -87.6298, "미국 시카고", 2.0),
    # 유럽
    (51.5074, -0.1278,  "영국 런던",  3.0),
    (48.8566, 2.3522,   "프랑스 파리", 3.0),
    (52.5200, 13.4050,  "독일 베를린", 2.0),
    (41.9028, 12.4964,  "이탈리아 로마", 2.0),
    (40.4168, -3.7038,  "스페인 마드리드", 2.0),
]


async def _wikimedia_geo_search(lat: float, lon: float, radius_km: float, max_results: int = 10) -> list[dict]:
    """Wikimedia Commons geo search API"""
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "geosearch",
        "gscoord": f"{lat}|{lon}",
        "gsradius": int(radius_km * 1000),
        "gslimit": max_results,
        "gsnamespace": 6,  # File namespace
        "format": "json",
    }
    headers = {"User-Agent": "EXXAS-VPR-Seeder/2.0"}
    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        return data.get("query", {}).get("geosearch", [])
    except Exception as e:
        logger.warning(f"Wikimedia geo search failed: {e}")
        return []


async def _wikimedia_image_url(page_id: int) -> tuple[str, float, float]:
    """페이지 ID → 이미지 URL + 실제 좌표"""
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "pageids": page_id,
        "prop": "imageinfo|coordinates",
        "iiprop": "url|mime",
        "iiurlwidth": 800,
        "format": "json",
    }
    headers = {"User-Agent": "EXXAS-VPR-Seeder/2.0"}
    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()

        page = data.get("query", {}).get("pages", {}).get(str(page_id), {})
        imageinfo = page.get("imageinfo", [{}])[0]
        img_url = imageinfo.get("thumburl") or imageinfo.get("url", "")
        mime = imageinfo.get("mime", "")
        if not img_url or mime not in ("image/jpeg", "image/png", "image/webp"):
            return "", 0.0, 0.0

        coords = page.get("coordinates", [{}])
        lat = float(coords[0].get("lat", 0)) if coords else 0.0
        lon = float(coords[0].get("lon", 0)) if coords else 0.0
        return img_url, lat, lon
    except Exception as e:
        logger.debug(f"Image URL fetch failed: {e}")
        return "", 0.0, 0.0


async def _download_image(url: str) -> bytes | None:
    headers = {"User-Agent": "EXXAS-VPR-Seeder/2.0"}
    try:
        async with aiohttp.ClientSession(headers=headers, connector=_connector()) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    return None
                return await resp.read()
    except Exception:
        return None


def _extract_embedding(image_bytes: bytes) -> np.ndarray | None:
    """CosPlace 임베딩 추출"""
    try:
        import torch
        import torchvision.transforms as T
        from app.pipeline.stage5_embedding import _get_cosplace_model, DEVICE

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        model = _get_cosplace_model()
        with torch.no_grad():
            emb = model(tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()[0]
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def _insert_to_milvus(records: list[dict]) -> int:
    """Milvus에 임베딩 일괄 저장 (MilvusClient API)"""
    try:
        from pymilvus import MilvusClient, DataType

        client = MilvusClient(uri="http://localhost:19530")

        col_name = "image_embeddings"
        if not client.has_collection(col_name):
            schema = client.create_schema()
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field("image_hash", DataType.VARCHAR, max_length=64)
            schema.add_field("latitude", DataType.FLOAT)
            schema.add_field("longitude", DataType.FLOAT)
            schema.add_field("location", DataType.VARCHAR, max_length=128)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=512)

            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 128},
            )
            client.create_collection(col_name, schema=schema, index_params=index_params)
            logger.info(f"Milvus collection '{col_name}' 생성됨")

        client.insert(col_name, records)
        client.flush(col_name)
        logger.info(f"Milvus 삽입: {len(records)}개")
        return len(records)

    except Exception as e:
        logger.error(f"Milvus 삽입 실패: {e}")
        return 0


async def seed_city(
    lat: float, lon: float, city_name: str, radius_km: float,
    max_per_city: int = 10,
) -> int:
    """도시 하나에 대해 이미지 검색 → 임베딩 → Milvus 저장"""
    logger.info(f"[{city_name}] 이미지 검색 중...")
    geo_results = await _wikimedia_geo_search(lat, lon, radius_km, max_per_city * 2)
    if not geo_results:
        logger.warning(f"[{city_name}] geo search 결과 없음")
        return 0

    records = []
    for item in geo_results[:max_per_city * 2]:
        page_id = item.get("pageid")
        if not page_id:
            continue
        img_url, actual_lat, actual_lon = await _wikimedia_image_url(page_id)
        if not img_url:
            continue
        # 좌표가 없으면 도시 중심 좌표 사용
        if actual_lat == 0 and actual_lon == 0:
            actual_lat, actual_lon = lat, lon

        img_bytes = await _download_image(img_url)
        if not img_bytes or len(img_bytes) < 10_000:
            await asyncio.sleep(1.0)
            continue

        emb = _extract_embedding(img_bytes)
        if emb is None:
            await asyncio.sleep(1.0)
            continue

        img_hash = hashlib.md5(img_bytes).hexdigest()
        records.append({
            "image_hash": img_hash,
            "latitude": float(actual_lat),
            "longitude": float(actual_lon),
            "location": city_name,
            "embedding": emb.tolist(),
        })
        logger.debug(f"  [{city_name}] {img_hash[:8]}... @ ({actual_lat:.4f},{actual_lon:.4f})")
        if len(records) >= max_per_city:
            break

        await asyncio.sleep(1.2)  # Wikimedia rate limit 준수

    if records:
        inserted = _insert_to_milvus(records)
        logger.info(f"[{city_name}] {inserted}개 저장 완료")
        return inserted
    return 0


async def main(only_korea: bool = False, max_per_city: int = 10):
    cities = REFERENCE_CITIES
    if only_korea:
        cities = [(lat, lon, name, r) for lat, lon, name, r in REFERENCE_CITIES if "한국" in name or "제주" in name]

    total = 0
    for lat, lon, city_name, radius in cities:
        try:
            n = await seed_city(lat, lon, city_name, radius, max_per_city)
            total += n
        except Exception as e:
            logger.error(f"[{city_name}] 실패: {e}")
        await asyncio.sleep(1.0)

    logger.info(f"\n=== 시딩 완료: 총 {total}개 임베딩 저장 ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXXAS VPR DB 시딩")
    parser.add_argument("--only-korea", action="store_true", help="한국 도시만 시딩")
    parser.add_argument("--max-per-city", type=int, default=10, help="도시당 최대 이미지 수")
    args = parser.parse_args()

    asyncio.run(main(only_korea=args.only_korea, max_per_city=args.max_per_city))

#!/usr/bin/env python3
"""
Milvus VPR DB 시드 스크립트 — 한국 주요 랜드마크 임베딩 사전 적재
실행: python3 -m app.data.seed_vpr_db

각 랜드마크의 GPS 좌표를 이용해:
1. Google Street View Static API로 실제 이미지 다운로드 (GOOGLE_STREET_VIEW_KEY 필요)
2. DINOv2-base로 768-dim 임베딩 생성
3. Milvus image_embeddings 컬렉션에 저장
"""
import asyncio
import hashlib
import logging
import os
import sys
from pathlib import Path

import httpx
import numpy as np
import torch
from PIL import Image

# path fix
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("seed_vpr")

# ── 랜드마크 좌표 (GPS + 방향각 포함) ──────────────────────────────────
# (이름, lat, lon, heading, pitch, city)
SEED_LANDMARKS = [
    # 서울 핵심
    ("롯데월드타워",        37.5126, 127.1028,  90,  0,  "서울 송파구"),
    ("롯데월드어드벤처",    37.5111, 127.0985, 180,  0,  "서울 송파구"),
    ("N서울타워",           37.5512, 126.9882, 270,  5,  "서울 용산구"),
    ("경복궁",              37.5796, 126.9770,  90, -5,  "서울 종로구"),
    ("광화문광장",          37.5722, 126.9769, 180,  0,  "서울 종로구"),
    ("명동성당",            37.5633, 126.9874,  45,  0,  "서울 중구"),
    ("명동거리",            37.5635, 126.9847, 270,  0,  "서울 중구"),
    ("동대문DDP",           37.5670, 127.0090, 135,  0,  "서울 중구"),
    ("홍대입구거리",        37.5572, 126.9253,  90,  0,  "서울 마포구"),
    ("강남역사거리",        37.4979, 127.0276,   0,  0,  "서울 강남구"),
    ("성수동카페거리",      37.5444, 127.0560, 180,  0,  "서울 성동구"),
    ("북촌한옥마을",        37.5817, 126.9835, 315,  0,  "서울 종로구"),
    ("청계천광교",          37.5691, 126.9820,  90,  0,  "서울 종로구"),
    ("여의도한강공원",      37.5284, 126.9341,  90, -5,  "서울 영등포구"),
    ("잠실한강공원",        37.5228, 127.0836,  90, -5,  "서울 송파구"),
    ("반포한강공원",        37.5130, 126.9941,  90, -5,  "서울 서초구"),
    ("올림픽공원",          37.5220, 127.1240, 180,  0,  "서울 송파구"),
    ("코엑스몰앞",          37.5115, 127.0598,  90,  0,  "서울 강남구"),
    ("이태원거리",          37.5344, 126.9943,  90,  0,  "서울 용산구"),
    ("인사동거리",          37.5742, 126.9839, 270,  0,  "서울 종로구"),
    ("압구정로데오",        37.5275, 127.0399, 180,  0,  "서울 강남구"),
    # 부산
    ("해운대해수욕장",      35.1588, 129.1603,  90,  0,  "부산 해운대구"),
    ("광안리해수욕장",      35.1531, 129.1186, 180,  0,  "부산 수영구"),
    ("광안대교",            35.1531, 129.1186,  90,  0,  "부산 수영구"),
    ("부산서면",            35.1572, 129.0589, 180,  0,  "부산 부산진구"),
    ("남포동자갈치시장",    35.0974, 129.0318,  90,  0,  "부산 중구"),
    ("감천문화마을",        35.0975, 129.0100, 180,  0,  "부산 사하구"),
    ("해동용궁사",          35.1877, 129.2236, 270,  0,  "부산 기장군"),
    # 제주
    ("성산일출봉",          33.4587, 126.9425, 270,  0,  "제주 서귀포시"),
    ("천지연폭포",          33.2475, 126.5569,  90,  0,  "제주 서귀포시"),
    ("제주도두해안도로",    33.5100, 126.4819,  90, -5,  "제주 제주시"),
    ("한라산",              33.3625, 126.5330, 180,  0,  "제주 서귀포시"),
    # 인천
    ("인천차이나타운",      37.4748, 126.6172, 180,  0,  "인천 중구"),
    ("송도국제도시",        37.3836, 126.6553,  90,  0,  "인천 연수구"),
    ("인천공항",            37.4602, 126.4407, 270,  0,  "인천 중구"),
    # 경기
    ("수원화성",            37.2853, 127.0125, 180,  0,  "경기 수원시"),
    ("판교테크노밸리",      37.3945, 127.1118,  90,  0,  "경기 성남시"),
    ("에버랜드",            37.2935, 127.2018, 180,  0,  "경기 용인시"),
    # 대구
    ("동성로",              35.8700, 128.5940, 180,  0,  "대구 중구"),
    ("서문시장",            35.8694, 128.5724, 270,  0,  "대구 중구"),
    # 광주
    ("광주518민주광장",     35.1474, 126.9216, 180,  0,  "광주 동구"),
    # 대전
    ("대전엑스포공원",      36.3730, 127.3858,  90,  0,  "대전 유성구"),
    # 강원
    ("강릉경포해변",        37.8034, 128.9010,  90, -5,  "강원 강릉시"),
    ("설악산대청봉",        38.1197, 128.4650, 180,  0,  "강원 속초시"),
]

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = "image_embeddings"
DIM = 768


async def _get_street_view_image(lat: float, lon: float, heading: int, pitch: int,
                                  api_key: str) -> bytes | None:
    """Google Street View Static API로 이미지 다운로드"""
    url = (
        f"https://maps.googleapis.com/maps/api/streetview"
        f"?size=640x480&location={lat},{lon}&heading={heading}"
        f"&pitch={pitch}&fov=90&key={api_key}"
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url)
            if r.status_code == 200 and len(r.content) > 5000:
                return r.content
    except Exception as e:
        logger.warning(f"Street View fetch failed ({lat},{lon}): {e}")
    return None


_LOCAL_CLIP_PATH = str(Path(__file__).resolve().parents[3] / "modelforder" / "model")

_clip_model_cache = None
_clip_processor_cache = None


def _load_local_clip():
    global _clip_model_cache, _clip_processor_cache
    if _clip_model_cache is None:
        from transformers import CLIPModel, CLIPProcessor
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Loading local fine-tuned CLIP: {_LOCAL_CLIP_PATH}")
        _clip_processor_cache = CLIPProcessor.from_pretrained(_LOCAL_CLIP_PATH)
        _clip_model_cache = CLIPModel.from_pretrained(_LOCAL_CLIP_PATH).to(device).eval()
    return _clip_model_cache, _clip_processor_cache


def _compute_dinov2_embedding(img_bytes: bytes) -> np.ndarray | None:
    """로컬 fine-tuned CLIP으로 768-dim 임베딩 계산"""
    try:
        from io import BytesIO
        model, processor = _load_local_clip()
        device = next(model.parameters()).device
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                if hasattr(features, "image_embeds"):
                    features = features.image_embeds
                elif hasattr(features, "last_hidden_state"):
                    features = features.last_hidden_state[:, 0]
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy().flatten()
    except Exception as e:
        logger.error(f"Local CLIP embedding failed: {e}")
        return None


async def _ensure_collection():
    """Milvus 컬렉션 생성/확인"""
    try:
        from pymilvus import MilvusClient, DataType
        client = MilvusClient(uri=MILVUS_URI)
        if not client.has_collection(COLLECTION_NAME):
            logger.info(f"Creating Milvus collection '{COLLECTION_NAME}' dim={DIM}")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                dimension=DIM,
                metric_type="COSINE",
                vector_field_name="embedding",
                id_field_name="id",
                auto_id=True,
            )
            # 추가 필드 스키마 정의
            client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=client.create_schema()
                    .add_field("id", DataType.INT64, is_primary=True, auto_id=True)
                    .add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)
                    .add_field("image_hash", DataType.VARCHAR, max_length=64)
                    .add_field("latitude", DataType.DOUBLE)
                    .add_field("longitude", DataType.DOUBLE)
                    .add_field("location", DataType.VARCHAR, max_length=200),
                index_params=client.prepare_index_params()
                    .add_index("embedding", index_type="IVF_FLAT",
                               metric_type="COSINE", params={"nlist": 128}),
            )
        logger.info("Milvus collection ready")
        return client
    except Exception as e:
        logger.error(f"Milvus setup failed: {e}")
        return None


async def seed():
    api_key = os.getenv("GOOGLE_STREET_VIEW_KEY", "")
    if not api_key:
        logger.warning("GOOGLE_STREET_VIEW_KEY not set — using placeholder embeddings (random noise)")

    client = await _ensure_collection()
    if not client:
        logger.error("Cannot connect to Milvus. Is it running?")
        return

    inserted = 0
    for name, lat, lon, heading, pitch, city in SEED_LANDMARKS:
        logger.info(f"Processing: {name} ({lat:.4f}, {lon:.4f})")

        # 이미지 취득
        img_bytes = None
        if api_key:
            img_bytes = await _get_street_view_image(lat, lon, heading, pitch, api_key)

        if img_bytes is None:
            # Google Street View 없을 때: 좌표 기반 결정론적 임베딩 생성 (테스트용)
            logger.warning(f"  No image for {name}, using coordinate-based dummy embedding")
            rng = np.random.default_rng(seed=int(abs(lat * 1000) + abs(lon * 1000)))
            dummy = rng.standard_normal(DIM).astype(np.float32)
            dummy = dummy / np.linalg.norm(dummy)
            emb = dummy
        else:
            emb = _compute_dinov2_embedding(img_bytes)
            if emb is None:
                continue

        img_hash = hashlib.md5(f"{name}_{lat}_{lon}".encode()).hexdigest()
        data = [{
            "embedding": emb.tolist(),
            "image_hash": img_hash,
            "latitude": float(lat),
            "longitude": float(lon),
            "location": f"{city} {name}",
        }]

        try:
            client.insert(collection_name=COLLECTION_NAME, data=data)
            inserted += 1
            logger.info(f"  Inserted: {name} ({city})")
        except Exception as e:
            logger.error(f"  Insert failed for {name}: {e}")

    # 인덱스 빌드
    try:
        client.create_index(
            collection_name=COLLECTION_NAME,
            field_name="embedding",
            index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE",
                          "params": {"nlist": min(128, max(16, inserted // 4))}},
        )
        client.load_collection(COLLECTION_NAME)
        logger.info(f"Index built and collection loaded. Total inserted: {inserted}")
    except Exception as e:
        logger.warning(f"Index build failed (may already exist): {e}")

    logger.info(f"Seeding complete: {inserted}/{len(SEED_LANDMARKS)} landmarks inserted")


if __name__ == "__main__":
    asyncio.run(seed())

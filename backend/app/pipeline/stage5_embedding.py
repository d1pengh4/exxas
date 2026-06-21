"""
Stage 5: AI 임베딩 & VPR — 한국 전용 v3
- GeoCLIP: GPS 좌표 직접 예측 (한국 범위 필터)
- Fine-tuned CLIP (로컬): 한국 세부 지역 분류 + VPR 임베딩 768d
- StreetCLIP: 장소 유형 분류 (폴백)
- DINOv2-base: VPR 시각 임베딩 (폴백)
"""
import io
import os as _os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from PIL import Image
import torch
from loguru import logger

DEVICE = _os.getenv("PYTORCH_DEVICE", "cpu")

# 로컬 fine-tuned CLIP 모델 경로 (backend/app/pipeline/ → 3단계 위 = EXXAS/)
_LOCAL_CLIP_PATH = str(
    Path(__file__).resolve().parents[3] / "modelforder" / "model"
)

VPR_DIM = 768  # fine-tuned CLIP projection_dim


@dataclass
class EmbeddingResult:
    # GeoCLIP
    geoclip_top_location: str = ""
    geoclip_latitude: float = 0.0
    geoclip_longitude: float = 0.0
    geoclip_score: float = 0.0
    geoclip_top5: list[dict] = field(default_factory=list)

    # StreetCLIP 한국 특화
    streetclip_country: str = ""
    streetclip_score: float = 0.0
    streetclip_top3: list[dict] = field(default_factory=list)

    # OpenCLIP ViT-L-14 한국 지역 분류 (NEW)
    openclip_country: str = ""        # 항상 "South Korea" 또는 "Unknown"
    openclip_score: float = 0.0
    openclip_city_hint: str = ""      # 세분화된 도시/구 힌트 (ex: "서울 강남구", "부산 해운대")
    openclip_top5: list[dict] = field(default_factory=list)

    # DINOv2 임베딩 (VPR)
    dinov2_embedding: list[float] = field(default_factory=list)

    # VPR
    vpr_matches: list[dict] = field(default_factory=list)
    best_vpr_location: str = ""
    best_vpr_similarity: float = 0.0

    # 저장용 임베딩
    embedding_vector: list[float] = field(default_factory=list)

    # 앙상블 최종 지역
    ensemble_country: str = "South Korea"
    ensemble_confidence: float = 0.0
    ensemble_region: str = ""         # 한국 세부 지역 (도시/구)


# ── 한국 세부 지역 프롬프트 ───────────────────────────────────────────────
# OpenCLIP ViT-L-14으로 한국 내 세부 지역 분류
# 각 항목: label(CLIP 프롬프트), region(출력 지역명), city(도시)

_KOREA_REGIONS: list[dict] = [
    # ── 서울 주요 지역 ──────────────────────────────────────
    {"label": "Gangnam Seocho Seoul luxury commercial high-rise office skyscraper", "region": "서울 강남구/서초구", "city": "서울"},
    {"label": "Hongdae Mapo Seoul university entertainment youth culture street", "region": "서울 마포구 홍대", "city": "서울"},
    {"label": "Jongno Insadong Bukchon Seoul traditional historic hanok cultural", "region": "서울 종로구 인사동", "city": "서울"},
    {"label": "Myeongdong Junggu Seoul shopping tourist dense crowd cosmetics", "region": "서울 중구 명동", "city": "서울"},
    {"label": "Itaewon Yongsan Seoul international multicultural foreign restaurants", "region": "서울 용산구 이태원", "city": "서울"},
    {"label": "Yeouido Seoul financial district Han River park skyscraper", "region": "서울 영등포구 여의도", "city": "서울"},
    {"label": "Jamsil Lotte World Tower Songpa Seoul 555m supertall skyscraper", "region": "서울 송파구 잠실", "city": "서울"},
    {"label": "Dongdaemun Seoul fashion market 24hours shopping DDP design plaza", "region": "서울 동대문", "city": "서울"},
    {"label": "Seongsu Seoul hipster industrial cafe brick building", "region": "서울 성수동", "city": "서울"},
    {"label": "Sinchon Eunpyeong Nowon North Seoul residential apartment university", "region": "서울 북부 (신촌/은평/노원)", "city": "서울"},
    {"label": "Gangdong Gildong Hanam Seoul east residential apartment", "region": "서울 강동구", "city": "서울"},
    {"label": "Gwanak Dongjak Noryangjin Sillim Seoul south university", "region": "서울 관악구/동작구", "city": "서울"},
    {"label": "Han River Hangang park Seoul riverside bicycle path promenade", "region": "서울 한강공원", "city": "서울"},
    {"label": "Seoul subway underground station platform train", "region": "서울 지하철역", "city": "서울"},
    {"label": "Seoul dense apartment complex residential tower block parking", "region": "서울 아파트단지", "city": "서울"},
    # ── 부산 ────────────────────────────────────────────────
    {"label": "Haeundae Beach Busan sandy beach sea coast resort hotel", "region": "부산 해운대", "city": "부산"},
    {"label": "Gwangalli Beach Busan diamond bridge suspension illuminated night ocean", "region": "부산 광안리", "city": "부산"},
    {"label": "Nampo Seomyeon Busan downtown commercial shopping BIFF", "region": "부산 남포동/서면", "city": "부산"},
    {"label": "Busan Port harbor container ship industrial", "region": "부산 항구/북항", "city": "부산"},
    {"label": "Gamcheon Busan colorful culture village hillside stairs", "region": "부산 감천문화마을", "city": "부산"},
    # ── 기타 주요 도시 ──────────────────────────────────────
    {"label": "Daegu Dongseongro downtown commercial street fashion", "region": "대구 동성로", "city": "대구"},
    {"label": "Daegu Apsan Park cable car mountain urban backdrop", "region": "대구 앞산공원", "city": "대구"},
    {"label": "Incheon Songdo new city international business district", "region": "인천 송도", "city": "인천"},
    {"label": "Incheon Airport international terminal flight", "region": "인천공항", "city": "인천"},
    {"label": "Incheon Chinatown gate arch colorful red lantern", "region": "인천 차이나타운", "city": "인천"},
    {"label": "Incheon Wolmido island amusement park seaside", "region": "인천 월미도", "city": "인천"},
    {"label": "Gwangju Asia Culture Center square modern architecture", "region": "광주 국립아시아문화전당", "city": "광주"},
    {"label": "Gwangju Mudeungsan mountain green forest park", "region": "광주 무등산", "city": "광주"},
    {"label": "Daejeon Expo Science Park tower", "region": "대전 엑스포과학공원", "city": "대전"},
    {"label": "Daejeon Yuseong hot spring resort spa", "region": "대전 유성온천", "city": "대전"},
    {"label": "Ulsan industrial factory POSCO Hyundai shipyard crane", "region": "울산 공업지역", "city": "울산"},
    {"label": "Ulsan Daewangam Park rock sea coast", "region": "울산 대왕암공원", "city": "울산"},
    {"label": "Suwon Hwaseong Fortress historic castle wall gate", "region": "수원 화성", "city": "수원"},
    {"label": "Seongnam Bundang Pangyo IT tech park high-rise", "region": "성남 판교/분당", "city": "성남"},
    {"label": "Goyang Ilsan Lake Park waterfront apartment", "region": "경기 고양 일산", "city": "고양"},
    {"label": "Yongin Everland theme park rides castle", "region": "경기 용인 에버랜드", "city": "용인"},
    {"label": "Namyangju Gapyeong Nami Island forest river", "region": "경기 남이섬", "city": "가평"},
    {"label": "Gangneung Gyeongpo Beach sea pine tree east coast", "region": "강원 강릉 경포대", "city": "강릉"},
    {"label": "Sokcho Seoraksan mountain snow rocky peak", "region": "강원 속초 설악산", "city": "속초"},
    {"label": "Chuncheon Uiam Lake skywalk pedestrian glass bridge", "region": "강원 춘천 의암호", "city": "춘천"},
    {"label": "Pyeongchang Alpensia ski resort alpine snow mountain", "region": "강원 평창 알펜시아", "city": "평창"},
    {"label": "Jeonju Hanok Village traditional roof tile alley", "region": "전북 전주 한옥마을", "city": "전주"},
    {"label": "Yeosu port night view Dolsan suspension bridge sea", "region": "전남 여수 돌산대교", "city": "여수"},
    {"label": "Suncheon Bay Garden reed field sunset estuary", "region": "전남 순천만정원", "city": "순천"},
    {"label": "Damyang bamboo forest green tall stalks", "region": "전남 담양 죽녹원", "city": "담양"},
    {"label": "Gyeongju Bulguksa temple pagoda historic Buddha", "region": "경북 경주 불국사", "city": "경주"},
    {"label": "Gyeongju Cheomseongdae observatory stone tower ancient", "region": "경북 경주 첨성대", "city": "경주"},
    {"label": "Andong Hahoe Village traditional hanok river bend", "region": "경북 안동 하회마을", "city": "안동"},
    {"label": "Pohang Homigot sunrise square bronze hands sea", "region": "경북 포항 호미곶", "city": "포항"},
    {"label": "Tongyeong cable car sea island panorama", "region": "경남 통영 케이블카", "city": "통영"},
    {"label": "Jeju Island volcanic basalt rock coast sea cliff", "region": "제주도 해안", "city": "제주"},
    {"label": "Jeju Hallasan mountain green lush misty crater", "region": "제주도 한라산", "city": "제주"},
    {"label": "Jeju Seongsan Ilchulbong crater rim sunrise ocean", "region": "제주도 성산일출봉", "city": "제주"},
    {"label": "Jeju Hyeopjae Beach turquoise shallow water white sand", "region": "제주도 협재해변", "city": "제주"},
    # ── 장소 유형 ────────────────────────────────────────────
    {"label": "Korean highway expressway rest stop convenience store parking", "region": "한국 고속도로 휴게소", "city": ""},
    {"label": "Korean school elementary middle high campus playground sports court", "region": "한국 학교", "city": ""},
    {"label": "Korean university campus building students", "region": "한국 대학교", "city": ""},
    {"label": "Korean traditional market outdoor stalls vendors", "region": "한국 전통시장", "city": ""},
    {"label": "Korean amusement theme park rides Lotte World Everland", "region": "한국 테마파크", "city": ""},
    {"label": "Korean indoor shopping mall department store COEX Lotte Starfield", "region": "한국 대형쇼핑몰", "city": ""},
    {"label": "Korean hospital medical center clinic", "region": "한국 병원", "city": ""},
    {"label": "Korean national park mountain hiking trail nature", "region": "한국 국립공원/등산로", "city": ""},
    {"label": "Korean river Han Nakdong Geum bridge promenade", "region": "한국 강변", "city": ""},
    {"label": "Korean rural countryside village rice field", "region": "한국 농촌/시골", "city": ""},
    {"label": "Korean apartment complex high-rise residential tower", "region": "한국 아파트단지", "city": ""},
    {"label": "Korean subway underground station platform tile", "region": "한국 지하철역", "city": ""},
    {"label": "Korean KTX high speed train station platform", "region": "한국 KTX역", "city": ""},
    {"label": "Korean beach resort east coast west coast", "region": "한국 해수욕장", "city": ""},
    {"label": "Korean temple Buddhist pagoda wooden gate", "region": "한국 사찰", "city": ""},
]

# StreetCLIP 한국 장소 유형 프롬프트
_STREETCLIP_KOREA: list[dict] = [
    {"label": "Gangnam luxury commercial district Seoul", "region": "서울 강남"},
    {"label": "Hongdae university entertainment Mapo Seoul", "region": "서울 홍대/마포"},
    {"label": "Myeongdong shopping tourist Seoul", "region": "서울 명동"},
    {"label": "Han River park waterfront Seoul", "region": "서울 한강공원"},
    {"label": "Seoul traditional historic neighborhood Jongno Bukchon", "region": "서울 종로/북촌"},
    {"label": "Seoul dense apartment residential complex", "region": "서울 아파트"},
    {"label": "Haeundae beach resort Busan", "region": "부산 해운대"},
    {"label": "Busan downtown Nampo Seomyeon commercial", "region": "부산 중심"},
    {"label": "Daegu downtown Dongseongro", "region": "대구"},
    {"label": "Incheon Songdo international city", "region": "인천 송도"},
    {"label": "Jeju Island nature volcanic beach", "region": "제주도"},
    {"label": "Daejeon university research district", "region": "대전"},
    {"label": "Korean suburban apartment satellite city", "region": "한국 신도시"},
    {"label": "Korean school playground sports court", "region": "한국 학교"},
    {"label": "Korean indoor facility building lobby", "region": "한국 실내"},
    {"label": "Korean mountain trail nature park", "region": "한국 자연/산"},
    {"label": "Korean traditional market stalls", "region": "한국 시장"},
    {"label": "Korean industrial factory area", "region": "한국 공업지역"},
]


# ── 메인 실행 ─────────────────────────────────────────────────────────────

async def run(image_bytes: bytes) -> EmbeddingResult:
    result = EmbeddingResult()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 1. GeoCLIP (한국 내 GPS 좌표 예측)
    try:
        geo = await _run_geoclip(img)
        result.geoclip_top_location = geo.get("top_location", "")
        result.geoclip_latitude = geo.get("latitude", 0.0)
        result.geoclip_longitude = geo.get("longitude", 0.0)
        result.geoclip_score = geo.get("score", 0.0)
        result.geoclip_top5 = geo.get("top5", [])
    except Exception as e:
        logger.warning(f"GeoCLIP failed: {e}")

    # 2. OpenCLIP ViT-L-14 한국 세부 지역 분류
    try:
        oc = await _run_openclip_korea(img)
        result.openclip_country = "South Korea"
        result.openclip_score = oc.get("score", 0.0)
        result.openclip_city_hint = oc.get("region", "")
        result.openclip_top5 = oc.get("top5", [])
    except Exception as e:
        logger.warning(f"OpenCLIP Korea failed: {e}")

    # 3. StreetCLIP 한국 장소 유형
    try:
        sc = await _run_streetclip_korea(img)
        result.streetclip_country = "South Korea"
        result.streetclip_score = sc.get("score", 0.0)
        result.streetclip_top3 = sc.get("top3", [])
        if sc.get("region"):
            # StreetCLIP 결과가 OpenCLIP보다 구체적이면 보완
            if not result.openclip_city_hint:
                result.openclip_city_hint = sc.get("region", "")
    except Exception as e:
        logger.warning(f"StreetCLIP Korea failed: {e}")

    # 4. VPR 임베딩 — 로컬 fine-tuned CLIP 768d (1순위)
    # 폴백: StreetCLIP 512d → DINOv2 768d → CosPlace 512d
    try:
        local_emb = await _get_local_clip_image_embedding(img)
        if local_emb:
            result.embedding_vector = local_emb
            result.dinov2_embedding = local_emb  # 호환성 유지
    except Exception as e:
        logger.warning(f"Local CLIP embedding failed ({e}), falling back to StreetCLIP")
        try:
            streetclip_emb = await _get_streetclip_image_embedding(img)
            if streetclip_emb:
                result.embedding_vector = streetclip_emb
        except Exception as e2:
            logger.warning(f"StreetCLIP embedding failed: {e2}")
            try:
                dino = await _run_dinov2(img)
                result.dinov2_embedding = dino.get("embedding", [])
                result.embedding_vector = result.dinov2_embedding
            except Exception:
                pass

    # 5. Milvus VPR 검색
    if result.embedding_vector:
        try:
            emb_arr = np.array(result.embedding_vector, dtype=np.float32)
            vpr = await _milvus_search(emb_arr)
            result.vpr_matches = vpr
            if vpr:
                result.best_vpr_location = vpr[0].get("location", "")
                result.best_vpr_similarity = vpr[0].get("similarity", 0.0)
        except Exception as e:
            logger.warning(f"VPR failed: {e}")

    # 6. 앙상블: 한국 세부 지역 결정
    result.ensemble_country = "South Korea"
    result.ensemble_region, result.ensemble_confidence = _ensemble_korea_region(result)

    return result


# ── 로컬 Fine-tuned CLIP — 한국 지역 분류 + VPR 임베딩 ───────────────────

_local_clip_cache: dict = {}


def _get_local_clip():
    """로컬 fine-tuned CLIP 로드 (modelforder/model)"""
    if "model" not in _local_clip_cache:
        from transformers import CLIPModel, CLIPProcessor
        from pathlib import Path as _Path
        path = _LOCAL_CLIP_PATH
        if not _Path(path).exists():
            raise FileNotFoundError(f"로컬 CLIP 모델 없음: {path}")
        model = CLIPModel.from_pretrained(path).to(DEVICE).eval()
        processor = CLIPProcessor.from_pretrained(path)
        _local_clip_cache["model"] = model
        _local_clip_cache["processor"] = processor
        logger.info(f"Local fine-tuned CLIP loaded ({path}) — projection_dim=768")
    return _local_clip_cache["model"], _local_clip_cache["processor"]


async def _run_openclip_korea(img: Image.Image) -> dict:
    """로컬 fine-tuned CLIP으로 한국 세부 지역 분류"""
    import asyncio

    def _sync():
        import torch as _t
        model, processor = _get_local_clip()
        prompts = [f"a photo of {r['label']} in South Korea" for r in _KOREA_REGIONS]
        inputs = processor(
            text=prompts, images=img, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)

        with _t.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().float()

        sorted_idx = probs.argsort(descending=True)
        best_i = int(sorted_idx[0])
        top5 = [
            {"region": _KOREA_REGIONS[int(i)]["region"],
             "city": _KOREA_REGIONS[int(i)]["city"],
             "score": round(float(probs[int(i)]), 4)}
            for i in sorted_idx[:5]
        ]
        return {
            "region": _KOREA_REGIONS[best_i]["region"],
            "city": _KOREA_REGIONS[best_i]["city"],
            "score": round(float(probs[best_i]), 4),
            "top5": top5,
        }

    return await asyncio.to_thread(_sync)


# ── StreetCLIP 한국 장소 유형 ─────────────────────────────────────────────

_streetclip_cache: dict = {}


def _get_streetclip():
    if "model" not in _streetclip_cache:
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("geolocal/StreetCLIP").to(DEVICE).eval()
        processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
        _streetclip_cache["model"] = model
        _streetclip_cache["processor"] = processor
        logger.info("StreetCLIP loaded for Korea place type classification")
    return _streetclip_cache["model"], _streetclip_cache["processor"]


async def _run_streetclip_korea(img: Image.Image) -> dict:
    import asyncio

    def _sync():
        import torch as _t
        model, processor = _get_streetclip()
        prompts = [f"Street view of {r['label']}" for r in _STREETCLIP_KOREA]
        inputs = processor(text=prompts, images=img, return_tensors="pt", padding=True).to(DEVICE)
        with _t.no_grad():
            probs = model(**inputs).logits_per_image.softmax(dim=1)[0]

        sorted_idx = probs.argsort(descending=True)
        best_i = int(sorted_idx[0])
        top3 = [
            {"region": _STREETCLIP_KOREA[int(i)]["region"], "score": round(float(probs[int(i)]), 4)}
            for i in sorted_idx[:3]
        ]
        return {
            "region": _STREETCLIP_KOREA[best_i]["region"],
            "score": round(float(probs[best_i]), 4),
            "top3": top3,
        }

    return await asyncio.to_thread(_sync)


# ── 로컬 CLIP 이미지 임베딩 (VPR용 768d) ─────────────────────────────────

async def _get_local_clip_image_embedding(img: Image.Image) -> list[float]:
    """로컬 fine-tuned CLIP 이미지 피처 768d 추출 — Milvus VPR 검색용"""
    import asyncio

    def _sync():
        import torch as _t
        model, processor = _get_local_clip()
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with _t.no_grad():
            features = model.get_image_features(**inputs)
            if not isinstance(features, _t.Tensor):
                if hasattr(features, "image_embeds"):
                    features = features.image_embeds
                elif hasattr(features, "pooler_output") and features.pooler_output is not None:
                    features = features.pooler_output
                elif hasattr(features, "last_hidden_state"):
                    features = features.last_hidden_state[:, 0]
            features = _t.nn.functional.normalize(features, p=2, dim=1)
        return features[0].cpu().tolist()

    return await asyncio.to_thread(_sync)


# ── StreetCLIP 이미지 임베딩 (VPR용 폴백 512d) ───────────────────────────

async def _get_streetclip_image_embedding(img: Image.Image) -> list[float]:
    """StreetCLIP 이미지 피처 512d 추출 — Milvus VPR 검색 및 시딩과 동일 모델"""
    import asyncio

    def _sync():
        import torch as _t
        model, processor = _get_streetclip()
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with _t.no_grad():
            features = model.get_image_features(**inputs)
            # transformers 5.x: get_image_features may return BaseModelOutputWithPooling
            if not isinstance(features, _t.Tensor):
                if hasattr(features, "image_embeds"):
                    features = features.image_embeds
                elif hasattr(features, "pooler_output") and features.pooler_output is not None:
                    features = features.pooler_output
                elif hasattr(features, "last_hidden_state"):
                    features = features.last_hidden_state[:, 0]
            features = _t.nn.functional.normalize(features, p=2, dim=1)
        return features[0].cpu().tolist()

    return await asyncio.to_thread(_sync)


# ── DINOv2 ────────────────────────────────────────────────────────────────

_dinov2_cache: dict = {}


def _get_dinov2():
    if "model" not in _dinov2_cache:
        from transformers import AutoImageProcessor, AutoModel
        try:
            processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            model = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()
            _dinov2_cache["model"] = model
            _dinov2_cache["processor"] = processor
            logger.info("DINOv2-base loaded (768-dim VPR)")
        except Exception as e:
            logger.warning(f"DINOv2 load failed: {e}")
            return None, None
    return _dinov2_cache.get("model"), _dinov2_cache.get("processor")


async def _run_dinov2(img: Image.Image) -> dict:
    import asyncio

    def _sync():
        import torch as _t
        model, processor = _get_dinov2()
        if model is None:
            return {}
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with _t.no_grad():
            out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :]
            emb = _t.nn.functional.normalize(emb, p=2, dim=1)
        return {"embedding": emb.cpu().numpy()[0].tolist()}

    return await asyncio.to_thread(_sync)


# ── GeoCLIP ───────────────────────────────────────────────────────────────

_geoclip_cache = None


def _get_geoclip_model():
    global _geoclip_cache
    if _geoclip_cache is None:
        from geoclip import GeoCLIP
        _geoclip_cache = GeoCLIP().to(DEVICE)
        _geoclip_cache.eval()
        logger.info("GeoCLIP loaded")
    return _geoclip_cache


async def _run_geoclip(img: Image.Image) -> dict:
    try:
        import asyncio

        def _sync():
            import tempfile, os as _os
            model = _get_geoclip_model()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                img.save(f, format="JPEG")
                tmp = f.name
            try:
                top_pred, top_scores = model.predict(tmp, top_k=5)
            finally:
                _os.unlink(tmp)
            top5 = []
            for i, (coords, score) in enumerate(zip(top_pred, top_scores)):
                lat, lon = float(coords[0]), float(coords[1])
                top5.append({
                    "rank": i + 1,
                    "latitude": round(lat, 4), "longitude": round(lon, 4),
                    "location": _coords_to_korea_region(lat, lon),
                    "score": round(float(score), 4),
                })
            return top5

        top5 = await asyncio.to_thread(_sync)
        if top5:
            b = top5[0]
            return {"top_location": b["location"], "latitude": b["latitude"],
                    "longitude": b["longitude"], "score": b["score"], "top5": top5}
    except ImportError:
        return await _geoclip_clip_fallback(img)
    except Exception as e:
        logger.error(f"GeoCLIP error: {e}")
    return {}


async def _geoclip_clip_fallback(img: Image.Image) -> dict:
    try:
        import asyncio
        from transformers import CLIPProcessor, CLIPModel
        # 한국 주요 도시 분류
        regions = ["Seoul Korea", "Busan Korea", "Jeju Korea", "Incheon Korea",
                   "Daegu Korea", "Daejeon Korea", "Gwangju Korea"]

        def _sync():
            proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
            inputs = proc(text=[f"a photo in {c}" for c in regions],
                         images=img, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                probs = model(**inputs).logits_per_image.softmax(dim=1)[0]
            bi = int(probs.argmax())
            return {"top_location": regions[bi], "latitude": 0.0, "longitude": 0.0,
                    "score": float(probs[bi]), "top5": []}

        return await asyncio.to_thread(_sync)
    except Exception as e:
        logger.error(f"CLIP fallback failed: {e}")
        return {}


# ── CosPlace 폴백 ─────────────────────────────────────────────────────────

_cosplace_cache = None


async def _get_cosplace_embedding(img: Image.Image) -> np.ndarray:
    import asyncio
    import torchvision.transforms as T

    global _cosplace_cache

    def _sync():
        global _cosplace_cache
        if _cosplace_cache is None:
            try:
                _cosplace_cache = torch.hub.load(
                    "gmberton/cosplace", "get_trained_model",
                    backbone="ResNet50", fc_output_dim=512, trust_repo=True,
                ).to(DEVICE).eval()
            except Exception:
                import torchvision.models as models
                m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                m.fc = torch.nn.Linear(2048, 512)
                _cosplace_cache = m.to(DEVICE).eval()
        t = T.Compose([
            T.Resize((512, 512)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = _cosplace_cache(t)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()[0]

    return await asyncio.to_thread(_sync)


# ── Milvus ────────────────────────────────────────────────────────────────

async def _milvus_search(embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri="http://localhost:19530")
        if not client.has_collection("image_embeddings"):
            return []
        results = client.search(
            collection_name="image_embeddings",
            data=[embedding.tolist()],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["image_hash", "latitude", "longitude", "location"],
        )
        matches = []
        for hit in results[0]:
            lat = hit.get("entity", {}).get("latitude", 0)
            lon = hit.get("entity", {}).get("longitude", 0)
            loc = hit.get("entity", {}).get("location") or _coords_to_korea_region(lat, lon)
            matches.append({
                "similarity": round(float(hit.get("distance", 0)), 4),
                "latitude": lat, "longitude": lon, "location": loc,
            })
        return matches
    except Exception as e:
        logger.debug(f"Milvus skipped: {type(e).__name__}")
        return []


# ── 한국 지역 앙상블 ──────────────────────────────────────────────────────

def _ensemble_korea_region(r: EmbeddingResult) -> tuple[str, float]:
    """OpenCLIP(0.5) + StreetCLIP(0.3) + GeoCLIP(0.2) 가중 투표로 한국 세부 지역 결정"""
    region_votes: dict[str, float] = {}

    # OpenCLIP 한국 지역 분류 (가중치 0.5)
    if r.openclip_city_hint and r.openclip_score > 0.03:
        region_votes[r.openclip_city_hint] = region_votes.get(r.openclip_city_hint, 0) + r.openclip_score * 0.5
    # top5도 반영 (약하게)
    for item in r.openclip_top5[1:3]:
        reg = item.get("region", "")
        sc = item.get("score", 0)
        if reg and sc > 0.02:
            region_votes[reg] = region_votes.get(reg, 0) + sc * 0.15

    # StreetCLIP 장소 유형 (가중치 0.3)
    for item in r.streetclip_top3[:2]:
        reg = item.get("region", "")
        sc = item.get("score", 0)
        if reg and sc > 0.03:
            w = 0.3 if item == r.streetclip_top3[0] else 0.1
            region_votes[reg] = region_votes.get(reg, 0) + sc * w

    # GeoCLIP 좌표 기반 지역 (가중치 0.2, 좌표가 한국 범위인 경우만)
    if r.geoclip_latitude and 33 <= r.geoclip_latitude <= 43 and 124 <= r.geoclip_longitude <= 132:
        geo_region = _coords_to_korea_region(r.geoclip_latitude, r.geoclip_longitude)
        if geo_region != "한국":
            region_votes[geo_region] = region_votes.get(geo_region, 0) + r.geoclip_score * 0.2

    if not region_votes:
        return "한국", 0.0
    best = max(region_votes, key=lambda k: region_votes[k])
    return best, round(region_votes[best], 4)


# ── 좌표 → 한국 지역명 ───────────────────────────────────────────────────

def _coords_to_korea_region(lat: float, lon: float) -> str:
    """좌표 → 한국 세부 지역명"""
    if not (33 <= lat <= 43 and 124 <= lon <= 132):
        return "한국"  # 한국 외 좌표도 한국으로 처리 (전용 서비스)

    # 서울 (37.4~37.7, 126.7~127.2)
    if 37.4 <= lat <= 37.72 and 126.7 <= lon <= 127.2:
        # 서울 주요 구
        if 37.47 <= lat <= 37.53 and 127.02 <= lon <= 127.09: return "서울 강남구"
        if 37.46 <= lat <= 37.52 and 126.97 <= lon <= 127.04: return "서울 서초구"
        if 37.54 <= lat <= 37.60 and 126.90 <= lon <= 126.97: return "서울 마포구"
        if 37.57 <= lat <= 37.63 and 126.97 <= lon <= 127.01: return "서울 종로구"
        if 37.55 <= lat <= 37.59 and 126.96 <= lon <= 127.01: return "서울 중구"
        if 37.52 <= lat <= 37.56 and 126.96 <= lon <= 127.00: return "서울 용산구"
        if 37.51 <= lat <= 37.56 and 126.90 <= lon <= 126.97: return "서울 영등포구"
        if 37.49 <= lat <= 37.54 and 127.09 <= lon <= 127.18: return "서울 송파구"
        if 37.54 <= lat <= 37.58 and 127.04 <= lon <= 127.09: return "서울 성동구"
        if 37.53 <= lat <= 37.57 and 127.00 <= lon <= 127.06: return "서울 동대문구"
        if 37.59 <= lat <= 37.65 and 127.02 <= lon <= 127.10: return "서울 노원구"
        if 37.59 <= lat <= 37.65 and 126.88 <= lon <= 126.95: return "서울 은평구"
        if 37.47 <= lat <= 37.52 and 126.85 <= lon <= 126.93: return "서울 관악구"
        return "서울"

    # 경기도 주요 도시
    if 37.25 <= lat <= 37.45 and 126.95 <= lon <= 127.20: return "경기 성남/수원"
    if 37.60 <= lat <= 37.80 and 126.70 <= lon <= 126.95: return "경기 고양/파주"
    if 37.30 <= lat <= 37.45 and 127.00 <= lon <= 127.20: return "경기 수원"
    if 37.20 <= lat <= 37.30 and 127.00 <= lon <= 127.20: return "경기 화성/오산"
    if 37.40 <= lat <= 37.55 and 127.15 <= lon <= 127.35: return "경기 하남/남양주"

    # 인천 (37.3~37.6, 126.4~126.7)
    if 37.30 <= lat <= 37.60 and 126.40 <= lon <= 126.80:
        if 37.35 <= lat <= 37.45 and 126.60 <= lon <= 126.73: return "인천 송도"
        return "인천"

    # 부산 (35.0~35.3, 128.9~129.3)
    if 35.00 <= lat <= 35.30 and 128.90 <= lon <= 129.30:
        if 35.15 <= lat <= 35.19 and 129.15 <= lon <= 129.23: return "부산 해운대"
        if 35.14 <= lat <= 35.17 and 129.11 <= lon <= 129.15: return "부산 광안리"
        if 35.09 <= lat <= 35.12 and 129.02 <= lon <= 129.06: return "부산 남포동"
        if 35.15 <= lat <= 35.19 and 129.05 <= lon <= 129.09: return "부산 서면"
        return "부산"

    # 대구 (35.8~36.0, 128.5~128.7)
    if 35.80 <= lat <= 36.00 and 128.50 <= lon <= 128.70: return "대구"
    # 광주 (35.1~35.2, 126.8~127.0)
    if 35.10 <= lat <= 35.25 and 126.80 <= lon <= 127.00: return "광주"
    # 대전 (36.3~36.4, 127.3~127.5)
    if 36.20 <= lat <= 36.50 and 127.30 <= lon <= 127.50: return "대전"
    # 울산 (35.5~35.7, 129.2~129.4)
    if 35.50 <= lat <= 35.70 and 129.20 <= lon <= 129.40: return "울산"
    # 제주 (33.3~33.6, 126.3~126.7)
    if 33.30 <= lat <= 33.60 and 126.30 <= lon <= 126.70: return "제주"

    return "한국"

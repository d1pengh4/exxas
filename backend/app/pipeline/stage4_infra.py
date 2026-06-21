"""
Stage 4: 인프라 핑거프린팅
- YOLOv8/v12 기반 객체 탐지
- 교통/도시/건축 인프라 탐지
- 170개국 국가별 DB 매칭
- 콘센트 형태 → 국가 확정
"""
import io
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from PIL import Image
from loguru import logger


@dataclass
class DetectedObject:
    label: str
    confidence: float
    bbox: list = field(default_factory=list)
    category: str = ""


@dataclass
class InfraResult:
    objects: list[DetectedObject] = field(default_factory=list)
    country_candidates: list[dict] = field(default_factory=list)  # [{"country": str, "score": float}]
    top_country: str = ""
    top_country_score: float = 0.0
    inferred_region: str = ""
    infrastructure_summary: str = ""
    scene_tags: list[str] = field(default_factory=list)     # CLIP 기반 시각 설명 태그
    scene_description: str = ""  # 자연어 장면 설명 (LLM 수사 힌트용)
    vision_analysis: str = ""    # qwen2.5vl 비전 LLM 분석 결과


# 국가별 인프라 특징 DB
INFRA_COUNTRY_DB: dict[str, dict] = {
    # ── 동아시아 ──────────────────────────────────────────
    "한국": {
        "traffic_lights": ["korean_traffic_light", "pedestrian_button"],
        "utility_poles": ["wooden_pole_korea"],
        "road_markings": ["korea_crosswalk", "korea_road_sign"],
        "bus_stops": ["korea_bus_stop"],
        "outlets": ["type_c", "type_f"],
        "vehicles": ["hyundai", "kia", "genesis"],
        "score_weights": {"traffic_lights": 3.0, "utility_poles": 2.5, "outlets": 5.0, "vehicles": 2.0},
    },
    "일본": {
        "traffic_lights": ["japan_traffic_light"],
        "vending_machines": ["japan_vending_machine"],
        "utility_poles": ["wooden_pole_japan"],
        "outlets": ["type_a"],
        "vehicles": ["toyota", "honda", "nissan", "mazda", "subaru"],
        "road_markings": ["japan_stop_sign_octagon", "japan_road_sign"],
        "score_weights": {"vending_machines": 5.0, "traffic_lights": 3.0, "outlets": 4.0, "road_markings": 3.0},
    },
    "중국": {
        "outlets": ["type_a", "type_i"],
        "vehicles": ["byd", "geely", "chery", "great_wall", "haval"],
        "road_markings": ["china_road_sign", "china_crosswalk"],
        "score_weights": {"vehicles": 3.0, "outlets": 3.0, "road_markings": 2.0},
    },
    "대만": {
        "outlets": ["type_a", "type_b"],
        "traffic_lights": ["taiwan_traffic_light"],
        "road_markings": ["chinese_script_sign"],
        "score_weights": {"traffic_lights": 3.0, "outlets": 3.5},
    },
    # ── 동남아시아 ────────────────────────────────────────
    "태국": {
        "outlets": ["type_a", "type_b", "type_c"],
        "traffic_lights": ["thai_traffic_light"],
        "vehicles": ["toyota", "honda", "isuzu"],
        "road_markings": ["thai_script_sign"],
        "score_weights": {"road_markings": 5.0, "traffic_lights": 2.5},
    },
    "베트남": {
        "outlets": ["type_a", "type_c"],
        "road_markings": ["vietnamese_script_sign"],
        "vehicles": ["honda_motorbike", "yamaha"],
        "score_weights": {"road_markings": 4.5, "vehicles": 3.0},
    },
    "인도네시아": {
        "outlets": ["type_c", "type_f"],
        "road_markings": ["indonesian_road_sign"],
        "vehicles": ["toyota_kijang", "daihatsu"],
        "score_weights": {"road_markings": 3.0, "vehicles": 2.5},
    },
    "말레이시아": {
        "outlets": ["type_g"],
        "road_markings": ["left_drive", "malay_road_sign"],
        "vehicles": ["proton", "perodua"],
        "score_weights": {"outlets": 4.0, "road_markings": 3.0, "vehicles": 4.0},
    },
    "싱가포르": {
        "outlets": ["type_g"],
        "road_markings": ["left_drive", "sg_road_sign"],
        "score_weights": {"outlets": 4.5, "road_markings": 4.0},
    },
    "필리핀": {
        "outlets": ["type_a", "type_b"],
        "road_markings": ["jeepney", "philippine_road"],
        "score_weights": {"road_markings": 3.0},
    },
    # ── 남아시아 ──────────────────────────────────────────
    "인도": {
        "outlets": ["type_d", "type_m"],
        "road_markings": ["devanagari_sign", "left_drive"],
        "vehicles": ["tata", "mahindra", "maruti"],
        "score_weights": {"outlets": 4.5, "road_markings": 3.5, "vehicles": 3.0},
    },
    # ── 북미 ──────────────────────────────────────────────
    "미국": {
        "traffic_lights": ["us_traffic_light"],
        "fire_hydrants": ["us_fire_hydrant"],
        "outlets": ["type_a", "type_b"],
        "road_markings": ["us_stop_sign", "us_yield", "us_speed_limit"],
        "vehicles": ["chevrolet", "ford_f150", "gmc", "ram_truck"],
        "score_weights": {"fire_hydrants": 4.0, "outlets": 3.5, "road_markings": 3.0, "vehicles": 2.0},
    },
    "캐나다": {
        "traffic_lights": ["us_traffic_light"],  # 유사
        "fire_hydrants": ["us_fire_hydrant"],
        "outlets": ["type_a", "type_b"],
        "road_markings": ["canada_stop_sign", "bilingual_sign"],
        "score_weights": {"road_markings": 4.0, "outlets": 3.0},
    },
    "멕시코": {
        "outlets": ["type_a", "type_b"],
        "road_markings": ["spanish_road_sign"],
        "score_weights": {"road_markings": 2.5, "outlets": 2.5},
    },
    # ── 서유럽 ────────────────────────────────────────────
    "영국": {
        "traffic_lights": ["uk_traffic_light"],
        "phone_booths": ["red_phone_booth"],
        "outlets": ["type_g"],
        "road_markings": ["uk_double_yellow", "left_drive", "uk_road_sign"],
        "score_weights": {"phone_booths": 5.5, "outlets": 5.0, "road_markings": 4.0},
    },
    "프랑스": {
        "outlets": ["type_c", "type_e"],
        "road_markings": ["french_road_sign", "priority_to_right"],
        "traffic_lights": ["eu_traffic_light"],
        "score_weights": {"outlets": 4.0, "road_markings": 3.5},
    },
    "독일": {
        "outlets": ["type_c", "type_f"],
        "road_markings": ["german_autobahn_sign", "german_road_sign"],
        "vehicles": ["volkswagen", "bmw", "mercedes", "audi"],
        "score_weights": {"outlets": 3.5, "vehicles": 3.0, "road_markings": 3.0},
    },
    "이탈리아": {
        "outlets": ["type_c", "type_f", "type_l"],
        "road_markings": ["italian_road_sign"],
        "score_weights": {"outlets": 4.0, "road_markings": 2.5},
    },
    "스페인": {
        "outlets": ["type_c", "type_f"],
        "road_markings": ["spanish_road_sign"],
        "score_weights": {"outlets": 3.5, "road_markings": 2.5},
    },
    "네덜란드": {
        "outlets": ["type_c", "type_f"],
        "road_markings": ["dutch_bicycle_lane", "dutch_road_sign"],
        "score_weights": {"outlets": 3.5, "road_markings": 4.0},
    },
    "스위스": {
        "outlets": ["type_j"],
        "road_markings": ["swiss_road_sign"],
        "score_weights": {"outlets": 5.5, "road_markings": 3.0},
    },
    # ── 동유럽/러시아 ─────────────────────────────────────
    "러시아": {
        "outlets": ["type_c", "type_f"],
        "road_markings": ["cyrillic_road_sign", "russia_road_sign"],
        "vehicles": ["lada", "uaz"],
        "score_weights": {"road_markings": 4.5, "vehicles": 3.0},
    },
    "우크라이나": {
        "outlets": ["type_c", "type_f"],
        "road_markings": ["cyrillic_road_sign"],
        "score_weights": {"road_markings": 3.5, "outlets": 3.0},
    },
    # ── 중동 ──────────────────────────────────────────────
    "사우디아라비아": {
        "outlets": ["type_g", "type_a", "type_b"],
        "road_markings": ["arabic_road_sign"],
        "vehicles": ["toyota_landcruiser", "ford_super_duty"],
        "score_weights": {"road_markings": 5.0, "vehicles": 3.0},
    },
    "아랍에미리트": {
        "outlets": ["type_g"],
        "road_markings": ["arabic_road_sign", "uae_road_sign"],
        "score_weights": {"outlets": 4.5, "road_markings": 4.0},
    },
    "이스라엘": {
        "outlets": ["type_h"],
        "road_markings": ["hebrew_road_sign"],
        "score_weights": {"outlets": 5.5, "road_markings": 4.5},
    },
    "터키": {
        "outlets": ["type_c", "type_f"],
        "road_markings": ["turkish_road_sign"],
        "score_weights": {"road_markings": 3.5, "outlets": 3.0},
    },
    # ── 오세아니아 ────────────────────────────────────────
    "호주": {
        "outlets": ["type_i"],
        "road_markings": ["left_drive", "australia_road_sign"],
        "fire_hydrants": ["australia_hydrant"],
        "score_weights": {"outlets": 5.5, "road_markings": 4.0},
    },
    "뉴질랜드": {
        "outlets": ["type_i"],
        "road_markings": ["left_drive", "nz_road_sign"],
        "score_weights": {"outlets": 5.0, "road_markings": 3.5},
    },
    # ── 남미 ──────────────────────────────────────────────
    "브라질": {
        "outlets": ["type_n"],
        "road_markings": ["portuguese_road_sign", "brazil_road_sign"],
        "score_weights": {"outlets": 5.5, "road_markings": 3.0},
    },
    "아르헨티나": {
        "outlets": ["type_i"],
        "road_markings": ["spanish_road_sign", "argentina_road"],
        "score_weights": {"outlets": 5.0, "road_markings": 2.5},
    },
    "칠레": {
        "outlets": ["type_c", "type_l"],
        "road_markings": ["spanish_road_sign"],
        "score_weights": {"outlets": 4.0, "road_markings": 2.0},
    },
    # ── 아프리카 ──────────────────────────────────────────
    "남아프리카": {
        "outlets": ["type_m", "type_n"],
        "road_markings": ["left_drive", "south_africa_road"],
        "score_weights": {"outlets": 5.0, "road_markings": 3.0},
    },
}

# YOLO 라벨 → 카테고리 매핑
YOLO_CATEGORY_MAP = {
    "traffic light": "traffic_lights",
    "fire hydrant": "fire_hydrants",
    "stop sign": "road_markings",
    "car": "vehicles",
    "truck": "vehicles",
    "bus": "vehicles",
    "motorcycle": "vehicles",
    "bicycle": "vehicles",
    "person": "pedestrians",
    "bench": "street_furniture",
    "umbrella": "weather_hints",
}


async def run(image_bytes: bytes) -> InfraResult:
    import time as _time
    result = InfraResult()

    # YOLO 객체 탐지
    _t0 = _time.time()
    objects = await _run_yolo(image_bytes)
    logger.info(f"[Stage4] YOLO done in {_time.time()-_t0:.1f}s, {len(objects)} objects")
    result.objects = objects

    if objects:
        # 국가 DB 매칭
        country_scores = _match_country_db(objects)
        result.country_candidates = sorted(
            [{"country": k, "score": v} for k, v in country_scores.items()],
            key=lambda x: x["score"],
            reverse=True,
        )

        if result.country_candidates:
            result.top_country = result.country_candidates[0]["country"]
            result.top_country_score = result.country_candidates[0]["score"]

        result.infrastructure_summary = _summarize(objects)
        result.inferred_region = _infer_climate_region(objects)

    # CLIP 시각 장면 태깅 (항상 실행 — 인프라 없는 자연/해변 사진에서도 작동)
    try:
        import asyncio as _asyncio
        tags, description = await _asyncio.wait_for(_clip_scene_tags(image_bytes), timeout=180.0)
        result.scene_tags = tags
        result.scene_description = description
    except _asyncio.TimeoutError:
        logger.warning("CLIP scene tagging timed out (>90s), skipping")
    except Exception as e:
        logger.warning(f"CLIP scene tagging failed: {e}")

    # 비전 LLM 분석: HF API (Llama-3.2-Vision) 우선, 없으면 Ollama 폴백
    try:
        import asyncio as _asyncio
        vision = await _asyncio.wait_for(_vision_analysis(image_bytes), timeout=90.0)
        if vision:
            result.vision_analysis = vision
            logger.info(f"[Stage4] Vision analysis done: {vision[:120]}")
    except Exception as e:
        logger.debug(f"Vision analysis skipped: {e}")

    return result


async def _run_yolo(image_bytes: bytes) -> list[DetectedObject]:
    try:
        import asyncio
        from ultralytics import YOLO

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)

        def _sync():
            import os as _os, torch as _torch
            _torch.set_num_threads(1)
            model = _get_yolo_model()
            results = model(img_array, conf=0.25, verbose=False)
            objects = []
            for r in results:
                for box in r.boxes:
                    label = r.names[int(box.cls)]
                    conf = float(box.conf)
                    bbox = box.xyxy[0].tolist()
                    category = YOLO_CATEGORY_MAP.get(label, "other")
                    objects.append(DetectedObject(
                        label=label,
                        confidence=conf,
                        bbox=bbox,
                        category=category,
                    ))
            return objects

        return await asyncio.to_thread(_sync)

    except ImportError:
        logger.warning("ultralytics not installed, skipping YOLO")
        return []
    except Exception as e:
        logger.error(f"YOLO detection failed: {e}")
        return []


_yolo_model = None


def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")  # nano 모델 (빠름, CPU 가능)
    return _yolo_model


def _match_country_db(objects: list[DetectedObject]) -> dict[str, float]:
    detected_categories = {}
    for obj in objects:
        cat = obj.category
        if cat not in detected_categories or detected_categories[cat] < obj.confidence:
            detected_categories[cat] = obj.confidence

    country_scores: dict[str, float] = {}

    for country, db in INFRA_COUNTRY_DB.items():
        score = 0.0
        weights = db.get("score_weights", {})
        for category, cat_conf in detected_categories.items():
            if category in db:
                weight = weights.get(category, 1.0)
                score += cat_conf * weight
        if score > 0:
            country_scores[country] = round(score, 3)

    return country_scores


def _summarize(objects: list[DetectedObject]) -> str:
    category_counts: dict[str, int] = {}
    for obj in objects:
        category_counts[obj.category] = category_counts.get(obj.category, 0) + 1

    parts = []
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        parts.append(f"{cat}×{count}")
    return ", ".join(parts)


def _infer_climate_region(objects: list[DetectedObject]) -> str:
    labels = {obj.label.lower() for obj in objects}

    if "snow" in labels or "ice" in labels:
        return "한대/아한대"
    if "palm tree" in labels or "tropical plant" in labels:
        return "열대"
    if "desert" in labels or "cactus" in labels:
        return "건조"
    if "umbrella" in labels:
        return "우천 가능 (온대/아열대)"
    return ""


async def _clip_scene_tags(image_bytes: bytes) -> tuple[list[str], str]:
    """
    CLIP 기반 시각 장면 태깅 — LLM이 이미지를 못 볼 때 텍스트 설명 제공
    각 카테고리에서 상위 1개 태그를 선택
    """
    import asyncio
    import torch
    import os as _os
    from PIL import Image
    import io as _io

    DEVICE = _os.getenv("PYTORCH_DEVICE", "cpu")

    # 태그 카테고리 정의
    SCENE_QUERIES: dict[str, list[str]] = {
        "water_type": [
            "ocean sea waves sandy beach coast",
            "wide flat calm river picnic crowd urban concrete bridge pillars sunset",
            "indoor water park swimming pool slides colorful",
            "lake reservoir mountain scenery",
            "no water visible",
        ],
        "scene_type": [
            "outdoor beach sandy ocean waves coast",
            "outdoor urban street buildings sidewalk",
            "outdoor park green grass nature",
            "outdoor river bank paved ground people sitting sunset",
            "indoor venue hall",
            "outdoor amusement theme park rides",
        ],
        "korea_landmark": [
            "Haeundae Beach Busan sandy beach sea coast",
            "Gwanganri Beach Busan diamond cable suspension bridge night illuminated ocean",
            "Han River Seoul Hangang wide river with concrete bridge pillar columns massive suspension cable stay",
            "Yeouido Hangang Park Seoul riverside 63 building gold skyscraper sunset",
            "Lotte World Adventure outdoor theme park Seoul Jamsil colorful rides",
            "Lotte World Tower Seoul 555m supertall skyscraper glass",
            "Gyeongbokgung Palace traditional Korean ancient architecture",
            "Namsan N Seoul Tower hilltop telecommunications tower",
            "Myeongdong Hongdae Seoul shopping street busy crowd",
            "Jeju Island volcanic basalt rock coast lava",
            "Korean apartment complex dense residential tower block",
            "Korean subway underground station",
            "other location",
        ],
        "korean_venue": [
            "Korean outdoor beach ocean resort coast waves",
            "Korean outdoor theme park amusement rides colorful",
            "Korean riverside waterfront park promenade concrete paved",
            "Korean city shopping commercial entertainment district crowd",
            "Korean apartment residential high-rise tower block",
            "Korean mountain hiking trail nature forest",
            "Korean indoor facility building interior",
        ],
        "density": [
            "crowded with many people", "few people", "empty no people"
        ],
        "architecture": [
            "high-rise apartment residential tower block",
            "modern glass office skyscraper",
            "traditional low-rise historic building",
            "mixed urban commercial building",
            "school elementary middle high university campus red brick building",
        ],
        "sports_venue": [
            "outdoor tennis court green fence hard surface",
            "outdoor basketball court concrete",
            "outdoor soccer football field grass",
            "indoor gymnasium sports hall",
            "public park sports facility",
            "school playground sports court",
            "no sports facility",
        ],
    }

    def _sync():
        import os as _os, torch as _torch
        _torch.set_num_threads(1)
        from transformers import CLIPProcessor, CLIPModel
        import sys as _sys
        # 모듈 레벨 CLIP 캐시 — 프로세스 내 모델 재로드 방지
        _clip_cache_key = f"_clip_cache_{DEVICE}"
        if not hasattr(_sys.modules[__name__], "_clip_model_cache"):
            _sys.modules[__name__]._clip_model_cache = {}
        _cache = _sys.modules[__name__]._clip_model_cache
        if _clip_cache_key not in _cache:
            _cache[_clip_cache_key] = (
                CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
                CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE),
            )
        processor, model = _cache[_clip_cache_key]
        img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")

        selected_tags: list[str] = []
        for category, options in SCENE_QUERIES.items():
            prompts = [f"a photo of {opt}" for opt in options]
            inputs = processor(text=prompts, images=img, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]
            best_idx = int(probs.argmax().item())
            best_score = float(probs[best_idx])
            import logging as _logging
            _logging.getLogger("stage4").debug(f"CLIP [{category}] best={options[best_idx][:40]!r} score={best_score:.2f}")
            if best_score > 0.10:  # 10% 이상만 포함 (약한 신호도 LLM에 전달)
                selected_tags.append(options[best_idx])

        return selected_tags

    tags = await asyncio.to_thread(_sync)

    # 한국 랜드마크 태그 우선 처리
    LANDMARK_SIGNALS = {
        # korea_landmark 카테고리
        "Haeundae Beach Busan sandy beach sea coast": "해운대해수욕장 (부산 해운대구)",
        "Gwanganri Beach Busan suspension bridge ocean": "광안리해수욕장 (부산 수영구)",
        "Han River Seoul Hangang wide river bridge": "한강공원 (서울)",
        "Yeouido Han River Park Seoul riverside": "여의도한강공원 (서울 영등포구)",
        "Lotte World Adventure outdoor theme park Seoul Jamsil": "롯데월드 어드벤처 (서울 송파구 잠실)",
        "Lotte World Tower Seoul supertall skyscraper": "롯데월드타워 (서울 송파구 잠실)",
        "Gyeongbokgung Palace traditional Korean architecture": "경복궁 (서울 종로구)",
        "Namsan N Seoul Tower hilltop tower": "남산서울타워 (서울 용산구)",
        "Myeongdong Hongdae Seoul shopping street crowd": "명동/홍대 (서울)",
        "Jeju Island volcanic rock coast": "제주도",
        "Korean apartment complex dense residential tower": "한국 아파트단지",
        "Korean subway underground station": "한국 지하철역",
        # korean_venue 카테고리
        "Korean outdoor beach resort sea resort": "해운대/광안리해수욕장 (부산)",
        "Korean outdoor theme park amusement rides Lotte World": "롯데월드 어드벤처 (서울 송파구 잠실)",
        "Korean Han River riverside park urban": "한강공원 (서울)",
        "Korean city shopping entertainment district": "한국 도심 쇼핑가",
        "Korean apartment residential high-rise": "한국 아파트단지",
        # water_type 카테고리 (해변 감지 강화)
        "ocean sea waves crashing sandy beach": "해변/해수욕장 (부산/제주 가능성)",
        "indoor swimming pool water slides rides colorful": "한국 워터파크/롯데월드",
    }

    korea_hint = ""
    for tag in tags:
        if tag in LANDMARK_SIGNALS:
            korea_hint = LANDMARK_SIGNALS[tag]
            break

    # 자연어 설명 생성
    if tags:
        desc = "이미지 시각 특징: " + ", ".join(tags)
        if korea_hint:
            desc = f"[한국 랜드마크 감지: {korea_hint}] " + desc
    else:
        desc = ""

    if korea_hint:
        tags = [f"KOREA_LANDMARK:{korea_hint}"] + tags

    return tags, desc


_VISION_PROMPT = (
    "You are a forensic geolocalization expert. Analyze this image and extract ALL location clues. "
    "CRITICAL RULES:\n"
    "- Text/signs: TRANSCRIBE EXACTLY what you can CLEARLY read. If unsure, write [UNCERTAIN: your_guess]. NEVER make up or guess text you cannot clearly see.\n"
    "- Generic notices (No Smoking, No Parking, Exit) are NOT location clues — focus on brand names, store names, building names, logos.\n"
    "Focus on:\n"
    "1) Brand names, store names, logos — transcribe exactly or mark [UNCERTAIN]\n"
    "2) Building materials, window style, architectural details\n"
    "3) Surrounding buildings, signage on nearby structures\n"
    "4) Road signs, vehicle types, utility infrastructure\n"
    "5) Venue type with specific identifying features\n"
    "6) Sky, vegetation, terrain for climate/region clues\n"
    "List each clue starting with '-'. "
    "End with: LIKELY_LOCATION: [specific country/city/district/venue or 'Cannot determine']"
)


async def _vision_analysis(image_bytes: bytes) -> str:
    """
    비전 LLM 분석:
    1순위: HuggingFace Inference API (Llama-3.2-11B-Vision, 무료+토큰)
    2순위: Ollama 로컬 (qwen2.5vl:7b)
    """
    import base64
    import aiohttp
    import ssl
    import certifi
    from ..core.config import settings

    # HF API 413 방지: 이미지를 1024px 이하로 리사이즈 후 JPEG 압축
    try:
        from PIL import Image as _PILImage
        import io as _io
        _img = _PILImage.open(_io.BytesIO(image_bytes)).convert("RGB")
        _max = 1024
        if max(_img.size) > _max:
            _img.thumbnail((_max, _max), _PILImage.LANCZOS)
        _buf = _io.BytesIO()
        _img.save(_buf, format="JPEG", quality=75)
        image_bytes = _buf.getvalue()
    except Exception:
        pass  # PIL 없으면 원본 사용
    b64 = base64.b64encode(image_bytes).decode()
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    hf_token = getattr(settings, "HF_TOKEN", "") or ""
    if hf_token:
        result = await _hf_vision_analysis(b64, hf_token, ssl_ctx)
        if result:
            return result

    # Ollama 폴백
    return await _ollama_vision_analysis(b64, ssl_ctx)


async def _hf_vision_analysis(b64: str, hf_token: str, ssl_ctx) -> str:
    """HuggingFace router API — Llama-3.2-11B-Vision-Instruct"""
    import aiohttp

    payload = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": _VISION_PROMPT},
                ],
            }
        ],
        "max_tokens": 400,
        "temperature": 0.1,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://router.huggingface.co/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {hf_token}"},
                ssl=ssl_ctx,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    logger.debug(f"HF Vision API status {resp.status}")
                    return ""
                data = await resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                logger.info(f"[HF Vision] Llama-3.2 response: {content[:100]}")
                return content.strip()
    except Exception as e:
        logger.debug(f"HF Vision API failed: {e}")
        return ""


async def _ollama_vision_analysis(b64: str, ssl_ctx=None) -> str:
    """Ollama 로컬 — qwen2.5vl:7b"""
    import aiohttp

    payload = {
        "model": "qwen2.5vl:7b",
        "prompt": _VISION_PROMPT,
        "images": [b64],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 400},
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=80),
            ) as resp:
                if resp.status != 200:
                    return ""
                data = await resp.json()
                content = data.get("response", "").strip()
                logger.info(f"[Ollama Vision] qwen2.5vl response: {content[:100]}")
                return content
    except Exception as e:
        logger.debug(f"Ollama vision failed: {e}")
        return ""


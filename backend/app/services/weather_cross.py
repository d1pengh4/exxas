"""
B. 날씨/계절 시각 교차검증
- CLIP으로 이미지에서 계절/날씨/복장 단서 감지
- Open-Meteo 과거 날씨 API로 지역 범위 좁히기
- 시간대 추정 (아침/낮/저녁/야간) 조명 분석
"""
import io
import asyncio
import datetime
import httpx
import numpy as np
from loguru import logger
from typing import Optional
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# CLIP 기반 시각 단서 분류기
# ─────────────────────────────────────────────────────────────────────────────
_SEASON_LABELS = [
    "cherry blossom spring flowers",          # 봄 (벚꽃)
    "green leaves summer hot",                # 여름
    "autumn fall foliage red orange leaves",  # 가을 (단풍)
    "winter snow cold bare trees",            # 겨울 (눈)
    "dry brown grass",                        # 건기
]

_WEATHER_LABELS = [
    "clear sunny blue sky",
    "cloudy overcast gray sky",
    "rainy wet umbrella",
    "snowing heavy snow",
    "foggy misty",
    "humid hazy",
]

_TIME_LABELS = [
    "sunrise dawn early morning golden hour",
    "daytime bright sunlight",
    "sunset dusk orange sky",
    "night dark artificial lights",
]

_CLOTHING_LABELS = [
    "people wearing thick winter coats heavy jackets",
    "people wearing light clothes t-shirts summer",
    "people wearing jackets spring autumn",
    "people wearing raincoats umbrellas",
]

# 계절명 (한국어)
_SEASON_NAMES = ["봄(벚꽃)", "여름", "가을(단풍)", "겨울(눈)", "건기"]
_WEATHER_NAMES = ["맑음", "흐림", "비", "눈", "안개", "황사/미세먼지"]
_TIME_NAMES = ["일출/새벽", "낮", "일몰/저녁", "야간"]
_CLOTHING_NAMES = ["두꺼운 겨울 외투", "반팔/여름옷", "자켓/봄가을", "우비/우산"]


def analyze_visual_climate(image_bytes: bytes) -> dict:
    """
    CLIP을 이용해 이미지에서 계절/날씨/시간대 시각 단서 추출
    (동기 함수 — asyncio.to_thread 래핑 필요)
    """
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((336, 336))

        model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)
        model.eval()

        results = {}
        for category, labels, names in [
            ("season", _SEASON_LABELS, _SEASON_NAMES),
            ("weather", _WEATHER_LABELS, _WEATHER_NAMES),
            ("time_of_day", _TIME_LABELS, _TIME_NAMES),
            ("clothing", _CLOTHING_LABELS, _CLOTHING_NAMES),
        ]:
            inputs = processor(text=labels, images=img, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(**inputs).logits_per_image[0]
            probs = torch.softmax(logits, dim=0).numpy()
            best_idx = int(probs.argmax())
            best_score = float(probs[best_idx])
            results[category] = {
                "name": names[best_idx],
                "confidence": round(best_score, 3),
                "all": {names[i]: round(float(p), 3) for i, p in enumerate(probs)},
            }

        # 조명 분석: 이미지 밝기/색온도
        arr = np.array(img)
        brightness = float(arr.mean() / 255.0)
        r_mean = float(arr[:, :, 0].mean())
        b_mean = float(arr[:, :, 2].mean())
        warm_cool = "따뜻한 빛 (황금시간대)" if r_mean > b_mean * 1.15 else (
            "차가운 빛 (흐리거나 그늘)" if b_mean > r_mean * 1.10 else "중성광"
        )

        results["brightness"] = round(brightness, 3)
        results["light_quality"] = warm_cool

        return results

    except Exception as e:
        logger.warning(f"[weather_cross] visual_climate 실패: {e}")
        return {}


async def analyze_visual_climate_async(image_bytes: bytes) -> dict:
    return await asyncio.to_thread(analyze_visual_climate, image_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# Open-Meteo 과거 날씨 역조회
# ─────────────────────────────────────────────────────────────────────────────
async def openmeteo_historical_match(
    lat: float,
    lon: float,
    season: str,
    weather: str,
) -> dict:
    """
    특정 좌표에서 주어진 계절/날씨 조건과 맞는 과거 날짜 범위 조회
    Open-Meteo free API 사용
    """
    # 계절 → 한국 기준 월 범위
    SEASON_MONTHS: dict[str, list[int]] = {
        "봄(벚꽃)": [3, 4, 5],
        "여름": [6, 7, 8],
        "가을(단풍)": [9, 10, 11],
        "겨울(눈)": [12, 1, 2],
        "건기": [10, 11, 12, 1, 2],  # 한국 기준
    }
    # 현재로부터 최근 3년 내 검색
    now = datetime.datetime.now()
    months = SEASON_MONTHS.get(season, [])
    if not months:
        return {"season_hint": season, "months": [], "note": "계절 매핑 없음"}

    try:
        # 현재 연도와 전년도 대상
        candidate_periods = []
        for year_offset in range(3):
            year = now.year - year_offset
            for month in months:
                m_year = year if month <= now.month or year < now.year else year - 1
                candidate_periods.append((m_year, month))

        if not candidate_periods:
            return {}

        year, month = candidate_periods[0]
        start_date = f"{year}-{month:02d}-01"
        last_day = 28 if month == 2 else (30 if month in (4, 6, 9, 11) else 31)
        end_date = f"{year}-{month:02d}-{last_day}"

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean,precipitation_sum,snowfall_sum",
            "timezone": "Asia/Seoul",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            data = resp.json()

        daily = data.get("daily", {})
        temps = daily.get("temperature_2m_mean", [])
        precip = daily.get("precipitation_sum", [])
        snow = daily.get("snowfall_sum", [])

        if not temps:
            return {"note": "Open-Meteo 데이터 없음"}

        avg_temp = round(sum(t for t in temps if t is not None) / max(len(temps), 1), 1)
        avg_precip = round(sum(p for p in precip if p is not None) / max(len(precip), 1), 1)
        snow_days = sum(1 for s in snow if s and s > 0.5) if snow else 0

        # 날씨 조건 매칭 점수
        match_score = 0.5  # 기본
        if weather == "눈" and snow_days > 3:
            match_score = 0.85
        elif weather == "비" and avg_precip > 3.0:
            match_score = 0.80
        elif weather == "맑음" and avg_precip < 1.0:
            match_score = 0.75
        elif weather in ("흐림", "안개") and avg_precip < 2.0:
            match_score = 0.65

        return {
            "lat": lat,
            "lon": lon,
            "period": f"{year}년 {month}월",
            "avg_temp_c": avg_temp,
            "avg_precip_mm": avg_precip,
            "snow_days": snow_days,
            "season": season,
            "weather_match_score": round(match_score, 3),
            "note": f"평균기온 {avg_temp}°C, 강수 {avg_precip}mm/일, 눈 {snow_days}일",
        }

    except Exception as e:
        logger.debug(f"[weather_cross] Open-Meteo 조회 실패: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 한국 지역별 계절 특성 DB (기후 단서로 지역 좁히기)
# ─────────────────────────────────────────────────────────────────────────────
KOREA_CLIMATE_HINTS: dict[str, dict] = {
    "봄(벚꽃)": {
        "jeju": {"city": "제주", "note": "전국 최초 벚꽃 (3월 하순)", "lat": 33.4996, "lon": 126.5312},
        "busan": {"city": "부산", "note": "남부 해안 벚꽃 (4월 초)", "lat": 35.1796, "lon": 129.0756},
        "seoul": {"city": "서울", "note": "여의도 벚꽃 (4월 초중순)", "lat": 37.5665, "lon": 126.9780},
    },
    "겨울(눈)": {
        "gangwon": {"city": "강원", "note": "대관령/평창 폭설 (12~3월)", "lat": 37.3622, "lon": 128.0350},
        "seoul": {"city": "서울", "note": "첫눈 (12월 초)", "lat": 37.5665, "lon": 126.9780},
        "jeju_snow": {"city": "제주 한라산", "note": "한라산 적설 (11~4월)", "lat": 33.3617, "lon": 126.5292},
    },
    "가을(단풍)": {
        "seoraksan": {"city": "강원 설악산", "note": "전국 첫 단풍 (9월 하순)", "lat": 38.1194, "lon": 128.4656},
        "naejangsan": {"city": "전북 내장산", "note": "단풍 명소 (10월)", "lat": 35.4873, "lon": 126.8875},
    },
}


async def weather_cross_check(
    image_bytes: bytes,
    candidate_lat: Optional[float] = None,
    candidate_lon: Optional[float] = None,
) -> dict:
    """
    종합 날씨/계절 교차검증
    1) 이미지에서 계절/날씨/시간대 추출
    2) Open-Meteo 역조회 (후보 좌표가 있을 때)
    3) 계절 특성으로 지역 힌트 제공
    """
    logger.debug("[weather_cross] 시각 기후 분석 시작")
    visual = await analyze_visual_climate_async(image_bytes)

    if not visual:
        return {"error": "visual_climate 분석 실패"}

    season = visual.get("season", {}).get("name", "")
    season_conf = visual.get("season", {}).get("confidence", 0.0)
    weather = visual.get("weather", {}).get("name", "")
    weather_conf = visual.get("weather", {}).get("confidence", 0.0)
    time_of_day = visual.get("time_of_day", {}).get("name", "")
    clothing = visual.get("clothing", {}).get("name", "")
    brightness = visual.get("brightness", 0.5)
    light_quality = visual.get("light_quality", "")

    result: dict = {
        "season": season,
        "season_confidence": season_conf,
        "weather": weather,
        "weather_confidence": weather_conf,
        "time_of_day": time_of_day,
        "clothing_hint": clothing,
        "brightness": brightness,
        "light_quality": light_quality,
    }

    # Open-Meteo 역조회 (후보 좌표 있을 때)
    if candidate_lat and candidate_lon and season_conf > 0.45:
        meteo = await openmeteo_historical_match(
            candidate_lat, candidate_lon, season, weather
        )
        result["meteo_match"] = meteo

    # 계절 기반 한국 지역 힌트
    if season and season_conf > 0.50:
        region_hints = KOREA_CLIMATE_HINTS.get(season, {})
        if region_hints:
            result["korea_region_hints"] = list(region_hints.values())

    # 온도 추정 (복장 기반)
    temp_estimate = ""
    if "겨울" in clothing:
        temp_estimate = "0°C 이하 ~ 10°C (겨울)"
    elif "반팔" in clothing or "여름" in clothing:
        temp_estimate = "25°C 이상 (여름)"
    elif "자켓" in clothing:
        temp_estimate = "10°C ~ 20°C (봄/가을)"
    elif "우비" in clothing or "우산" in clothing:
        temp_estimate = "강수 중 (온도 무관)"
    if temp_estimate:
        result["temperature_estimate"] = temp_estimate

    logger.debug(f"[weather_cross] 완료: {season}/{weather}/{time_of_day} → bright={brightness:.2f}")
    return result

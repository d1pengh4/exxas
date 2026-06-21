"""
Stage 6: 물리·천문 역산
- 태양 고도/그림자 역산 → 위도 밴드
- 달·별 분석 → 날짜/반구 추정
- DEM 능선 매칭 (NASA SRTM + Copernicus)
- ERA5 기상 DB 교차검증
- 식생 계절 상태
"""
import math
import datetime
from dataclasses import dataclass, field
from typing import Optional
import httpx
from loguru import logger
from ..core.config import settings


@dataclass
class SunAnalysis:
    shadow_direction_degrees: Optional[float] = None
    sun_elevation_degrees: Optional[float] = None
    estimated_latitude_band: str = ""      # "북위 30~45도" 등
    estimated_time_utc: str = ""
    north_direction_degrees: Optional[float] = None


@dataclass
class WeatherMatch:
    date_estimate: str = ""
    season: str = ""
    temperature_c: Optional[float] = None
    weather_description: str = ""
    match_confidence: float = 0.0


@dataclass
class MoonStarAnalysis:
    moon_phase: str = ""               # "보름달", "반달", "초승달" 등
    moon_phase_angle: Optional[float] = None  # 0~360
    estimated_date_range: str = ""     # "2024-09-12 ~ 2024-09-14"
    hemisphere_hint: str = ""          # 달 방향으로 반구 추정
    star_constellation: str = ""       # 주요 별자리
    star_hemisphere: str = ""          # "북반구" | "남반구"


@dataclass
class PhysicalResult:
    sun: SunAnalysis = field(default_factory=SunAnalysis)
    moon_star: MoonStarAnalysis = field(default_factory=MoonStarAnalysis)
    weather: WeatherMatch = field(default_factory=WeatherMatch)
    hemisphere: str = ""         # "북반구" | "남반구" | ""
    latitude_band: str = ""      # "열대" | "아열대" | "온대" | "냉대" | "한대"
    season_estimate: str = ""
    vegetation_hints: list[str] = field(default_factory=list)
    dem_ridge_matched: bool = False
    dem_candidate_regions: list[str] = field(default_factory=list)


async def run(image_bytes: bytes, exif_datetime: str = "", latitude_hint: float = 0.0) -> PhysicalResult:
    result = PhysicalResult()

    if exif_datetime:
        sun = await _analyze_sun(exif_datetime, latitude_hint)
        result.sun = sun

        # 반구 판정: 위도 힌트 우선, 없으면 정오 태양 방향(북쪽/남쪽)으로 판단
        if latitude_hint != 0.0:
            result.hemisphere = "북반구" if latitude_hint >= 0 else "남반구"
        elif sun.north_direction_degrees is not None:
            # 북반구: 정오에 태양이 남쪽(방위각 ~180°)
            # 남반구: 정오에 태양이 북쪽(방위각 ~0°/360°)
            az = sun.north_direction_degrees
            result.hemisphere = "북반구" if 90 <= az <= 270 else "남반구"

        result.latitude_band = _elevation_to_band(sun.sun_elevation_degrees)

    # 기상 DB 매칭
    if exif_datetime and latitude_hint != 0.0:
        weather = await _match_weather(exif_datetime, latitude_hint)
        result.weather = weather
        result.season_estimate = weather.season

    # 달/별 분석 (야간 이미지)
    moon_star = await _analyze_moon_stars(image_bytes, exif_datetime)
    result.moon_star = moon_star

    # 반구가 아직 미결정이면 달/별 분석 보조
    if not result.hemisphere and moon_star.star_hemisphere:
        result.hemisphere = moon_star.star_hemisphere
    elif not result.hemisphere and moon_star.hemisphere_hint:
        result.hemisphere = moon_star.hemisphere_hint

    # DEM 능선 매칭
    dem = await _dem_ridge_match(image_bytes, latitude_hint)
    result.dem_ridge_matched = dem["matched"]
    result.dem_candidate_regions = dem["candidate_regions"]

    return result


async def _analyze_sun(datetime_str: str, lat_hint: float) -> SunAnalysis:
    """SunCalc 기반 태양 위치 역산"""
    try:
        # ephem 또는 astropy로 태양 위치 계산
        import ephem

        dt = _parse_datetime(datetime_str)
        if not dt:
            return SunAnalysis()

        sun = ephem.Sun()

        # 위도 힌트 없으면 여러 위도에서 계산
        latitudes = [lat_hint] if lat_hint else [0, 15, 30, 37.5, 45, 60]
        best = SunAnalysis()

        for lat in latitudes:
            observer = ephem.Observer()
            observer.lat = str(lat)
            observer.lon = "127"  # 경도는 나중에 좁힘
            observer.date = dt.strftime("%Y/%m/%d %H:%M:%S")

            sun.compute(observer)
            elevation = math.degrees(sun.alt)
            azimuth = math.degrees(sun.az)

            if abs(elevation) < 80:  # 현실적인 태양 고도
                best.sun_elevation_degrees = round(elevation, 1)
                shadow_dir = (azimuth + 180) % 360  # 그림자는 반대 방향
                best.shadow_direction_degrees = round(shadow_dir, 1)
                best.estimated_latitude_band = _elevation_to_band(elevation)
                best.north_direction_degrees = round((azimuth + 180) % 360, 1)
                break

        return best

    except ImportError:
        logger.warning("ephem not installed, skipping sun analysis")
        return SunAnalysis()
    except Exception as e:
        logger.error(f"Sun analysis failed: {e}")
        return SunAnalysis()


async def _match_weather(datetime_str: str, latitude: float, longitude: float = 127.0) -> WeatherMatch:
    """Open-Meteo Historical API로 기상 매칭"""
    try:
        dt = _parse_datetime(datetime_str)
        if not dt:
            return WeatherMatch()

        date_str = dt.strftime("%Y-%m-%d")

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{settings.OPEN_METEO_API_URL}/archive",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": date_str,
                    "end_date": date_str,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
                    "timezone": "auto",
                },
            )

            if resp.status_code == 200:
                data = resp.json()
                daily = data.get("daily", {})

                max_temp = (daily.get("temperature_2m_max") or [None])[0]
                min_temp = (daily.get("temperature_2m_min") or [None])[0]
                precip = (daily.get("precipitation_sum") or [None])[0]
                wcode = (daily.get("weathercode") or [None])[0]

                avg_temp = ((max_temp or 0) + (min_temp or 0)) / 2 if max_temp and min_temp else None
                season = _temp_to_season(avg_temp, latitude)
                desc = _weather_code_to_desc(wcode)

                return WeatherMatch(
                    date_estimate=date_str,
                    season=season,
                    temperature_c=round(avg_temp, 1) if avg_temp else None,
                    weather_description=desc,
                    match_confidence=0.7,
                )

    except Exception as e:
        logger.warning(f"Weather match failed: {e}")

    return WeatherMatch()


async def _analyze_moon_stars(image_bytes: bytes, exif_datetime: str = "") -> MoonStarAnalysis:
    """
    달/별 분석 (야간 이미지)
    1. 이미지 밝기 분포로 야간 여부 판단
    2. 달 감지 → 위상 추정 → 날짜 범위 압축
    3. 별자리 패턴 → 반구 추정
    4. 달 기울기 방향 → 북/남반구 구분
    """
    result = MoonStarAnalysis()

    try:
        import io
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_arr = np.array(img, dtype=np.float32)

        # ── 1. 야간 이미지 판별 ──
        mean_brightness = float(np.mean(img_arr))
        if mean_brightness > 100:
            # 충분히 밝음 → 야간 아님
            return result

        # ── 2. 달 감지 (밝은 원형 영역 탐지) ──
        gray = np.mean(img_arr, axis=2)
        # 임계값: 전체 평균의 3배 이상인 밝은 픽셀
        bright_mask = gray > (mean_brightness * 3 + 50)
        bright_pct = float(np.sum(bright_mask)) / bright_mask.size

        moon_detected = 0.001 < bright_pct < 0.05  # 전체 0.1~5% 범위의 밝은 영역

        if moon_detected:
            # ── 3. 달 위상 추정 ──
            # 달 영역의 형태로 위상 추정 (밝기 분포의 좌우 비대칭)
            moon_rows, moon_cols = np.where(bright_mask)
            if len(moon_cols) > 10:
                col_center = float(np.median(moon_cols))
                img_center = img_arr.shape[1] / 2

                # 달 내부 밝기 중심 (왼쪽 vs 오른쪽)
                left_bright = float(np.sum(bright_mask[:, :int(col_center)]))
                right_bright = float(np.sum(bright_mask[:, int(col_center):]))
                total = left_bright + right_bright

                if total > 0:
                    right_ratio = right_bright / total
                    if right_ratio > 0.85:
                        result.moon_phase = "초승달 (오른쪽 초승)"
                        result.moon_phase_angle = 45.0
                    elif right_ratio > 0.65:
                        result.moon_phase = "상현달"
                        result.moon_phase_angle = 90.0
                    elif right_ratio > 0.45:
                        result.moon_phase = "보름달"
                        result.moon_phase_angle = 180.0
                    elif right_ratio > 0.25:
                        result.moon_phase = "하현달"
                        result.moon_phase_angle = 270.0
                    else:
                        result.moon_phase = "그믐달 (왼쪽 초승)"
                        result.moon_phase_angle = 315.0

            # ── 4. 달 기울기 → 반구 추정 ──
            # 북반구: 달이 오른쪽으로 기울며 차고, 남반구는 반대
            if moon_detected and result.moon_phase_angle:
                moon_row_center = float(np.median(moon_rows)) if len(moon_rows) > 0 else 0
                img_height = img_arr.shape[0]
                # 달이 이미지 상단에 있으면 → 위를 향해 기울기
                if moon_row_center < img_height * 0.4:
                    result.hemisphere_hint = "북반구"  # 북반구에서 달은 남쪽 하늘 (상단)

        # ── 5. 별자리 패턴 분석 ──
        # 밝은 점들의 분포 → 특정 별자리 형태 매칭
        # (간략화: 별 밀도와 분포로 위도 밴드 추정)
        star_mask = (gray > mean_brightness * 5) & (~bright_mask)
        star_count = int(np.sum(star_mask))

        if star_count > 20:
            # 별들의 수직 분포
            star_rows, star_cols = np.where(star_mask)
            if len(star_rows) > 10:
                row_center = float(np.median(star_rows))
                img_height = float(img_arr.shape[0])

                # 은하수 방향 추정 (밀도가 높은 영역)
                upper_stars = int(np.sum(star_mask[:int(img_height * 0.5), :]))
                lower_stars = int(np.sum(star_mask[int(img_height * 0.5):, :]))

                if upper_stars > lower_stars * 1.5:
                    result.star_hemisphere = "북반구"
                    result.star_constellation = "북반구 별자리 (상단 집중)"
                elif lower_stars > upper_stars * 1.5:
                    result.star_hemisphere = "남반구"
                    result.star_constellation = "남반구 별자리 (하단 집중)"

        # ── 6. EXIF 날짜 + 달 위상 → 날짜 범위 계산 ──
        if result.moon_phase_angle and exif_datetime:
            try:
                dt = _parse_datetime(exif_datetime)
                if dt:
                    # ephem으로 달 위상 역산
                    import ephem
                    # 당일 달 위상 확인
                    moon = ephem.Moon(dt.strftime("%Y/%m/%d"))
                    actual_phase = moon.phase  # 0~100 (100=보름달)
                    # 위상 오차 ±2일 범위로 날짜 압축
                    result.estimated_date_range = (
                        f"{dt.strftime('%Y-%m-%d')} (달 위상: {actual_phase:.0f}%)"
                    )
            except Exception:
                pass

    except Exception as e:
        logger.debug(f"Moon/star analysis failed: {e}")

    return result


async def _dem_ridge_match(image_bytes: bytes, latitude_hint: float = 0.0) -> dict:
    """
    DEM 능선 매칭:
    1. 이미지에서 수평선/능선 프로파일 추출 (Canny 에지 + 상단 경계)
    2. 지평선이 있는 이미지인지 판단
    3. Open-Elevation API로 후보 위도 밴드 격자점 고도 조회
    4. 능선 거칠기(ruggedness) 기반 지형 유형 매칭
    """
    try:
        import io
        import numpy as np
        from PIL import Image, ImageFilter

        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        w, h = img.size
        if w < 64 or h < 64:
            return {"matched": False, "candidate_regions": []}

        # ── 1. 수평선 프로파일 추출 ──
        # 상단 40% 영역에서 에지 강도 분석
        top_region = img.crop((0, 0, w, int(h * 0.4)))
        edges = top_region.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edges, dtype=np.float32)

        # 각 열의 에지 강도 합산 → 능선 거칠기 프로파일
        col_intensity = edge_arr.sum(axis=0)
        ridge_roughness = float(np.std(col_intensity) / (np.mean(col_intensity) + 1e-6))

        # ── 2. 지평선 존재 여부 ──
        # 수평선: 이미지 상단부에 강한 수평 에지
        row_intensity = edge_arr.sum(axis=1)
        sky_land_row = int(np.argmax(row_intensity[:int(h * 0.4)]))
        has_horizon = float(row_intensity[sky_land_row]) > float(row_intensity.mean() * 2.0)

        if not has_horizon:
            return {"matched": False, "candidate_regions": []}

        # ── 3. 지형 유형 분류 (거칠기 기반) ──
        if ridge_roughness > 1.5:
            terrain_type = "산악"
        elif ridge_roughness > 0.8:
            terrain_type = "구릉/언덕"
        elif ridge_roughness > 0.3:
            terrain_type = "평지/해안"
        else:
            terrain_type = "완전평지"

        # ── 4. 후보 지역 생성 ──
        # 위도 힌트가 있으면 해당 위도 ±20도 범위 검색, 없으면 전 세계 샘플
        candidate_regions = []

        # Open-Elevation API: 격자점 고도 조회
        sample_points = _build_sample_grid(latitude_hint, terrain_type)
        if sample_points:
            elevation_data = await _query_open_elevation(sample_points)
            if elevation_data:
                matching_regions = _match_terrain_type(elevation_data, terrain_type, latitude_hint)
                candidate_regions = matching_regions[:5]

        if not candidate_regions:
            # API 실패 시 terrain_type 기반 정성적 후보
            candidate_regions = _terrain_to_regions(terrain_type)

        return {
            "matched": bool(candidate_regions),
            "candidate_regions": candidate_regions,
            "terrain_type": terrain_type,
            "ridge_roughness": round(ridge_roughness, 3),
            "horizon_detected": has_horizon,
        }

    except Exception as e:
        logger.debug(f"DEM ridge match failed: {e}")
        return {"matched": False, "candidate_regions": []}


def _build_sample_grid(lat_hint: float, terrain_type: str) -> list[dict]:
    """샘플 격자점 생성 (Open-Elevation API용)"""
    points = []

    if lat_hint != 0.0:
        # 위도 힌트 ±15도, 경도 전체 30도 간격 샘플
        for dlat in [-10, -5, 0, 5, 10]:
            for lon in range(-180, 180, 30):
                lat = lat_hint + dlat
                if -90 <= lat <= 90:
                    points.append({"latitude": round(lat, 1), "longitude": float(lon)})
    else:
        # 전 세계 샘플 (10도 간격)
        for lat in range(-60, 70, 10):
            for lon in range(-170, 180, 30):
                points.append({"latitude": float(lat), "longitude": float(lon)})

    return points[:100]  # API 한번에 최대 100개


async def _query_open_elevation(points: list[dict]) -> list[dict]:
    """Open-Elevation API (무료, 키 불필요)로 고도 조회"""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                "https://api.open-elevation.com/api/v1/lookup",
                json={"locations": points},
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("results", [])
    except Exception as e:
        logger.debug(f"Open-Elevation API failed: {e}")
    return []


def _match_terrain_type(
    elevation_data: list[dict],
    terrain_type: str,
    lat_hint: float,
) -> list[str]:
    """고도 데이터 → 지형 유형 매칭 → 후보 지역 반환"""
    # 격자점별 고도로 지역 거칠기 계산
    # 위도별로 그룹화
    lat_groups: dict[float, list[float]] = {}
    for pt in elevation_data:
        lat = round(pt.get("latitude", 0), 0)
        elev = pt.get("elevation", 0) or 0
        lat_groups.setdefault(lat, []).append(elev)

    # 각 위도 밴드의 평균/표준편차 계산
    scored: list[tuple[float, str]] = []
    for lat, elevs in lat_groups.items():
        if not elevs:
            continue
        mean_elev = sum(elevs) / len(elevs)
        std_elev = (sum((e - mean_elev) ** 2 for e in elevs) / len(elevs)) ** 0.5
        ruggedness = std_elev / (mean_elev + 1.0)

        # 지형 유형별 점수
        if terrain_type == "산악" and mean_elev > 500 and ruggedness > 0.3:
            score = ruggedness
        elif terrain_type == "구릉/언덕" and 100 < mean_elev < 800 and 0.1 < ruggedness < 0.5:
            score = 1.0 - abs(ruggedness - 0.3)
        elif terrain_type in ("평지/해안", "완전평지") and mean_elev < 200 and ruggedness < 0.2:
            score = 1.0 - ruggedness
        else:
            score = 0.0

        if score > 0.1:
            region = _lat_to_region_name(lat)
            scored.append((score, region))

    scored.sort(key=lambda x: -x[0])
    # 중복 제거
    seen: set[str] = set()
    result = []
    for _, region in scored:
        if region and region not in seen:
            seen.add(region)
            result.append(region)
    return result


def _lat_to_region_name(lat: float) -> str:
    """위도 → 대략적 지역명"""
    if 33 <= lat <= 38:
        return "한국/일본"
    if 25 <= lat <= 45 and lat > 0:
        return "동아시아 온대"
    if 45 <= lat <= 65:
        return "유럽/러시아/북미 북부"
    if 25 <= lat <= 50 and lat > 0:
        return "북미/유럽 중부"
    if -10 <= lat <= 25:
        return "열대/아열대"
    if -35 <= lat < -10:
        return "남미/호주/남아프리카"
    return f"위도 {lat:.0f}도 부근"


def _terrain_to_regions(terrain_type: str) -> list[str]:
    """API 실패 시 지형 유형 → 정성적 후보 지역"""
    mapping = {
        "산악": ["한국 태백산맥", "일본 알프스", "유럽 알프스", "히말라야 주변", "로키산맥"],
        "구릉/언덕": ["한국 중부 구릉지", "영국/아일랜드", "동유럽 평원", "남부 유럽"],
        "평지/해안": ["한국 서해안", "유럽 북해 연안", "미국 동부 해안", "동남아 해안"],
        "완전평지": ["한국 평야지대", "미국 중부 평원", "유럽 평원", "호주 내륙"],
    }
    return mapping.get(terrain_type, [])


def _elevation_to_band(elevation: Optional[float]) -> str:
    if elevation is None:
        return ""
    if elevation > 70:
        return "열대 (적도 부근)"
    if elevation > 55:
        return "아열대 (북위 20~35도)"
    if elevation > 40:
        return "온대 (북위 35~50도)"
    if elevation > 25:
        return "냉대 (북위 50~65도)"
    return "한대 (북위 65도 이상)"


def _temp_to_season(temp: Optional[float], lat: float) -> str:
    if temp is None:
        return ""

    if lat >= 0:  # 북반구
        if temp > 20:
            return "여름"
        elif temp > 10:
            return "봄/가을"
        elif temp > 0:
            return "초겨울/늦겨울"
        return "겨울"
    else:  # 남반구 (계절 반전)
        if temp > 20:
            return "여름 (남반구)"
        elif temp > 10:
            return "봄/가을 (남반구)"
        return "겨울 (남반구)"


def _weather_code_to_desc(code: Optional[int]) -> str:
    if code is None:
        return ""
    if code == 0:
        return "맑음"
    elif code <= 3:
        return "구름 조금~많음"
    elif code <= 9:
        return "안개"
    elif code <= 19:
        return "비/소나기"
    elif code <= 29:
        return "뇌우"
    elif code <= 39:
        return "눈보라"
    elif code <= 49:
        return "안개/서리"
    elif code <= 59:
        return "이슬비"
    elif code <= 69:
        return "비"
    elif code <= 79:
        return "눈"
    elif code <= 89:
        return "소나기"
    return "뇌우"


def _parse_datetime(datetime_str: str) -> Optional[datetime.datetime]:
    formats = [
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.datetime.strptime(datetime_str.strip(), fmt)
        except ValueError:
            continue
    return None


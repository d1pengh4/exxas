"""
Stage 1: EXIF 디지털 포렌식
- GPS/시각/기기 파싱
- UTC 오프셋 → 시간대 추정
- PRNU 핑거프린트 (센서 고유 패턴)
- JPEG 압축 팩터 → 플랫폼 경유 탐지
- 썸네일 vs 현재 이미지 불일치 탐지
"""
import io
import math
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
import piexif
import exifread


@dataclass
class GPSInfo:
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    direction: Optional[float] = None
    timestamp: Optional[str] = None


@dataclass
class ExifResult:
    # GPS
    gps: Optional[GPSInfo] = None
    has_gps: bool = False

    # 촬영 정보
    datetime_original: str = ""
    datetime_digitized: str = ""
    utc_offset: str = ""
    timezone_estimate: str = ""  # 시간대 → 국가군

    # 기기 정보
    make: str = ""
    model: str = ""
    software: str = ""
    device_country_hint: str = ""  # 기기 주요 판매국

    # JPEG 압축 팩터
    jpeg_quality: int = -1
    platform_hint: str = ""  # Instagram/Twitter/KakaoTalk 등

    # PRNU 카메라 핑거프린트
    prnu_fingerprint: str = ""       # 센서 노이즈 해시
    prnu_anomaly_score: float = 0.0  # 0~1 (높을수록 합성/변조 의심)

    # 조작 탐지
    thumbnail_mismatch: bool = False
    has_thumbnail: bool = False
    orientation_modified: bool = False

    # 원본 태그 전체 (디버깅용)
    raw_tags: dict = field(default_factory=dict)


async def run(image_bytes: bytes) -> ExifResult:
    result = ExifResult()

    # exifread로 전체 태그 파싱
    stream = io.BytesIO(image_bytes)
    tags = exifread.process_file(stream, details=True)
    result.raw_tags = {k: str(v) for k, v in tags.items()}

    # GPS
    gps = _extract_gps(tags)
    if gps:
        result.gps = gps
        result.has_gps = True

    # 촬영 시간
    result.datetime_original = str(tags.get("EXIF DateTimeOriginal", ""))
    result.datetime_digitized = str(tags.get("EXIF DateTimeDigitized", ""))
    result.utc_offset = str(tags.get("EXIF OffsetTimeOriginal", ""))

    if result.utc_offset:
        result.timezone_estimate = _utc_to_region(result.utc_offset)

    # 기기 정보
    result.make = str(tags.get("Image Make", "")).strip()
    result.model = str(tags.get("Image Model", "")).strip()
    result.software = str(tags.get("Image Software", "")).strip()
    result.device_country_hint = _device_to_country(result.make, result.model)

    # JPEG 압축 팩터
    result.jpeg_quality = _estimate_jpeg_quality(image_bytes)
    result.platform_hint = _quality_to_platform(result.jpeg_quality)

    # 썸네일 vs 본체 불일치 탐지
    thumb_check = _check_thumbnail_mismatch(image_bytes)
    result.has_thumbnail = thumb_check["has_thumbnail"]
    result.thumbnail_mismatch = thumb_check["mismatch"]

    # 방향 수정 탐지
    orientation = str(tags.get("Image Orientation", ""))
    result.orientation_modified = orientation not in ["Horizontal (normal)", "1", ""]

    # PRNU 카메라 핑거프린트
    prnu = _prnu_analysis(image_bytes)
    result.prnu_fingerprint = prnu["fingerprint"]
    result.prnu_anomaly_score = prnu["anomaly_score"]

    return result


def _extract_gps(tags: dict) -> Optional[GPSInfo]:
    lat_ref = str(tags.get("GPS GPSLatitudeRef", ""))
    lon_ref = str(tags.get("GPS GPSLongitudeRef", ""))
    lat_tag = tags.get("GPS GPSLatitude")
    lon_tag = tags.get("GPS GPSLongitude")

    if not (lat_tag and lon_tag):
        return None

    try:
        lat = _dms_to_decimal(lat_tag.values)
        lon = _dms_to_decimal(lon_tag.values)

        if lat_ref == "S":
            lat = -lat
        if lon_ref == "W":
            lon = -lon

        alt = None
        alt_tag = tags.get("GPS GPSAltitude")
        if alt_tag:
            alt = float(alt_tag.values[0].num) / float(alt_tag.values[0].den)

        direction = None
        dir_tag = tags.get("GPS GPSImgDirection")
        if dir_tag:
            direction = float(dir_tag.values[0].num) / float(dir_tag.values[0].den)

        ts = str(tags.get("GPS GPSTimeStamp", ""))

        return GPSInfo(
            latitude=round(lat, 6),
            longitude=round(lon, 6),
            altitude=round(alt, 1) if alt is not None else None,
            direction=round(direction, 1) if direction is not None else None,
            timestamp=ts,
        )
    except Exception:
        return None


def _dms_to_decimal(values) -> float:
    d = float(values[0].num) / float(values[0].den)
    m = float(values[1].num) / float(values[1].den)
    s = float(values[2].num) / float(values[2].den)
    return d + m / 60 + s / 3600


def _utc_to_region(utc_offset: str) -> str:
    UTC_REGION = {
        "+09:00": "한국/일본",
        "+09": "한국/일본",
        "+08:00": "중국/대만/싱가포르",
        "+08": "중국/대만/싱가포르",
        "+07:00": "태국/베트남/인도네시아",
        "+05:30": "인도",
        "+01:00": "유럽 중부",
        "+00:00": "영국/서아프리카",
        "-05:00": "미국 동부",
        "-08:00": "미국 서부",
    }
    return UTC_REGION.get(utc_offset.strip(), f"UTC{utc_offset}")


def _device_to_country(make: str, model: str) -> str:
    make_lower = make.lower()
    model_lower = model.lower()

    if "samsung" in make_lower:
        return "한국 (삼성)"
    elif "lg" in make_lower:
        return "한국 (LG)"
    elif "apple" in make_lower:
        return "글로벌"
    elif "huawei" in make_lower or "honor" in make_lower:
        return "중국"
    elif "xiaomi" in make_lower or "redmi" in make_lower:
        return "중국"
    elif "sony" in make_lower:
        return "일본"
    elif "canon" in make_lower or "nikon" in make_lower or "fujifilm" in make_lower:
        return "일본"
    return ""


def _estimate_jpeg_quality(image_bytes: bytes) -> int:
    """
    JPEG 양자화 테이블 분석으로 압축 품질 추정
    Instagram≈85, Twitter≈80, KakaoTalk≈72
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if hasattr(img, "quantization"):
            q = img.quantization[0]
            # 첫 번째 양자화 테이블의 평균으로 품질 역산
            avg = sum(q.values()) / len(q) if q else 50
            quality = max(1, min(100, int(100 - avg * 0.8)))
            return quality
    except Exception:
        pass
    return -1


def _quality_to_platform(quality: int) -> str:
    if quality < 0:
        return ""
    elif quality <= 75:
        return "카카오톡/라인 (강한 압축)"
    elif quality <= 82:
        return "Twitter/X"
    elif quality <= 87:
        return "Instagram/Facebook"
    elif quality <= 92:
        return "네이버 블로그/카페"
    return "원본 또는 경미한 압축"


def _check_thumbnail_mismatch(image_bytes: bytes) -> dict:
    try:
        exif_data = piexif.load(image_bytes)
        thumb = exif_data.get("thumbnail")
        if not thumb:
            return {"has_thumbnail": False, "mismatch": False}

        # 썸네일과 본체 해시 비교
        main_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        main_img.thumbnail((64, 64))

        thumb_img = Image.open(io.BytesIO(thumb)).convert("RGB")
        thumb_img = thumb_img.resize((64, 64))

        import imagehash
        h1 = imagehash.phash(main_img)
        h2 = imagehash.phash(thumb_img)
        diff = h1 - h2

        return {
            "has_thumbnail": True,
            "mismatch": diff > 10,
            "hash_diff": diff,
        }
    except Exception:
        return {"has_thumbnail": False, "mismatch": False}


def _prnu_analysis(image_bytes: bytes) -> dict:
    """
    PRNU (Photo Response Non-Uniformity) 기반 카메라 핑거프린팅
    - 카메라 센서 고유 노이즈 패턴 추출
    - 노이즈 패턴의 공간적 일관성 분석 → 합성/변조 탐지
    - 실제 PRNU는 다수 이미지 + 카메라 DB 필요하므로
      여기서는 단일 이미지의 노이즈 이상 탐지 (Anomaly Detection)
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter

        img = Image.open(io.BytesIO(image_bytes)).convert("L")  # 그레이스케일

        # 너무 작은 이미지 제외
        w, h = img.size
        if w < 64 or h < 64:
            return {"fingerprint": "", "anomaly_score": 0.0}

        img_array = np.array(img, dtype=np.float32)

        # ── 1. 센서 노이즈 추출 (Wavelet Denoise 근사: Gaussian 차분) ──
        # σ=2.0 가우시안으로 스무딩 → 원본과 차이 = 노이즈 잔차
        blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32)
        noise = img_array - blurred  # PRNU 잔차 (센서 노이즈 근사)

        # ── 2. 노이즈 패턴 해시 (핑거프린트) ──
        # 128×128로 다운샘플 후 중앙값 기준 이진화 → 64비트 해시
        from PIL import Image as PILImage
        noise_img = PILImage.fromarray(np.clip(noise + 128, 0, 255).astype(np.uint8))
        noise_thumb = noise_img.resize((128, 128), PILImage.BILINEAR)
        noise_arr = np.array(noise_thumb, dtype=np.float32)
        median = float(np.median(noise_arr))
        bits = (noise_arr > median).flatten()[:256]
        fingerprint = hex(int("".join("1" if b else "0" for b in bits), 2))[:32]

        # ── 3. 공간적 일관성 → 이상 탐지 ──
        # 정상 이미지: 노이즈 분산이 공간적으로 균일
        # 합성 이미지: 조작 영역의 노이즈 분산이 다름
        block_size = 32
        variances = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = noise[y:y+block_size, x:x+block_size]
                variances.append(float(np.var(block)))

        if len(variances) < 4:
            return {"fingerprint": fingerprint, "anomaly_score": 0.0}

        var_arr = np.array(variances)
        overall_mean = float(np.mean(var_arr))
        overall_std = float(np.std(var_arr))

        if overall_mean == 0:
            return {"fingerprint": fingerprint, "anomaly_score": 0.0}

        # Coefficient of Variation: 높을수록 불균일 (합성 의심)
        cv = overall_std / (overall_mean + 1e-6)

        # 정규화: cv 0~2 범위를 0~1로
        anomaly_score = float(min(cv / 2.0, 1.0))

        return {
            "fingerprint": fingerprint,
            "anomaly_score": round(anomaly_score, 4),
        }

    except Exception:
        return {"fingerprint": "", "anomaly_score": 0.0}

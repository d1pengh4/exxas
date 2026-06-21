"""
Stage 0: 입력 전처리
- 포맷 정규화
- 품질 평가
- pHash/dHash/aHash 생성
- ELA 조작 선행 탐지
"""
import io
import math
from dataclasses import dataclass, field
from PIL import Image, ImageChops, ImageEnhance
import imagehash
import numpy as np


@dataclass
class PreprocessResult:
    # 해시
    phash: str = ""
    dhash: str = ""
    ahash: str = ""

    # 품질
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    format: str = ""
    mode: str = ""
    quality_score: float = 0.0  # 0~1, 낮을수록 품질 불량

    # ELA 조작 탐지
    ela_max_difference: float = 0.0
    ela_mean_difference: float = 0.0
    manipulation_suspected: bool = False
    manipulation_score: float = 0.0  # 0~1

    # AI 생성 탐지 (F 업그레이드)
    ai_generated_score: float = 0.0
    ai_generated_suspected: bool = False
    ai_detection_evidence: list = None  # type: ignore

    # 야간/저조도 탐지
    is_night_scene: bool = False
    brightness_mean: float = 0.0
    night_enhanced_bytes: bytes = b""  # CLAHE 보정된 ��미지

    def __post_init__(self):
        if self.ai_detection_evidence is None:
            self.ai_detection_evidence = []

    # 전처리된 이미지
    normalized_image_bytes: bytes = b""
    thumbnail_bytes: bytes = b""


async def run(image_bytes: bytes) -> PreprocessResult:
    result = PreprocessResult()
    result.file_size_bytes = len(image_bytes)

    img = Image.open(io.BytesIO(image_bytes))
    result.format = img.format or "UNKNOWN"
    result.mode = img.mode
    result.width, result.height = img.size

    # RGB 정규화
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 품질 평가 (해상도 기반)
    pixels = result.width * result.height
    if pixels >= 1_000_000:
        result.quality_score = 1.0
    elif pixels >= 500_000:
        result.quality_score = 0.7
    elif pixels >= 200_000:
        result.quality_score = 0.5
    else:
        result.quality_score = 0.3

    # 해시 생성
    result.phash = str(imagehash.phash(img))
    result.dhash = str(imagehash.dhash(img))
    result.ahash = str(imagehash.average_hash(img))

    # ELA (Error Level Analysis) — JPEG 재압축 오차 분석
    ela_result = _ela_analysis(img)
    result.ela_max_difference = ela_result["max"]
    result.ela_mean_difference = ela_result["mean"]
    result.manipulation_score = ela_result["score"]
    result.manipulation_suspected = ela_result["score"] > 0.15

    # 정규화된 이미지 저장
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    result.normalized_image_bytes = buf.getvalue()

    # 썸네일
    thumb = img.copy()
    thumb.thumbnail((256, 256))
    tbuf = io.BytesIO()
    thumb.save(tbuf, format="JPEG", quality=80)
    result.thumbnail_bytes = tbuf.getvalue()

    # ── 야간/저조도 감지 및 CLAHE 보정 ─────────────────────
    brightness = _calc_brightness(img)
    result.brightness_mean = brightness
    result.is_night_scene = brightness < 60.0  # 0~255 중 60 이하 = 어두운 장면

    if result.is_night_scene:
        enhanced = _clahe_enhance(img)
        ebuf = io.BytesIO()
        enhanced.save(ebuf, format="JPEG", quality=92)
        result.night_enhanced_bytes = ebuf.getvalue()

    # ── F. AI 생성 이미지 탐지 ────���───────────���─────────────
    try:
        from ..services.ai_detector import detect_ai_generated
        ai_result = detect_ai_generated(image_bytes)
        result.ai_generated_score = ai_result.get("ai_generated_score", 0.0)
        result.ai_generated_suspected = ai_result.get("ai_generated_suspected", False)
        result.ai_detection_evidence = ai_result.get("evidence", [])
    except Exception:
        pass  # 탐지 실패해도 파이프라인 계속

    return result


def _quick_phash(image_bytes: bytes) -> str:
    """캐시 키용 빠른 phash 계산 (8글자)"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return str(imagehash.phash(img))


def _calc_brightness(img: Image.Image) -> float:
    """이미지 평균 밝기 (0~255)"""
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)
    return float(arr.mean())


def _clahe_enhance(img: Image.Image) -> Image.Image:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    야간/저조도 이미지 대비 향상 — OpenCV 없이 PIL로 구��
    """
    # YCbCr 변환 후 Y채널만 히스토그램 평활화
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()

    y_arr = np.array(y, dtype=np.float32)

    # CLAHE 근사: 타일별 히스토그램 평활화
    tile_h, tile_w = max(y_arr.shape[0] // 8, 1), max(y_arr.shape[1] // 8, 1)
    result_arr = np.zeros_like(y_arr)

    for i in range(0, y_arr.shape[0], tile_h):
        for j in range(0, y_arr.shape[1], tile_w):
            tile = y_arr[i:i+tile_h, j:j+tile_w]
            flat = tile.flatten()

            # 히스토그램
            hist, bins = np.histogram(flat, bins=256, range=(0, 256))

            # Clip limit (과도한 대비 제한)
            clip_limit = max(int(flat.size * 0.01), 1)
            excess = np.maximum(hist - clip_limit, 0)
            hist = np.minimum(hist, clip_limit)
            hist += excess.sum() // 256

            # 누적 분포 → 맵핑
            cdf = hist.cumsum().astype(np.float32)
            if cdf[-1] > 0:
                cdf = (cdf - cdf.min()) / (cdf[-1] - cdf.min() + 1e-6) * 255
            result_arr[i:i+tile_h, j:j+tile_w] = cdf[tile.astype(np.int32).clip(0, 255)]

    # 밝기 50% 증폭 (야간 이미지용)
    result_arr = np.clip(result_arr * 1.5, 0, 255).astype(np.uint8)

    y_new = Image.fromarray(result_arr, mode="L")
    enhanced = Image.merge("YCbCr", (y_new, cb, cr)).convert("RGB")
    return enhanced


def _ela_analysis(img: Image.Image) -> dict:
    """
    ELA: 원본과 재압축본의 차이를 분석
    조작된 영역은 재압축 오차가 낮게 나타남 (이미 최저 품질에 도달)
    """
    # 90% 품질로 재압축
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")

    diff = ImageChops.difference(img.convert("RGB"), compressed)
    diff_array = np.array(diff, dtype=np.float32)

    max_diff = float(diff_array.max())
    mean_diff = float(diff_array.mean())

    # 10배 증폭으로 차이 가시화
    amplified = np.clip(diff_array * 10, 0, 255).astype(np.uint8)

    # 조작 점수: 고오차 픽셀 비율
    high_error_pixels = np.sum(amplified > 50) / amplified.size
    manipulation_score = float(high_error_pixels)

    return {
        "max": round(max_diff, 2),
        "mean": round(mean_diff, 4),
        "score": round(manipulation_score, 4),
    }

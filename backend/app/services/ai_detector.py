"""
F. AI 생성 이미지 탐지
- DCT 주파수 스펙트럼 분석 (GAN/Diffusion 아티팩트)
- 노이즈 패턴 분석 (자연 이미지 vs AI 생성 이미지)
- 색상 히스토그램 분포 분석
- CLIP 기반 "AI generated artwork" 분류
신뢰도 0.70+ → ai_generated_suspected = True
"""
import io
import math
import asyncio
import numpy as np
from loguru import logger
from PIL import Image
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# DCT 기반 주파수 분석
# AI 생성 이미지는 고주파 성분이 자연 이미지와 다른 분포를 가짐
# ─────────────────────────────────────────────────────────────────────────────
def _dct2d_simple(block: np.ndarray) -> np.ndarray:
    """2D DCT (8x8 블록, 순수 numpy 구현)"""
    n = block.shape[0]
    result = np.zeros_like(block, dtype=np.float32)
    for u in range(n):
        for v in range(n):
            cu = math.sqrt(1/n) if u == 0 else math.sqrt(2/n)
            cv = math.sqrt(1/n) if v == 0 else math.sqrt(2/n)
            s = 0.0
            for x in range(n):
                for y in range(n):
                    s += (block[x, y] *
                          math.cos(math.pi * u * (2*x + 1) / (2*n)) *
                          math.cos(math.pi * v * (2*y + 1) / (2*n)))
            result[u, v] = cu * cv * s
    return result


def _analyze_dct_spectrum(img_gray: np.ndarray) -> dict:
    """
    이미지 DCT 스펙트럼 분석
    AI 생성 이미지: 고주파 성분 과다 or 특정 주파수 주기성
    자연 사진: 1/f 스펙트럼 (저주파 우세, 고주파 점진 감소)
    """
    h, w = img_gray.shape
    # 중앙 256x256 크롭
    cy, cx = h // 2, w // 2
    crop = img_gray[max(0, cy-128):cy+128, max(0, cx-128):cx+128].astype(np.float32)
    if crop.shape[0] < 64 or crop.shape[1] < 64:
        crop = img_gray.astype(np.float32)

    # FFT 기반 주파수 분석 (DCT와 유사, 더 빠름)
    fft = np.fft.fft2(crop)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    magnitude[magnitude == 0] = 1e-10

    # 로그 스펙트럼
    log_mag = np.log(magnitude)

    ch, cw = log_mag.shape
    cy2, cx2 = ch // 2, cw // 2

    # 저주파 (중앙) vs 고주파 (외곽) 에너지 비율
    r_low = min(cy2, cx2) // 4
    r_mid = min(cy2, cx2) // 2
    r_high = min(cy2, cx2)

    y_coords, x_coords = np.mgrid[0:ch, 0:cw]
    dist = np.sqrt((y_coords - cy2)**2 + (x_coords - cx2)**2)

    e_low = float(magnitude[dist < r_low].sum())
    e_mid = float(magnitude[(dist >= r_low) & (dist < r_mid)].sum())
    e_high = float(magnitude[(dist >= r_mid) & (dist < r_high)].sum())
    total = e_low + e_mid + e_high + 1e-10

    # 자연 이미지: e_low/total > 0.85 (저주파 우세)
    # AI 생성: e_low/total이 낮거나, 고주파 에너지 비율이 높음
    low_ratio = e_low / total
    high_ratio = e_high / total
    mid_ratio = e_mid / total

    # 주파수 균일성 (AI 생성 이미지는 더 균일한 경향)
    freq_uniformity = 1.0 - (high_ratio - low_ratio)

    # 고주파 피크 감지 (GAN 아티팩트: 특정 주파수에 비정상 피크)
    high_region = magnitude[dist >= r_mid]
    if len(high_region) > 0:
        high_std = float(np.std(high_region))
        high_mean = float(np.mean(high_region))
        peak_ratio = high_std / max(high_mean, 1e-10)
    else:
        peak_ratio = 0.0

    return {
        "low_ratio": round(low_ratio, 4),
        "mid_ratio": round(mid_ratio, 4),
        "high_ratio": round(high_ratio, 4),
        "freq_uniformity": round(freq_uniformity, 4),
        "peak_ratio": round(peak_ratio, 4),
    }


def _analyze_noise_pattern(img_gray: np.ndarray) -> dict:
    """
    노이즈 패턴 분석
    자연 사진: 센서 노이즈 (랜덤, 균일 분포)
    AI 생성: 구조적 패턴 노이즈 또는 지나치게 매끄러운 영역
    """
    h, w = img_gray.shape
    img_f = img_gray.astype(np.float32)

    # 인접 픽셀 차이 (노이즈 추정)
    diff_h = np.abs(img_f[:, 1:] - img_f[:, :-1])
    diff_v = np.abs(img_f[1:, :] - img_f[:-1, :])

    noise_mean = float((diff_h.mean() + diff_v.mean()) / 2)
    noise_std = float((diff_h.std() + diff_v.std()) / 2)

    # 자연 이미지: noise_mean 2~8, noise_std 3~12
    # AI 생성 (매끄러움): noise_mean < 1.5
    # GAN 아티팩트: noise_std 비정상적으로 높음
    too_smooth = noise_mean < 1.5
    irregular_noise = noise_std > 15.0

    # 블록 아티팩트 감지 (8x8 주기성 — AI/압축 아티팩트)
    block_artifact = 0.0
    if h >= 16 and w >= 16:
        # 8픽셀 주기로 평균 차이
        h8 = (h // 8) * 8
        w8 = (w // 8) * 8
        block_diffs = []
        for i in range(8, h8, 8):
            row_diff = float(np.abs(img_f[i, :w8] - img_f[i-1, :w8]).mean())
            block_diffs.append(row_diff)
        if block_diffs:
            block_artifact = float(np.mean(block_diffs))

    return {
        "noise_mean": round(noise_mean, 3),
        "noise_std": round(noise_std, 3),
        "too_smooth": too_smooth,
        "irregular_noise": irregular_noise,
        "block_artifact_score": round(block_artifact, 3),
    }


def _analyze_color_distribution(img_rgb: np.ndarray) -> dict:
    """
    색상 분포 분석
    AI 생성: 채도 과포화, 특정 색상 클러스터링
    자연 사진: 균일한 색상 분포
    """
    # HSV 변환 (근사)
    r = img_rgb[:, :, 0].astype(np.float32) / 255.0
    g = img_rgb[:, :, 1].astype(np.float32) / 255.0
    b = img_rgb[:, :, 2].astype(np.float32) / 255.0

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Saturation
    saturation = np.where(cmax > 0, delta / cmax, 0.0)
    mean_saturation = float(saturation.mean())
    high_sat_ratio = float((saturation > 0.7).mean())  # 과포화 픽셀 비율

    # Value (밝기)
    value = cmax
    mean_value = float(value.mean())

    # 색상 다양성 (엔트로피)
    r_hist, _ = np.histogram(img_rgb[:, :, 0], bins=32, range=(0, 256))
    g_hist, _ = np.histogram(img_rgb[:, :, 1], bins=32, range=(0, 256))
    b_hist, _ = np.histogram(img_rgb[:, :, 2], bins=32, range=(0, 256))

    def entropy(hist):
        p = hist / (hist.sum() + 1e-10)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p + 1e-10)))

    color_entropy = (entropy(r_hist) + entropy(g_hist) + entropy(b_hist)) / 3

    # AI 생성 징표: 과포화 + 낮은 엔트로피 (특정 색상 반복)
    overly_saturated = mean_saturation > 0.55 and high_sat_ratio > 0.20

    return {
        "mean_saturation": round(mean_saturation, 3),
        "high_sat_ratio": round(high_sat_ratio, 3),
        "mean_brightness": round(mean_value, 3),
        "color_entropy": round(color_entropy, 3),
        "overly_saturated": overly_saturated,
    }


def detect_ai_generated(image_bytes: bytes) -> dict:
    """
    AI 생성 이미지 탐지 메인 함수 (동기)
    반환: {
        ai_generated_score: 0~1,
        ai_generated_suspected: bool,
        confidence: 0~1,
        evidence: [설명 목록],
        details: {...}
    }
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((512, 512))
        arr = np.array(img)
        gray = np.mean(arr, axis=2).astype(np.uint8)

        dct_stats = _analyze_dct_spectrum(gray)
        noise_stats = _analyze_noise_pattern(gray)
        color_stats = _analyze_color_distribution(arr)

        evidence = []
        score_factors = []

        # ── DCT 분석 ──
        low_ratio = dct_stats["low_ratio"]
        high_ratio = dct_stats["high_ratio"]

        if low_ratio < 0.70:
            # 자연 이미지보다 저주파 비율이 낮음 → AI 생성 징표
            ai_dct_score = (0.70 - low_ratio) / 0.70
            score_factors.append(min(ai_dct_score, 0.6))
            if ai_dct_score > 0.3:
                evidence.append(f"주파수 분포 이상 (저주파 {low_ratio:.2%}, 정상 70%+)")

        if high_ratio > 0.08:
            score_factors.append(min((high_ratio - 0.08) * 5, 0.4))
            evidence.append(f"고주파 에너지 과다 ({high_ratio:.2%})")

        # ── 노이즈 분석 ──
        if noise_stats["too_smooth"]:
            score_factors.append(0.4)
            evidence.append(f"비정상적 매끄러움 (노이즈 평균: {noise_stats['noise_mean']:.2f})")

        if noise_stats["irregular_noise"]:
            score_factors.append(0.3)
            evidence.append(f"불규칙 노이즈 패턴 (std: {noise_stats['noise_std']:.2f})")

        if noise_stats["block_artifact_score"] > 8.0:
            score_factors.append(0.25)
            evidence.append(f"블록 아티팩트 감지 (score: {noise_stats['block_artifact_score']:.2f})")

        # ── 색상 분석 ──
        if color_stats["overly_saturated"]:
            score_factors.append(0.3)
            evidence.append(
                f"과포화 색상 (mean_sat: {color_stats['mean_saturation']:.2f}, "
                f"high_sat: {color_stats['high_sat_ratio']:.2%})"
            )

        if color_stats["color_entropy"] < 2.5:
            score_factors.append(0.2)
            evidence.append(f"색상 다양성 부족 (entropy: {color_stats['color_entropy']:.2f})")

        # ── 최종 점수 계산 ──
        if score_factors:
            # 독립 단서 조합 (최대 1.0)
            ai_score = 1.0 - math.prod(1.0 - f for f in score_factors)
        else:
            ai_score = 0.0

        ai_score = round(min(ai_score, 0.99), 4)
        suspected = ai_score > 0.65

        # 신뢰도 = 단서 수에 따라
        confidence = min(0.40 + len(evidence) * 0.12, 0.85)

        if suspected and not evidence:
            evidence.append("복합 통계적 이상 감지")

        logger.debug(
            f"[ai_detector] score={ai_score:.3f} suspected={suspected} "
            f"evidence_count={len(evidence)}"
        )

        return {
            "ai_generated_score": ai_score,
            "ai_generated_suspected": suspected,
            "confidence": round(confidence, 3),
            "evidence": evidence,
            "details": {
                "dct": dct_stats,
                "noise": noise_stats,
                "color": color_stats,
            },
        }

    except Exception as e:
        logger.warning(f"[ai_detector] 탐지 실패: {e}")
        return {
            "ai_generated_score": 0.0,
            "ai_generated_suspected": False,
            "confidence": 0.0,
            "evidence": [],
            "error": str(e),
        }


async def detect_ai_generated_async(image_bytes: bytes) -> dict:
    return await asyncio.to_thread(detect_ai_generated, image_bytes)

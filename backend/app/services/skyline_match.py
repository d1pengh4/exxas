"""
C. 스카이라인 실루엣 매칭
- Canny Edge로 이미지 지평선/스카이라인 추출
- 한국 주요 도시 스카이라인 특징 DB와 DTW 유사도 비교
- 광안대교, 롯데타워, N서울타워 등 랜드마크 실루엣 식별
"""
import io
import math
import asyncio
import numpy as np
from loguru import logger
from PIL import Image
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 한국 주요 도시 스카이라인 특징 DB
# 각 도시/지역별 특징 벡터:
#   - height_profile: 정규화된 건물 높이 프로파일 (0~1, 20개 구간)
#   - has_bridge: 교량 특징 여부
#   - has_tower: 뾰족한 타워 여부
#   - coastal: 해안선/수평선 여부
#   - density: 건물 밀도 (sparse/medium/dense)
# ─────────────────────────────────────────────────────────────────────────────
SKYLINE_DB: dict[str, dict] = {
    "서울_남산타워": {
        "city": "서울",
        "area": "남산/용산",
        "lat": 37.5512, "lon": 126.9882,
        "features": {
            "has_tower": True,       # N서울타워 뾰족한 실루엣
            "tower_position": "center",
            "has_bridge": False,
            "coastal": False,
            "density": "dense",
            "mountain_backdrop": True,  # 남산
            "height_profile": [0.3, 0.4, 0.6, 0.7, 0.9, 1.0, 0.8, 0.5, 0.4, 0.3,
                                0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        },
        "description": "N서울타워가 중앙에 있는 서울 도심 스카이라인",
    },
    "서울_강남": {
        "city": "서울",
        "area": "강남/삼성",
        "lat": 37.5013, "lon": 127.0595,
        "features": {
            "has_tower": True,      # 롯데월드타워 (송파)
            "tower_position": "right",
            "has_bridge": False,
            "coastal": False,
            "density": "dense",
            "mountain_backdrop": False,
            "height_profile": [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.7, 0.8, 0.9,
                                1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3, 0.3],
        },
        "description": "고층 빌딩 밀집, 롯데월드타워 실루엣",
    },
    "서울_여의도": {
        "city": "서울",
        "area": "여의도/영등포",
        "lat": 37.5219, "lon": 126.9244,
        "features": {
            "has_tower": False,
            "has_bridge": True,    # 한강대교/원효대교
            "coastal": True,       # 한강변
            "density": "medium",
            "mountain_backdrop": False,
            "height_profile": [0.4, 0.5, 0.7, 0.8, 0.8, 0.7, 0.8, 0.9, 0.8, 0.7,
                                0.6, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1],
        },
        "description": "여의도 금융가 + 한강변 + 교량",
    },
    "부산_해운대": {
        "city": "부산",
        "area": "해운대/마린시티",
        "lat": 35.1634, "lon": 129.1610,
        "features": {
            "has_tower": True,      # 해운대 LCT 타워
            "tower_position": "left",
            "has_bridge": False,
            "coastal": True,        # 해운대 해수욕장
            "density": "dense",
            "mountain_backdrop": True,  # 달맞이고개
            "height_profile": [0.9, 1.0, 0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.3, 0.2,
                                0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        },
        "description": "해운대 초고층 아파트 + 해수욕장",
    },
    "부산_광안대교": {
        "city": "부산",
        "area": "수영/광안리",
        "lat": 35.1502, "lon": 129.1237,
        "features": {
            "has_tower": False,
            "has_bridge": True,     # 광안대교 (다이아몬드 케이블교)
            "bridge_type": "cable_stayed",  # 사장교 특징
            "coastal": True,        # 광안리 해수욕장
            "density": "medium",
            "mountain_backdrop": False,
            "height_profile": [0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9,
                                0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3],
        },
        "description": "광안대교 다이아몬드 케이블교 야경",
    },
    "부산_남포동": {
        "city": "부산",
        "area": "남포동/자갈치",
        "lat": 35.0984, "lon": 129.0319,
        "features": {
            "has_tower": False,
            "has_bridge": False,
            "coastal": True,        # 부산항
            "density": "medium",
            "mountain_backdrop": True,  # 용두산공원
            "height_profile": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7,
                                0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
        },
        "description": "부산항 + 용두산공원 + 자갈치시장",
    },
    "인천_송도": {
        "city": "인천",
        "area": "송도국제도시",
        "lat": 37.3797, "lon": 126.6561,
        "features": {
            "has_tower": True,      # 포스코타워 등 초고층
            "tower_position": "center",
            "has_bridge": False,
            "coastal": True,        # 송도 갯벌/바다
            "density": "medium",
            "mountain_backdrop": False,
            "height_profile": [0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3,
                                0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        },
        "description": "송도 첨단 신도시 고층 빌딩 + 바다",
    },
    "서울_한강": {
        "city": "서울",
        "area": "한강공원",
        "lat": 37.5172, "lon": 126.9790,
        "features": {
            "has_tower": False,
            "has_bridge": True,     # 한강 교량 다수
            "bridge_type": "truss",
            "coastal": True,        # 한강
            "density": "sparse",
            "mountain_backdrop": False,
            "height_profile": [0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.4, 0.3,
                                0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        },
        "description": "한강 교량 + 낮은 스카이라인",
    },
    "제주": {
        "city": "제주",
        "area": "제주시/서귀포",
        "lat": 33.4996, "lon": 126.5312,
        "features": {
            "has_tower": False,
            "has_bridge": False,
            "coastal": True,        # 해안
            "density": "sparse",
            "mountain_backdrop": True,  # 한라산
            "height_profile": [0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1],
        },
        "description": "한라산 배경 + 낮은 건물 + 해안",
    },
}


def _extract_skyline(image_bytes: bytes) -> dict:
    """
    이미지에서 스카이라인 특징 추출 (동기)
    - Canny Edge로 윤곽 추출
    - 상단 1/3 영역에서 수평 프로파일 생성
    - 교량/타워/해안 특징 감지
    """
    try:
        import cv2

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((640, 480))
        arr = np.array(img)

        # 그레이스케일 변환
        gray = np.mean(arr, axis=2).astype(np.uint8)
        h, w = gray.shape

        # Canny Edge 검출
        blurred = _gaussian_blur(gray)
        edges = _canny_edge(blurred)

        # 상단 40% 영역 (스카이라인)
        sky_region = edges[:int(h * 0.40), :]

        # 수평 프로파일 (20개 구간으로 분할)
        n_bins = 20
        bin_width = w // n_bins
        height_profile = []
        for i in range(n_bins):
            col_start = i * bin_width
            col_end = min((i + 1) * bin_width, w)
            col_edges = sky_region[:, col_start:col_end]
            # 에지가 있는 최상단 행 위치 (0=top, 1=bottom)
            edge_rows = np.where(col_edges > 0)
            if len(edge_rows[0]) > 0:
                # 최상단 에지 위치 → 0(맨위)~1(바닥) 정규화
                top_edge = float(edge_rows[0].min()) / (h * 0.40)
                height_profile.append(round(1.0 - top_edge, 3))
            else:
                height_profile.append(0.0)

        # 전체 이미지 밝기 분포 (야간/주간)
        top_brightness = float(arr[:int(h * 0.3), :, :].mean() / 255.0)
        bottom_brightness = float(arr[int(h * 0.7):, :, :].mean() / 255.0)

        # 수평선/교량 특징 탐지 (하단 30%에 강한 수평 에지)
        bottom_edges = edges[int(h * 0.55):int(h * 0.75), :]
        h_line_score = float(bottom_edges.sum()) / max(bottom_edges.size, 1)
        has_bridge = h_line_score > 0.03

        # 타워 특징 (상단에 뾰족한 고립 에지)
        top_edges = edges[:int(h * 0.20), :]
        v_score = 0.0
        for col in range(0, w, 10):
            col_data = top_edges[:, max(0, col-5):col+5]
            if col_data.sum() > 50:
                v_score += 1.0
        v_score /= max(w // 10, 1)
        has_tower = v_score > 0.15

        # 해안선/수평선 (하늘과 바다의 경계 — 명도 변화가 균일한 가로 에지)
        sky_brightness_var = float(np.std(arr[:int(h * 0.3), :, :].mean(axis=2)))
        coastal = sky_brightness_var < 15.0 and h_line_score > 0.01

        return {
            "height_profile": height_profile,
            "has_bridge": has_bridge,
            "has_tower": has_tower,
            "coastal": coastal,
            "top_brightness": round(top_brightness, 3),
            "bottom_brightness": round(bottom_brightness, 3),
            "h_line_score": round(h_line_score, 4),
            "v_score": round(v_score, 4),
        }

    except ImportError:
        logger.warning("[skyline] cv2 없음 — PIL 폴백 사용")
        return _extract_skyline_pil(image_bytes)
    except Exception as e:
        logger.warning(f"[skyline] 특징 추출 실패: {e}")
        return {}


def _extract_skyline_pil(image_bytes: bytes) -> dict:
    """cv2 없을 때 PIL 기반 단순 분석"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        img.thumbnail((320, 240))
        arr = np.array(img)
        h, w = arr.shape

        # 단순 프로파일: 각 열에서 밝기 임계값(100) 초과 첫 행
        n_bins = 20
        bin_width = w // n_bins
        height_profile = []
        for i in range(n_bins):
            col = arr[:, i * bin_width:(i + 1) * bin_width]
            bright_rows = np.where(col < 100)  # 어두운 건물 픽셀
            if len(bright_rows[0]) > 0:
                height_profile.append(round(1.0 - float(bright_rows[0].min()) / h, 3))
            else:
                height_profile.append(0.0)

        return {
            "height_profile": height_profile,
            "has_bridge": False,
            "has_tower": max(height_profile) - min(height_profile) > 0.5,
            "coastal": False,
        }
    except Exception as e:
        logger.warning(f"[skyline_pil] 실패: {e}")
        return {}


def _gaussian_blur(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """단순 가우시안 블러 (순수 numpy)"""
    sigma = 1.0
    k = kernel_size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    h, w = img.shape
    result = np.zeros_like(img, dtype=np.float32)
    padded = np.pad(img.astype(np.float32), k, mode='edge')
    for i in range(h):
        for j in range(w):
            result[i, j] = (padded[i:i+kernel_size, j:j+kernel_size] * kernel).sum()
    return np.clip(result, 0, 255).astype(np.uint8)


def _canny_edge(img: np.ndarray) -> np.ndarray:
    """단순 Sobel 기반 에지 검출"""
    img_f = img.astype(np.float32)
    # Sobel X
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    # Sobel Y
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    h, w = img_f.shape
    gx = np.zeros_like(img_f)
    gy = np.zeros_like(img_f)
    padded = np.pad(img_f, 1, mode='edge')

    for i in range(h):
        for j in range(w):
            patch = padded[i:i+3, j:j+3]
            gx[i, j] = (patch * kx).sum()
            gy[i, j] = (patch * ky).sum()

    magnitude = np.sqrt(gx**2 + gy**2)
    threshold = magnitude.max() * 0.2
    edges = (magnitude > threshold).astype(np.uint8) * 255
    return edges


def _dtw_distance(a: list[float], b: list[float]) -> float:
    """Dynamic Time Warping 거리"""
    n, m = len(a), len(b)
    dtw = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    dtw[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i-1] - b[j-1])
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
    return dtw[n][m]


def _feature_similarity(extracted: dict, db_features: dict) -> float:
    """이진 특징 유사도 (0~1)"""
    score = 0.0
    total = 0

    for key in ("has_bridge", "has_tower", "coastal"):
        if key in extracted and key in db_features:
            score += 1.0 if extracted[key] == db_features[key] else 0.0
            total += 1

    # 산 배경 - 하늘 밝기로 추정 (단순 휴리스틱)
    if db_features.get("mountain_backdrop") and extracted.get("top_brightness", 0.5) < 0.4:
        score += 0.5
        total += 1

    return score / max(total, 1)


def match_skyline(image_bytes: bytes) -> dict:
    """
    스카이라인 매칭 메인 함수
    반환: {best_match, candidates, confidence, description}
    """
    extracted = _extract_skyline(image_bytes)
    if not extracted or not extracted.get("height_profile"):
        return {"error": "스카이라인 추출 실패", "candidates": []}

    profile = extracted["height_profile"]
    candidates = []

    for location_id, db_entry in SKYLINE_DB.items():
        db_profile = db_entry["features"]["height_profile"]
        db_features = db_entry["features"]

        # DTW 프로파일 거리 (낮을수록 유사)
        dtw_dist = _dtw_distance(profile, db_profile)
        dtw_score = max(0.0, 1.0 - dtw_dist / 5.0)

        # 이진 특징 유사도
        feat_score = _feature_similarity(extracted, db_features)

        # 종합 점수
        combined = dtw_score * 0.5 + feat_score * 0.5

        candidates.append({
            "location_id": location_id,
            "city": db_entry["city"],
            "area": db_entry["area"],
            "lat": db_entry["lat"],
            "lon": db_entry["lon"],
            "description": db_entry["description"],
            "score": round(combined, 3),
            "dtw_score": round(dtw_score, 3),
            "feature_score": round(feat_score, 3),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[0] if candidates else {}

    # 신뢰도 결정
    confidence = 0.0
    if top:
        raw_score = top["score"]
        # 2위와 격차가 클수록 신뢰도 높음
        gap = raw_score - (candidates[1]["score"] if len(candidates) > 1 else 0)
        confidence = min(raw_score * 0.85 + gap * 0.3, 0.88)

    return {
        "best_match": top.get("location_id", ""),
        "city": top.get("city", ""),
        "area": top.get("area", ""),
        "lat": top.get("lat"),
        "lon": top.get("lon"),
        "description": top.get("description", ""),
        "confidence": round(confidence, 3),
        "score": top.get("score", 0),
        "candidates": candidates[:3],
        "extracted_features": {
            "has_bridge": extracted.get("has_bridge"),
            "has_tower": extracted.get("has_tower"),
            "coastal": extracted.get("coastal"),
        },
    }


async def match_skyline_async(image_bytes: bytes) -> dict:
    return await asyncio.to_thread(match_skyline, image_bytes)

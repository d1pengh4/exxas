"""
Stage 7: 앙상블 판정 + Explainable AI 리포트
- 모든 Stage 결과를 Bayesian Fusion
- 입력 유형 자동 분류 → 동적 가중치
- 신뢰도 최종 계산
- XAI 리포트 생성
"""
from dataclasses import dataclass, field
from typing import Optional
import math


# ── 입력 유형별 동적 가중치 ─────────────────────────────────
DYNAMIC_WEIGHTS: dict[str, dict[str, float]] = {
    "urban": {
        "exif": 5.0, "ocr": 4.5, "reverse_search": 4.0,
        "object_detect": 3.0, "geoclip": 2.5,
        "physical": 0.5, "dem": 0.0,
    },
    "nature": {
        "exif": 5.0, "physical": 4.0, "dem": 3.5,
        "geoclip": 3.0, "object_detect": 1.5,
        "ocr": 0.3, "reverse_search": 1.0,
    },
    "night": {
        "exif": 5.0, "physical": 4.5, "ocr": 3.0,
        "object_detect": 2.0, "geoclip": 1.5,
        "dem": 1.0, "reverse_search": 2.0,
    },
    "indoor": {
        "exif": 5.0, "ocr": 4.5, "object_detect": 1.5,
        "reverse_search": 4.5, "geoclip": 0.3,
        "physical": 0.0, "dem": 0.0,
    },
    "sns_compressed": {
        "exif": 2.0, "reverse_search": 5.0, "ocr": 4.0,
        "object_detect": 3.0, "geoclip": 2.5,
        "physical": 1.0, "dem": 0.5,
    },
    "default": {
        "exif": 4.0, "ocr": 3.5, "reverse_search": 3.0,
        "object_detect": 2.5, "geoclip": 2.5,
        "physical": 2.0, "dem": 1.5,
    },
}

# 신뢰도 상한 — confidence.py의 CONFIDENCE_CEILING을 단일 소스로 사용 (값 중복 제거)
from ..services.confidence import CONFIDENCE_CEILING as _CC
EVIDENCE_CEILINGS = {
    "gps_exif":               _CC["gps_exif_verified"],        # 0.99
    "street_view_vpr":        _CC["street_view_match"],        # 0.95
    "ocr_poi_verified":       _CC["ocr_poi_verified"],         # 0.92
    "reverse_search_confirmed": _CC["reverse_search_blog"],    # 0.85
    "geoclip_top1":           _CC["geoclip_single"],           # 0.75
    "infra_fingerprint":      _CC["infra_fingerprint"],        # 0.70
    "llm_osint_confirmed":    _CC["llm_osint_confirmed"],      # 0.72
    "physical_calc":          _CC["physical_calculation"],     # 0.60
    "vegetation_terrain":     _CC["vegetation_terrain"],       # 0.50
    "inference_only":         _CC["inference_only"],           # 0.40
}

# 독립 소스 수렴 보너스
CONVERGENCE_BONUS = {1: 1.0, 2: 1.1, 3: 1.2, 4: 1.3}
MAX_CONFIDENCE = 0.99


@dataclass
class EnsembleInput:
    """각 Stage의 핵심 결과를 담는 통합 입력"""
    # Stage 1 (EXIF)
    has_gps: bool = False
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_verified: bool = False  # ELA 조작탐지 통과
    exif_timezone: str = ""
    exif_platform: str = ""

    # Stage 2 (Internet)
    reverse_search_hits: int = 0
    reverse_search_location: str = ""
    reverse_search_confidence: float = 0.0

    # Stage 3 (OCR/GIS)
    ocr_texts: list[str] = field(default_factory=list)
    detected_languages: list[str] = field(default_factory=list)
    poi_matched: bool = False
    poi_location: str = ""
    poi_lat: Optional[float] = None
    poi_lon: Optional[float] = None
    poi_source: str = ""
    license_plate_country: str = ""

    # Stage 4 (Infra)
    infra_top_country: str = ""
    infra_score: float = 0.0
    scene_type: str = "default"  # urban/nature/night/indoor/sns_compressed

    # Stage 5 (AI Embedding)
    geoclip_location: str = ""
    geoclip_lat: float = 0.0
    geoclip_lon: float = 0.0
    geoclip_score: float = 0.0
    streetclip_country: str = ""
    # OpenCLIP 한국 세부 지역 (NEW)
    openclip_korea_region: str = ""
    openclip_korea_score: float = 0.0
    ensemble_korea_region: str = ""
    ensemble_korea_confidence: float = 0.0
    vpr_location: str = ""
    vpr_similarity: float = 0.0

    # Stage 6 (Physical)
    latitude_band: str = ""
    hemisphere: str = ""
    season: str = ""
    shadow_analysis: str = ""

    # LLM 수사관 결론
    llm_location: str = ""
    llm_lat: Optional[float] = None
    llm_lon: Optional[float] = None
    llm_confidence: float = 0.0
    llm_evidence_count: int = 0
    llm_contradiction_count: int = 0
    llm_independent_sources: int = 0
    llm_poi_source: str = ""  # interior_osint/naver_place 등 고신뢰 OSINT 소스

    # OSINT 확장 (web_search, naver_blog, osm_poi, street_view)
    osint_web_hits: int = 0
    osint_blog_hits: int = 0
    osint_poi_hits: int = 0
    osint_street_view_matched: bool = False

    # ── Round 3 신규 도구 결과 ──
    transit_match: str = ""               # A. 대중교통 DB 매칭 결과 (도시명)
    skyline_match: str = ""               # C. 스카이라인 매칭 결과 (도시명)
    skyline_confidence: float = 0.0       # C. 스카이라인 매칭 신뢰도
    weather_season: str = ""              # B. 계절/날씨 분석 결과
    clova_entities: dict = field(default_factory=dict)  # E. CLOVA NER 결과

    # ── 한국 전용 분석 결과 ──
    korea_confidence: float = 0.0          # korea_specializer 결과 신뢰도
    korea_location: str = ""               # 한국 특정 위치 문자열
    korea_lat: Optional[float] = None
    korea_lon: Optional[float] = None
    korea_subway_station: str = ""         # 지하철역 확정 시
    korea_roadview_available: bool = False # 네이버 로드뷰 사용 가능


@dataclass
class EnsembleResult:
    # 최종 결론
    final_location: str = ""
    final_lat: Optional[float] = None
    final_lon: Optional[float] = None
    final_confidence: float = 0.0
    confidence_label: str = ""  # HIGH/MEDIUM/LOW/UNKNOWN
    confidence_breakdown: dict = field(default_factory=dict)

    # XAI 리포트
    dominant_evidence: str = ""       # 가장 강력한 단서
    evidence_summary: list[str] = field(default_factory=list)
    contradiction_notes: list[str] = field(default_factory=list)
    scene_type: str = ""
    exploration_quality: str = ""     # "충분" | "제한적" | "불충분"

    # 메타
    input_type_detected: str = ""
    weights_used: dict = field(default_factory=dict)


def _load_weights_from_redis() -> dict:
    """Redis에 RLHF 업데이트된 가중치가 있으면 로드, 없으면 기본값 사용"""
    try:
        import redis as _redis_sync
        import json as _json
        from ..core.config import settings
        r = _redis_sync.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=1)
        raw = r.get("ensemble:weights")
        r.close()
        if raw:
            loaded = _json.loads(raw)
            if isinstance(loaded, dict) and len(loaded) >= 3:
                return loaded
    except Exception:
        pass
    return DYNAMIC_WEIGHTS


def run(inp: EnsembleInput) -> EnsembleResult:
    result = EnsembleResult()
    result.scene_type = inp.scene_type

    # 1. 입력 유형 감지 → 가중치 선택 (RLHF 업데이트된 가중치 우선)
    input_type = _detect_input_type(inp)
    result.input_type_detected = input_type
    active_weights = _load_weights_from_redis()
    weights = active_weights.get(input_type, active_weights.get("default", DYNAMIC_WEIGHTS["default"]))
    result.weights_used = weights

    # 2. GPS가 있고 조작 탐지 통과 → 즉시 최고 신뢰도
    if inp.has_gps and inp.gps_verified and inp.gps_lat and inp.gps_lon:
        result.final_lat = inp.gps_lat
        result.final_lon = inp.gps_lon
        result.final_confidence = EVIDENCE_CEILINGS["gps_exif"]
        result.final_location = f"GPS ({inp.gps_lat:.6f}, {inp.gps_lon:.6f})"
        result.dominant_evidence = "GPS EXIF 좌표 (조작 탐지 통과)"
        result.confidence_label = "HIGH"
        result.evidence_summary = ["GPS 원본 좌표 확인 — 최고 신뢰도 달성"]
        return result

    # 3. 베이지안 퓨전으로 신뢰도 계산
    ceiling = _compute_ceiling(inp)
    independent_sources = _count_independent_sources(inp)
    convergence = CONVERGENCE_BONUS.get(min(independent_sources, 4), 1.3)
    contradiction_penalty = _compute_contradiction_penalty(inp)

    # LLM 수사관 결론 신뢰도를 기반으로
    base = inp.llm_confidence

    # 가중치 적용한 다중 소스 보정
    weighted_score = _compute_weighted_score(inp, weights)

    # 최종 신뢰도: 두 방식의 가중 평균
    raw_confidence = (base * 0.6 + weighted_score * 0.4) * convergence * contradiction_penalty
    final_confidence = min(raw_confidence, min(ceiling, MAX_CONFIDENCE))

    result.final_confidence = round(final_confidence, 4)
    result.confidence_breakdown = {
        "base_llm": round(base, 4),
        "weighted_score": round(weighted_score, 4),
        "ceiling": ceiling,
        "convergence_bonus": convergence,
        "contradiction_penalty": contradiction_penalty,
        "independent_sources": independent_sources,
    }

    # 4. 위치 결정 (우선순위: POI > VPR > GeoCLIP > LLM > Infra)
    result.final_location, result.final_lat, result.final_lon = _pick_best_location(inp)

    # 5. 신뢰도 레이블
    result.confidence_label = _confidence_label(final_confidence)

    # 6. XAI 리포트 생성
    result.dominant_evidence = _dominant_evidence(inp)
    result.evidence_summary = _build_evidence_summary(inp)
    result.contradiction_notes = _build_contradiction_notes(inp)
    result.exploration_quality = _assess_quality(inp)

    return result


def _detect_input_type(inp: EnsembleInput) -> str:
    if inp.exif_platform and ("카카오" in inp.exif_platform or "Instagram" in inp.exif_platform or "Twitter" in inp.exif_platform):
        return "sns_compressed"
    # scene_type을 우선 확인 (LLM/OCR 단계에서 파생)
    if inp.scene_type in ("urban", "nature", "indoor", "night", "sns_compressed", "default"):
        if inp.scene_type != "default":
            return inp.scene_type
    # scene_type이 default이면 데이터로 판단
    if inp.ocr_texts or inp.poi_matched:
        return "urban"
    if not inp.ocr_texts and not inp.poi_matched and inp.latitude_band:
        return "nature"
    return "default"


def _compute_ceiling(inp: EnsembleInput) -> float:
    # 한국 전용: 지하철역 확정 → OCR POI 수준 신뢰도
    if inp.korea_subway_station:
        return max(EVIDENCE_CEILINGS["ocr_poi_verified"], 0.92)
    # 한국 전용: 주소 지오코딩 성공 → 높은 신뢰도
    if inp.korea_confidence >= 0.88 and inp.korea_lat:
        return EVIDENCE_CEILINGS["ocr_poi_verified"]
    # 한국 전용: 랜드마크/우편번호 수준
    if inp.korea_confidence >= 0.70 and inp.korea_lat:
        return EVIDENCE_CEILINGS["reverse_search_confirmed"]

    if inp.vpr_similarity > 0.85:
        return EVIDENCE_CEILINGS["street_view_vpr"]
    if inp.poi_matched and inp.poi_lat:
        return EVIDENCE_CEILINGS["ocr_poi_verified"]
    if inp.reverse_search_hits > 0 and inp.reverse_search_location:
        # GPS/OCR/독립 소스 없이 역방향 검색 단독 → 0.55 상한
        # 시각적으로 유사한 다른 장소가 매칭될 수 있음 (테니스장, 공원 등 일반 장소)
        independent = _count_independent_sources(inp)
        if not inp.has_gps and not inp.poi_matched and independent < 2:
            return 0.55
        return EVIDENCE_CEILINGS["reverse_search_confirmed"]
    # LLM OSINT 3+ 독립 소스 → 높은 천장
    if inp.llm_independent_sources >= 3 and inp.llm_evidence_count >= 3:
        return EVIDENCE_CEILINGS["llm_osint_confirmed"]
    # OpenCLIP 한국 세부 지역 강한 신호 (높은 신뢰도)
    if inp.ensemble_korea_confidence >= 0.25 and inp.ensemble_korea_region:
        return EVIDENCE_CEILINGS["geoclip_top1"]
    if inp.openclip_korea_score >= 0.20 and inp.openclip_korea_region:
        return EVIDENCE_CEILINGS["infra_fingerprint"]
    if inp.geoclip_score > 0.25:
        return EVIDENCE_CEILINGS["geoclip_top1"]
    if inp.infra_score > 1.0:
        return EVIDENCE_CEILINGS["infra_fingerprint"]
    if inp.latitude_band:
        return EVIDENCE_CEILINGS["physical_calc"]
    if inp.llm_evidence_count > 0:
        return EVIDENCE_CEILINGS["inference_only"]
    return 0.20


def _count_independent_sources(inp: EnsembleInput) -> int:
    count = 0
    if inp.has_gps: count += 1
    if inp.ocr_texts and inp.poi_matched: count += 1
    if inp.reverse_search_hits > 0 or inp.osint_web_hits > 0 or inp.osint_blog_hits > 0:
        count += 1  # 인터넷 검색 그룹
    # OpenCLIP + GeoCLIP + StreetCLIP = AI 임베딩 그룹 (하나의 독립 소스)
    if inp.ensemble_korea_confidence >= 0.15 or inp.geoclip_score > 0.3 or inp.openclip_korea_score > 0.15:
        count += 1
    if inp.infra_top_country: count += 1
    if inp.vpr_similarity > 0.5 or inp.osint_street_view_matched: count += 1
    if inp.latitude_band: count += 1
    if inp.osint_poi_hits > 0: count += 1
    if inp.korea_confidence >= 0.70: count += 1
    if inp.transit_match: count += 1        # A. 대중교통 DB — map_api 독립 소스
    if inp.skyline_match: count += 1        # C. 스카이라인 — visual_analysis 독립 소스
    return count


def _compute_contradiction_penalty(inp: EnsembleInput) -> float:
    n = inp.llm_contradiction_count
    if n == 0: return 1.0
    if n == 1: return 0.7
    if n == 2: return 0.4
    return 0.1  # 3개 이상 → 거의 불신


def _compute_weighted_score(inp: EnsembleInput, weights: dict) -> float:
    total_weight = 0.0
    weighted_sum = 0.0

    if inp.has_gps:
        w = weights.get("exif", 4.0)
        weighted_sum += 0.95 * w
        total_weight += w

    if inp.poi_matched:
        w = weights.get("ocr", 3.5)
        weighted_sum += 0.90 * w
        total_weight += w

    if inp.reverse_search_hits > 0:
        w = weights.get("reverse_search", 3.0)
        score = min(inp.reverse_search_confidence, 0.85)
        weighted_sum += score * w
        total_weight += w

    if inp.infra_score > 0:
        w = weights.get("object_detect", 2.5)
        score = min(inp.infra_score / 5.0, 0.70)
        weighted_sum += score * w
        total_weight += w

    if inp.geoclip_score > 0:
        w = weights.get("geoclip", 2.5)
        weighted_sum += inp.geoclip_score * 0.75 * w
        total_weight += w

    # OpenCLIP 한국 지역 분류 (geoclip 그룹으로 절반 가중치)
    if inp.openclip_korea_score > 0.25:
        w = weights.get("geoclip", 2.5) * 0.5
        weighted_sum += inp.openclip_korea_score * w
        total_weight += w

    if inp.vpr_similarity > 0:
        w = weights.get("geoclip", 2.5)
        weighted_sum += inp.vpr_similarity * 0.90 * w
        total_weight += w * 0.5  # 중복 방지

    if inp.latitude_band:
        w = weights.get("physical", 2.0)
        weighted_sum += 0.50 * w
        total_weight += w

    # OSINT 히트 (web_search + naver_blog)
    osint_hits = inp.osint_web_hits + inp.osint_blog_hits
    if osint_hits > 0:
        w = weights.get("reverse_search", 3.0) * 0.6
        score = min(0.65, 0.20 * osint_hits)
        weighted_sum += score * w
        total_weight += w
    # OSM POI 히트 (독립 소스로 가산)
    if inp.osint_poi_hits > 0:
        w = weights.get("ocr", 3.5) * 0.5
        weighted_sum += 0.60 * w
        total_weight += w

    # Round 3 신규 도구
    if inp.transit_match:                   # A. 대중교통 DB — 고신뢰 (ocr_poi 수준)
        w = weights.get("ocr", 3.5) * 0.9
        weighted_sum += 0.88 * w
        total_weight += w
    if inp.skyline_match and inp.skyline_confidence > 0:  # C. 스카이라인
        w = weights.get("geoclip", 2.5) * 0.8
        weighted_sum += min(inp.skyline_confidence, 0.90) * w
        total_weight += w

    if total_weight == 0:
        return inp.llm_confidence
    return min(weighted_sum / total_weight, 1.0)


def _pick_best_location(inp: EnsembleInput) -> tuple[str, Optional[float], Optional[float]]:
    # 한국 전용: 지하철역/주소 지오코딩이 확정된 경우 최우선
    if inp.korea_subway_station and inp.korea_lat:
        return inp.korea_location, inp.korea_lat, inp.korea_lon
    if inp.korea_confidence >= 0.88 and inp.korea_lat:
        return inp.korea_location, inp.korea_lat, inp.korea_lon

    if inp.poi_matched and inp.poi_lat:
        # poi_lon이 유효한 한국 경도(124-132)인지 검증
        if inp.poi_lon and 124.0 <= inp.poi_lon <= 132.0:
            return inp.poi_location, inp.poi_lat, inp.poi_lon
        elif inp.llm_lon and 124.0 <= inp.llm_lon <= 132.0:
            return inp.poi_location, inp.poi_lat, inp.llm_lon
        else:
            return inp.poi_location, inp.poi_lat, inp.poi_lon
    if inp.vpr_location and inp.vpr_similarity > 0.7:
        return inp.vpr_location, None, None
    if inp.geoclip_location and inp.geoclip_score > 0.25 and inp.geoclip_lon != 0.0:
        return inp.geoclip_location, inp.geoclip_lat, inp.geoclip_lon
    # LLM이 OSINT로 좌표 확보한 경우 우선
    if inp.llm_location and inp.llm_lat and inp.llm_lon:
        return inp.llm_location, inp.llm_lat, inp.llm_lon
    if inp.llm_location:
        return inp.llm_location, None, None
    # OpenCLIP 한국 세부 지역 — LLM 실패 시 도시 수준 폴백
    if inp.openclip_korea_region and inp.openclip_korea_score > 0.25:
        return inp.openclip_korea_region, None, None
    # 한국 낮은 신뢰도라도 도시 수준 힌트 활용
    if inp.korea_location and inp.korea_confidence >= 0.50:
        return inp.korea_location, inp.korea_lat, inp.korea_lon
    if inp.infra_top_country:
        return inp.infra_top_country, None, None
    return "위치 특정 불가", None, None


def _confidence_label(conf: float) -> str:
    if conf >= 0.90: return "HIGH"
    if conf >= 0.70: return "MEDIUM"
    if conf >= 0.30: return "LOW"
    return "UNKNOWN"


def _dominant_evidence(inp: EnsembleInput) -> str:
    if inp.has_gps and inp.gps_verified:
        return "GPS EXIF 원본 좌표"
    if inp.vpr_similarity > 0.85:
        return f"VPR Street View 매칭 (유사도 {inp.vpr_similarity:.0%})"
    if inp.poi_matched:
        return f"OCR → {inp.poi_source} 플레이스 매칭"
    if inp.reverse_search_hits > 0:
        return f"역방향 이미지 검색 ({inp.reverse_search_hits}건 매칭)"
    if inp.geoclip_score > 0.5:
        return f"GeoCLIP AI 임베딩 (점수 {inp.geoclip_score:.2f})"
    if inp.infra_top_country:
        return f"인프라 핑거프린팅 → {inp.infra_top_country}"
    return "LLM 추론"


def _build_evidence_summary(inp: EnsembleInput) -> list[str]:
    lines = []
    if inp.has_gps:
        lines.append(f"GPS 좌표: {inp.gps_lat:.5f}, {inp.gps_lon:.5f} ({'검증 완료' if inp.gps_verified else '조작 의심'})")
    if inp.detected_languages:
        lines.append(f"텍스트 언어: {', '.join(inp.detected_languages)} → {_languages_to_countries(inp.detected_languages)}")
    if inp.korea_subway_station:
        lines.append(f"지하철역 확정: {inp.korea_subway_station} → 정밀 좌표 확정")
    elif inp.korea_location and inp.korea_confidence >= 0.70:
        lines.append(f"한국 위치 분석: {inp.korea_location} (신뢰도 {inp.korea_confidence:.0%})")
    if inp.poi_matched:
        lines.append(f"장소 확인: {inp.poi_location} ({inp.poi_source})")
    if inp.license_plate_country:
        lines.append(f"번호판 국가: {inp.license_plate_country}")
    if inp.infra_top_country:
        lines.append(f"인프라 분석: {inp.infra_top_country} (점수 {inp.infra_score:.1f})")
    if inp.geoclip_location:
        lines.append(f"GeoCLIP: {inp.geoclip_location} (점수 {inp.geoclip_score:.2f})")
    if inp.vpr_similarity > 0:
        lines.append(f"VPR 매칭: {inp.vpr_location} (유사도 {inp.vpr_similarity:.0%})")
    if inp.latitude_band:
        lines.append(f"물리 역산: {inp.latitude_band}, {inp.hemisphere}")
    if inp.season:
        lines.append(f"계절 추정: {inp.season}")
    return lines


def _build_contradiction_notes(inp: EnsembleInput) -> list[str]:
    notes = []
    # 언어와 인프라 불일치 체크
    if inp.detected_languages and inp.infra_top_country:
        langs = set(inp.detected_languages)
        country = inp.infra_top_country
        if "ko" in langs and country not in ("한국", ""):
            notes.append(f"주의: 한국어 텍스트 vs 인프라 분석 ({country}) 불일치")
        if "ja" in langs and country not in ("일본", ""):
            notes.append(f"주의: 일본어 텍스트 vs 인프라 분석 ({country}) 불일치")
    return notes


def _assess_quality(inp: EnsembleInput) -> str:
    sources = sum([
        bool(inp.has_gps),
        bool(inp.ocr_texts),
        bool(inp.poi_matched),
        bool(inp.reverse_search_hits or inp.osint_web_hits or inp.osint_blog_hits),
        bool(inp.geoclip_score > 0.3),
        bool(inp.infra_top_country),
        bool(inp.vpr_similarity > 0.5 or inp.osint_street_view_matched),
        bool(inp.osint_poi_hits > 0),
        bool(inp.korea_confidence >= 0.70),
        bool(inp.transit_match),
        bool(inp.skyline_match),
    ])
    if sources >= 4: return "충분"
    if sources >= 2: return "제한적"
    return "불충분"


def _languages_to_countries(languages: list[str]) -> str:
    m = {"ko": "한국", "ja": "일본", "zh": "중국", "en": "영어권", "ar": "아랍권", "th": "태국"}
    return ", ".join(m.get(l, l) for l in languages if l in m)

"""
신뢰도 계산 설계
신뢰도 = min(독립소스 수렴도, 최강단서 상한) × 반증 패널티
독립성 원칙: 같은 출처 N개 단서 = 독립 소스 1개
"""
from ..agents.hypothesis_tree import Hypothesis, Evidence


# 단서 유형별 신뢰도 상한 (단일 소스 — stage7_ensemble.py도 이 dict를 참조)
CONFIDENCE_CEILING = {
    "gps_exif_verified": 0.99,
    "street_view_match": 0.95,
    "ocr_poi_verified": 0.92,
    "reverse_search_blog": 0.85,
    "geoclip_single": 0.75,
    "infra_fingerprint": 0.70,
    "llm_osint_confirmed": 0.72,   # LLM이 OSINT로 3+ 독립 소스 확인
    "physical_calculation": 0.60,
    "vegetation_terrain": 0.50,
    "inference_only": 0.40,
}

# 독립 소스 수렴 보너스
CONVERGENCE_BONUS = {
    1: 1.0,
    2: 1.1,
    3: 1.2,
    4: 1.3,
}

# 반증 패널티
CONTRADICTION_PENALTY = {
    0: 1.0,
    1: 0.7,
    2: 0.4,
}

MAX_CONFIDENCE = 0.99


class ConfidenceCalculator:
    def calculate(
        self,
        hypothesis: Hypothesis,
        evidence_log: list[Evidence],
    ) -> float:
        if not evidence_log:
            return 0.0

        # 1. 최강 단서의 상한 결정
        ceiling = self._get_ceiling(evidence_log)

        # 2. 독립 소스 수렴 보너스
        independent_sources = self._count_independent_sources(evidence_log)
        bonus = CONVERGENCE_BONUS.get(min(independent_sources, 4), 1.3)

        # 3. 반증 패널티
        contradictions = sum(1 for ev in evidence_log if ev.is_contradiction)
        penalty = CONTRADICTION_PENALTY.get(min(contradictions, 2), 0.0)
        if contradictions >= 3:
            return 0.0  # 결론 보류

        # 4. 가설 자체 확률 반영
        base = hypothesis.probability

        # 5. 최종 계산
        confidence = min(base, ceiling) * bonus * penalty
        return min(confidence, MAX_CONFIDENCE)

    def _get_ceiling(self, evidence_log: list[Evidence]) -> float:
        # elif 대신 if/max 사용: 복수 소스가 있을 때 최고 상한을 올바르게 선택
        ceiling = 0.40  # 기본값: 추론만
        high_evidence = [ev for ev in evidence_log if ev.confidence_level == "HIGH"]

        if any("GPS" in ev.description for ev in high_evidence):
            ceiling = max(ceiling, CONFIDENCE_CEILING["gps_exif_verified"])
        if any("Street View" in ev.description or "VPR" in ev.description or
               ev.source in ("street_view", "vpr") for ev in high_evidence):
            ceiling = max(ceiling, CONFIDENCE_CEILING["street_view_match"])
        if any(ev.source == "skyline_match" for ev in high_evidence):
            ceiling = max(ceiling, CONFIDENCE_CEILING["street_view_match"])
        if any("POI" in ev.description or "플레이스" in ev.description or
               ev.source in ("naver_place", "kakao_place", "osm_poi",
                             "interior_osint", "auto_chain", "biz_reg_lookup",
                             "phone_lookup", "korea_analyze", "juso_lookup",
                             "korea_specializer", "naver_local", "kakao_local",
                             "roadview_check") for ev in high_evidence):
            ceiling = max(ceiling, CONFIDENCE_CEILING["ocr_poi_verified"])
        if any(ev.source in ("transit_db", "clova_ocr") for ev in high_evidence):
            ceiling = max(ceiling, CONFIDENCE_CEILING["ocr_poi_verified"])
        if any(ev.source == "license_plate" for ev in high_evidence):
            ceiling = max(ceiling, CONFIDENCE_CEILING["infra_fingerprint"])
        if any(ev.source in ("reverse_search", "web_search", "naver_blog", "url_crawl",
                             "crawl_social", "reverse_chain", "auto_chain",
                             "naver_vision_ocr", "kakao_vision_ocr",
                             "naver_news", "flickr")
               for ev in evidence_log):
            ceiling = max(ceiling, CONFIDENCE_CEILING["reverse_search_blog"])
        if any(ev.source == "geoclip" for ev in evidence_log):
            ceiling = max(ceiling, CONFIDENCE_CEILING["geoclip_single"])
        if any(ev.source == "object_detect" for ev in evidence_log):
            ceiling = max(ceiling, CONFIDENCE_CEILING["infra_fingerprint"])
        if any(ev.source == "physical" for ev in evidence_log):
            ceiling = max(ceiling, CONFIDENCE_CEILING["physical_calculation"])

        return ceiling

    def _count_independent_sources(self, evidence_log: list[Evidence]) -> int:
        """같은 source_group의 단서들은 하나로 카운트"""
        source_groups: set[str] = set()
        for ev in evidence_log:
            if not ev.is_contradiction:
                # 소스 그룹화 (같은 블로그 여러 단서 = 1개)
                group = self._get_source_group(ev.source)
                source_groups.add(group)
        return len(source_groups)

    def _get_source_group(self, source: str) -> str:
        GROUP_MAP = {
            "exif": "exif",
            "ocr": "text_analysis",
            "naver_place": "map_api",
            "kakao_place": "map_api",
            "google_place": "map_api",
            "geoclip": "ai_embedding",
            "vpr": "ai_embedding",
            "reverse_search": "internet_search",
            "object_detect": "visual_analysis",
            "physical": "physical_analysis",
            # OSINT 확장 도구 — 인터넷 검색 그룹으로 통합
            "web_search": "internet_search",
            "naver_blog": "internet_search",
            "osm_poi": "map_api",
            "street_view": "ai_embedding",
            "url_crawl": "internet_search",
            # 체인 OSINT 도구
            "interior_osint": "map_api",   # Naver Place 체인 → map_api 독립 소스
            "auto_chain": "internet_search",
            "biz_reg_lookup": "internet_search",
            "phone_lookup": "internet_search",
            "crawl_social": "internet_search",
            "reverse_chain": "internet_search",
            # 한국 전용 도구
            "korea_analyze": "map_api",
            "juso_lookup": "map_api",
            "roadview_check": "ai_embedding",
            "korea_specializer": "map_api",
            # 신규 도구
            "license_plate": "visual_analysis",   # 번호판 지역 확정 (시각 분석 독립 소스)
            "naver_vision_ocr": "text_analysis",  # Naver Vision OCR
            "kakao_vision_ocr": "text_analysis",  # Kakao Vision OCR
            "kakao_vision_scene": "visual_analysis",
            # 신규 OSINT 도구 (A~G 업그레이드)
            "naver_news": "internet_search",       # 네이버 뉴스 (internet_search 그룹)
            "naver_local": "map_api",              # 네이버 로컬 API (map_api 독립 소스)
            "kakao_local": "map_api",              # 카카오 로컬 (기존 kakao_place와 동일 그룹)
            "flickr": "internet_search",           # Flickr 지오태그 (internet_search 그룹)
            "gdelt": "internet_search",            # GDELT 뉴스 이미지
            "osint_fuse": "map_api",               # OSINT 융합 (고신뢰 출처 취급)
            # Round 3 신규 도구 (A~G)
            "transit_db": "map_api",               # A. 대중교통 DB (map_api 독립 소스)
            "weather_cross": "physical_analysis",  # B. 날씨/계절 교차검증 (물리 분석 그룹)
            "skyline_match": "visual_analysis",    # C. 스카이라인 매칭 (시각 분석 그룹)
            "clova_ocr": "text_analysis",          # E. CLOVA OCR+NER (text_analysis 그룹)
            # F. AI 탐지는 source가 없음 (PreprocessResult 필드로 처리)
            # G. shadow_analysis는 physical 그룹 (기존 물리 분석과 동일)
        }
        return GROUP_MAP.get(source, source)

"""
파이프라인 오케스트레이터 v2
Stage 0~7 전체 + Stage 7 앙상블 통합 + 지식 그래프 업데이트
"""
import asyncio
import json
from loguru import logger

from .stage0_preprocess import run as stage0
from .stage1_exif import run as stage1
from .stage2_internet import run as stage2
from .stage3_ocr_gis import run as stage3
from .stage4_infra import run as stage4
from .stage5_embedding import run as stage5
from .stage6_physical import run as stage6
from .stage7_ensemble import run as stage7, EnsembleInput
from ..agents.investigator import EXXASInvestigator, InvestigationResult


class EXXASOrchestrator:
    def __init__(self):
        self.investigator = EXXASInvestigator()
        self._image_bytes: bytes = b""
        self._stage0_cache = None
        self._stage1_cache = None
        self._stage2_cache = None
        self._stage3_cache = None
        self._stage4_cache = None
        self._stage5_cache = None
        self._stage6_cache = None
        self._job_id: str = ""  # Celery 태스크에서 주입 가능
        self._register_tools()

    def _register_tools(self):
        inv = self.investigator

        inv.register_tool(
            "exif_extract",
            "EXIF 포렌식 완전 분석. GPS 좌표, 촬영 시간, 기기 정보, 플랫폼 경유 탐지, 썸네일 조작 탐지.",
            {"type": "object", "properties": {}},
            self._t_exif,
        )
        inv.register_tool(
            "ocr_extract",
            "다국어 OCR (PaddleOCR+EasyOCR). 간판/번호판/우편번호 → 네이버/카카오/Google 플레이스 교차검증.",
            {"type": "object", "properties": {
                "search_poi": {"type": "boolean", "description": "POI 검색 여부 (기본 True)"},
            }},
            self._t_ocr,
        )
        inv.register_tool(
            "object_detect",
            "YOLOv8 인프라 탐지. 신호등/소화전/차량/간판 등 170개국 DB 매칭 → 국가 추정.",
            {"type": "object", "properties": {}},
            self._t_infra,
        )
        inv.register_tool(
            "geoclip_embed",
            "GeoCLIP + StreetCLIP + VPR 임베딩. 전 세계 수억 장 벡터 DB와 유사도 매칭.",
            {"type": "object", "properties": {}},
            self._t_geoclip,
        )
        inv.register_tool(
            "reverse_image_search",
            "역방향 이미지 검색 8종 병렬 (Naver Vision/Kakao Vision/Google Lens/Yandex/TinEye/Bing/SmartLens) + Wayback Machine. 인터넷에서 동일/유사 이미지 탐색.",
            {"type": "object", "properties": {}},
            self._t_reverse,
        )
        inv.register_tool(
            "naver_place_search",
            "네이버 플레이스 검색. OCR로 탐지한 가게명/도로명으로 정확한 주소와 좌표 획득.",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "검색할 장소명"},
            }, "required": ["query"]},
            self._t_naver,
        )
        inv.register_tool(
            "kakao_place_search",
            "카카오맵 장소 검색. 로드뷰 매칭에 활용.",
            {"type": "object", "properties": {
                "query": {"type": "string"},
                "x": {"type": "number"},
                "y": {"type": "number"},
            }, "required": ["query"]},
            self._t_kakao,
        )
        inv.register_tool(
            "sun_analysis",
            "태양 그림자/고도 역산 → 위도 밴드 + 기상 교차검증. 달/별 야간 이미지도 분석.",
            {"type": "object", "properties": {
                "shadow_direction": {"type": "number", "description": "그림자 방향 (도)"},
                "shadow_length_ratio": {"type": "number", "description": "그림자/물체 길이 비율"},
                "lat": {"type": "number", "description": "후보 위도 (알고 있으면 입력)"},
                "lon": {"type": "number", "description": "후보 경도 (알고 있으면 입력)"},
            }},
            self._t_sun,
        )
        inv.register_tool(
            "vpr_compare",
            "Visual Place Recognition. Milvus VPR 벡터 DB + Street View 직접 매칭.",
            {"type": "object", "properties": {
                "target_address": {"type": "string"},
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "radius_m": {"type": "number"},
            }},
            self._t_vpr,
        )
        inv.register_tool(
            "web_search",
            "DuckDuckGo/SerpAPI 웹 검색. 장소명·텍스트·번호판 키워드로 위치 정보 탐색. "
            "OCR로 간판명 발견 시 '<간판명> 위치 주소' 형태로 즉시 검색하라.",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "검색 쿼리"},
            }, "required": ["query"]},
            self._t_web_search,
        )
        inv.register_tool(
            "search_naver_blog",
            "네이버 블로그 검색. 장소 이름·가게명으로 방문 후기/사진 검색 → 주소·좌표 획득. "
            "OCR 텍스트나 POI 이름 발견 시 반드시 실행하라.",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "검색어 (가게명, 장소명 등)"},
            }, "required": ["query"]},
            self._t_naver_blog,
        )
        inv.register_tool(
            "osm_poi_search",
            "OpenStreetMap Overpass API. 좌표 반경 내 POI 검색. "
            "후보 좌표 확보 후 주변 특징물로 위치 확정 시 사용.",
            {"type": "object", "properties": {
                "query": {"type": "string"},
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "radius_m": {"type": "number", "description": "기본 500"},
            }, "required": ["query", "lat", "lon"]},
            self._t_osm_poi,
        )
        inv.register_tool(
            "street_view_fetch",
            "Mapillary API로 후보 좌표 근처 실제 거리 사진 수집. "
            "좌표 후보 확보 시 시각 비교/VPR 매칭을 위해 반드시 실행.",
            {"type": "object", "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "radius_m": {"type": "number", "description": "기본 150"},
            }, "required": ["lat", "lon"]},
            self._t_street_view,
        )
        inv.register_tool(
            "deep_crawl_url",
            "URL 딥 크롤. geo.position/OG 태그, 네이버/카카오 지도 링크, 주소 키워드 파싱. "
            "역방향 검색 결과 URL에 즉시 적용하라.",
            {"type": "object", "properties": {
                "url": {"type": "string"},
            }, "required": ["url"]},
            self._t_deep_crawl,
        )
        inv.register_tool(
            "knowledge_graph_query",
            "과거 분석 지식 그래프 조회. 가설 위치명을 입력하면 이전 수사에서 같은 위치가 몇 번 확인됐는지, "
            "평균 신뢰도는 얼마인지 반환. 위치 후보를 좁힐 때 활용.",
            {"type": "object", "properties": {
                "location": {"type": "string", "description": "조회할 위치명"},
            }, "required": ["location"]},
            self._t_knowledge_graph,
        )
        inv.register_tool(
            "receipt_parse",
            "영수증/메뉴/명함/라벨 이미지 전문 분석. OCR 텍스트에서 전화번호→지역코드, "
            "사업자등록번호, 도로명주소, 상호명, 브랜드, 통화/가격 형식을 추출해 위치 특정. "
            "실내/소형 객체 이미지에 최우선 실행하라.",
            {"type": "object", "properties": {}},
            self._t_receipt_parse,
        )
        inv.register_tool(
            "brand_locate",
            "브랜드명/상호명으로 본사·지점 위치 추적. DB 즉시 조회 후 웹 검색으로 주소·좌표 획득. "
            "간판/포장지/영수증에서 상호명 발견 시 즉시 실행.",
            {"type": "object", "properties": {
                "brand": {"type": "string", "description": "브랜드명 또는 상호명"},
            }, "required": ["brand"]},
            self._t_brand_locate,
        )
        inv.register_tool(
            "barcode_lookup",
            "바코드/QR코드 번호로 제품 정보·제조국 조회 (Open Food Facts + GS1 국가코드). "
            "포장지·영수증·라벨에서 숫자 코드 발견 시 즉시 실행.",
            {"type": "object", "properties": {
                "barcode": {"type": "string", "description": "바코드 또는 QR 코드 값"},
            }, "required": ["barcode"]},
            self._t_barcode_lookup,
        )
        inv.register_tool(
            "interior_osint",
            "실내/소형 객체 이미지 전용 통합 OSINT. OCR 텍스트 신호(전화번호·브랜드·메뉴·가격)로 "
            "최적화된 검색 쿼리를 자동 생성 후 Naver Place + 웹 검색 + 블로그를 병렬 실행. "
            "구체적인 장소명·주소·좌표까지 획득. 실내/문서/소형객체 이미지 최우선 실행.",
            {"type": "object", "properties": {}},
            self._t_interior_osint,
        )
        inv.register_tool(
            "auto_chain",
            "단서 하나에서 OSINT 체인을 자동 실행. phone→지역코드→상호검색→Naver Place, "
            "biz_reg→사업자조회→상호→위치, store/brand→Naver Place+웹+블로그+2차검색, "
            "address→Naver Place+OSM, menu→맛집검색, url→SNS크롤+위치추출. "
            "발견한 단서가 있으면 즉시 이 도구로 체인을 시작하라.",
            {"type": "object", "properties": {
                "clue_type": {"type": "string",
                    "description": "phone/biz_reg/store/brand/address/menu/url 중 하나"},
                "value": {"type": "string", "description": "단서 값"},
                "region_hint": {"type": "string", "description": "지역 힌트 (선택)"},
            }, "required": ["clue_type", "value"]},
            self._t_auto_chain,
        )
        inv.register_tool(
            "biz_reg_lookup",
            "한국 사업자등록번호(000-00-00000)로 상호명·주소 조회. "
            "영수증/문서에서 사업자번호 발견 시 즉시 실행.",
            {"type": "object", "properties": {
                "reg_number": {"type": "string", "description": "사업자등록번호"},
            }, "required": ["reg_number"]},
            self._t_biz_reg_lookup,
        )
        inv.register_tool(
            "phone_lookup",
            "전화번호로 상호명·주소 조회 + 지역코드 → 도시 확정. "
            "영수증/간판/명함에서 전화번호 발견 시 즉시 실행.",
            {"type": "object", "properties": {
                "phone": {"type": "string", "description": "전화번호"},
            }, "required": ["phone"]},
            self._t_phone_lookup,
        )
        inv.register_tool(
            "crawl_social",
            "SNS/블로그 URL에서 위치 정보 심층 추출. Instagram 위치태그, Naver 블로그 내장지도, "
            "YouTube 메타데이터, 카카오맵 링크 파싱. 역방향 검색에서 SNS URL 발견 시 즉시 실행.",
            {"type": "object", "properties": {
                "url": {"type": "string", "description": "SNS 또는 블로그 URL"},
            }, "required": ["url"]},
            self._t_crawl_social,
        )
        inv.register_tool(
            "reverse_chain",
            "역방향 이미지 검색 결과 URL들을 자동으로 체인 크롤. "
            "SNS→블로그→뉴스 순서로 우선 처리, 각 페이지에서 좌표·위치태그·지도링크 추출. "
            "reverse_image_search 실행 후 결과 URL이 있으면 반드시 이 도구를 실행하라.",
            {"type": "object", "properties": {}},
            self._t_reverse_chain,
        )
        inv.register_tool(
            "korea_analyze",
            "한국 전용 위치 특화 분석. OCR 텍스트에서 지하철역/랜드마크/우편번호/도로명주소 탐지 → "
            "행안부 Juso API + 네이버/카카오 지오코딩으로 정밀 좌표 확정. "
            "한국어 텍스트가 감지되면 반드시 이 도구를 ocr_extract 직후 실행하라.",
            {"type": "object", "properties": {
                "texts": {"type": "array", "items": {"type": "string"},
                          "description": "OCR로 추출한 텍스트 목록"},
                "lat": {"type": "number", "description": "현재 후보 위도 (없으면 생략)"},
                "lon": {"type": "number", "description": "현재 후보 경도 (없으면 생략)"},
            }, "required": ["texts"]},
            self._t_korea_analyze,
        )
        inv.register_tool(
            "juso_lookup",
            "행정안전부 도로명주소 API로 한국 주소 → 정밀 GPS 좌표 변환. "
            "도로명주소/지번주소/건물명이 있으면 즉시 이 도구로 정밀 좌표를 획득하라.",
            {"type": "object", "properties": {
                "address": {"type": "string", "description": "변환할 한국 주소"},
            }, "required": ["address"]},
            self._t_juso_lookup,
        )
        inv.register_tool(
            "roadview_check",
            "네이버 로드뷰(Street View)로 후보 좌표 시각 검증. "
            "좌표 확정 후 실제 로드뷰 이미지 존재 여부를 확인해 신뢰도를 높인다.",
            {"type": "object", "properties": {
                "lat": {"type": "number", "description": "확인할 위도"},
                "lon": {"type": "number", "description": "확인할 경도"},
                "radius_m": {"type": "integer", "description": "탐색 반경 미터 (기본 100m)"},
            }, "required": ["lat", "lon"]},
            self._t_roadview_check,
        )
        inv.register_tool(
            "license_plate_lookup",
            "OCR에서 감지된 한국 번호판 패턴으로 시/도 즉시 확정. "
            "번호판(12가3456 형식)이 보이면 즉시 실행 → 지역 좌표 + 반경 반환. "
            "신형/구형/전기차/영업용 번호판 모두 지원.",
            {"type": "object", "properties": {
                "texts": {"type": "array", "items": {"type": "string"},
                          "description": "OCR로 추출한 텍스트 목록 (번호판 포함 가능성)"},
            }, "required": ["texts"]},
            self._t_license_plate,
        )
        inv.register_tool(
            "naver_news_search",
            "네이버 뉴스 검색 — 장소명 관련 보도 기사에서 날짜/위치 교차확인. "
            "장소명이나 이벤트가 확인됐을 때 실행해 날짜·위치 교차검증.",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "검색 쿼리 (장소명, 사건명 등)"},
            }, "required": ["query"]},
            self._t_naver_news,
        )
        inv.register_tool(
            "naver_local_search",
            "네이버 로컬(장소) 검색 API — POI명/도로명주소/좌표를 직접 반환. "
            "naver_place_search 보완 — 정형화된 좌표와 전화번호까지 획득.",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "장소명 또는 주소"},
            }, "required": ["query"]},
            self._t_naver_local,
        )
        inv.register_tool(
            "kakao_local_search",
            "카카오 키워드 장소 검색 — 영업중 POI, 도로명주소, 좌표 반환. "
            "카카오맵 DB를 직접 조회해 네이버와 교차검증.",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "검색 키워드"},
                "lat": {"type": "number", "description": "중심 위도 (선택)"},
                "lon": {"type": "number", "description": "중심 경도 (선택)"},
            }, "required": ["query"]},
            self._t_kakao_local,
        )
        inv.register_tool(
            "flickr_search",
            "Flickr 지오태그 사진 검색 — 좌표 반경 내 공개 사진 태그/제목에서 지명 추출. "
            "후보 좌표 확보 후 주변 랜드마크 교차검증에 활용.",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "텍스트 검색어 (선택)"},
                "lat": {"type": "number", "description": "중심 위도 (선택)"},
                "lon": {"type": "number", "description": "중심 경도 (선택)"},
                "radius_km": {"type": "number", "description": "검색 반경 km (기본 5)"},
            }},
            self._t_flickr,
        )
        inv.register_tool(
            "news_image_search",
            "뉴스 이미지 역추적 — GDELT/네이버 뉴스 이미지 인덱스로 동일 장소 보도 사진 탐색. "
            "시위/사고/행사 사진에서 날짜+위치 동시 확정.",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "장소명 또는 이벤트 키워드"},
            }, "required": ["query"]},
            self._t_news_image,
        )
        inv.register_tool(
            "osint_fuse",
            "복수 OSINT 결과 융합 — 여러 도구의 결과를 DBSCAN 클러스터링으로 통합. "
            "독립 소스 3개 이상 수렴 시 신뢰도 자동 승격. "
            "수사 후반부 모든 결과를 취합할 때 실행.",
            {"type": "object", "properties": {
                "results": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "각 OSINT 도구 결과 배열 "
                                   "[{source, name?, address?, lat?, lon?}, ...]",
                },
            }, "required": ["results"]},
            self._t_osint_fuse,
        )
        # ── A. 대중교통 DB ─────────────────────────────────
        inv.register_tool(
            "transit_lookup",
            "한국 대중교통 DB 직접 매칭. OCR 텍스트에서 버스 번호/지하철역 탐지 → "
            "운행 도시/좌표 즉시 확정. 이미지에 버스/전철 관련 텍스트나 표지판 있으면 즉시 실행.",
            {"type": "object", "properties": {
                "texts": {"type": "array", "items": {"type": "string"},
                          "description": "OCR 텍스트 목록"},
                "query": {"type": "string", "description": "역명/정류장 검색어 (선택)"},
            }, "required": ["texts"]},
            self._t_transit_lookup,
        )
        # ── B. 날씨/계절 교차검증 ──────────────────────────
        inv.register_tool(
            "weather_cross_check",
            "이미지에서 계절/날씨/시간대/복장 분석 → Open-Meteo 기상 DB 교차검증. "
            "계절이나 날씨 단서로 촬영 지역/시기를 좁힐 때 실행. "
            "벚꽃/단풍/눈 등 뚜렷한 계절 특징이 있으면 즉시 실행.",
            {"type": "object", "properties": {
                "lat": {"type": "number", "description": "후보 위도 (없으면 생략)"},
                "lon": {"type": "number", "description": "후보 경도 (없으면 생략)"},
            }},
            self._t_weather_cross,
        )
        # ── C. 스카이라인 실루엣 매칭 ──────────────────────
        inv.register_tool(
            "skyline_match",
            "이미지 스카이라인 실루엣 → 한국 주요 도시 DB 비교. "
            "교량(광안대교)/타워(롯데월드타워/N서울타워)/해안선 패턴으로 도시 특정. "
            "도시 스카이라인이 보이는 이미지에서 실행.",
            {"type": "object", "properties": {}},
            self._t_skyline,
        )
        # ── E. CLOVA OCR + NER ─────────────────────────────
        inv.register_tool(
            "clova_ocr",
            "CLOVA OCR + NER 엔진 (한국어 특화). "
            "브랜드명/전화번호/도로명주소/지하철역/사업자번호를 자동 분류(NER). "
            "일반 OCR보다 한국어 정확도 높음. ocr_extract 실패/불충분 시 실행.",
            {"type": "object", "properties": {}},
            self._t_clova_ocr,
        )
        # ── G. 그림자 방위각 분석 ──────────────────────────
        inv.register_tool(
            "shadow_analysis",
            "이미지에서 그림자 방향 추출 → 태양 방위각 역산 → 반구/위도 범위 좁히기. "
            "야외 주간 이미지에서 뚜렷한 그림자가 있을 때 실행. "
            "현재 위치 후보 검증에도 활용.",
            {"type": "object", "properties": {
                "exif_datetime": {"type": "string", "description": "EXIF 촬영 시각 (있으면 정확도 향상)"},
                "lat": {"type": "number", "description": "후보 위도 (선택)"},
                "lon": {"type": "number", "description": "후보 경도 (선택)"},
            }},
            self._t_shadow_analysis,
        )

        inv.register_tool(
            "similar_location_search",
            "VPR 임베딩으로 유사한 외관의 한국 위치를 Milvus DB에서 검색. "
            "후보 위치 확보 후 시각적으로 비슷한 주변 지역을 탐색할 때 실행. "
            "좌표 후보가 있을 때 반경 내 유사 장소 비교로 최종 위치를 좁힌다.",
            {"type": "object", "properties": {
                "lat": {"type": "number", "description": "중심 위도"},
                "lon": {"type": "number", "description": "중심 경도"},
                "radius_km": {"type": "number", "description": "검색 반경 km (기본 5)"},
                "top_k": {"type": "integer", "description": "결과 개수 (기본 5)"},
            }, "required": ["lat", "lon"]},
            self._t_similar_location,
        )

        # ── D 병렬 investigator용 레지스트리 구축 ───────────
        self._tool_registry = [
            (schema["name"], schema, inv._tools[schema["name"]])
            for schema in inv._tool_schemas
            if schema["name"] in inv._tools
        ]

    # progress_callback: async fn(stage_id, stage_name, status) 옵셔널
    _progress_cb = None

    async def analyze(
        self, image_bytes: bytes, image_media_type: str = "image/jpeg"
    ) -> InvestigationResult:
        self._image_bytes = image_bytes

        async def _stage_pub(sid: str, name: str, status: str):
            if self._progress_cb:
                try:
                    await self._progress_cb(sid, name, status)
                except Exception:
                    pass

        # ── Stage 0: 전처리 ──────────────────────────────
        logger.info("[Stage 0] Preprocessing")
        await _stage_pub("stage0", "이미지 검증 (ELA·해시)", "running")
        pre = await stage0(image_bytes)
        self._stage0_cache = pre
        await _stage_pub("stage0", "이미지 검증 (ELA·해시)", "done")

        # 야간 이미지 보정: CLAHE로 향상된 이미지를 나머지 파이프라인에 사용
        if pre.is_night_scene and pre.night_enhanced_bytes:
            logger.info(f"[Stage 0] Night scene (brightness={pre.brightness_mean:.1f}), CLAHE applied")
            analysis_bytes = pre.night_enhanced_bytes
        else:
            analysis_bytes = image_bytes

        # ── Stage 1: EXIF ────────────────────────────────
        logger.info("[Stage 1] EXIF forensics")
        await _stage_pub("stage1", "EXIF 포렌식 분석", "running")
        exif = await stage1(image_bytes)  # EXIF는 원본으로 (메타데이터 보존)
        self._stage1_cache = exif
        await _stage_pub("stage1", "EXIF 포렌식 분석", "done")

        manipulation_suspected = pre.manipulation_suspected or exif.prnu_anomaly_score > 0.5

        # ── GPS 즉시 확정 경로 ────────────────────────────
        if exif.has_gps and exif.gps and not manipulation_suspected:
            logger.info(f"[GPS Fast-path] {exif.gps.latitude:.6f},{exif.gps.longitude:.6f}")
            initial_ctx = {
                "exif": {"gps": exif.gps.__dict__},
                "has_gps": True,
                "manipulation_suspected": False,
            }
            result = await self.investigator.investigate(
                image_data=image_bytes,
                image_media_type=image_media_type,
                initial_context=initial_ctx,
            )
            result = await self._run_ensemble(result, exif, pre)
            asyncio.create_task(self._update_knowledge_graph(result, pre.phash, self._job_id))
            return result

        # ── Stage 3: OCR ──────────────────────────────────
        logger.info("[Stage 3] OCR + GIS")
        await _stage_pub("stage3", "OCR 텍스트·GIS 분석", "running")
        try:
            import asyncio as _asyncio
            ocr = await _asyncio.wait_for(stage3(analysis_bytes), timeout=30.0)
            self._stage3_cache = ocr
        except Exception as e:
            logger.warning(f"Stage 3 failed: {e}")
            ocr = None
        await _stage_pub("stage3", "OCR 텍스트·GIS 분석", "done")

        # ── Stage 4: YOLO 인프라 탐지 ─────────────────────
        logger.info("[Stage 4] YOLO infra detection")
        await _stage_pub("stage4", "인프라·비전 AI 분석", "running")
        try:
            import asyncio as _asyncio
            infra = await _asyncio.wait_for(stage4(analysis_bytes), timeout=320.0)
            self._stage4_cache = infra
        except _asyncio.TimeoutError:
            logger.warning("Stage 4 timed out (>320s), skipping YOLO/CLIP")
            from .stage4_infra import InfraResult as _InfraResult
            infra = _InfraResult()
            self._stage4_cache = infra
        except Exception as e:
            logger.warning(f"Stage 4 failed: {e}")
            from .stage4_infra import InfraResult as _InfraResult
            infra = _InfraResult()
            self._stage4_cache = infra
        await _stage_pub("stage4", "인프라·비전 AI 분석", "done")

        # ── Stage 5: AI Embedding (geoclip/streetclip) ────
        logger.info("[Stage 5] AI embeddings")
        await _stage_pub("stage5", "AI 위치 임베딩 (GeoCLIP)", "running")
        try:
            import asyncio as _asyncio
            emb = await _asyncio.wait_for(stage5(analysis_bytes), timeout=360.0)
            self._stage5_cache = emb
        except Exception as e:
            logger.warning(f"Stage 5 failed or timed out: {e}")
            from .stage5_embedding import EmbeddingResult as _EmbResult
            emb = _EmbResult()
            self._stage5_cache = emb
        await _stage_pub("stage5", "AI 위치 임베딩 (GeoCLIP)", "done")

        # ── Stage 6: 물리 분석 ────────────────────────────
        logger.info("[Stage 6] Physical analysis")
        await _stage_pub("stage6", "물리 환경 분석 (태양·DEM)", "running")
        try:
            phys = await stage6(image_bytes)
            self._stage6_cache = phys
        except Exception as e:
            logger.warning(f"Stage 6 failed: {e}")
            phys = None

        # ── Stage 2: 역방향 이미지 검색 ──────────────────
        # GPS/OCR/번호판이 있는 경우에만 스킵 (AI 모델만으로는 신뢰 불가)
        has_definitive_signal = (
            exif.has_gps or
            (ocr and ocr.best_match) or          # POI 확인된 경우만 (텍스트 개수 기준 제거)
            (ocr and ocr.plate_country) or
            (ocr and len(ocr.all_texts) >= 5)    # 텍스트 5개 이상 → 강력한 간판 단서
        )
        await _stage_pub("stage6", "물리 환경 분석 (태양·DEM)", "done")

        if not has_definitive_signal:
            logger.info("[Stage 2] Reverse image search (always run without GPS/OCR)")
            await _stage_pub("stage2", "역방향 이미지 검색", "running")
            try:
                rev = await stage2(image_bytes, pre.phash)
                self._stage2_cache = rev
            except Exception as e:
                logger.warning(f"Stage 2 failed: {e}")
                rev = None
            await _stage_pub("stage2", "역방향 이미지 검색", "done")
        else:
            logger.info("[Stage 2] Skipped (GPS/OCR/plate detected)")
            await _stage_pub("stage2", "역방향 이미지 검색", "skipped")
            rev = None

        # ── F. AI 생성 이미지 조기 차단 ──────────────────────
        if pre.ai_generated_suspected and pre.ai_generated_score > 0.75:
            logger.warning(
                f"[Stage 0] AI 생성 이미지 의심 (score={pre.ai_generated_score:.2f}) "
                f"— 수사 계속하나 신뢰도 패널티 적용"
            )

        # ── 전체 분석 결과 → 초기 컨텍스트 구성 ──────────
        scene_type = _detect_scene_type(ocr, infra, exif)
        initial_ctx = _build_initial_context(exif, pre, ocr, infra, emb, phys, rev, manipulation_suspected, scene_type)
        # AI 생성 탐지 결과 컨텍스트에 추가
        initial_ctx["ai_generated_suspected"] = pre.ai_generated_suspected
        initial_ctx["ai_generated_score"] = pre.ai_generated_score

        # Fix I: Naver Vision Landmark 결과 → strong signal로 주입
        if rev and rev.naver_landmark:
            initial_ctx["naver_landmark"] = rev.naver_landmark
            initial_ctx["naver_landmark_lat"] = rev.naver_landmark_lat
            initial_ctx["naver_landmark_lon"] = rev.naver_landmark_lon
            logger.info(f"[Stage 2] Naver Landmark 감지: {rev.naver_landmark} → initial_ctx 주입")

        # ── LLM 수사관: 목표 검색만 수행 (D. 병렬 investigator 시도) ──
        await _stage_pub("investigate", "AI 수사관 OSINT 추론", "running")
        # 단서가 불명확할 때만 병렬 investigator 실행 (단서 충분하면 단일 investigator)
        _has_strong_initial = (
            exif.has_gps or
            (ocr and len(ocr.all_texts) >= 3) or
            (emb and emb.geoclip_score > 0.50)
        )
        # 병렬 investigator는 Groq 429 rate limit 유발 → 단일 investigator로 안정화
        # 추후 OpenAI/Claude 사용 시 병렬 재활성화 가능
        _ = _has_strong_initial  # noqa: F841
        result = await self.investigator.investigate(
            image_data=image_bytes,
            image_media_type=image_media_type,
            initial_context=initial_ctx,
        )
        await _stage_pub("investigate", "AI 수사관 OSINT 추론", "done")

        # ── Stage 7: 앙상블 최종 판정 ─────────────────────
        await _stage_pub("stage7", "앙상블 최종 판정", "running")
        result = await self._run_ensemble(result, exif, pre)
        await _stage_pub("stage7", "앙상블 최종 판정", "done")

        asyncio.create_task(self._update_knowledge_graph(result, pre.phash, self._job_id))
        return result

    async def _run_ensemble(self, result: InvestigationResult, exif, pre) -> InvestigationResult:
        """Stage 7: 모든 캐시된 Stage 결과를 앙상블"""
        try:
            s2 = self._stage2_cache
            s3 = self._stage3_cache
            s4 = self._stage4_cache
            s5 = self._stage5_cache
            s6 = self._stage6_cache

            inp = EnsembleInput(
                has_gps=exif.has_gps,
                gps_lat=exif.gps.latitude if exif.gps else None,
                gps_lon=exif.gps.longitude if exif.gps else None,
                gps_verified=not pre.manipulation_suspected,
                exif_timezone=exif.timezone_estimate,
                exif_platform=exif.platform_hint,

                reverse_search_hits=len(s2.reverse_search_results) if s2 else 0,
                reverse_search_location=s2.location_hints[0] if s2 and s2.location_hints else "",
                reverse_search_confidence=s2.best_match.confidence if s2 and s2.best_match else 0.0,

                ocr_texts=[t.text for t in s3.all_texts[:10]] if s3 else [],
                detected_languages=s3.detected_languages if s3 else [],
                poi_matched=bool(s3 and s3.best_match),
                poi_location=s3.best_match.address if s3 and s3.best_match else "",
                poi_lat=s3.best_match.latitude if s3 and s3.best_match else None,
                poi_lon=s3.best_match.longitude if s3 and s3.best_match else None,
                poi_source=s3.best_match.source if s3 and s3.best_match else "",
                license_plate_country=s3.plate_country if s3 else "",

                infra_top_country=s4.top_country if s4 else "",
                infra_score=s4.top_country_score if s4 else 0.0,
                scene_type=_detect_scene_type(s3, s4, exif),

                geoclip_location=s5.geoclip_top_location if s5 else "",
                geoclip_lat=s5.geoclip_latitude if s5 else 0.0,
                geoclip_lon=s5.geoclip_longitude if s5 else 0.0,
                geoclip_score=s5.geoclip_score if s5 else 0.0,
                streetclip_country=s5.streetclip_country if s5 else "",
                # OpenCLIP 한국 세부 지역 결과 (NEW)
                openclip_korea_region=(s5.openclip_city_hint if s5 else ""),
                openclip_korea_score=(s5.openclip_score if s5 else 0.0),
                ensemble_korea_region=(getattr(s5, 'ensemble_region', '') if s5 else ""),
                ensemble_korea_confidence=(getattr(s5, 'ensemble_confidence', 0.0) if s5 else 0.0),
                vpr_location=s5.best_vpr_location if s5 else "",
                vpr_similarity=s5.best_vpr_similarity if s5 else 0.0,

                latitude_band=s6.sun.estimated_latitude_band if s6 else "",
                hemisphere=s6.hemisphere if s6 else "",
                season=s6.season_estimate if s6 else "",
                shadow_analysis=(
                    f"달:{s6.moon_star.moon_phase} 별:{s6.moon_star.star_constellation}"
                    if s6 and (s6.moon_star.moon_phase or s6.moon_star.star_constellation) else ""
                ),

                llm_location=result.location,
                llm_lat=result.latitude,
                llm_lon=result.longitude,
                llm_confidence=result.confidence,
                llm_evidence_count=len(result.evidence_chain),
                llm_contradiction_count=sum(
                    1 for e in result.evidence_chain if e.get("is_contradiction")
                ),
                llm_independent_sources=len({e["source"] for e in result.evidence_chain}),
                # OSINT 확장 도구 히트 수 집계
                osint_web_hits=sum(
                    1 for e in result.evidence_chain
                    if e.get("source") in ("web_search",) and not e.get("is_contradiction")
                ),
                osint_blog_hits=sum(
                    1 for e in result.evidence_chain
                    if e.get("source") in ("naver_blog",) and not e.get("is_contradiction")
                ),
                osint_poi_hits=sum(
                    1 for e in result.evidence_chain
                    if e.get("source") in ("osm_poi",) and not e.get("is_contradiction")
                ),
                osint_street_view_matched=any(
                    e.get("source") == "street_view" and
                    e.get("confidence_level") == "HIGH"
                    for e in result.evidence_chain
                ),

                # ── Round 3 도구 결과 ──
                transit_match=next(
                    (e.get("metadata", {}).get("city", "") for e in result.evidence_chain
                     if e.get("source") == "transit_db" and not e.get("is_contradiction")), ""
                ),
                skyline_match=next(
                    (e.get("metadata", {}).get("city", "") for e in result.evidence_chain
                     if e.get("source") == "skyline_match" and not e.get("is_contradiction")), ""
                ),
                skyline_confidence=next(
                    (e.get("metadata", {}).get("confidence", 0.0) for e in result.evidence_chain
                     if e.get("source") == "skyline_match" and not e.get("is_contradiction")), 0.0
                ),
                weather_season=next(
                    (e.get("metadata", {}).get("season", "") for e in result.evidence_chain
                     if e.get("source") == "weather_cross" and not e.get("is_contradiction")), ""
                ),
                clova_entities=next(
                    (e.get("metadata", {}).get("entities", {}) for e in result.evidence_chain
                     if e.get("source") == "clova_ocr" and not e.get("is_contradiction")), {}
                ),

                # ── 한국 전용 분석 (stage3 Korea specializer 결과) ──
                **_extract_korea_ensemble(s3),
            )

            # LLM OSINT 고신뢰 증거(naver_place / interior_osint / brand) → poi 필드 승격
            # ※ S3 OCR이 먼저 poi_matched=True로 설정해도, LLM 고신뢰 증거가 있으면 항상 덮어씀
            if result.latitude and result.longitude:
                _high_osint_sources = {"interior_osint", "naver_place", "brand", "receipt"}
                _high_ev = next(
                    (e for e in result.evidence_chain
                     if e.get("source") in _high_osint_sources
                     and e.get("confidence_level") == "HIGH"),
                    None,
                )
                if _high_ev:
                    inp.poi_matched = True
                    inp.poi_location = result.location
                    inp.poi_lat = result.latitude
                    inp.poi_lon = result.longitude
                    inp.poi_source = _high_ev.get("source", "llm_osint")
                    inp.llm_poi_source = inp.poi_source

            ens = stage7(inp)

            # 앙상블 결과로 업데이트 (신뢰도가 더 높을 때만)
            if ens.final_confidence >= result.confidence:
                result.confidence = ens.final_confidence
                result.confidence_label = ens.confidence_label
                if ens.final_lat is not None:
                    result.latitude = ens.final_lat
                if ens.final_lon is not None:
                    result.longitude = ens.final_lon
                if ens.final_location and ens.final_location != "위치 특정 불가":
                    result.location = ens.final_location

            logger.info(f"[Stage 7] Ensemble: conf={ens.final_confidence:.2%} quality={ens.exploration_quality}")

            # 고신뢰 결과는 VPR DB에 자동 저장 (자가학습)
            if ens.final_confidence >= 0.65 and ens.final_lat and ens.final_lon:
                s5 = self._stage5_cache
                if s5 and s5.embedding_vector:
                    asyncio.create_task(self._store_vpr_embedding(
                        embedding=s5.embedding_vector,
                        lat=ens.final_lat,
                        lon=ens.final_lon,
                        location=ens.final_location,
                    ))

        except Exception as e:
            logger.warning(f"Stage 7 ensemble failed (non-critical): {e}")

        return result

    async def _parallel_investigate(
        self,
        image_bytes: bytes,
        image_media_type: str,
        initial_ctx: dict,
    ) -> "InvestigationResult":
        """
        D. 멀티-에이전트 병렬 가설 탐색
        단서가 불명확할 때 3개 investigator가 서로 다른 초기 가설로 동시 탐색 후 앙상블
        """
        logger.info("[D] 병렬 investigator 3개 동시 실행")

        # 각 investigator에 다른 우선순위 힌트 주입
        ctx_geo = {**initial_ctx, "_focus": "geoclip"}   # GeoCLIP 좌표 중심
        ctx_ocr = {**initial_ctx, "_focus": "ocr"}       # OCR 텍스트 중심

        async def run_agent(ctx: dict, label: str):
            try:
                inv = EXXASInvestigator()
                # 기존 investigator의 도구 스키마/함수 공유 (재등록)
                for name, schema, fn in self._tool_registry:
                    inv.register_tool(name, schema.get("description", ""), schema, fn)
                result = await inv.investigate(
                    image_data=image_bytes,
                    image_media_type=image_media_type,
                    initial_context=ctx,
                )
                logger.debug(
                    f"[D] {label}: {result.location} conf={result.confidence:.2f}"
                )
                return result
            except Exception as e:
                logger.warning(f"[D] {label} 실패: {e}")
                return None

        # 병렬 실행 — 2개로 줄여 Groq rate limit 부하 감소, 타임아웃 150s
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    run_agent(ctx_geo, "agent_geo"),
                    run_agent(ctx_ocr, "agent_ocr"),
                    return_exceptions=True,
                ),
                timeout=150.0,
            )
        except asyncio.TimeoutError:
            logger.warning("[D] 병렬 investigator 타임아웃 — 단일 investigator로 폴백")
            return await self.investigator.investigate(
                image_data=image_bytes,
                image_media_type=image_media_type,
                initial_context=initial_ctx,
            )

        # 유효 결과만 필터링
        valid = [r for r in results if r and not isinstance(r, Exception) and r.confidence > 0]
        if not valid:
            return await self.investigator.investigate(
                image_data=image_bytes,
                image_media_type=image_media_type,
                initial_context=initial_ctx,
            )

        # 최고 신뢰도 결과 선택
        best = max(valid, key=lambda r: r.confidence)

        # 결과 앙상블 — osint_fuse로 좌표 수렴
        coord_results = [
            {"source": f"agent_{i}", "lat": r.latitude, "lon": r.longitude,
             "confidence": r.confidence, "name": r.location}
            for i, r in enumerate(valid)
            if r.latitude and r.longitude
        ]
        if len(coord_results) >= 2:
            try:
                from ..services.osint_chain import osint_fuse
                fused = await osint_fuse(coord_results)
                if fused.get("confidence", 0) > best.confidence:
                    best.latitude = fused["best_lat"]
                    best.longitude = fused["best_lon"]
                    best.confidence = fused["confidence"]
                    best.location = fused.get("best_location", best.location)
                    logger.info(
                        f"[D] 앙상블 결과: {best.location} "
                        f"conf={best.confidence:.2f} "
                        f"(agents={len(valid)})"
                    )
            except Exception as e:
                logger.debug(f"[D] osint_fuse 실패: {e}")

        return best

    # 도구 레지스트리 (D 병렬 investigator용 — 도구 함수 참조 저장)
    _tool_registry: list = []

    async def _store_vpr_embedding(self, embedding: list, lat: float, lon: float, location: str):
        """고신뢰 분석 결과 → Milvus VPR DB 자동 저장"""
        try:
            from pymilvus import MilvusClient
            import hashlib, json
            client = MilvusClient(uri="http://localhost:19530")
            col_name = "image_embeddings"
            if not client.has_collection(col_name):
                return
            img_hash = hashlib.md5(json.dumps(embedding[:16]).encode()).hexdigest()
            client.insert(col_name, [{
                "image_hash": img_hash,
                "latitude": float(lat),
                "longitude": float(lon),
                "location": location,
                "embedding": embedding,
            }])
            logger.debug(f"VPR self-learn: {location} ({lat:.4f},{lon:.4f}) 저장")
        except Exception as e:
            logger.debug(f"VPR store skipped: {type(e).__name__}")

    async def _update_knowledge_graph(self, result: InvestigationResult, image_hash: str, job_id: str = ""):
        try:
            from ..services.knowledge_graph import add_analysis_to_graph
            await asyncio.wait_for(
                add_analysis_to_graph(
                    job_id=job_id,
                    image_hash=image_hash,
                    location=result.location,
                    latitude=result.latitude,
                    longitude=result.longitude,
                    evidence_chain=result.evidence_chain,
                    confidence=result.confidence,
                ),
                timeout=5.0,
            )
        except Exception as e:
            logger.debug(f"Knowledge graph update: {e}")

    # ── 도구 구현 ──────────────────────────────────────────

    async def _t_exif(self, _):
        if self._stage1_cache:
            e = self._stage1_cache
            return {
                "gps": e.gps.__dict__ if e.gps else None,
                "datetime": e.datetime_original,
                "utc_offset": e.utc_offset,
                "timezone": e.timezone_estimate,
                "device": f"{e.make} {e.model}".strip(),
                "device_country": e.device_country_hint,
                "platform": e.platform_hint,
                "jpeg_quality": e.jpeg_quality,
                "manipulation": {
                    "thumbnail_mismatch": e.thumbnail_mismatch,
                    "prnu_anomaly_score": e.prnu_anomaly_score,
                    "prnu_fingerprint": e.prnu_fingerprint[:16] if e.prnu_fingerprint else "",
                },
            }
        e = await stage1(self._image_bytes)
        self._stage1_cache = e
        return {"gps": e.gps.__dict__ if e.gps else None, "timezone": e.timezone_estimate}

    async def _t_ocr(self, args):
        r = await stage3(self._image_bytes)
        self._stage3_cache = r
        return {
            "texts": [{"text": t.text, "confidence": round(t.confidence, 2)} for t in r.all_texts[:20]],
            "languages": r.detected_languages,
            "country_hints": r.country_hints,
            "license_plates": r.license_plates,
            "plate_country": r.plate_country,
            "postal_codes": r.postal_codes,
            "poi_matches": [
                {"name": p.name, "address": p.address, "lat": p.latitude, "lon": p.longitude, "source": p.source}
                for p in r.poi_matches[:5]
            ],
            "best_match": {
                "name": r.best_match.name, "address": r.best_match.address,
                "lat": r.best_match.latitude, "lon": r.best_match.longitude,
            } if r.best_match else None,
        }

    async def _t_infra(self, _):
        # stage4 캐시 사용 (파이프라인 사전 계산 결과 재활용)
        if self._stage4_cache is not None:
            r = self._stage4_cache
        else:
            import asyncio as _asyncio
            try:
                r = await _asyncio.wait_for(stage4(self._image_bytes), timeout=240.0)
            except _asyncio.TimeoutError:
                logger.warning("[_t_infra] stage4 timeout (>120s), returning empty result")
                from .stage4_infra import InfraResult
                r = InfraResult()
            self._stage4_cache = r
        return {
            "objects": [{"label": o.label, "confidence": round(o.confidence, 2), "category": o.category}
                        for o in r.objects[:30]],
            "top_country": r.top_country,
            "top_country_score": round(r.top_country_score, 3),
            "country_candidates": r.country_candidates[:5],
            "summary": r.infrastructure_summary,
            "climate_region": r.inferred_region,
        }

    async def _t_geoclip(self, _):
        if self._stage5_cache is not None:
            r = self._stage5_cache
        else:
            import asyncio as _asyncio
            try:
                r = await _asyncio.wait_for(stage5(self._image_bytes), timeout=360.0)
            except _asyncio.TimeoutError:
                logger.warning("[_t_geoclip] stage5 timeout (>180s), returning empty result")
                from .stage5_embedding import EmbeddingResult
                r = EmbeddingResult()
            self._stage5_cache = r
        return {
            "top_location": r.geoclip_top_location,
            "latitude": r.geoclip_latitude,
            "longitude": r.geoclip_longitude,
            "score": r.geoclip_score,
            "top5": r.geoclip_top5,
            "streetclip_country": r.streetclip_country,
            "vpr_best_match": r.best_vpr_location,
            "vpr_similarity": r.best_vpr_similarity,
        }

    async def _t_reverse(self, _):
        hash_val = self._stage0_cache.phash if self._stage0_cache else ""
        r = await stage2(self._image_bytes, hash_val)
        self._stage2_cache = r
        return {
            "results": [{"source": x.source, "url": x.url, "title": x.title, "location_hint": x.location_hint}
                        for x in r.reverse_search_results[:10]],
            "location_hints": r.location_hints,
            "wayback_first_seen": r.wayback_first_seen,
            "best_url": r.best_match.url if r.best_match else "",
        }

    async def _t_naver(self, args):
        from .stage3_ocr_gis import _naver_place_search
        places = await _naver_place_search(args.get("query", ""))
        return {"places": [{"name": p.name, "address": p.address, "lat": p.latitude, "lon": p.longitude}
                           for p in places[:5]], "found": bool(places)}

    async def _t_kakao(self, args):
        # 카카오 비즈앱 미전환 — 네이버로 위임
        from .stage3_ocr_gis import _naver_place_search
        places = await _naver_place_search(args.get("query", ""))
        return {"places": [{"name": p.name, "address": p.address, "lat": p.latitude, "lon": p.longitude}
                           for p in places[:5]], "found": bool(places)}

    async def _t_sun(self, args):
        exif_dt = self._stage1_cache.datetime_original if self._stage1_cache else ""
        # lat/lon이 args에 있으면 latitude_hint로 활용 (좌표 후보 보조)
        lat = args.get("lat", args.get("lat_hint", 0.0))
        r = await stage6(self._image_bytes, exif_datetime=exif_dt, latitude_hint=lat)
        self._stage6_cache = r
        return {
            "sun_elevation": r.sun.sun_elevation_degrees,
            "shadow_direction": r.sun.shadow_direction_degrees,
            "latitude_band": r.sun.estimated_latitude_band,
            "hemisphere": r.hemisphere,
            "season": r.season_estimate,
            "weather": {"temp_c": r.weather.temperature_c, "desc": r.weather.weather_description},
            "moon_star": {
                "moon_phase": r.moon_star.moon_phase,
                "estimated_date_range": r.moon_star.estimated_date_range,
                "star_hemisphere": r.moon_star.star_hemisphere,
                "star_constellation": r.moon_star.star_constellation,
                "hemisphere_hint": r.moon_star.hemisphere_hint,
            } if r.moon_star.moon_phase or r.moon_star.star_hemisphere else None,
            "dem": {
                "matched": r.dem_ridge_matched,
                "candidate_regions": r.dem_candidate_regions[:3],
            },
        }

    async def _t_vpr(self, args):
        # stage5 캐시 재사용 (geoclip_embed 이미 호출했으면 재실행 불필요)
        r = self._stage5_cache if self._stage5_cache else await stage5(self._image_bytes)
        self._stage5_cache = r
        return {
            "matches": r.vpr_matches[:5],
            "best_location": r.best_vpr_location,
            "similarity": r.best_vpr_similarity,
        }

    async def _t_web_search(self, args):
        from ..services.osint_chain import web_search, _extract_location_from_text
        query = args.get("query", "")
        if not query:
            return {"error": "query 파라미터 필요"}
        results = await web_search(query, max_results=6)
        # 위치 힌트 빠른 집계 (전체 텍스트 + 개별)
        all_text = " ".join(r.get("title", "") + " " + r.get("snippet", "") for r in results)
        location_hints = []
        all_hint = _extract_location_from_text(all_text)
        if all_hint:
            location_hints.append(all_hint)
        for r in results:
            hint = _extract_location_from_text(r.get("snippet", "") + " " + r.get("title", ""))
            if hint and hint not in location_hints:
                location_hints.append(hint)
        return {
            "results": [{"title": r.get("title",""), "url": r.get("url",""), "snippet": r.get("snippet","")[:200]} for r in results],
            "location_hints": list(dict.fromkeys(location_hints))[:5],
            "count": len(results),
        }

    async def _t_naver_blog(self, args):
        from ..services.osint_chain import search_naver_blog, _extract_location_from_text
        query = args.get("query", "")
        if not query:
            return {"error": "query 파라미터 필요"}
        results = await search_naver_blog(query, max_results=5)
        # 위치 힌트 집계
        all_text = " ".join(r.get("title", "") + " " + r.get("description", "") for r in results)
        top_hint = _extract_location_from_text(all_text)
        items = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "description": r.get("description", "")[:150],
                "location_hint": r.get("location_hint", ""),
            }
            for r in results
        ]
        return {
            "results": items,
            "top_location_hint": top_hint,
            "count": len(results),
        }

    async def _t_osm_poi(self, args):
        from ..services.osint_chain import osm_poi_search
        query = args.get("query", "")
        lat = args.get("lat", 0.0)
        lon = args.get("lon", 0.0)
        radius_m = int(args.get("radius_m", 500))
        if not query or (lat == 0 and lon == 0):
            return {"error": "query, lat, lon 파라미터 필요"}
        pois = await osm_poi_search(query, lat, lon, radius_m)
        return {
            "pois": pois[:8],
            "count": len(pois),
            "center": {"lat": lat, "lon": lon},
        }

    async def _t_street_view(self, args):
        from ..services.osint_chain import mapillary_nearby
        lat = args.get("lat", 0.0)
        lon = args.get("lon", 0.0)
        radius_m = int(args.get("radius_m", 150))
        if lat == 0 and lon == 0:
            return {"error": "lat, lon 파라미터 필요"}
        images = await mapillary_nearby(lat, lon, radius_m)
        return {
            "images": images[:5],
            "count": len(images),
            "note": (
                "이미지 thumb_url을 vpr_compare 또는 시각 비교에 활용하세요."
                if images else "이 반경에 Mapillary 이미지 없음. radius_m을 늘려 재시도하세요."
            ),
        }

    async def _t_deep_crawl(self, args):
        from ..services.osint_chain import deep_crawl_url
        url = args.get("url", "")
        if not url or not url.startswith("http"):
            return {"error": "유효한 URL 필요"}
        result = await deep_crawl_url(url)
        return result

    async def _t_auto_chain(self, args):
        """단서 → OSINT 자동 체인"""
        from ..services.osint_chain import auto_chain_from_clue
        clue_type = args.get("clue_type", "")
        value = args.get("value", "")
        region_hint = args.get("region_hint", "")
        if not clue_type or not value:
            return {"error": "clue_type과 value 필수"}
        return await auto_chain_from_clue(clue_type, value, region_hint)

    async def _t_biz_reg_lookup(self, args):
        from ..services.osint_chain import biz_reg_lookup
        reg = args.get("reg_number", "")
        if not reg:
            return {"error": "reg_number 필수"}
        return await biz_reg_lookup(reg)

    async def _t_phone_lookup(self, args):
        from ..services.osint_chain import phone_lookup
        phone = args.get("phone", "")
        if not phone:
            return {"error": "phone 필수"}
        return await phone_lookup(phone)

    async def _t_crawl_social(self, args):
        from ..services.osint_chain import crawl_social_location
        url = args.get("url", "")
        if not url or not url.startswith("http"):
            return {"error": "유효한 URL 필요"}
        return await crawl_social_location(url)

    async def _t_reverse_chain(self, _):
        """역방향 이미지 검색 결과 URL 자동 체인 크롤"""
        from ..services.osint_chain import chain_from_reverse_search
        s2 = self._stage2_cache
        if not s2 or not s2.reverse_search_results:
            return {"error": "역방향 검색 결과 없음 — reverse_image_search 먼저 실행"}
        urls = [{"url": r.url, "title": r.title} for r in s2.reverse_search_results if r.url]
        return await chain_from_reverse_search(urls)

    async def _t_receipt_parse(self, _):
        """영수증/문서/라벨 전문 파싱 — stage3 캐시 재사용"""
        from ..services.indoor_analyzer import (
            extract_receipt_signals, brand_quick_lookup,
            detect_document_type,
        )
        s3 = self._stage3_cache
        if not s3:
            try:
                s3 = await stage3(self._image_bytes)
                self._stage3_cache = s3
            except Exception as e:
                return {"error": f"OCR 실패: {e}"}

        text_list = [t.text for t in s3.all_texts]
        signals = extract_receipt_signals(text_list)
        brands = brand_quick_lookup(text_list)
        doc_type = detect_document_type(text_list)

        result = {
            "document_type": doc_type,
            "phone_numbers": signals["phone_numbers"],
            "phone_regions": signals["phone_regions"],      # ← 핵심: 지역코드 → 도시
            "business_reg_numbers": signals["business_reg_numbers"],
            "store_names": signals["store_names"],
            "addresses": signals["addresses"],
            "currency_hints": signals["currency_hints"],    # ← 통화 → 국가
            "brands": [{"brand": b, "location": l} for b, l in brands],
            "barcodes": s3.barcodes,
            "all_texts": [t.text for t in s3.all_texts[:20]],
        }
        # 분석 힌트 생성
        hints = []
        if signals["phone_regions"]:
            hints.append(f"전화번호 지역: {', '.join(signals['phone_regions'])}")
        if signals["addresses"]:
            hints.append(f"주소 발견: {signals['addresses'][0]}")
        if signals["store_names"]:
            hints.append(f"상호명: {', '.join(signals['store_names'][:3])}")
        if brands:
            hints.append(f"브랜드: {', '.join(b for b, _ in brands[:3])}")
        if signals["currency_hints"]:
            hints.append(f"통화/국가: {', '.join(signals['currency_hints'][:2])}")
        result["location_hints"] = hints
        return result

    async def _t_brand_locate(self, args):
        from ..services.indoor_analyzer import brand_web_search
        brand = args.get("brand", "")
        if not brand:
            return {"error": "brand 파라미터 필요"}
        result = await brand_web_search(brand)
        # 위치 힌트가 있으면 POI 검색도 시도
        if result.get("location_hints"):
            from .stage3_ocr_gis import _naver_place_search
            places = await _naver_place_search(brand)
            if places:
                p = places[0]
                result["poi"] = {"name": p.name, "address": p.address,
                                 "lat": p.latitude, "lon": p.longitude}
        return result

    async def _t_barcode_lookup(self, args):
        from ..services.indoor_analyzer import barcode_lookup
        barcode = args.get("barcode", "")
        if not barcode:
            return {"error": "barcode 파라미터 필요"}
        return await barcode_lookup(barcode)

    async def _t_interior_osint(self, _):
        """실내/소형 객체 통합 OSINT — stage3 캐시 재사용, 병렬 다각도 검색"""
        from ..services.indoor_analyzer import run_interior_osint
        s3 = self._stage3_cache
        if not s3:
            try:
                s3 = await stage3(self._image_bytes)
                self._stage3_cache = s3
            except Exception as e:
                return {"error": f"OCR 실패: {e}"}

        texts = [t.text for t in s3.all_texts]
        doc_type = getattr(s3, "document_type", "")
        brands = getattr(s3, "brand_names", [])
        phone_regions = getattr(s3, "phone_regions", [])
        currency_hints = getattr(s3, "currency_hints", [])
        barcodes = getattr(s3, "barcodes", [])

        result = await run_interior_osint(
            texts=texts,
            doc_type=doc_type,
            brands=brands,
            phone_regions=phone_regions,
            currency_hints=currency_hints,
            barcodes=barcodes,
        )

        # 요약 힌트 생성
        hints = []
        if result.get("best_guess"):
            hints.append(f"최유력 장소: {result['best_guess']}")
        for p in result.get("naver_places", [])[:3]:
            if p.get("address"):
                hints.append(f"{p['name']} — {p['address']}")
        for c in result.get("location_candidates", [])[:3]:
            hints.append(f"{c['location']} (score={c['score']:.1f})")
        result["location_hints"] = hints
        return result

    async def _t_knowledge_graph(self, args):
        location = args.get("location", "")
        if not location:
            return {"error": "location 파라미터 필요"}
        try:
            from ..services.knowledge_graph import (
                get_location_confidence_history,
                query_similar_locations,
            )
            history = await get_location_confidence_history(location)
            similar = await query_similar_locations(location, limit=5)
            return {
                "location": location,
                "past_analyses": history.get("total_analyses", 0),
                "avg_confidence": round(history.get("avg_confidence", 0) or 0, 3),
                "max_confidence": round(history.get("max_confidence", 0) or 0, 3),
                "similar_locations": similar,
                "note": (
                    f"이 위치는 과거 {history.get('total_analyses', 0)}회 분석에서 "
                    f"평균 {(history.get('avg_confidence') or 0)*100:.0f}% 신뢰도로 확인됨."
                    if history.get("total_analyses")
                    else "이 위치에 대한 과거 분석 이력 없음 (신규 위치)."
                ),
            }
        except Exception as e:
            return {"error": str(e), "location": location}

    async def _t_license_plate(self, args):
        """번호판 OCR → 시/도 즉시 지오코딩"""
        from ..services.license_plate import parse_license_plates_from_texts
        texts = args.get("texts", [])
        if not texts:
            return {"found": False, "message": "texts 파라미터 필요"}
        results = parse_license_plates_from_texts(texts)
        if not results:
            return {"found": False, "message": "번호판 패턴 미감지"}
        best = results[0]
        return {
            "found": True,
            "raw_text": best.raw_text,
            "region": best.region_code,
            "region_detail": best.region_detail,
            "latitude": best.latitude,
            "longitude": best.longitude,
            "radius_km": best.radius_km,
            "confidence": best.confidence,
            "plate_type": best.plate_type,
            "all_results": [
                {"raw": r.raw_text, "region": r.region_code, "lat": r.latitude, "lon": r.longitude}
                for r in results
            ],
        }

    # ── 한국 전용 도구 구현 ────────────────────────────────

    async def _t_korea_analyze(self, args):
        """한국 전용 위치 특화 분석"""
        from ..services.korea_specializer import analyze_korea_location
        texts = args.get("texts", [])
        lat = args.get("lat") or None
        lon = args.get("lon") or None
        if not texts:
            # stage3 캐시에서 가져오기
            s3 = self._stage3_cache
            if s3:
                texts = [t.text for t in s3.all_texts]
        if not texts:
            return {"error": "texts 파라미터 필요 또는 ocr_extract 먼저 실행"}
        result = await analyze_korea_location(texts, lat, lon)
        # 핵심 정보만 요약해서 반환
        summary = {
            "best_location": result.get("best_location", ""),
            "lat": result.get("lat"),
            "lon": result.get("lon"),
            "confidence": round(result.get("confidence", 0), 3),
            "address": result.get("address", ""),
            "city_hint": result.get("city_hint", ""),
        }
        if result.get("subway_station"):
            s = result["subway_station"]
            summary["subway_station"] = f"{s['name']}역 ({s['line']}, {s['city']})"
            summary["note"] = f"지하철역 정밀 확정: 신뢰도 {summary['confidence']:.0%}"
        if result.get("landmark"):
            lm = result["landmark"]
            summary["landmark"] = f"{lm['name']} ({lm['city']})"
        clues = result.get("clues", {})
        if clues.get("brands"):
            summary["brands"] = [b["brand"] for b in clues["brands"][:5]]
        if clues.get("postal_codes"):
            summary["postal_codes"] = [p["code"] for p in clues["postal_codes"][:3]]
        return summary

    async def _t_juso_lookup(self, args):
        """행정안전부 도로명주소 API + 지오코딩 → GPS 좌표"""
        from ..services.korea_specializer import search_juso_api, naver_geocode
        from ..core.config import settings
        address = args.get("address", "")
        if not address:
            return {"error": "address 파라미터 필요"}
        geo = None
        # 1순위: Juso API → 도로명주소 정규화 → 지오코딩
        if settings.JUSO_API_KEY:
            juso = await search_juso_api(address, settings.JUSO_API_KEY)
            if juso:
                road_addr = juso[0].get("road_address", "")
                if road_addr:
                    geo = await naver_geocode(road_addr)
                    if geo:
                        geo["juso_normalized"] = road_addr
                        geo["building_name"] = juso[0].get("building_name", "")
                        geo["zipcode"] = juso[0].get("zipcode", "")
        # 2순위: 직접 네이버 지오코딩
        if not geo:
            geo = await naver_geocode(address)

        if not geo or not geo.get("lat"):
            return {"error": f"'{address}' 주소 좌표 변환 실패", "address": address}
        return {
            "address": geo.get("address", address),
            "lat": geo.get("lat"),
            "lon": geo.get("lon"),
            "building_name": geo.get("building_name", ""),
            "zipcode": geo.get("zipcode", ""),
            "note": f"주소 지오코딩 성공: {geo.get('address','')} → ({geo.get('lat'):.4f},{geo.get('lon'):.4f})"
        }

    async def _t_roadview_check(self, args):
        """네이버 로드뷰 좌표 검증"""
        from ..services.korea_specializer import naver_roadview_check
        lat = args.get("lat", 0.0)
        lon = args.get("lon", 0.0)
        radius_m = int(args.get("radius_m", 100))
        if not lat or not lon:
            return {"error": "lat, lon 파라미터 필요"}
        result = await naver_roadview_check(lat, lon, radius_m)
        if result.get("available"):
            return {
                "available": True,
                "pano_id": result.get("pano_id"),
                "roadview_lat": result.get("lat"),
                "roadview_lon": result.get("lon"),
                "date": result.get("date", ""),
                "note": f"네이버 로드뷰 확인됨 — 이 위치 시각 검증 가능 (촬영: {result.get('date','')})",
            }
        return {
            "available": False,
            "note": f"이 좌표({lat:.4f},{lon:.4f}) 반경 {radius_m}m 내 로드뷰 없음. radius_m 늘려 재시도 권장.",
        }

    # ── 신규 OSINT 도구 구현 (A~G 업그레이드) ──────────────

    async def _t_naver_news(self, args):
        """네이버 뉴스 검색"""
        from ..services.osint_chain import naver_news_search
        query = args.get("query", "")
        if not query:
            return {"error": "query 필요"}
        results = await naver_news_search(query, max_results=5)
        location_hints = [r["location_hint"] for r in results if r.get("location_hint")]
        return {
            "results": results,
            "location_hints": location_hints,
            "best_hint": location_hints[0] if location_hints else "",
        }

    async def _t_naver_local(self, args):
        """네이버 로컬(장소) 검색 API"""
        from ..services.osint_chain import naver_local_search
        query = args.get("query", "")
        if not query:
            return {"error": "query 필요"}
        results = await naver_local_search(query, max_results=5)
        # 좌표 있는 결과 우선
        with_coords = [r for r in results if r.get("lat") and r.get("lon")]
        best = with_coords[0] if with_coords else (results[0] if results else {})
        return {
            "results": results,
            "best": best,
            "found": bool(best),
        }

    async def _t_kakao_local(self, args):
        """카카오 키워드 장소 검색"""
        from ..services.osint_chain import kakao_local_search
        query = args.get("query", "")
        lat = args.get("lat", 0.0) or 0.0
        lon = args.get("lon", 0.0) or 0.0
        if not query:
            return {"error": "query 필요"}
        results = await kakao_local_search(query, lat=lat, lon=lon, max_results=5)
        with_coords = [r for r in results if r.get("lat") and r.get("lon")]
        best = with_coords[0] if with_coords else (results[0] if results else {})
        return {
            "results": results,
            "best": best,
            "found": bool(best),
        }

    async def _t_flickr(self, args):
        """Flickr 지오태그 사진 검색"""
        from ..services.osint_chain import flickr_geo_search, flickr_text_search
        query = args.get("query", "")
        lat = args.get("lat") or 0.0
        lon = args.get("lon") or 0.0
        radius_km = args.get("radius_km", 5.0)

        results = []
        if lat and lon and 33 <= lat <= 38 and 124 <= lon <= 132:
            geo_results = await flickr_geo_search(lat, lon, radius_km=radius_km, max_results=6)
            results.extend(geo_results)
        if query:
            text_results = await flickr_text_search(query, max_results=4)
            results.extend(text_results)

        location_hints = [r["location_hint"] for r in results if r.get("location_hint")]
        with_coords = [r for r in results if r.get("lat") and r.get("lon")]
        return {
            "total": len(results),
            "with_coords": len(with_coords),
            "location_hints": list(dict.fromkeys(location_hints))[:5],
            "best_coords": {"lat": with_coords[0]["lat"], "lon": with_coords[0]["lon"]}
                if with_coords else {},
        }

    async def _t_news_image(self, args):
        """뉴스 이미지 역추적"""
        from ..services.osint_chain import naver_news_image_search
        query = args.get("query", "")
        if not query:
            return {"error": "query 필요"}
        results = await naver_news_image_search(query, max_results=5)
        location_hints = [r["location_hint"] for r in results if r.get("location_hint")]
        return {
            "results": results,
            "location_hints": list(dict.fromkeys(location_hints))[:5],
            "best_hint": location_hints[0] if location_hints else "",
        }

    async def _t_osint_fuse(self, args):
        """복수 OSINT 결과 융합"""
        from ..services.osint_chain import osint_fuse
        results = args.get("results", [])
        if not results:
            return {"error": "results 배열 필요"}
        fused = await osint_fuse(results)
        return fused

    # ── A. 대중교통 DB ──────────────────────────────────────
    async def _t_transit_lookup(self, args):
        """한국 대중교통 DB 매칭"""
        from ..services.transit_db import match_transit, transit_search_api
        texts = args.get("texts", [])
        query = args.get("query", "")
        if isinstance(texts, str):
            texts = [texts]
        # OCR 텍스트 직접 매칭
        match = match_transit(texts)
        if match and match.get("confidence", 0) >= 0.80:
            logger.debug(f"[transit] 직접 매칭: {match.get('name')} → {match.get('city')}")
            return match
        # 역명 검색
        if query:
            api_results = await transit_search_api(query)
            if api_results:
                return api_results[0]
        # OCR 텍스트에서 역명 추출 후 API 검색
        for text in texts:
            import re
            m = re.search(r'([가-힣]{2,6}역)', text)
            if m:
                api_results = await transit_search_api(m.group(1).rstrip("역"))
                if api_results:
                    return api_results[0]
        return {"found": False, "note": "대중교통 단서 없음"}

    # ── B. 날씨/계절 교차검증 ────────────────────────────────
    async def _t_weather_cross(self, args):
        """날씨/계절 시각 교차검증"""
        from ..services.weather_cross import weather_cross_check
        lat = args.get("lat") or (
            self.investigator._tools and None
        )
        lon = args.get("lon") or None
        result = await weather_cross_check(self._image_bytes, lat, lon)
        return result

    # ── C. 스카이라인 실루엣 매칭 ───────────────────────────
    async def _t_skyline(self, args):
        """스카이라인 실루엣 매칭"""
        from ..services.skyline_match import match_skyline_async
        result = await match_skyline_async(self._image_bytes)
        return result

    # ── E. CLOVA OCR + NER ──────────────────────────────────
    async def _t_clova_ocr(self, args):
        """CLOVA OCR + NER (한국어 특화)"""
        from ..services.osint_chain import clova_ocr
        result = await clova_ocr(self._image_bytes)
        # 추출된 NER 결과로 자동 OSINT 체인 트리거
        entities = result.get("entities", {})
        auto_results = {}
        # 브랜드 힌트 → naver_place_search
        brands = entities.get("brands", [])
        if brands:
            from .stage3_ocr_gis import _naver_place_search
            places = await _naver_place_search(brands[0])
            auto_results["brand_place"] = [
                {"name": p.name, "address": p.address, "lat": p.latitude, "lon": p.longitude}
                for p in places[:3]
            ]
        # 전화번호 → 지역 확정
        phones = entities.get("phones", [])
        if phones:
            from ..services.osint_chain import search_phone_region
            try:
                region = await search_phone_region(phones[0])
                auto_results["phone_region"] = region
            except Exception:
                pass
        result["auto_results"] = auto_results
        return result

    # ── 유사 위치 검색 ──────────────────────────────────────
    async def _t_similar_location(self, args: dict) -> dict:
        """VPR 임베딩으로 Milvus DB에서 시각적으로 유사한 위치 검색"""
        lat = args.get("lat")
        lon = args.get("lon")
        radius_km = float(args.get("radius_km", 5.0))
        top_k = int(args.get("top_k", 5))

        if lat is None or lon is None:
            return {"error": "lat, lon 파라미터 필요"}

        # stage5 캐시에서 임베딩 벡터 가져오기
        s5 = self._stage5_cache
        if s5 is None:
            try:
                import asyncio as _asyncio
                s5 = await _asyncio.wait_for(stage5(self._image_bytes), timeout=120.0)
                self._stage5_cache = s5
            except Exception as e:
                logger.warning(f"[similar_location] stage5 실패: {e}")

        embedding = s5.embedding_vector if s5 else []

        # Milvus VPR 검색
        milvus_results = []
        if embedding:
            try:
                from pymilvus import MilvusClient
                import math

                client = MilvusClient(uri="http://localhost:19530")
                col_name = "image_embeddings"
                if client.has_collection(col_name):
                    search_res = client.search(
                        collection_name=col_name,
                        data=[embedding],
                        limit=top_k * 3,  # 반경 필터 전 여유분
                        output_fields=["latitude", "longitude", "location", "image_hash"],
                        search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
                    )
                    if search_res and search_res[0]:
                        for hit in search_res[0]:
                            h_lat = hit.get("entity", {}).get("latitude", 0)
                            h_lon = hit.get("entity", {}).get("longitude", 0)
                            # 반경 필터 (Haversine 근사)
                            dlat = math.radians(h_lat - lat)
                            dlon = math.radians(h_lon - lon)
                            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat)) * math.cos(math.radians(h_lat)) * math.sin(dlon/2)**2
                            dist_km = 6371 * 2 * math.asin(math.sqrt(a))
                            if dist_km <= radius_km:
                                milvus_results.append({
                                    "location": hit.get("entity", {}).get("location", ""),
                                    "lat": h_lat,
                                    "lon": h_lon,
                                    "similarity": round(1 - hit.get("distance", 1), 3),
                                    "distance_km": round(dist_km, 2),
                                    "source": "vpr_milvus",
                                })
                        milvus_results = sorted(milvus_results, key=lambda x: -x["similarity"])[:top_k]
            except Exception as e:
                logger.debug(f"[similar_location] Milvus 검색 실패: {type(e).__name__}: {e}")

        # Milvus 결과 없으면 OSM + Naver 폴백
        if not milvus_results:
            try:
                from ..services.osint_chain import osm_poi_search
                radius_m = int(min(radius_km * 1000, 2000))
                pois = await osm_poi_search("", lat, lon, radius_m)
                for poi in pois[:top_k]:
                    milvus_results.append({
                        "location": poi.get("name", ""),
                        "lat": poi.get("lat", lat),
                        "lon": poi.get("lon", lon),
                        "similarity": 0.0,
                        "distance_km": poi.get("distance_m", 0) / 1000,
                        "source": "osm_fallback",
                    })
            except Exception as e:
                logger.debug(f"[similar_location] OSM 폴백 실패: {e}")

        return {
            "similar_locations": milvus_results,
            "count": len(milvus_results),
            "center": {"lat": lat, "lon": lon},
            "radius_km": radius_km,
            "note": (
                f"반경 {radius_km}km 내 시각적으로 유사한 장소 {len(milvus_results)}개 발견."
                if milvus_results
                else f"반경 {radius_km}km 내 유사 장소 없음 — radius_km를 늘려 재시도하거나 VPR DB 부족."
            ),
        }

    # ── G. 그림자 방위각 분석 ────────────────────────────────
    async def _t_shadow_analysis(self, args):
        """그림자 방위각 + 렌즈 지문 분석"""
        from ..pipeline.stage6_physical import analyze_shadow_azimuth, analyze_lens_fingerprint
        shadow = await analyze_shadow_azimuth(self._image_bytes)
        # 렌즈 지문 (EXIF에서)
        lens_info = {}
        if self._stage1_cache:
            exif = self._stage1_cache
            lens_info = analyze_lens_fingerprint({
                "make": exif.make,
                "model": exif.model,
                "focal_length": getattr(exif, "focal_length", 0),
                "focal_length_35": getattr(exif, "focal_length_35mm", 0),
                "max_aperture": getattr(exif, "max_aperture", 0),
            })
        return {
            "shadow": shadow,
            "lens": lens_info,
            "combined_hint": (
                f"그림자 방위각: {shadow.get('sun_azimuth_estimate', '?')}° "
                f"→ {shadow.get('hemisphere', '?')} / "
                f"카메라: {lens_info.get('brand_hint', '?')} ({lens_info.get('country_hint', '?')})"
                if shadow.get("shadow_detected") else "그림자 미감지"
            ),
        }


def _extract_korea_ensemble(s3) -> dict:
    """stage3 Korea specializer 결과를 EnsembleInput kwargs로 변환"""
    defaults = {
        "korea_confidence": 0.0,
        "korea_location": "",
        "korea_lat": None,
        "korea_lon": None,
        "korea_subway_station": "",
        "korea_roadview_available": False,
    }
    if not s3 or not s3.korea_analysis:
        return defaults
    kr = s3.korea_analysis
    subway = kr.get("subway_station")
    return {
        "korea_confidence": float(kr.get("confidence", 0.0)),
        "korea_location": kr.get("best_location", ""),
        "korea_lat": kr.get("lat"),
        "korea_lon": kr.get("lon"),
        "korea_subway_station": subway["name"] if subway else "",
        "korea_roadview_available": False,
    }


def _detect_scene_type(s3, s4, exif) -> str:
    """Stage 결과에서 장면 유형 파생"""
    # SNS 압축 플랫폼
    if exif and exif.platform_hint and any(
        k in exif.platform_hint for k in ("카카오", "Instagram", "Twitter", "라인")
    ):
        return "sns_compressed"

    # 텍스트가 많으면 도시
    if s3 and len(s3.all_texts) >= 3:
        return "urban"

    # 인프라 탐지 결과에서 자연/실내 힌트
    if s4:
        region = s4.inferred_region
        if "열대" in region or "한대" in region or "건조" in region:
            return "nature"
        labels = {obj.label.lower() for obj in s4.objects}
        if any(k in labels for k in ("bed", "chair", "couch", "refrigerator", "dining table")):
            return "indoor"
        if s4.top_country_score > 1.0:
            return "urban"

    return "default"


def _build_initial_context(exif, pre, ocr, infra, emb, phys, rev, manipulation_suspected, scene_type) -> dict:
    """모든 Stage 결과를 investigator 초기 컨텍스트로 변환"""
    ctx: dict = {
        "exif": {"gps": exif.gps.__dict__ if exif.gps else None},
        "has_gps": exif.has_gps,
        "manipulation_suspected": manipulation_suspected,
        "ela_score": pre.manipulation_score,
        "prnu_anomaly_score": exif.prnu_anomaly_score,
        "scene_type": scene_type,
        "device": f"{exif.make} {exif.model}".strip(),
        "platform": exif.platform_hint,
        "device_country_hint": exif.device_country_hint,
        "timezone_estimate": exif.timezone_estimate,
        # 한국 전용 서비스 — 기본 컨텍스트
        "country_context": "Korea",
        "service_mode": "korea_only",
    }

    # OCR 텍스트
    if ocr:
        ctx["ocr_texts"] = [t.text for t in ocr.all_texts[:15]]
        ctx["detected_languages"] = ocr.detected_languages
        ctx["license_plate_country"] = ocr.plate_country
        ctx["has_text_detected"] = len(ocr.all_texts) >= 3  # 3개 이상 텍스트일 때만 fast mode 활성화
        if ocr.best_match:
            ctx["poi_name"] = ocr.best_match.name
            ctx["poi_address"] = ocr.best_match.address
            ctx["poi_lat"] = ocr.best_match.latitude
            ctx["poi_lon"] = ocr.best_match.longitude
            ctx["poi_source"] = ocr.best_match.source

    # 인프라 (YOLO) + CLIP 시각 태그
    if infra:
        ctx["infra_top_country"] = infra.top_country
        ctx["infra_score"] = round(infra.top_country_score, 3)
        ctx["infra_objects"] = [obj.label for obj in infra.objects[:10]]
        ctx["infra_region"] = infra.inferred_region
        ctx["infra_candidates"] = infra.country_candidates[:3]
        ctx["scene_tags"] = infra.scene_tags
        ctx["scene_description"] = infra.scene_description
        if infra.vision_analysis:
            ctx["vision_analysis"] = infra.vision_analysis
            _lg = __import__("loguru").logger
            _lg.info(f"[Stage 4] Vision LLM analysis: {infra.vision_analysis[:150]}")
        # CLIP 한국 랜드마크 즉시 추출 — 즉시 결론 가능 신호
        from loguru import logger as _lg
        _lg.info(f"[Stage 4] CLIP scene tags: {infra.scene_tags}")
        for tag in infra.scene_tags:
            if tag.startswith("KOREA_LANDMARK:"):
                ctx["korea_landmark_clip"] = tag[len("KOREA_LANDMARK:"):]
                _lg.info(f"[Stage 4] KOREA_LANDMARK detected: {ctx["korea_landmark_clip"]}")
                break

    # AI 임베딩 (geoclip/streetclip)
    if emb:
        ctx["geoclip_location"] = emb.geoclip_top_location
        ctx["geoclip_score"] = round(emb.geoclip_score, 3)
        ctx["geoclip_lat"] = emb.geoclip_latitude
        ctx["geoclip_lon"] = emb.geoclip_longitude
        ctx["geoclip_top5"] = emb.geoclip_top5[:3]
        ctx["streetclip_country"] = emb.streetclip_country
        ctx["streetclip_score"] = round(emb.streetclip_score, 3)
        # OpenCLIP 한국 세부 지역 힌트 (NEW)
        ctx["openclip_korea_region"] = getattr(emb, 'openclip_city_hint', '')
        ctx["openclip_korea_score"] = round(getattr(emb, 'openclip_score', 0.0), 3)
        ctx["ensemble_korea_region"] = getattr(emb, 'ensemble_region', '')
        ctx["ensemble_korea_confidence"] = round(getattr(emb, 'ensemble_confidence', 0.0), 3)
        ctx["streetclip_top3"] = getattr(emb, 'streetclip_top3', [])

    # 물리 분석
    if phys:
        ctx["hemisphere"] = phys.hemisphere
        ctx["latitude_band"] = phys.sun.estimated_latitude_band if phys.sun else ""
        ctx["season"] = phys.season_estimate

    # 역방향 이미지 검색
    if rev:
        ctx["reverse_search_hints"] = rev.location_hints[:5]
        ctx["reverse_search_count"] = len(rev.reverse_search_results)
        top_titles = [
            r.title for r in rev.reverse_search_results
            if r.title and len(r.title) > 3
        ][:5]
        if top_titles:
            ctx["reverse_search_titles"] = top_titles
        # Naver Vision API 랜드마크 감지
        if getattr(rev, 'naver_landmark', ''):
            ctx["naver_landmark"] = rev.naver_landmark
            ctx["naver_landmark_lat"] = rev.naver_landmark_lat
            ctx["naver_landmark_lon"] = rev.naver_landmark_lon

    # ── 실내/소형 객체 특화 신호 ─────────────────────────────
    if ocr:
        if ocr.phone_regions:
            ctx["phone_regions"] = ocr.phone_regions
        if ocr.business_reg_numbers:
            ctx["business_reg_numbers"] = ocr.business_reg_numbers
        if ocr.brand_names:
            ctx["brand_names"] = [(b, l) for b, l in ocr.brand_names[:5]]
        if ocr.currency_hints:
            ctx["currency_hints"] = ocr.currency_hints
        if ocr.address_fragments:
            ctx["address_fragments"] = ocr.address_fragments[:3]
        if ocr.document_type and ocr.document_type != "unknown":
            ctx["document_type"] = ocr.document_type
        if ocr.barcodes:
            ctx["barcodes"] = ocr.barcodes[:5]

    # 실내 이미지 여부 감지 (CLIP 태그 기반)
    if infra and infra.scene_tags:
        indoor_tags = [t for t in infra.scene_tags if "indoor" in t.lower() or
                       t in ("receipt document paper", "food packaging product label",
                              "business card", "menu board", "small object close-up macro")]
        if indoor_tags:
            ctx["is_indoor"] = True
            ctx["indoor_tags"] = indoor_tags

    # ── 한국 전용 분석 컨텍스트 ──────────────────────────
    if ocr and ocr.korea_analysis:
        kr = ocr.korea_analysis
        ctx["korea_analysis"] = {
            "best_location": kr.get("best_location", ""),
            "lat": kr.get("lat"),
            "lon": kr.get("lon"),
            "confidence": round(kr.get("confidence", 0), 3),
            "city_hint": kr.get("city_hint", ""),
            "address": kr.get("address", ""),
        }
        if kr.get("subway_station"):
            s = kr["subway_station"]
            ctx["korea_subway_station"] = f"{s['name']}역 ({s.get('line','')}, {s.get('city','')})"
        if kr.get("landmark"):
            lm = kr["landmark"]
            ctx["korea_landmark"] = f"{lm['name']} ({lm.get('city','')})"
        clues = kr.get("clues", {})
        if clues.get("brands"):
            ctx["korea_brands"] = [b["brand"] for b in clues["brands"][:5]]
        if kr.get("best_location"):
            ctx["korea_city"] = kr["best_location"]

    return ctx

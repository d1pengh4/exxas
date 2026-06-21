"""
EXXAS LLM 수사관 v2 — 완전 업그레이드
ReAct 패턴 + 가설 트리 + Hallucination 방지 4종 + 탐색 모드 3종
"""
import json
import time
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Optional
from loguru import logger

from .hypothesis_tree import HypothesisTree, Evidence
from .llm_provider import get_llm_provider, LLMMessage
from ..services.confidence import ConfidenceCalculator
from ..core.config import settings


# ── Layer 1: 정체성 프롬프트 (고정) ───────────────────────
IDENTITY_PROMPT = """당신은 EXXAS — 사진 기반 위치 수사 AI입니다.
임무: 사진의 촬영 위치를 GPS 좌표까지 특정하세요. 모든 사진은 한국에서 촬영된 것으로 가정합니다.

■ 핵심 원칙
• 증거 우선순위: 번호판 > GPS > 간판텍스트 > POI검색 > AI추론
• 반드시 3개 이상의 독립 소스(지도API + 웹검색 + 로드뷰 또는 블로그)가 같은 장소를 확인해야 CONCLUDE 가능
• 도구를 한 번도 쓰지 않고 CONCLUDE 불가
• 좌표(위도/경도)가 없으면 CONCLUDE 불가 — 반드시 숫자 좌표를 확보할 것

■ 장면 유형별 최적 수사 순서
[해변/바다]: ①naver_place_search(해변명) → ②roadview_check(lat,lon) → ③web_search(해변명+특징) → CONCLUDE
[간판/텍스트]: ①naver_place_search(텍스트) → ②kakao_place_search(텍스트) → ③roadview_check(lat,lon) → CONCLUDE
[도시/건물]: ①web_search(특징+한국) → ②naver_place_search(추정지명) → ③roadview_check(lat,lon) → CONCLUDE
[좌표 획득 후]: roadview_check(lat,lon) 필수 실행 — 로드뷰로 3번째 독립 소스 확보

■ 독립 소스 그룹 (각 그룹에서 1개만 독립 소스로 인정)
- 지도API 그룹: naver_place_search, kakao_place_search, osm_poi_search (이 중 하나만 카운트)
- 웹검색 그룹: web_search, search_naver_blog (이 중 하나만 카운트)
- 로드뷰/거리뷰 그룹: roadview_check, street_view_fetch (독립 소스 추가)
→ 3그룹 사용 시 신뢰도 대폭 상승

■ 응답 형식
THINK: <단서목록 → 현재 가설 → 다음 행동 이유>
HYPOTHESIS: <"부산 해운대구 (75%)" 형식, 최대 3개>
ACTION: <도구명({"파라미터": "값"})>

결론 시:
ACTION: CONCLUDE
LOCATION: <최종 위치, 가능하면 도로명 주소>
LATITUDE: <숫자 위도, 예: 35.1581>
LONGITUDE: <숫자 경도, 예: 129.1584>
REASONING: <각 도구 결과를 인용하며 결론 도달 과정>"""


# ── 탐색 종료 조건 ────────────────────────────────────────
STOP_CONDITIONS = {
    "high_confidence": 0.95,    # 즉시 종료
    "medium_confidence": 0.70,  # 자기평가 후 종료
    "low_confidence": 0.30,     # 단서 소진 후 "불가" 출력
}

Tool = Callable[[dict], Awaitable[dict]]


@dataclass
class InvestigationStep:
    step_num: int
    think: str = ""
    hypothesis_update: str = ""
    action: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: Any = None
    evidence: Optional[Evidence] = None
    expected: str = ""
    elapsed_ms: int = 0


@dataclass
class InvestigationResult:
    location: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: str = ""
    confidence: float = 0.0
    confidence_label: str = ""
    steps: list[InvestigationStep] = field(default_factory=list)
    evidence_chain: list[dict] = field(default_factory=list)
    hypothesis_tree: dict = field(default_factory=dict)
    total_steps: int = 0
    elapsed_seconds: float = 0.0
    exploration_mode: str = ""
    hallucination_check_passed: bool = False
    final_reasoning: str = ""


class EXXASInvestigator:
    def __init__(self):
        self.llm = get_llm_provider()
        self.confidence_calc = ConfidenceCalculator()
        self._tools: dict[str, Tool] = {}
        self._tool_schemas: list[dict] = []
        # 공식 콜백 — tasks/analysis.py 에서 몽키패칭 없이 주입
        self.on_tool_start: Optional[Callable] = None   # async fn(step_num, tool_name)
        self.restricted_tools: set[str] = set()         # 플랜 제한 도구명

    def register_tool(self, name: str, description: str, parameters: dict, fn: Tool) -> None:
        self._tools[name] = fn
        self._tool_schemas.append({"name": name, "description": description, "parameters": parameters})

    async def investigate(
        self,
        image_data: bytes,
        image_media_type: str = "image/jpeg",
        initial_context: dict | None = None,
    ) -> InvestigationResult:
        start_time = time.time()
        result = InvestigationResult()
        tree = HypothesisTree()
        ctx = initial_context or {}

        # 초기 가설 — 한국 내 지역으로만 (균등)
        tree.initialize([
            "서울", "경기", "인천",
            "부산", "경남", "울산",
            "대구", "경북",
            "광주", "전남", "전북",
            "대전", "충남", "충북", "세종",
            "강원", "제주",
        ])

        mode = self._detect_mode(ctx)
        result.exploration_mode = mode
        logger.info(f"Investigation start — mode={mode}, llm={settings.LLM_PROVIDER}")

        messages: list[LLMMessage] = []
        used_tools: set[str] = set()
        all_steps: list[InvestigationStep] = []

        # ── GPS 즉시 확정 경로 ────────────────────────────
        if ctx.get("has_gps") and ctx.get("exif", {}).get("gps") and not ctx.get("manipulation_suspected"):
            gps = ctx["exif"]["gps"]
            lat, lon = gps["latitude"], gps["longitude"]
            result.latitude = lat
            result.longitude = lon
            result.confidence = 0.99
            result.confidence_label = "HIGH"
            result.hallucination_check_passed = True

            # 역방향 지오코딩으로 실제 주소 획득
            try:
                from ..services.geocoding import reverse_geocode
                addr = await reverse_geocode(lat, lon)
                if addr:
                    result.location = addr
                    result.address = addr
                else:
                    result.location = f"GPS ({lat:.6f}, {lon:.6f})"
            except Exception:
                result.location = f"GPS ({lat:.6f}, {lon:.6f})"

            result.final_reasoning = f"GPS EXIF 원본 좌표 확인 ({lat:.6f}, {lon:.6f}). 조작 탐지 통과. 최고 신뢰도."
            result.elapsed_seconds = round(time.time() - start_time, 1)
            return result

        # ── Groq vision pre-step: scout 모델로 이미지 → 텍스트 설명 ──
        vision_desc = await self._vision_describe_image(image_data, image_media_type)
        vision_search_candidate: str = ""
        if vision_desc:
            logger.info(f"[Vision pre-step] {vision_desc[:80]}...")
            # [불확실] 태그가 포함된 경우 해당 줄을 낮은 신뢰도로 표시
            _uncertain_lines = [l for l in vision_desc.split('\n') if '[불확실' in l or '[UNCERTAIN' in l]
            if _uncertain_lines:
                logger.debug(f"[Vision pre-step] 불확실 텍스트: {_uncertain_lines[:2]}")
            # Vision 설명에서 "X일 가능성", "X같은", "X처럼 보이는" 패턴의 불확실 지명 추출
            # [불확실] 태그 포함 줄은 검색 후보에서 제외
            _vision_for_candidate = '\n'.join(
                l for l in vision_desc.split('\n')
                if '[불확실' not in l and '[UNCERTAIN' not in l
            )
            _hedged_candidate = _extract_hedged_place_candidate(_vision_for_candidate)
            if _hedged_candidate:
                vision_search_candidate = _hedged_candidate
                logger.info(f"[Vision pre-step] 검색 후보(미확인): {_hedged_candidate}")

        # ── 초기 컨텍스트 메시지 주입 ─────────────────────
        if ctx and any(k in ctx for k in ("ocr_texts", "infra_top_country", "geoclip_location", "streetclip_country")):
            initial_msg = _build_context_summary(ctx)
            if vision_desc:
                candidate_hint = (
                    f"\n[Vision 검색 후보(미확인, 반드시 naver_place_search로 검증 필요): {vision_search_candidate}]"
                    if vision_search_candidate else ""
                )
                vision_header = (
                    "[시각 분석 (AI 장면 묘사) — 장소명은 부정확할 수 있음, 검색 도구로 확인 필수]\n"
                    + vision_desc
                    + candidate_hint
                )
                initial_msg = f"{vision_header}\n\n{initial_msg}"
            messages.append(LLMMessage(role="user", content=initial_msg))
            # 초기 가설을 컨텍스트 기반으로 업데이트
            _update_hypotheses_from_context(tree, ctx)
        elif vision_desc:
            # 파이프라인 결과가 없어도 이미지 설명은 주입
            candidate_hint2 = (
                f"\n[Vision 검색 후보(미확인, naver_place_search로 검증 필요): {vision_search_candidate}]"
                if vision_search_candidate else ""
            )
            vision_header = (
                "[시각 분석 (AI 장면 묘사) — 장소명은 부정확할 수 있음, 검색 도구로 확인 필수]\n"
                + vision_desc
                + candidate_hint2
            )
            messages.append(LLMMessage(role="user", content=vision_header))

        # ── ReAct 루프 ────────────────────────────────────
        last_coords: tuple[float, float] | None = None   # 마지막으로 확보된 좌표
        stall_count = 0

        for step_num in range(1, settings.MAX_INVESTIGATION_STEPS + 1):
            tree.step_count = step_num
            step_start = time.time()

            state = tree.to_state_prompt()
            action_prompt = self._build_action_prompt(
                step_num, state, used_tools, mode, last_coords=last_coords
            )

            # 413 방지: 초기 메시지(0번) 보존 + 최근 8개만 유지
            if len(messages) > 9:
                messages = messages[:1] + messages[-8:]

            # 첫 스텝: 이미지 포함 (vision 지원 모델용)
            img_bytes = image_data if (step_num == 1 and settings.LLM_PROVIDER in ("claude", "ollama")) else None
            messages.append(LLMMessage(role="user", content=action_prompt))

            response = await self.llm.complete(
                messages=messages,
                system=IDENTITY_PROMPT,
                tools=self._tool_schemas,
                image_data=img_bytes,
                image_media_type=image_media_type,
            )
            messages.append(LLMMessage(role="assistant", content=response.content))

            step = self._parse_step(response.content, step_num)
            step.elapsed_ms = int((time.time() - step_start) * 1000)

            # ── 도구 실행 헬퍼 (중복 방지) ──────────────────
            async def _run_tool_call(tc_name: str, tc_args, tc_id: str = "") -> None:
                nonlocal last_coords
                if not isinstance(tc_args, dict):
                    tc_args = {}
                used_tools.add(tc_name)
                step.tool_name = tc_name
                step.tool_args = tc_args
                logger.info(f"  [{step_num}] {tc_name}({list(tc_args.keys())})")
                if self.on_tool_start:
                    try:
                        await self.on_tool_start(step_num, tc_name)
                    except Exception:
                        pass
                tool_result = await self._exec_tool(tc_name, tc_args)
                step.tool_result = tool_result
                ev = self._to_evidence(tc_name, tool_result, tc_args)
                if ev:
                    tree.bayesian_update(ev)
                    step.evidence = ev
                self._inject_sub_hypotheses(response.content, tree)
                # ── 좌표 자동 추출 ──────────────────────────
                _lat = _lon = None
                if isinstance(tool_result, dict):
                    # naver_place_search / kakao_place_search
                    places = tool_result.get("places", [])
                    if places and isinstance(places[0], dict):
                        _lat = places[0].get("lat") or places[0].get("latitude")
                        _lon = places[0].get("lon") or places[0].get("longitude")
                    # roadview_check
                    if not _lat:
                        _lat = tool_result.get("roadview_lat") or tool_result.get("lat")
                        _lon = tool_result.get("roadview_lon") or tool_result.get("lon")
                    # juso_lookup / korea_analyze
                    if not _lat:
                        _lat = tool_result.get("latitude") or tool_result.get("lat")
                        _lon = tool_result.get("longitude") or tool_result.get("lon")
                if _lat and _lon:
                    try:
                        last_coords = (float(_lat), float(_lon))
                        logger.debug(f"  [{step_num}] 좌표 추출: {last_coords}")
                    except (ValueError, TypeError):
                        pass
                messages.append(LLMMessage(
                    role="user",
                    content=f"[도구결과:{tc_id}] {tc_name} → {json.dumps(tool_result, ensure_ascii=False)[:3000]}",
                ))

            # 네이티브 Function Calling (Claude / OpenAI)
            if response.has_tool_call:
                stall_count = 0
                for tc in response.tool_calls:
                    await _run_tool_call(tc["name"], tc.get("arguments", {}), tc.get("id", ""))

            # 텍스트 파싱 폴백 — Ollama 등 네이티브 tool call 미지원 모델
            elif not self._is_conclude(response.content):
                text_tc = self._parse_text_tool_call(response.content)
                if text_tc:
                    stall_count = 0
                    await _run_tool_call(text_tc["name"], text_tc["arguments"])
                else:
                    # stall 방지: 도구 호출도 CONCLUDE도 없는 응답
                    stall_count += 1
                    logger.debug(f"[Step {step_num}] stall #{stall_count} — no tool call or CONCLUDE")
                    # 2회 연속 stall 시 강제 도구 실행
                    if stall_count >= 2 and not used_tools:
                        _forced = _pick_forced_tool(ctx, used_tools, last_coords)
                        logger.info(f"  [{step_num}] FORCED tool: {_forced['name']}")
                        await _run_tool_call(_forced["name"], _forced["args"])
                        stall_count = 0
                    else:
                        _coord_hint = (
                            f" 좌표 ({last_coords[0]:.4f}, {last_coords[1]:.4f})가 확보됨 →"
                            f" roadview_check({{\"lat\": {last_coords[0]:.4f}, \"lon\": {last_coords[1]:.4f}}}) 실행 또는"
                            if last_coords else ""
                        )
                        messages.append(LLMMessage(
                            role="user",
                            content=(
                                f"[시스템]{_coord_hint} "
                                "ACTION: 도구명({\"파라미터\": \"값\"}) 형식으로 도구를 호출하거나 "
                                "ACTION: CONCLUDE로 결론을 내리세요."
                            )
                        ))

            # CONCLUDE 감지
            if self._is_conclude(response.content) and not response.has_tool_call:
                # 도구를 한 번도 사용하지 않았으면 CONCLUDE 차단
                if len(used_tools) == 0:
                    logger.debug(f"[Step {step_num}] CONCLUDE blocked — no tools used yet")
                    messages.append(LLMMessage(
                        role="user",
                        content=(
                            "[시스템 경고] 도구를 한 번도 사용하지 않고 결론을 낼 수 없습니다. "
                            "naver_place_search 또는 web_search로 위치를 먼저 확인한 뒤 결론을 내리세요. "
                            "위에 제공된 CLIP 장면 태그와 AI 추정 결과를 검색 쿼리로 활용하세요."
                        )
                    ))
                    all_steps.append(step)
                    continue

                parsed = self._parse_conclude(response.content)
                step.action = "CONCLUDE"
                all_steps.append(step)

                if parsed.get("location"):
                    # 오염 필터: LATITUDE:/LONGITUDE:/REASONING: 이후 내용 제거 (실제 줄바꿈 + \\n 모두 처리)
                    loc = parsed["location"]
                    loc = re.split(r'(?:\n|\\n)(?:LATITUDE|LONGITUDE|REASONING|ACTION)\s*:', loc, maxsplit=1)[0]
                    # 추가 방어: 줄바꿈 자체에서 잘라내기
                    loc = loc.split('\n')[0].strip()
                    if loc:
                        result.location = loc
                if parsed.get("latitude") and parsed["latitude"] != "UNKNOWN":
                    try:
                        result.latitude = float(parsed["latitude"])
                    except (ValueError, TypeError):
                        pass
                if parsed.get("longitude") and parsed["longitude"] != "UNKNOWN":
                    try:
                        result.longitude = float(parsed["longitude"])
                    except (ValueError, TypeError):
                        pass
                result.final_reasoning = parsed.get("reasoning", "")

                # ── 좌표 역방향 지오코딩 검증 ─────────────────────────
                # last_coords 또는 CONCLUDE 좌표로 실제 주소 확인
                _verify_lat = result.latitude or (last_coords[0] if last_coords else None)
                _verify_lon = result.longitude or (last_coords[1] if last_coords else None)
                if _verify_lat and _verify_lon:
                    # 한국 범위(124~132°E, 33~39°N) 내인지 먼저 체크
                    if 33.0 <= _verify_lat <= 39.0 and 124.0 <= _verify_lon <= 132.0:
                        try:
                            from ..services.geocoding import reverse_geocode
                            _real_addr = await reverse_geocode(_verify_lat, _verify_lon)
                            if _real_addr:
                                logger.info(f"[Reverse geocode] {_verify_lat:.4f},{_verify_lon:.4f} → {_real_addr}")
                                result.address = _real_addr
                                # LLM 제시 location과 역지오코딩 주소가 크게 다르면 역지오코딩 우선
                                if not result.location or len(_real_addr) > len(result.location):
                                    result.location = _real_addr
                        except Exception as _ge:
                            logger.debug(f"Reverse geocode failed: {_ge}")
                break

            all_steps.append(step)

            # 종료 조건
            max_conf = tree.get_max_confidence()
            if max_conf >= STOP_CONDITIONS["high_confidence"]:
                # 최소 2개 독립 소스 그룹 없이는 조기 종료 차단
                _has_map = bool(used_tools & {"naver_place_search", "kakao_place_search", "osm_poi_search", "juso_lookup", "korea_analyze"})
                _has_web = bool(used_tools & {"web_search", "search_naver_blog", "naver_news_search"})
                _has_view = bool(used_tools & {"roadview_check", "street_view_fetch", "vpr_compare"})
                _independent_groups = sum([_has_map, _has_web, _has_view])
                if _independent_groups < 2:
                    logger.info(f"High conf {max_conf:.2%} but only {_independent_groups} source group(s) — need ≥2, continuing")
                    all_steps.append(step)
                    continue
                logger.info(f"High confidence {max_conf:.2%} — forcing CONCLUDE")
                # 고신뢰도 조기종료: CONCLUDE를 LLM에 강제 요청
                if not result.location and step_num < settings.MAX_INVESTIGATION_STEPS:
                    _ev_summary = "; ".join(
                        f"{e.source}:{e.description[:50]}" for e in tree.evidence_log[:5]
                    )
                    _force_msg = (
                        f"[시스템] 신뢰도 {max_conf:.0%} 달성. 수집된 단서: {_ev_summary}\n"
                        f"지금 바로 ACTION: CONCLUDE 로 결론을 내리세요. "
                        f"LOCATION, LATITUDE, LONGITUDE, REASONING을 모두 포함하세요."
                    )
                    messages.append(LLMMessage(role="user", content=_force_msg))
                    _cr = await self.llm.complete(
                        messages=messages, system=IDENTITY_PROMPT,
                        tools=None,
                    )
                    _parsed = self._parse_conclude(_cr.content)
                    if _parsed.get("location"):
                        result.location = _parsed["location"]
                    if _parsed.get("latitude"):
                        try:
                            result.latitude = float(_parsed["latitude"])
                        except (ValueError, TypeError):
                            pass
                    if _parsed.get("longitude"):
                        try:
                            result.longitude = float(_parsed["longitude"])
                        except (ValueError, TypeError):
                            pass
                    if _parsed.get("reasoning"):
                        result.final_reasoning = _parsed["reasoning"]
                break
            if self._clues_exhausted(used_tools):
                logger.info("Clues exhausted")
                break

        result.steps = all_steps

        # ── 최종 신뢰도 계산 & 검증 ──────────────────────
        result = await self._finalize(result, tree, messages, image_data, image_media_type)
        result.elapsed_seconds = round(time.time() - start_time, 1)
        result.total_steps = len(all_steps)

        logger.info(
            f"Investigation done: {result.location} | conf={result.confidence:.2%} "
            f"| steps={result.total_steps} | {result.elapsed_seconds}s"
        )
        return result

    # ── 내부 헬퍼 ─────────────────────────────────────────

    def _detect_mode(self, ctx: dict) -> str:
        if ctx.get("has_gps"):
            return "fast"
        if ctx.get("has_text_detected") or ctx.get("has_license_plate"):
            return "fast"
        scene = ctx.get("scene_type", "")
        if scene in ("nature", "indoor"):
            return "inductive"
        return "elimination"

    def _build_action_prompt(self, step: int, state: str, used: set[str], mode: str,
                             last_coords: tuple[float, float] | None = None) -> str:
        mode_hint = {
            "fast": "강력 단서 존재. 해당 단서를 즉시 추적하세요.",
            "elimination": "불가능 지역을 소거해 범위를 좁히세요.",
            "inductive": "물리적/환경적 단서의 교집합으로 후보를 선정하세요.",
        }.get(mode, "")

        used_str = f"\n사용 완료 도구: {', '.join(sorted(used))}" if used else ""
        remaining = settings.MAX_INVESTIGATION_STEPS - step

        # 사용된 도구 기반 후속 가이드
        next_hint = ""
        has_map_api = bool(used & {"naver_place_search", "kakao_place_search", "osm_poi_search"})
        has_web = bool(used & {"web_search", "search_naver_blog", "naver_news_search"})
        has_roadview = bool(used & {"roadview_check", "street_view_fetch"})
        independent_count = sum([has_map_api, has_web, has_roadview])

        # 좌표 확보 시 roadview 유도 힌트
        coord_hint = ""
        if last_coords and not has_roadview and remaining > 0:
            coord_hint = (
                f"\n📍 확보된 좌표: ({last_coords[0]:.4f}, {last_coords[1]:.4f})"
                f" → roadview_check({{\"lat\": {last_coords[0]:.4f}, \"lon\": {last_coords[1]:.4f}}}) 즉시 실행"
            )

        if has_map_api and not has_roadview and remaining > 1:
            next_hint = "\n⚡ 권장 다음 단계: 좌표 확보 후 roadview_check(lat, lon) 실행 → 3번째 독립 소스"
        elif has_map_api and has_web and not has_roadview and remaining > 0:
            next_hint = "\n⚡ roadview_check로 최종 확인 후 CONCLUDE 가능"
        elif independent_count >= 3 or (has_map_api and has_web and has_roadview):
            next_hint = "\n✅ 3개 독립 소스 확보 — ACTION: CONCLUDE로 결론 내리세요"
        elif not has_map_api and remaining > 2:
            next_hint = "\n⚡ 우선: naver_place_search 또는 web_search로 장소 후보 특정"

        # Ollama/Groq는 네이티브 function calling 미사용 → 텍스트 형식 명시
        tool_format_hint = ""
        if settings.LLM_PROVIDER in ("ollama", "groq"):
            tool_format_hint = (
                "\n\n[도구 호출 형식]\n"
                "ACTION: naver_place_search({\"query\": \"해운대 해수욕장\"})\n"
                "ACTION: roadview_check({\"lat\": 35.1581, \"lon\": 129.1584})\n"
                "ACTION: web_search({\"query\": \"검색어\"})\n"
                "ACTION: CONCLUDE  ← 결론 시"
            )

        return (
            f"{state}\n\n"
            f"탐색 모드: {mode} — {mode_hint}{used_str}{next_hint}{coord_hint}\n"
            f"남은 스텝: {remaining}{tool_format_hint}\n\n"
            f"최적의 다음 행동을 선택하세요."
        )

    def _parse_step(self, content: str, step_num: int) -> InvestigationStep:
        step = InvestigationStep(step_num=step_num)
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("THINK:"):
                step.think = line[6:].strip()
            elif line.startswith("HYPOTHESIS:"):
                step.hypothesis_update = line[11:].strip()
            elif line.startswith("ACTION:"):
                step.action = line[7:].strip()
            elif line.startswith("EXPECTED:"):
                step.expected = line[9:].strip()
        return step

    def _parse_text_tool_call(self, content: str) -> dict | None:
        """
        Ollama/Groq 등 네이티브 tool call 미지원 모델 폴백.
        'ACTION: tool_name({"key": "val (괄호포함)"})' 형태도 올바르게 파싱.
        [^)]* 대신 balanced brace matching으로 JSON 중첩 처리.
        """
        name_m = re.search(r"ACTION\s*:\s*(\w+)", content, re.IGNORECASE)
        if not name_m:
            return None
        name = name_m.group(1).strip()
        if name.upper() == "CONCLUDE" or name not in self._tools:
            return None

        args: dict = {}
        after = content[name_m.end():]

        # 1순위: balanced brace matching으로 JSON 객체 추출 (중첩 처리)
        brace_pos = after.find('{')
        if brace_pos != -1:
            depth = 0
            end_pos = None
            for i in range(brace_pos, len(after)):
                if after[i] == '{':
                    depth += 1
                elif after[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break
            if end_pos is not None:
                try:
                    args = json.loads(after[brace_pos:end_pos])
                except json.JSONDecodeError:
                    pass

        # 2순위: key="value" 형태
        if not args:
            for k, v in re.findall(r'(\w+)\s*=\s*"([^"]*)"', after[:200]):
                args[k] = v

        # 3순위: 단순 인용 문자열 → 첫 required 파라미터에 매핑
        if not args:
            plain_m = re.search(r'[("\'](.*?)["\')]', after[:100])
            if plain_m:
                schema = next((s for s in self._tool_schemas if s["name"] == name), None)
                required = schema["parameters"].get("required", []) if schema else []
                if required:
                    args[required[0]] = plain_m.group(1)

        return {"name": name, "arguments": args}

    def _is_conclude(self, content: str) -> bool:
        return bool(re.search(r"ACTION\s*:\s*CONCLUDE", content, re.IGNORECASE))

    def _parse_conclude(self, content: str) -> dict:
        # <think>...</think> 태그 및 \\n 정규화
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = content.replace('\\n', '\n')
        result: dict = {}
        patterns = {
            "location": r"LOCATION\s*:\s*([^\n]+)",
            "latitude": r"LATITUDE\s*:\s*([0-9.\-]+)",
            "longitude": r"LONGITUDE\s*:\s*([0-9.\-]+)",
            "reasoning": r"REASONING\s*:\s*(.+?)(?=\n[A-Z]+:|$)",
        }
        for key, pat in patterns.items():
            m = re.search(pat, content, re.IGNORECASE | re.DOTALL)
            if m:
                val = m.group(1).strip()
                # LOCATION 값이 다른 키워드로 오염된 경우 정제
                if key == "location":
                    val = re.split(r'(?:\n|\\n)(?:LATITUDE|LONGITUDE|REASONING|ACTION|THINK|HYPOTHESIS)\s*:', val, maxsplit=1)[0]
                    val = val.split('\n')[0].strip()
                    # THINK:/HYPOTHESIS: 등 LLM 메타 텍스트가 location으로 들어오면 무효화
                    if re.match(r'^(THINK|HYPOTHESIS|ACTION|EXPECTED)\s*:', val, re.I):
                        continue
                result[key] = val
        return result

    def _inject_sub_hypotheses(self, content: str, tree: HypothesisTree) -> None:
        """LLM 응답에서 하위 지역 후보를 파싱해 트리에 주입"""
        # "분기: 은평구(45%), 마포구(35%)" 형태 파싱
        m = re.search(r"분기\s*:\s*(.+?)(?:\n|$)", content)
        if not m:
            return

        parts = re.findall(r"([가-힣a-zA-Z]+(?:\s[가-힣a-zA-Z]+)?)\s*\((\d+)%\)", m.group(1))
        if not parts:
            return

        top = tree.get_top_hypothesis()
        if top and top.probability >= 0.60 and len(top.children_ids) == 0:
            sub_locs = [loc for loc, _ in parts]
            if sub_locs:
                tree.inject_sub_hypotheses(top.id, sub_locs)

    async def _exec_tool(self, name: str, args: dict) -> dict:
        if name in self.restricted_tools:
            return {"error": "이 기능은 현재 플랜에서 사용할 수 없습니다", "plan_restricted": True}
        fn = self._tools.get(name)
        if not fn:
            return {"error": f"도구 '{name}' 없음"}
        try:
            return await fn(args)
        except Exception as e:
            logger.error(f"Tool {name} error: {e}")
            return {"error": str(e)}

    def _to_evidence(self, tool_name: str, result: dict, args: dict) -> Optional[Evidence]:
        if "error" in result:
            return None

        # 도구별 단서 추출 규칙
        TOOL_MAP = {
            # 핵심 도구
            "exif_extract": ("exif", "EXIF"),
            "ocr_extract": ("ocr", "OCR 텍스트"),
            "object_detect": ("object_detect", "인프라 탐지"),
            "geoclip_embed": ("geoclip", "GeoCLIP"),
            "reverse_image_search": ("reverse_search", "역방향 검색"),
            "naver_place_search": ("naver_place", "네이버 플레이스"),
            "kakao_place_search": ("kakao_place", "카카오맵"),
            "sun_analysis": ("physical", "태양 역산"),
            "vpr_compare": ("vpr", "VPR 매칭"),
            "web_search": ("web_search", "웹 검색"),
            "search_naver_blog": ("naver_blog", "네이버 블로그"),
            "osm_poi_search": ("osm_poi", "OSM POI"),
            "street_view_fetch": ("street_view", "Street View"),
            "deep_crawl_url": ("url_crawl", "URL 크롤"),
            # 한국 특화 도구
            "korea_analyze": ("korea_analyze", "한국 특화 분석"),
            "juso_lookup": ("juso_lookup", "Juso 주소→GPS"),
            "roadview_check": ("roadview_check", "네이버 로드뷰"),
            "license_plate_lookup": ("license_plate", "번호판 지역 확정"),
            "transit_lookup": ("transit_db", "대중교통 DB"),
            "naver_local_search": ("naver_local", "네이버 로컬"),
            "kakao_local_search": ("kakao_local", "카카오 로컬"),
            # OSINT 확장 도구
            "receipt_parse": ("interior_osint", "영수증/명함 분석"),
            "brand_locate": ("web_search", "브랜드 위치 추적"),
            "auto_chain": ("auto_chain", "OSINT 자동 체인"),
            "crawl_social": ("crawl_social", "SNS 크롤"),
            "reverse_chain": ("reverse_chain", "역방향 URL 체인"),
            "naver_news_search": ("naver_news", "네이버 뉴스"),
            "flickr_search": ("flickr", "Flickr 지오태그"),
            "clova_ocr": ("clova_ocr", "CLOVA OCR+NER"),
        }
        if tool_name not in TOOL_MAP:
            return None

        source, prefix = TOOL_MAP[tool_name]

        # 신뢰도 & likelihood_ratio 결정
        if tool_name == "exif_extract":
            if result.get("gps"):
                return Evidence(source=source, description=f"GPS 좌표: {result['gps']}",
                                confidence_level="HIGH", likelihood_ratio=50.0)
            if result.get("timezone"):
                return Evidence(source=source, description=f"시간대: {result['timezone']}",
                                confidence_level="MED", likelihood_ratio=3.0)

        elif tool_name == "ocr_extract":
            texts = result.get("texts", [])
            langs = result.get("languages", [])
            if texts and langs:
                desc = f"언어={langs}, 텍스트={[t['text'] for t in texts[:3]]}"
                lr = 10.0 if "ko" in langs else 5.0
                return Evidence(source=source, description=desc, confidence_level="HIGH", likelihood_ratio=lr)

        elif tool_name == "object_detect":
            country = result.get("top_country")
            score = result.get("top_country_score", 0)
            if country:
                return Evidence(source=source, description=f"인프라→{country} (점수:{score:.1f})",
                                confidence_level="MED" if score > 2.0 else "LOW",
                                likelihood_ratio=min(score, 5.0))

        elif tool_name == "geoclip_embed":
            loc = result.get("top_location")
            score = result.get("score", 0)
            if loc:
                return Evidence(source=source, description=f"GeoCLIP→{loc} (점수:{score:.2f})",
                                confidence_level="MED" if score > 0.5 else "LOW",
                                likelihood_ratio=score * 5)

        elif tool_name in ("naver_place_search", "kakao_place_search"):
            places = result.get("places", [])
            if places:
                p = places[0]
                return Evidence(source=source,
                                description=f"플레이스 확인: {p['name']} @ {p['address']}",
                                confidence_level="HIGH", likelihood_ratio=15.0)

        elif tool_name == "reverse_image_search":
            hints = result.get("location_hints", [])
            hits = len(result.get("results", []))
            if hints or hits > 0:
                return Evidence(source=source, description=f"역방향 검색 {hits}건, 위치힌트={hints}",
                                confidence_level="MED" if hits > 2 else "LOW",
                                likelihood_ratio=min(hits * 2, 8.0))

        elif tool_name == "vpr_compare":
            sim = result.get("similarity", 0)
            loc = result.get("best_location", "")
            if sim > 0.5:
                return Evidence(source=source, description=f"VPR→{loc} (유사도:{sim:.2f})",
                                confidence_level="HIGH" if sim > 0.85 else "MED",
                                likelihood_ratio=sim * 20)

        elif tool_name == "sun_analysis":
            band = result.get("latitude_band")
            if band:
                return Evidence(source=source, description=f"물리역산: {band}, {result.get('hemisphere','')}",
                                confidence_level="LOW", likelihood_ratio=2.0)

        elif tool_name == "web_search":
            items = result.get("results", [])
            hints = result.get("location_hints", [])
            # 스니펫에서 한국 주소 패턴 추출 시 신뢰도 상승
            _kr_addr_found = any(
                any(kw in str(r.get("snippet", "")) + str(r.get("title", ""))
                    for kw in ("부산", "서울", "인천", "대구", "광주", "해운대", "명동", "강남", "제주"))
                for r in (items or [])
            )
            if hints:
                lr = 6.0 if _kr_addr_found else 4.0
                return Evidence(source=source, description=f"웹검색 위치힌트: {hints[:2]}",
                                confidence_level="MED", likelihood_ratio=lr)
            if items:
                lr = 3.0 if _kr_addr_found else 1.5
                return Evidence(source=source, description=f"웹검색 {len(items)}건 결과",
                                confidence_level="LOW", likelihood_ratio=lr)

        elif tool_name == "search_naver_blog":
            items = result.get("results", [])
            hints = [r.get("location_hint") for r in items if r.get("location_hint")]
            if hints:
                return Evidence(source=source, description=f"블로그 위치힌트: {hints[:2]}",
                                confidence_level="MED", likelihood_ratio=5.0)

        elif tool_name == "osm_poi_search":
            pois = result.get("pois", [])
            if pois:
                p = pois[0]
                return Evidence(source=source,
                                description=f"OSM POI: {p.get('name')} @ ({p.get('lat'):.4f},{p.get('lon'):.4f})",
                                confidence_level="MED", likelihood_ratio=6.0)

        elif tool_name == "street_view_fetch":
            imgs = result.get("images", [])
            matched = result.get("visual_match", False)
            if matched:
                return Evidence(source=source, description=f"Street View 시각 매칭 확인",
                                confidence_level="HIGH", likelihood_ratio=20.0)
            if imgs:
                return Evidence(source=source, description=f"Street View {len(imgs)}개 후보 이미지",
                                confidence_level="LOW", likelihood_ratio=2.0)

        elif tool_name == "deep_crawl_url":
            hints = result.get("location_hints", [])
            lat = result.get("lat")
            lon = result.get("lon")
            if lat and lon:
                return Evidence(source=source, description=f"URL 좌표: ({lat:.4f},{lon:.4f})",
                                confidence_level="HIGH", likelihood_ratio=12.0)
            if hints:
                return Evidence(source=source, description=f"URL 위치힌트: {hints[:2]}",
                                confidence_level="MED", likelihood_ratio=3.0)

        # ── 한국 특화 도구 ────────────────────────────────────
        elif tool_name == "korea_analyze":
            lat = result.get("lat")
            lon = result.get("lon")
            loc = result.get("best_location", "")
            conf = result.get("confidence", 0)
            if lat and lon:
                return Evidence(source=source,
                                description=f"한국 특화 분석: {loc} ({lat:.4f},{lon:.4f})",
                                confidence_level="HIGH", likelihood_ratio=15.0)
            if loc and conf > 0.3:
                return Evidence(source=source, description=f"한국 특화 분석: {loc}",
                                confidence_level="MED", likelihood_ratio=8.0)

        elif tool_name == "juso_lookup":
            lat = result.get("lat")
            lon = result.get("lon")
            addr = result.get("address", "")
            if lat and lon:
                return Evidence(source=source,
                                description=f"Juso 주소 좌표: {addr} → ({lat:.4f},{lon:.4f})",
                                confidence_level="HIGH", likelihood_ratio=18.0)

        elif tool_name == "roadview_check":
            if result.get("available"):
                rlat = result.get("roadview_lat", 0)
                rlon = result.get("roadview_lon", 0)
                addr = result.get("address", "") or result.get("road_address", "")
                # 주소까지 반환된 경우만 MED, 단순 가용 확인은 LOW
                # 로드뷰 "available"은 해당 좌표에 로드뷰가 '존재'한다는 것일 뿐
                # 사진과 시각적 매칭 검증이 없으므로 신뢰도 낮게 유지
                if addr:
                    return Evidence(source=source,
                                    description=f"로드뷰+주소 확인: {addr} ({rlat:.4f},{rlon:.4f})",
                                    confidence_level="MED", likelihood_ratio=6.0)
                return Evidence(source=source,
                                description=f"로드뷰 가용 확인 ({rlat:.4f},{rlon:.4f}) — 시각 매칭 미검증",
                                confidence_level="LOW", likelihood_ratio=3.0)
            elif result.get("checked"):
                return Evidence(source=source,
                                description="로드뷰 미확인 — 해당 좌표 주변 로드뷰 없음",
                                confidence_level="LOW", likelihood_ratio=0.8)

        elif tool_name == "license_plate_lookup":
            if result.get("found"):
                region = result.get("region", "")
                plat = result.get("latitude", 0)
                plon = result.get("longitude", 0)
                return Evidence(source=source,
                                description=f"번호판 지역 확정: {region} ({plat:.4f},{plon:.4f})",
                                confidence_level="HIGH", likelihood_ratio=20.0)

        elif tool_name == "transit_lookup":
            tname = result.get("name", "")
            city = result.get("city", "")
            conf = result.get("confidence", 0)
            if tname and city:
                lr = 12.0 if conf >= 0.8 else 6.0
                lvl = "HIGH" if conf >= 0.8 else "MED"
                return Evidence(source=source,
                                description=f"대중교통 매칭: {tname} ({city})",
                                confidence_level=lvl, likelihood_ratio=lr)

        elif tool_name in ("naver_local_search", "kakao_local_search"):
            best = result.get("best", {})
            if best and best.get("lat") and best.get("lon"):
                label = "네이버 로컬" if tool_name == "naver_local_search" else "카카오 로컬"
                return Evidence(source=source,
                                description=f"{label}: {best.get('name','')} @ {best.get('address','')}",
                                confidence_level="HIGH", likelihood_ratio=15.0)
            if result.get("found"):
                return Evidence(source=source,
                                description=f"로컬 검색 {len(result.get('results',[]))}건",
                                confidence_level="MED", likelihood_ratio=6.0)

        # ── OSINT 확장 도구 ──────────────────────────────────
        elif tool_name == "receipt_parse":
            addrs = result.get("addresses", [])
            phones = result.get("phone_regions", [])
            hints = result.get("location_hints", [])
            if addrs:
                return Evidence(source=source, description=f"영수증 주소: {addrs[0]}",
                                confidence_level="HIGH", likelihood_ratio=12.0)
            if phones:
                return Evidence(source=source,
                                description=f"전화번호 지역코드: {', '.join(phones)}",
                                confidence_level="MED", likelihood_ratio=8.0)
            if hints:
                return Evidence(source=source, description=f"영수증 위치힌트: {hints[0]}",
                                confidence_level="MED", likelihood_ratio=5.0)

        elif tool_name == "brand_locate":
            poi = result.get("poi", {})
            hints = result.get("location_hints", [])
            if poi and poi.get("lat") and poi.get("lon"):
                return Evidence(source=source,
                                description=f"브랜드 위치: {poi.get('name','')} @ {poi.get('address','')}",
                                confidence_level="HIGH", likelihood_ratio=10.0)
            if hints:
                return Evidence(source=source, description=f"브랜드 위치힌트: {hints[0]}",
                                confidence_level="MED", likelihood_ratio=5.0)

        elif tool_name == "auto_chain":
            loc = result.get("location", "")
            clat = result.get("lat")
            clon = result.get("lon")
            conf = result.get("confidence", 0)
            if clat and clon:
                return Evidence(source=source,
                                description=f"OSINT 체인 결과: {loc} ({clat:.4f},{clon:.4f})",
                                confidence_level="HIGH" if conf >= 0.7 else "MED",
                                likelihood_ratio=12.0)
            if loc:
                return Evidence(source=source, description=f"OSINT 체인 위치: {loc}",
                                confidence_level="MED", likelihood_ratio=5.0)

        elif tool_name == "crawl_social":
            slat = result.get("lat")
            slon = result.get("lon")
            hints = result.get("location_hints", [])
            if slat and slon:
                return Evidence(source=source, description=f"SNS 좌표: ({slat:.4f},{slon:.4f})",
                                confidence_level="HIGH", likelihood_ratio=8.0)
            if hints:
                return Evidence(source=source, description=f"SNS 위치힌트: {hints[:2]}",
                                confidence_level="MED", likelihood_ratio=4.0)

        elif tool_name == "reverse_chain":
            hints = result.get("location_hints", [])
            chain_results = result.get("results", [])
            if hints:
                return Evidence(source=source, description=f"URL 체인 위치힌트: {hints[:2]}",
                                confidence_level="MED", likelihood_ratio=5.0)
            if chain_results:
                return Evidence(source=source,
                                description=f"URL 체인 크롤 {len(chain_results)}건",
                                confidence_level="LOW", likelihood_ratio=2.0)

        elif tool_name == "naver_news_search":
            hints = result.get("location_hints", [])
            best = result.get("best_hint", "")
            if best or hints:
                return Evidence(source=source,
                                description=f"뉴스 위치힌트: {best or hints[0]}",
                                confidence_level="MED", likelihood_ratio=4.0)

        elif tool_name == "flickr_search":
            coords = result.get("best_coords", {})
            hints = result.get("location_hints", [])
            if coords.get("lat") and coords.get("lon"):
                flat, flon = coords["lat"], coords["lon"]
                return Evidence(source=source,
                                description=f"Flickr 지오태그: ({flat:.4f},{flon:.4f})",
                                confidence_level="MED", likelihood_ratio=6.0)
            if hints:
                return Evidence(source=source, description=f"Flickr 위치힌트: {hints[:2]}",
                                confidence_level="LOW", likelihood_ratio=3.0)

        elif tool_name == "clova_ocr":
            entities = result.get("entities", {})
            texts = result.get("texts", [])
            # NER에서 위치 엔티티 (LC = 위치명, address 키)
            addrs = entities.get("addresses", []) or entities.get("LC", [])
            if addrs:
                return Evidence(source=source, description=f"CLOVA NER 주소: {addrs[0]}",
                                confidence_level="HIGH", likelihood_ratio=12.0)
            if texts:
                return Evidence(source=source,
                                description=f"CLOVA OCR {len(texts)}개 텍스트 추출",
                                confidence_level="MED", likelihood_ratio=6.0)

        return None

    async def _vision_describe_image(self, image_data: bytes, image_media_type: str) -> str:
        """
        Groq 사용 시 vision 미지원 메인 모델(qwen3) 전 사전 단계:
        llama-4-scout(vision 지원)로 이미지를 한 번 보고 텍스트 설명 생성.
        이후 qwen3가 이 설명을 컨텍스트로 받아 추론한다.
        """
        if settings.LLM_PROVIDER != "groq" or not image_data:
            return ""
        try:
            import base64
            import io as _io
            from PIL import Image as _PILImage
            from openai import AsyncOpenAI

            # 이미지 리사이즈: Groq scout 33MP 제한 대응 → 최대 1920px 긴 변
            try:
                _img = _PILImage.open(_io.BytesIO(image_data)).convert("RGB")
                _w, _h = _img.size
                _max_side = 1920
                if max(_w, _h) > _max_side:
                    _ratio = _max_side / max(_w, _h)
                    _img = _img.resize((int(_w * _ratio), int(_h * _ratio)), _PILImage.LANCZOS)
                _buf = _io.BytesIO()
                _img.save(_buf, format="JPEG", quality=85)
                _img_bytes = _buf.getvalue()
                image_media_type = "image/jpeg"
            except Exception:
                _img_bytes = image_data

            client = AsyncOpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
                timeout=30.0,
            )
            b64 = base64.standard_b64encode(_img_bytes).decode()
            resp = await client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "이 사진에서 위치 수사에 유용한 시각적 단서를 한국어로 설명하세요.\n"
                            "규칙:\n"
                            "1) 간판/텍스트는 보이는 글자를 정확히 전사하세요. 잘 안 보이거나 확신 없으면 반드시 [불확실] 표기.\n"
                            "2) '흡연금지', '주차금지' 같은 일반 안내문은 위치 단서가 아님 — 건물명/상호명/브랜드명만 중점 기술.\n"
                            "3) 건물 외관, 주변 건물, 도로/교통 인프라, 랜드마크, 자연환경.\n"
                            "4) 번호판, 사람 복장, 날씨/계절.\n"
                            "확신 없는 내용은 절대 단정하지 말고 [불확실] 표기. 200자 이내."
                        )},
                        {"type": "image_url",
                         "image_url": {"url": f"data:{image_media_type};base64,{b64}"}},
                    ],
                }],
                max_tokens=300,
                temperature=0.1,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"[Vision pre-step] scout 모델 실패: {e}")
            return ""

    def _clues_exhausted(self, used: set[str]) -> bool:
        # 핵심 도구 + OSINT 확장 도구(web_search) 모두 시도해야 종료 허용
        # 단, restricted_tools(플랜 제한)는 체크에서 제외
        core = {"exif_extract", "ocr_extract", "object_detect",
                "geoclip_embed", "reverse_image_search",
                "naver_place_search", "sun_analysis"}
        effective_core = core - self.restricted_tools  # 제한 도구 제외
        osint = {"web_search"}
        return effective_core.issubset(used) and osint.issubset(used)

    async def _finalize(
        self,
        result: InvestigationResult,
        tree: HypothesisTree,
        messages: list[LLMMessage],
        image_data: bytes,
        media_type: str,
    ) -> InvestigationResult:
        top = tree.get_top_hypothesis()
        if not top and not result.location:
            result.location = "위치 특정 불가"
            result.confidence = 0.0
            result.confidence_label = "UNKNOWN"
            return result

        # ── 좌표 교차검증: evidence에서 좌표 수집 후 이상값 제거 ─────────
        _coord_pool: list[tuple[float, float]] = []
        _coord_re = re.compile(r'\((\d{2,3}\.\d{2,6})\s*,\s*(\d{2,3}\.\d{2,6})\)')
        for ev in tree.evidence_log:
            if ev.is_contradiction:
                continue
            m = _coord_re.search(ev.description)
            if m:
                try:
                    _clat, _clon = float(m.group(1)), float(m.group(2))
                    if 33.0 <= _clat <= 39.0 and 124.0 <= _clon <= 132.0:
                        _coord_pool.append((_clat, _clon))
                except (ValueError, TypeError):
                    pass
        if len(_coord_pool) >= 2:
            # 중앙값 계산 후 500m(약 0.005°) 초과 이상값 제거
            _med_lat = sorted(c[0] for c in _coord_pool)[len(_coord_pool) // 2]
            _med_lon = sorted(c[1] for c in _coord_pool)[len(_coord_pool) // 2]
            _valid = [(la, lo) for la, lo in _coord_pool
                      if abs(la - _med_lat) < 0.005 and abs(lo - _med_lon) < 0.005]
            if _valid:
                _avg_lat = sum(c[0] for c in _valid) / len(_valid)
                _avg_lon = sum(c[1] for c in _valid) / len(_valid)
                # CONCLUDE 좌표가 없거나 이상값이면 교차검증 평균으로 보정
                if not result.latitude or not result.longitude:
                    result.latitude = round(_avg_lat, 6)
                    result.longitude = round(_avg_lon, 6)
                    logger.info(f"[Coord cross-val] 좌표 보정: ({_avg_lat:.4f},{_avg_lon:.4f}) from {len(_valid)}개 소스")
                elif (abs(result.latitude - _avg_lat) > 0.01 or abs(result.longitude - _avg_lon) > 0.01):
                    logger.info(f"[Coord cross-val] CONCLUDE 좌표({result.latitude:.4f},{result.longitude:.4f}) vs 교차검증({_avg_lat:.4f},{_avg_lon:.4f}) — 교차검증 채택")
                    result.latitude = round(_avg_lat, 6)
                    result.longitude = round(_avg_lon, 6)

        # 신뢰도 계산
        confidence = self.confidence_calc.calculate(top, tree.evidence_log) if top else 0.0

        # Hallucination 검증 (반증 탐색 강제)
        contradictions = [ev for ev in tree.evidence_log if ev.is_contradiction]
        if len(contradictions) == 0:
            result.hallucination_check_passed = True
        elif len(contradictions) == 1:
            confidence *= 0.7
            result.hallucination_check_passed = False
        elif len(contradictions) == 2:
            confidence *= 0.4
            result.hallucination_check_passed = False
        else:
            confidence *= 0.1
            result.hallucination_check_passed = False

        result.confidence = round(min(confidence, 0.99), 4)

        # 레이블
        if result.confidence >= 0.90:
            result.confidence_label = "HIGH"
        elif result.confidence >= 0.70:
            result.confidence_label = "MEDIUM"
        elif result.confidence >= 0.30:
            result.confidence_label = "LOW"
        else:
            result.confidence_label = "UNKNOWN"

        # 위치 (CONCLUDE가 없었으면 트리 최고 가설 사용)
        if not result.location and top:
            result.location = top.location

        # 가설 트리의 초기 후보(시/도 수준)만 남은 경우 LLM에 구체화 요청
        INITIAL_CANDIDATES = {
            "서울", "경기", "인천", "부산", "경남", "울산",
            "대구", "경북", "광주", "전남", "전북",
            "대전", "충남", "충북", "세종", "강원", "제주",
        }
        if result.location in INITIAL_CANDIDATES and tree.evidence_log:
            ev_summary = "; ".join(f"{e.source}:{e.description[:60]}" for e in tree.evidence_log[:6])
            refine = await self.llm.complete(
                messages=[LLMMessage(role="user", content=(
                    f"수집된 단서: {ev_summary}\n"
                    f"현재 추정 지역: {result.location}\n"
                    f"단서를 바탕으로 가능한 구체적인 시/군/구 수준 지역명을 한 줄로 답하세요. "
                    f"확신할 수 없으면 '{result.location} (정확한 위치 미상)'으로 답하세요."
                ))],
                system=IDENTITY_PROMPT,
            )
            # THINK:/HYPOTHESIS:/ACTION: 줄 제거 후 첫 번째 실제 지명 줄 추출
            _skip_prefixes = re.compile(r'^(THINK|HYPOTHESIS|ACTION|EXPECTED|REASONING|LATITUDE|LONGITUDE)\s*:', re.I)
            refined_lines = [
                l.strip() for l in re.sub(r'<think>.*?</think>', '', refine.content, flags=re.DOTALL).split('\n')
                if l.strip() and not _skip_prefixes.match(l.strip())
            ]
            if refined_lines:
                result.location = refined_lines[0][:200]

        # 30% 미만 → 불가
        if result.confidence < 0.30:
            result.location = "위치 특정 불가"
            result.confidence_label = "UNKNOWN"

        result.hypothesis_tree = tree.to_dict()
        result.evidence_chain = [
            {"id": ev.id, "source": ev.source, "description": ev.description,
             "confidence_level": ev.confidence_level, "is_contradiction": ev.is_contradiction}
            for ev in tree.evidence_log
        ]

        # LLM 최종 보고서 (아직 없으면 생성)
        if not result.final_reasoning:
            summary = await self.llm.complete(
                messages=[LLMMessage(role="user", content=(
                    f"수사 결과를 3~5문장으로 요약하세요.\n"
                    f"위치: {result.location}\n"
                    f"신뢰도: {result.confidence:.1%}\n"
                    f"단서 {len(tree.evidence_log)}개 수집\n"
                    f"각 단서가 어떻게 이 결론을 지지하는지 포함."
                ))],
                system=IDENTITY_PROMPT,
            )
            result.final_reasoning = summary.content

        return result


# ── 모듈 레벨 헬퍼 ────────────────────────────────────────────────────────────


def _pick_forced_tool(ctx: dict, used: set[str], last_coords: tuple[float, float] | None) -> dict:
    """stall 방지: 2회 연속 도구 미사용 시 강제 실행할 도구+인수 결정.
    Returns {"name": tool_name, "args": {...}}
    """
    # 1. 좌표가 확보됐고 roadview를 아직 안 썼으면 → roadview_check 최우선
    if last_coords and "roadview_check" not in used:
        return {
            "name": "roadview_check",
            "args": {"lat": last_coords[0], "lon": last_coords[1]},
        }

    # 2. 역방향 검색 힌트가 있으면 → naver_place_search
    rev_hints = ctx.get("reverse_search_hints", [])
    if rev_hints and "naver_place_search" not in used:
        query = rev_hints[0]
        return {"name": "naver_place_search", "args": {"query": query}}

    # 3. 장면이 해변이면 → naver_place_search('해운대 해수욕장')
    scene_tags = ctx.get("scene_tags", [])
    scene_str = " ".join(scene_tags).lower()
    if any(w in scene_str for w in ("beach", "ocean", "resort", "해수욕")) and "naver_place_search" not in used:
        return {"name": "naver_place_search", "args": {"query": "해운대 해수욕장"}}

    # 4. 강변이면 → naver_place_search('한강공원')
    if ("river" in scene_str or "riverside" in scene_str) and "naver_place_search" not in used:
        return {"name": "naver_place_search", "args": {"query": "한강공원"}}

    # 5. naver_place_search 미사용이면 geoclip 추정 지역으로
    geo_loc = ctx.get("geoclip_location", "")
    if geo_loc and "naver_place_search" not in used:
        return {"name": "naver_place_search", "args": {"query": geo_loc}}

    # 6. 기본 폴백 → web_search
    ocr_texts = ctx.get("ocr_texts", [])
    query = ocr_texts[0] if ocr_texts else "한국 유명 관광지 위치"
    return {"name": "web_search", "args": {"query": query}}


# Vision pre-step에서 추출할 한국 지명 사전
_KOREA_PLACE_DICT: dict[str, str] = {
    # 해수욕장/해변
    "해운대": "부산 해운대", "Haeundae": "부산 해운대",
    "광안리": "부산 광안리", "Gwangalli": "부산 광안리",
    "광안대교": "부산 광안리", "Gwangan Bridge": "부산 광안리",
    "해동용궁사": "부산 해동용궁사", "송정해수욕장": "부산 송정",
    "을왕리": "인천 을왕리", "대천해수욕장": "충남 대천",
    "경포대": "강릉 경포대", "속초해수욕장": "속초",
    # 서울 주요 명소
    "한강": "서울 한강", "Hangang": "서울 한강", "Han River": "서울 한강",
    "여의도": "서울 여의도", "Yeouido": "서울 여의도",
    "남산": "서울 남산", "Namsan": "서울 남산",
    "명동": "서울 명동", "Myeongdong": "서울 명동",
    "홍대": "서울 홍대", "Hongdae": "서울 홍대",
    "강남": "서울 강남", "Gangnam": "서울 강남",
    "잠실": "서울 잠실", "Jamsil": "서울 잠실",
    "동대문": "서울 동대문", "Dongdaemun": "서울 동대문",
    "인사동": "서울 인사동", "Insadong": "서울 인사동",
    "성수동": "서울 성수동", "Seongsu": "서울 성수동",
    "북한산": "서울 북한산", "Bukhansan": "서울 북한산",
    "경복궁": "서울 경복궁", "Gyeongbokgung": "서울 경복궁",
    "청계천": "서울 청계천", "Cheonggyecheon": "서울 청계천",
    "이태원": "서울 이태원", "Itaewon": "서울 이태원",
    "신촌": "서울 신촌", "코엑스": "서울 코엑스",
    # 부산
    "부산": "부산", "Busan": "부산",
    "감천": "부산 감천문화마을", "Gamcheon": "부산 감천문화마을",
    "자갈치": "부산 자갈치시장", "Jagalchi": "부산 자갈치시장",
    "남포동": "부산 남포동", "해운대구": "부산 해운대",
    # 경기/인천
    "인천": "인천", "Incheon": "인천",
    "송도": "인천 송도", "Songdo": "인천 송도",
    "수원": "수원", "Suwon": "수원",
    "수원화성": "수원 화성", "화성": "수원 화성",
    # 제주
    "제주": "제주", "Jeju": "제주",
    "한라산": "제주 한라산", "Hallasan": "제주 한라산",
    "성산일출봉": "제주 성산일출봉",
    # 기타 유명 지역
    "설악산": "강원 설악산", "Seoraksan": "강원 설악산",
    "경주": "경주", "Gyeongju": "경주",
    "전주": "전주", "Jeonju": "전주",
    "전주한옥마을": "전주 한옥마을",
    "대전": "대전", "대구": "대구",
    "광주": "광주", "울산": "울산",
}


def _extract_hedged_place_candidate(text: str) -> str:
    """Vision 설명에서 '~일 가능성', '~같은', '~처럼 보이는' 패턴의 불확실 지명 추출.
    반환값은 naver_place_search의 첫 번째 검색 쿼리로만 사용 (확정 신호 아님)."""
    import re as _re
    _BAD = {"장소", "곳", "위치", "한국", "바다", "해변", "도시", "지역", "나라",
            "해수욕장", "공원", "광장", "거리", "시장", "건물", "아파트", "산", "강"}
    # 패턴: 공백 없는 한국어 2-8자 직전에 불확실 표현 등장
    hedged_patterns = [
        r'([가-힣]{2,8})(?:일\s*가능성|처럼\s*보이|로\s*추정|인\s*것\s*같|같아\s*보)',
        r'([가-힣]{2,8})\s*(?:같은\s*(?:곳|장소|해변|해수욕장|공원))',
        r'(?:아마도|probably|likely|possibly)\s*([가-힣]{2,8})',
    ]
    for pat in hedged_patterns:
        m = _re.search(pat, text)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) >= 2 and candidate not in _BAD:
                return candidate
    # 패턴 없어도 사전 단어 있으면 첫 번째 매칭 반환
    found = _extract_korea_places_from_vision(text)
    return found[0] if found else ""


def _extract_korea_places_from_vision(text: str) -> list[str]:
    """Vision pre-step 텍스트에서 한국 지명 추출 (CLIP보다 우선 적용)"""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for keyword, kor_name in _KOREA_PLACE_DICT.items():
        if keyword in text and kor_name not in seen:
            found.append(kor_name)
            seen.add(kor_name)
    return found


def _build_context_summary(ctx: dict) -> str:
    """Pre-computed 분석 결과를 LLM 초기 메시지로 변환"""
    lines = ["[자동 분석 완료 — 아래 결과를 바탕으로 구체적 장소를 특정하세요]\n"]

    strong_signals: list[str] = []
    weak_signals: list[str] = []

    # GPS (최강 신호)
    if ctx.get("has_gps"):
        g = ctx.get("exif", {}).get("gps", {})
        lat, lon = g.get("latitude", 0), g.get("longitude", 0)
        lines.append(f"● GPS (확정): {lat:.6f}, {lon:.6f}")
        strong_signals.append(f"GPS좌표")

    # 기기/플랫폼
    device = ctx.get("device", "")
    platform = ctx.get("platform", "")
    tz = ctx.get("timezone_estimate", "")
    device_country = ctx.get("device_country_hint", "")
    if device or platform or device_country:
        lines.append(f"● 기기: {device} | 플랫폼: {platform or '원본'} | 시간대: {tz} | 주 판매국: {device_country}")
        if device_country:
            strong_signals.append(f"기기판매국={device_country}")

    # OCR
    ocr_texts = ctx.get("ocr_texts", [])
    langs = ctx.get("detected_languages", [])
    plate = ctx.get("license_plate_country", "")
    # 위치 무관 일반 안내문 필터 — 이런 텍스트는 수사에 불필요
    _NOISE_OCR = {
        "흡연금지", "금연", "주차금지", "주차금지구역", "출입금지", "촬영금지",
        "관계자외출입금지", "비상구", "화장실", "남", "여", "push", "pull",
        "no smoking", "no parking", "exit", "entrance", "restroom",
        "화재위험", "위험", "조심", "주의", "안전", "소화기",
    }
    ocr_useful = [t for t in ocr_texts if t.lower().strip() not in _NOISE_OCR and len(t.strip()) >= 2]
    if ocr_useful:
        lines.append(f"● OCR 텍스트: {', '.join(repr(t) for t in ocr_useful[:8])} | 언어: {langs}")
        strong_signals.append(f"OCR={ocr_useful[:2]}")
        # OCR 텍스트로 즉시 장소 검색 지시
        for t in ocr_useful[:3]:
            if len(t) >= 2 and not t.isdigit():
                lines.append(f"  → OCR '{t}' 발견 — naver_place_search('{t}') 즉시 실행")
                break
    elif ocr_texts:
        # 필터링 후 남은 텍스트가 없으면 (흡연금지 등만 있는 경우) 건물 외관으로 추론
        lines.append(f"● OCR 텍스트: 일반 안내문만 ({', '.join(repr(t) for t in ocr_texts[:3])}) — 위치 단서 없음, 건물 외관으로 수사")
    if plate:
        lines.append(f"● 번호판 국가: {plate}")
        strong_signals.append(f"번호판국가={plate}")

    # POI 매칭
    if ctx.get("poi_name"):
        lines.append(f"● POI 매칭: {ctx['poi_name']} | 주소: {ctx.get('poi_address','')} "
                     f"| 좌표: ({ctx.get('poi_lat', '')}, {ctx.get('poi_lon', '')})")
        strong_signals.append(f"POI={ctx['poi_name']}")

    # 인프라 (YOLO) + CLIP 시각 태그
    infra_country = ctx.get("infra_top_country", "")
    infra_score = ctx.get("infra_score", 0)
    objects = ctx.get("infra_objects", [])
    scene_tags = ctx.get("scene_tags", [])
    scene_description = ctx.get("scene_description", "")
    if objects:
        lines.append(f"● 인프라/객체 탐지: 국가={infra_country}({infra_score:.2f}) | 객체: {', '.join(objects[:8])}")
        if infra_score > 0.3:
            strong_signals.append(f"인프라={infra_country}")
    if scene_tags:
        # CLIP 태그에서 구체 한국 지명 필터링 — LLM 앵커링 방지
        # "Hangang", "Haeundae" 등 지명은 별도 낮은신뢰도 섹션으로 처리
        _PLACE_FILTER = ("Hangang", "Han River", "Haeundae", "Gwangalli", "Yeouido",
                         "KOREA_LANDMARK:", "Lotte World", "Namsan", "Seoul Tower",
                         "Gyeongbokgung", "Myeongdong", "Hongdae", "Jamsil",
                         "Jeju", "Busan", "Incheon", "Daegu")
        _display_tags = [t for t in scene_tags if not any(kw in t for kw in _PLACE_FILTER)]
        _landmark_tags = [t for t in scene_tags if t.startswith("KOREA_LANDMARK:")]
        if _display_tags:
            lines.append(f"● 시각 장면 특징(CLIP): {', '.join(_display_tags)}")
        if _landmark_tags:
            for lt in _landmark_tags:
                lines.append(f"● CLIP 장소 추정(낮은신뢰도, 반드시 검색으로 확인): {lt.replace('KOREA_LANDMARK:','')}")
        # CLIP 장면 유형 분류 → 구체적 한국 장소 유형 검색 제안
        # 주의: CLIP은 지명 오분류 잦음 → 장소 유형(beach/river/urban)만 활용
        _scene_str = " ".join(scene_tags).lower()
        _is_beach = any(w in _scene_str for w in ("beach", "ocean", "resort", "해수욕"))
        _is_river = "river" in _scene_str or "riverside" in _scene_str
        _is_mountain = any(w in _scene_str for w in ("mountain", "hiking", "forest"))
        _is_urban = any(w in _scene_str for w in ("urban", "high-rise", "apartment", "city"))

        if _is_beach:
            lines.append("  → 장면 유형: 해변/해수욕장 — 후보: 해운대, 광안리, 을왕리, 경포대, 속초, 대천")
            lines.append("  → naver_place_search('해운대 해수욕장') 또는 web_search('Korea beach high-rise buildings resort')")
            weak_signals.append("장면=한국해변")
        elif _is_river and not _is_beach:
            lines.append("  → 장면 유형: 강변/수변공원 — 한강, 낙동강, 금강 등")
            lines.append("  → naver_place_search('한강공원') 또는 web_search('한국 강변 고층아파트 공원')")
            weak_signals.append("장면=강변")
        elif _is_mountain:
            lines.append("  → 장면 유형: 산/자연 — 설악산, 북한산, 한라산 등")
            weak_signals.append("장면=산")
        elif _is_urban:
            lines.append("  → 장면 유형: 도시/고층")
            weak_signals.append("장면=도시")
        else:
            weak_signals.append(f"장면={'+'.join(scene_tags[:3])}")

    # AI 임베딩 — 신뢰도 임계값 적용
    geo_loc = ctx.get("geoclip_location", "")
    geo_score = ctx.get("geoclip_score", 0)
    street = ctx.get("streetclip_country", "")
    street_score = ctx.get("streetclip_score", 0)
    geo5 = ctx.get("geoclip_top5", [])

    # GeoCLIP: 분류 확률이 너무 낮으면 신뢰 불가 (100개 후보에 균등분포 = ~0.01)
    GEO_RELIABLE = geo_score >= 0.08   # 8% 이상만 신뢰
    STREET_RELIABLE = street_score >= 0.40  # 40% 이상만 신뢰

    if geo_loc or street:
        geo_note = "" if GEO_RELIABLE else " [⚠️신뢰도매우낮음]"
        street_note = "" if STREET_RELIABLE else " [⚠️불확실]"
        lines.append(f"● AI 지역 추정: GeoCLIP={geo_loc}({geo_score:.3f}{geo_note}) | StreetCLIP={street}({street_score:.2f}{street_note})")

        if GEO_RELIABLE and geo_loc:
            strong_signals.append(f"GeoCLIP={geo_loc}")
        elif geo_loc:
            weak_signals.append(f"GeoCLIP추정={geo_loc}(신뢰도낮음:{geo_score:.3f})")

        # 역방향 검색 힌트가 있으면 StreetCLIP은 약한 신호로 격하
        _rev_hints_early = ctx.get("reverse_search_hints", [])
        _has_rev_country = bool(_rev_hints_early)
        if STREET_RELIABLE and street and not _has_rev_country:
            strong_signals.append(f"StreetCLIP={street}")
        elif street:
            weak_signals.append(f"StreetCLIP추정={street}(역방향검색이 우선)"  if _has_rev_country else f"StreetCLIP추정={street}(불확실:{street_score:.2f})")

    if geo5:
        scores = [g.get("score", 0) for g in geo5[:3]]
        max_s = max(scores) if scores else 0
        geo5_str = " | ".join(f"{g.get('location','?')}({g.get('score',0):.3f})" for g in geo5[:3])
        if max_s < 0.08:
            lines.append(f"  GeoCLIP Top3: {geo5_str} ← 전체 점수 낮음(무작위 수준), 다른 단서 우선")
        else:
            lines.append(f"  GeoCLIP Top3: {geo5_str}")

    # 물리 분석
    hemi = ctx.get("hemisphere", "")
    lat_band = ctx.get("latitude_band", "")
    season = ctx.get("season", "")
    if hemi or lat_band:
        lines.append(f"● 물리 분석: 반구={hemi} | 위도대={lat_band} | 계절={season}")
        if hemi:
            weak_signals.append(f"반구={hemi}")

    # Naver Vision Landmark (최강 신호)
    naver_lm = ctx.get("naver_landmark", "")
    naver_lm_lat = ctx.get("naver_landmark_lat", 0.0)
    naver_lm_lon = ctx.get("naver_landmark_lon", 0.0)
    if naver_lm:
        lines.append(f"● ⚡ Naver Vision 랜드마크 직접 감지: {naver_lm} ({naver_lm_lat:.4f},{naver_lm_lon:.4f})")
        lines.append(f"  → juso_lookup('{naver_lm}') 또는 naver_place_search('{naver_lm}')로 정밀 좌표 즉시 확인")
        strong_signals.insert(0, f"Naver랜드마크={naver_lm}")

    # 역방향 검색
    rev_hints = ctx.get("reverse_search_hints", [])
    rev_titles = ctx.get("reverse_search_titles", [])
    if rev_hints:
        lines.append(f"● 역방향 검색 힌트: {', '.join(rev_hints[:5])}")
        strong_signals.extend(rev_hints[:3])
    if rev_titles:
        lines.append(f"● 역방향 검색 제목: {' | '.join(rev_titles[:5])}")
        # 제목에서 추가 위치 힌트 추출
        for t in rev_titles:
            if any(kw in t for kw in ("해운대", "한강", "명동", "홍대", "강남", "Haeundae", "Seoul", "Busan")):
                strong_signals.append(f"제목힌트={t[:30]}")

    # ── 수사 지시 생성 ──
    lines.append("\n[수사 지시]")
    has_strong = bool(strong_signals)

    if not has_strong:
        lines.append("⚠️ 강한 신호 없음 — 시각 장면 분석 기반으로 수사:")
        _scene_all = " ".join(scene_tags).lower() if scene_tags else ""
        if "beach" in _scene_all or "ocean" in _scene_all or "resort" in _scene_all:
            lines.append("  1. naver_place_search('해운대 해수욕장') — 한국 대표 해변")
            lines.append("  2. web_search('Korea beach high-rise buildings apartment complex') — 건물+해변 조합")
            lines.append("  3. naver_place_search('광안리 해수욕장'), naver_place_search('을왕리 해수욕장')")
        elif "river" in _scene_all or "riverside" in _scene_all:
            lines.append("  1. naver_place_search('한강공원') — 한강 공원 후보")
            lines.append("  2. web_search('한국 강변 고층아파트 단지')")
        elif "mountain" in _scene_all or "hiking" in _scene_all:
            lines.append("  1. web_search('한국 유명 등산 명소 케이블카')")
        elif scene_tags:
            lines.append(f"  1. web_search('{' '.join(scene_tags[:2])} 한국 명소 위치')")
        else:
            lines.append("  1. web_search('한국 도심 유명 관광지')")
        lines.append("  2. object_detect → 탐지 객체로 국가/지역 추론")
        lines.append("  3. osm_poi_search로 후보 좌표 주변 POI 확인")
        lines.append("  4. 여러 후보 비교 후 가장 유력한 곳으로 CONCLUDE")
    else:
        lines.append(f"강한 신호: {', '.join(strong_signals)}")
        if ocr_texts:
            lines.append(f"  → '{ocr_texts[0]}'로 naver_place_search, kakao_place_search 즉시 실행")
        # 역방향 검색에서 구체적인 지명이 나온 경우 직접 검색 지시
        for hint in rev_hints:
            if len(hint) > 3 and hint not in ("한국", "일본", "중국", "미국"):
                lines.append(f"  → 역방향검색 '{hint}' → naver_place_search('{hint}') 또는 kakao_place_search('{hint}') 즉시 실행")
                break
        for title in rev_titles:
            if any(kw in title for kw in ("해운대", "한강", "Haeundae", "Hangang", "Han River")):
                lines.append(f"  → 제목 '{title[:30]}' 발견 → naver_place_search('해운대 해수욕장') 즉시 실행하여 좌표 확보")
                break
        if STREET_RELIABLE and "Korea" in (street or "") and not rev_hints:
            lines.append(f"  → StreetCLIP 한국 확인 → 한국 지명/랜드마크 web_search 실행")
        if ctx.get("poi_name"):
            lines.append(f"  → POI '{ctx['poi_name']}' 확인됨 → 좌표 확보 후 ACTION: CONCLUDE")
        lines.append("  → 좌표 확보 시 즉시 ACTION: CONCLUDE")

    if weak_signals:
        lines.append(f"참고(낮은신뢰): {', '.join(weak_signals)} — 검증 필요")

    return "\n".join(lines)


def _update_hypotheses_from_context(tree, ctx: dict) -> None:
    """컨텍스트 기반으로 초기 가설 확률 업데이트 (한국 시/도 레벨)"""
    from ..agents.hypothesis_tree import Evidence

    # ── 한국 지역별 부스트 ─────────────────────────────────
    region_boosts: dict[str, float] = {}  # {시/도명: score}

    # StreetCLIP "South Korea" → 한국 전체 신호 (모든 가설 균등 상승, 후처리)
    street = ctx.get("streetclip_country", "")
    street_score = ctx.get("streetclip_score", 0)
    is_korea_signal = (street == "South Korea" and street_score >= 0.40)

    # GeoCLIP top5에서 한국 지역 특정 → 해당 시/도 부스트
    geo5 = ctx.get("geoclip_top5", [])
    KR_REGION_HINT = {
        "Seoul": "서울", "Busan": "부산", "Daegu": "대구", "Incheon": "인천",
        "Gwangju": "광주", "Daejeon": "대전", "Ulsan": "울산", "Sejong": "세종",
        "Gyeonggi": "경기", "Gangwon": "강원", "Chungbuk": "충북", "Chungnam": "충남",
        "Jeonbuk": "전북", "Jeonnam": "전남", "Gyeongbuk": "경북", "Gyeongnam": "경남",
        "Jeju": "제주",
        # 한글 포함
        "서울": "서울", "부산": "부산", "대구": "대구", "인천": "인천",
        "광주": "광주", "대전": "대전", "울산": "울산", "세종": "세종",
        "경기": "경기", "강원": "강원", "충북": "충북", "충남": "충남",
        "전북": "전북", "전남": "전남", "경북": "경북", "경남": "경남", "제주": "제주",
    }
    for g in geo5[:3]:
        g_loc = g.get("location", "")
        g_score = g.get("score", 0)
        if g_score >= 0.05:
            for key, region in KR_REGION_HINT.items():
                if key in g_loc:
                    region_boosts[region] = max(region_boosts.get(region, 0), g_score * 0.8)
                    break

    # OCR 텍스트에서 지역명 직접 탐지
    ocr_texts = ctx.get("ocr_texts", [])
    combined_ocr = " ".join(ocr_texts)
    for region in ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "경기",
                   "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주", "세종"]:
        if region in combined_ocr:
            region_boosts[region] = max(region_boosts.get(region, 0), 0.5)

    # 번호판에서 지역명
    plate_region = ctx.get("license_plate_region", "")
    if plate_region:
        for region in region_boosts.keys() | {plate_region}:
            if plate_region == region:
                region_boosts[region] = max(region_boosts.get(region, 0), 0.8)

    # POI 지역 힌트
    poi_addr = ctx.get("poi_address", "")
    if poi_addr:
        for region in ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "경기",
                       "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주", "세종"]:
            if region in poi_addr:
                region_boosts[region] = max(region_boosts.get(region, 0), 0.7)

    # ── 가설 트리 업데이트 ─────────────────────────────────
    # 1) 한국 전체 신호 → 모든 가설 직접 확률 균등 상승
    if is_korea_signal:
        active = [h for h in tree.hypotheses.values() if not h.is_rejected]
        for h in active:
            h.probability += street_score * 0.1  # 균등하게 약간 상승
        # 정규화
        total = sum(h.probability for h in active)
        if total > 0:
            for h in active:
                h.probability /= total
        ev = Evidence(
            source="pre_analysis",
            description=f"자동 분석 결과: 한국 (점수: {street_score:.2f})",
            confidence_level="MEDIUM",
            likelihood_ratio=1.0,
        )
        tree.evidence_log.append(ev)

    # 2) 특정 지역 부스트
    for region, score in region_boosts.items():
        for h in tree.hypotheses.values():
            if h.location == region:
                h.probability += score * 0.5
        ev = Evidence(
            source="pre_analysis",
            description=f"자동 분석 결과: {region} (점수: {score:.2f})",
            confidence_level="HIGH" if score >= 0.5 else "MEDIUM",
            likelihood_ratio=1.0 + score,
        )
        tree.bayesian_update(ev)

    # 정규화
    active = [h for h in tree.hypotheses.values() if not h.is_rejected]
    total = sum(h.probability for h in active)
    if total > 0:
        for h in active:
            h.probability /= total


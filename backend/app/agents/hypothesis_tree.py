"""
가설 트리 관리 — Bayesian 업데이트, 가지치기, 분기
사전 확률: Uninformative Prior (완전 균등, 인구 통계 가중치 없음)
"""
from dataclasses import dataclass, field
from typing import Optional
import uuid
import math


PRUNE_THRESHOLD = 0.05       # 5% 미만 → 기각 태그
BRANCH_THRESHOLD = 0.60      # 60% 이상 → 하위 분기
MAX_HYPOTHESES = 20          # 활성 가설 최대 수


@dataclass
class Evidence:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source: str = ""          # "ocr", "geoclip", "object_detect" 등
    description: str = ""
    confidence_level: str = "LOW"   # HIGH / MED / LOW / SPEC
    likelihood_ratio: float = 1.0   # P(단서|위치) / P(단서|비위치)
    is_contradiction: bool = False


@dataclass
class Hypothesis:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    location: str = ""         # "한국 서울 은평구" 등 계층 표현
    probability: float = 0.0
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    supporting_evidence: list[Evidence] = field(default_factory=list)
    contradicting_evidence: list[Evidence] = field(default_factory=list)
    is_rejected: bool = False
    level: int = 0             # 0=시/도, 1=시/군/구, 2=읍/면/동, 3=세부주소


class HypothesisTree:
    def __init__(self):
        self.hypotheses: dict[str, Hypothesis] = {}
        self.evidence_log: list[Evidence] = []
        self.step_count: int = 0

    def initialize(self, candidates: list[str]) -> None:
        """균등 사전 확률로 초기 가설 생성"""
        n = len(candidates)
        if n == 0:
            return
        prob = 1.0 / n
        for loc in candidates:
            h = Hypothesis(location=loc, probability=prob, level=0)
            self.hypotheses[h.id] = h

    def bayesian_update(self, evidence: Evidence) -> None:
        """
        P(위치_i | 단서) ∝ P(단서 | 위치_i) × P(위치_i)
        likelihood_ratio: 해당 단서가 특정 위치를 지지하는 비율
        """
        self.evidence_log.append(evidence)
        active = self._active_hypotheses()
        if not active:
            return

        # 각 가설에 likelihood 적용
        new_probs: dict[str, float] = {}
        for h in active:
            if self._hypothesis_matches_evidence(h, evidence):
                if evidence.is_contradiction:
                    new_probs[h.id] = h.probability * (1.0 / max(evidence.likelihood_ratio, 1.0))
                else:
                    new_probs[h.id] = h.probability * evidence.likelihood_ratio
            else:
                # 단서가 이 가설과 무관하면 확률 소폭 감소
                new_probs[h.id] = h.probability * 0.8 if not evidence.is_contradiction else h.probability

        # 정규화
        total = sum(new_probs.values())
        if total > 0:
            for hid, prob in new_probs.items():
                self.hypotheses[hid].probability = prob / total
                if evidence.is_contradiction:
                    self.hypotheses[hid].contradicting_evidence.append(evidence)
                else:
                    self.hypotheses[hid].supporting_evidence.append(evidence)

        # 가지치기
        self._prune()

        # 분기: 상위 가설 60% 초과 → 하위 분기
        self._branch_if_needed()

    # 한국 광역시도 → 관련 키워드 매핑
    _KR_REGION_KEYWORDS: dict[str, list[str]] = {
        "서울": ["서울", "seoul", "강남", "강북", "종로", "마포", "용산", "성동", "송파", "강서", "관악"],
        "경기": ["경기", "수원", "성남", "고양", "용인", "부천", "안산", "안양", "화성", "평택", "의정부"],
        "인천": ["인천", "incheon", "부평", "계양", "연수", "미추홀"],
        "부산": ["부산", "busan", "해운대", "수영", "남구", "동래", "북구", "사하", "사상", "기장"],
        "경남": ["경남", "경상남도", "창원", "진주", "통영", "사천", "김해", "밀양", "거제", "양산"],
        "울산": ["울산", "ulsan", "중구", "남구", "동구", "북구", "울주"],
        "대구": ["대구", "daegu", "달서", "달성", "수성", "동구", "서구", "남구", "북구", "중구"],
        "경북": ["경북", "경상북도", "포항", "경주", "안동", "구미", "영주", "영천", "상주", "문경"],
        "광주": ["광주", "gwangju", "동구", "서구", "남구", "북구", "광산"],
        "전남": ["전남", "전라남도", "목포", "여수", "순천", "나주", "광양"],
        "전북": ["전북", "전라북도", "전주", "익산", "군산", "정읍", "남원"],
        "대전": ["대전", "daejeon", "동구", "중구", "서구", "유성", "대덕"],
        "충남": ["충남", "충청남도", "천안", "공주", "보령", "아산", "서산", "논산"],
        "충북": ["충북", "충청북도", "청주", "충주", "제천", "보은", "옥천"],
        "세종": ["세종", "sejong"],
        "강원": ["강원", "춘천", "원주", "강릉", "동해", "태백", "속초", "삼척"],
        "제주": ["제주", "jeju", "서귀포", "제주시"],
    }

    def _hypothesis_matches_evidence(self, h: Hypothesis, evidence: Evidence) -> bool:
        """단서가 해당 가설을 지지하는지 키워드 매칭"""
        desc_lower = evidence.description.lower()

        # 지역 키워드 매핑으로 매칭
        keywords = self._KR_REGION_KEYWORDS.get(h.location, [h.location])
        for kw in keywords:
            if kw.lower() in desc_lower:
                return True
        return False

    def _prune(self) -> None:
        for h in self.hypotheses.values():
            if not h.is_rejected and h.probability < PRUNE_THRESHOLD:
                h.is_rejected = True

    def _branch_if_needed(self) -> None:
        for h in self._active_hypotheses():
            if h.probability >= BRANCH_THRESHOLD and len(h.children_ids) == 0:
                self._create_sub_hypotheses(h)

    def _create_sub_hypotheses(self, parent: Hypothesis) -> None:
        """상위 가설에서 하위 지역 분기 생성 (LLM이 실제 후보 제공, 여기선 placeholder)"""
        # 실제로는 LLM이 하위 지역 후보를 생성해서 주입
        # 여기서는 빈 분기로 표시하고 LLM에게 채우도록 요청
        pass

    def inject_sub_hypotheses(self, parent_id: str, sub_locations: list[str]) -> None:
        """LLM이 생성한 하위 지역 가설 주입"""
        parent = self.hypotheses.get(parent_id)
        if not parent:
            return

        n = len(sub_locations)
        if n == 0:
            return

        # 부모 확률을 균등 분배
        sub_prob = parent.probability / n
        parent.probability = 0.0  # 부모는 합산용으로만
        parent.is_rejected = True

        for loc in sub_locations:
            h = Hypothesis(
                location=loc,
                probability=sub_prob,
                parent_id=parent_id,
                level=parent.level + 1,
            )
            self.hypotheses[h.id] = h
            parent.children_ids.append(h.id)

    def get_top_hypothesis(self) -> Optional[Hypothesis]:
        active = self._active_hypotheses()
        if not active:
            return None
        return max(active, key=lambda h: h.probability)

    def get_top_n(self, n: int = 3) -> list[Hypothesis]:
        active = self._active_hypotheses()
        return sorted(active, key=lambda h: h.probability, reverse=True)[:n]

    def get_max_confidence(self) -> float:
        top = self.get_top_hypothesis()
        return top.probability if top else 0.0

    def _active_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses.values() if not h.is_rejected]

    def to_state_prompt(self) -> str:
        """Layer 2 상태 프롬프트 생성"""
        lines = ["[현재 가설 트리]"]
        top = self.get_top_n(5)
        for h in top:
            pct = f"{h.probability * 100:.1f}%"
            lines.append(f"  {'  ' * h.level}{h.location} ({pct})")

        lines.append("")
        lines.append("[수집된 단서]")
        for ev in self.evidence_log[-10:]:  # 최근 10개
            tag = f"[{ev.confidence_level}]"
            contra = " [CONTRA]" if ev.is_contradiction else ""
            lines.append(f"  {ev.source}: {ev.description} {tag}{contra}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "hypotheses": [
                {
                    "id": h.id,
                    "location": h.location,
                    "probability": round(h.probability, 4),
                    "level": h.level,
                    "is_rejected": h.is_rejected,
                    "supporting_evidence_count": len(h.supporting_evidence),
                    "contradicting_evidence_count": len(h.contradicting_evidence),
                }
                for h in self.hypotheses.values()
            ],
            "evidence_count": len(self.evidence_log),
            "max_confidence": round(self.get_max_confidence(), 4),
            "step_count": self.step_count,
        }


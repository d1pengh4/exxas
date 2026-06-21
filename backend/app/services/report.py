"""
Explainable AI 리포트 생성
수사 결과를 사람이 읽을 수 있는 형태로 변환
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class InvestigationReport:
    # 헤더
    job_id: str = ""
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # 결론
    verdict: str = ""               # "확정" | "유력 추정" | "불확실" | "특정 불가"
    location: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    confidence_pct: str = ""        # "94.1%"
    confidence_label: str = ""

    # 수사 요약
    total_steps: int = 0
    elapsed_seconds: float = 0.0
    exploration_mode: str = ""
    tools_used: list[str] = field(default_factory=list)

    # 근거 체인 (단계별)
    evidence_chain: list[dict] = field(default_factory=list)
    key_evidence: list[str] = field(default_factory=list)   # 핵심 3개

    # 상세 분석
    final_reasoning: str = ""
    contradiction_notes: list[str] = field(default_factory=list)
    hallucination_checked: bool = False

    # 메타데이터
    image_manipulation_suspected: bool = False
    llm_model: str = ""


def build_report(analysis_data: dict) -> InvestigationReport:
    """DB/Redis 분석 결과에서 리포트 생성"""
    report = InvestigationReport(
        job_id=analysis_data.get("job_id", ""),
        location=analysis_data.get("location", "위치 특정 불가"),
        latitude=analysis_data.get("latitude"),
        longitude=analysis_data.get("longitude"),
        confidence_pct=f"{(analysis_data.get('confidence', 0) * 100):.1f}%",
        confidence_label=analysis_data.get("confidence_label", "UNKNOWN"),
        total_steps=analysis_data.get("total_steps", 0),
        elapsed_seconds=analysis_data.get("elapsed_seconds", 0),
        exploration_mode=analysis_data.get("exploration_mode", ""),
        final_reasoning=analysis_data.get("final_reasoning", ""),
        hallucination_checked=analysis_data.get("hallucination_check_passed", False),
    )

    # 결론 레이블
    conf = analysis_data.get("confidence", 0)
    if conf >= 0.90:
        report.verdict = "확정"
    elif conf >= 0.70:
        report.verdict = "유력 추정"
    elif conf >= 0.30:
        report.verdict = "불확실"
    else:
        report.verdict = "특정 불가"

    # 근거 체인
    chain = analysis_data.get("evidence_chain", [])
    report.evidence_chain = chain

    # 핵심 증거 추출 (HIGH 레벨 상위 3개)
    high_evidence = [
        ev["description"]
        for ev in chain
        if ev.get("confidence_level") == "HIGH" and not ev.get("is_contradiction")
    ]
    report.key_evidence = high_evidence[:3]

    # 사용된 도구 추출
    tools = list({ev["source"] for ev in chain if "source" in ev})
    report.tools_used = tools

    return report


def report_to_pdf_bytes(report: InvestigationReport) -> bytes:
    """리포트를 PDF bytes로 변환 (fpdf2 사용)"""
    try:
        from fpdf import FPDF
    except ImportError:
        raise RuntimeError("fpdf2 패키지 필요: pip install fpdf2")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    # 한글 폰트가 없으면 로마자만 사용
    # fpdf2는 기본적으로 latin-1이므로 한글은 인코딩 대체
    def safe(text: str) -> str:
        """한글을 ASCII로 대체 (폰트 미설치 환경)"""
        if text is None:
            return ""
        try:
            return text.encode("latin-1", errors="replace").decode("latin-1")
        except Exception:
            return text.encode("ascii", errors="replace").decode("ascii")

    # 제목
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "EXXAS Investigation Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Generated: {report.generated_at}  |  Job: {report.job_id}", ln=True, align="C")
    pdf.ln(4)

    # 구분선
    pdf.set_draw_color(100, 120, 200)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(6)

    # 결론 박스
    pdf.set_fill_color(20, 30, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, f"  Verdict: {safe(report.verdict)}  |  {safe(report.location)}", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    # 신뢰도 + 좌표
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"Confidence: {report.confidence_pct} ({safe(report.confidence_label)})", ln=True)
    if report.latitude is not None and report.longitude is not None:
        pdf.cell(0, 7, f"Coordinates: {report.latitude:.6f}, {report.longitude:.6f}", ln=True)
    pdf.ln(4)

    # 수사 요약
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Investigation Summary", ln=True)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Steps: {report.total_steps}   |   Elapsed: {report.elapsed_seconds:.1f}s   |   Mode: {safe(report.exploration_mode)}", ln=True)
    pdf.cell(0, 6, f"Hallucination Check: {'PASSED' if report.hallucination_checked else 'WARNING'}", ln=True)
    if report.image_manipulation_suspected:
        pdf.set_text_color(200, 50, 50)
        pdf.cell(0, 6, "! Image manipulation suspected (ELA+PRNU)", ln=True)
        pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # 핵심 근거
    if report.key_evidence:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Key Evidence", ln=True)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10)
        for i, ev in enumerate(report.key_evidence, 1):
            pdf.multi_cell(0, 6, f"{i}. {safe(ev)}")
        pdf.ln(4)

    # 사용 도구
    if report.tools_used:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Tools Used", ln=True)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 9)
        tools_str = ", ".join(safe(t) for t in report.tools_used)
        pdf.multi_cell(0, 5, tools_str)
        pdf.ln(4)

    # 최종 분석
    if report.final_reasoning:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Final Analysis", ln=True)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, safe(report.final_reasoning))
        pdf.ln(4)

    # 근거 체인 (최대 20개)
    chain = report.evidence_chain[:20]
    if chain:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, f"Evidence Chain ({len(report.evidence_chain)} total, showing {len(chain)})", ln=True)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(2)
        for ev in chain:
            pdf.set_font("Helvetica", "B", 9)
            level = ev.get("confidence_level", "")
            source = safe(ev.get("source", ""))
            pdf.cell(0, 6, f"[{level}] {source}", ln=True)
            pdf.set_font("Helvetica", "", 9)
            desc = safe(ev.get("description", ""))
            pdf.multi_cell(0, 5, desc)
            pdf.ln(1)

    # 푸터
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, f"EXXAS v2.0  |  {report.generated_at}", align="C")

    return bytes(pdf.output())


def report_to_markdown(report: InvestigationReport) -> str:
    """리포트를 Markdown으로 변환"""
    lines = [
        f"# EXXAS 수사 리포트",
        f"",
        f"**결론**: {report.verdict} — {report.location}",
        f"**신뢰도**: {report.confidence_pct} ({report.confidence_label})",
        f"**좌표**: {report.latitude}, {report.longitude}" if report.latitude else "",
        f"",
        f"## 수사 요약",
        f"- 탐색 단계: {report.total_steps}",
        f"- 소요 시간: {report.elapsed_seconds}초",
        f"- 탐색 모드: {report.exploration_mode}",
        f"- Hallucination 검증: {'통과' if report.hallucination_checked else '주의'}",
        f"",
        f"## 핵심 근거",
    ]

    for i, ev in enumerate(report.key_evidence, 1):
        lines.append(f"{i}. {ev}")

    lines += [
        f"",
        f"## 수사관 최종 분석",
        report.final_reasoning,
        f"",
        f"---",
        f"*EXXAS v2.0 · {report.generated_at}*",
    ]

    return "\n".join(lines)

"use client";
import dynamic from "next/dynamic";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AnalysisResult, EvidenceItem, submitFeedback, downloadReport } from "@/lib/api";
import HypothesisTreeView from "./HypothesisTreeView";

const MapView = dynamic(() => import("./MapView"), { ssr: false });

interface Props {
  result: AnalysisResult;
  imagePreview: string | null;
}

const SOURCE_LABEL: Record<string, string> = {
  exif: "EXIF",
  ocr: "OCR",
  geoclip: "GeoCLIP",
  vpr: "VPR",
  reverse_search: "역방향 검색",
  naver_place: "네이버 플레이스",
  kakao_place: "카카오맵",
  object_detect: "객체 탐지",
  physical: "물리 분석",
  web_search: "웹 검색",
  naver_blog: "네이버 블로그",
  osm_poi: "OSM POI",
  street_view: "스트리트뷰",
  url_crawl: "URL 크롤",
  korea_analyze: "한국 분석",
  juso_lookup: "도로명주소",
  roadview_check: "로드뷰",
  korea_specializer: "한국 특화",
};

const CONF_COLOR: Record<string, string> = {
  HIGH: "#22c55e", MEDIUM: "#f59e0b", LOW: "#ef4444", UNKNOWN: "#64748b",
};

const LEVEL_BG: Record<string, string> = {
  HIGH: "rgba(34,197,94,0.1)", MED: "rgba(245,158,11,0.1)", LOW: "rgba(100,116,139,0.1)", SPEC: "rgba(42,64,96,0.1)",
};
const LEVEL_COLOR: Record<string, string> = {
  HIGH: "#22c55e", MED: "#f59e0b", LOW: "#64748b", SPEC: "#475569",
};

type Tab = "result" | "evidence" | "hypothesis" | "reasoning";

function ConfRing({ value, label }: { value: number; label: string }) {
  const r = 38;
  const circ = 2 * Math.PI * r;
  const c = CONF_COLOR[label] || "#64748b";
  return (
    <div className="relative inline-flex items-center justify-center w-24 h-24">
      <svg width="96" height="96" style={{ transform: "rotate(-90deg)", position: "absolute" }}>
        <circle cx="48" cy="48" r={r} fill="none" stroke="var(--border)" strokeWidth="6" />
        <motion.circle cx="48" cy="48" r={r} fill="none" stroke={c} strokeWidth="6"
          strokeLinecap="round"
          initial={{ strokeDasharray: `0 ${circ}` }}
          animate={{ strokeDasharray: `${circ * Math.min(value, 1)} ${circ}` }}
          transition={{ duration: 1.2, ease: "easeOut", delay: 0.3 }} />
      </svg>
      <div className="relative z-10 text-center">
        <motion.div initial={{ opacity: 0, scale: 0.5 }} animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5, duration: 0.4 }}
          className="font-mono font-bold text-base leading-none" style={{ color: c }}>
          {(value * 100).toFixed(0)}%
        </motion.div>
        <div className="text-[9px] mt-0.5 font-medium" style={{ color: c }}>{label}</div>
      </div>
    </div>
  );
}

export default function AnalysisResultView({ result, imagePreview }: Props) {
  const [tab, setTab] = useState<Tab>("result");
  const [feedbackSent, setFeedbackSent] = useState<boolean | null>(null);
  const confColor = CONF_COLOR[result.confidence_label || "UNKNOWN"];
  const failed = result.status === "failed";

  if (failed) {
    return (
      <div className="rounded-xl border p-5" style={{ borderColor: "rgba(239,68,68,0.2)", background: "rgba(239,68,68,0.04)" }}>
        <div className="text-sm font-medium text-red-400 mb-2">분석 실패</div>
        <div className="text-sm" style={{ color: "var(--muted)" }}>{result.error}</div>
      </div>
    );
  }

  return (
    <div className="space-y-3">

      {/* Location result card */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
        className="rounded-xl border overflow-hidden"
        style={{ borderColor: `${confColor}30`, background: "var(--surface)" }}>

        <div className="px-4 py-2.5 border-b flex items-center justify-between"
          style={{ borderColor: `${confColor}18`, background: `${confColor}06` }}>
          <span className="text-xs font-medium" style={{ color: confColor }}>위치 특정</span>
          <div className="flex items-center gap-3 text-xs" style={{ color: "var(--muted)" }}>
            {result.total_steps && <span>{result.total_steps}단계</span>}
            {result.elapsed_seconds && <span>{result.elapsed_seconds}s</span>}
            {result.hallucination_check_passed !== undefined && (
              <span style={{ color: result.hallucination_check_passed ? "var(--success)" : "var(--danger)" }}>
                {result.hallucination_check_passed ? "검증됨" : "미검증"}
              </span>
            )}
          </div>
        </div>

        <div className="p-4 flex items-start gap-4">
          <ConfRing value={result.confidence || 0} label={result.confidence_label || "UNKNOWN"} />
          <div className="flex-1 min-w-0">
            <motion.div initial={{ opacity: 0, x: 6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}>
              <div className="text-lg font-semibold text-white leading-tight mb-1">
                {result.location || "위치 특정 불가"}
              </div>
              {result.address && result.address !== result.location && (
                <div className="text-sm mb-1.5" style={{ color: "var(--muted)" }}>{result.address}</div>
              )}
              {result.latitude && result.longitude && (
                <div className="font-mono text-xs" style={{ color: "var(--accent)" }}>
                  {result.latitude.toFixed(6)}°N, {result.longitude.toFixed(6)}°E
                </div>
              )}
            </motion.div>
            <div className="flex items-center gap-2 mt-2.5">
              {result.exploration_mode && (
                <span className="text-xs px-2 py-0.5 rounded border"
                  style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
                  {result.exploration_mode === "fast" ? "⚡ 빠른 경로" : result.exploration_mode === "elimination" ? "소거법" : "귀납법"}
                </span>
              )}
              {result.evidence_chain && (
                <span className="text-xs px-2 py-0.5 rounded border"
                  style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
                  증거 {result.evidence_chain.length}개
                </span>
              )}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Manipulation warning */}
      {result.image_manipulation_suspected && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          className="rounded-lg border p-3 flex gap-2.5"
          style={{ borderColor: "rgba(245,158,11,0.25)", background: "rgba(245,158,11,0.04)" }}>
          <span className="text-sm shrink-0">⚠️</span>
          <div>
            <div className="text-xs font-medium mb-0.5" style={{ color: "var(--warning)" }}>이미지 편집 감지</div>
            <p className="text-xs" style={{ color: "var(--muted)" }}>ELA 분석 또는 PRNU 핑거프린팅에서 편집·합성 흔적 탐지. 신뢰도가 저하될 수 있습니다.</p>
          </div>
        </motion.div>
      )}

      {/* AI generated warning */}
      {result.ai_generated_suspected && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          className="rounded-lg border p-3 flex gap-2.5"
          style={{ borderColor: "rgba(239,68,68,0.25)", background: "rgba(239,68,68,0.04)" }}>
          <span className="text-sm shrink-0">🤖</span>
          <div>
            <div className="text-xs font-medium mb-0.5" style={{ color: "var(--danger)" }}>AI 생성 이미지 의심</div>
            <p className="text-xs" style={{ color: "var(--muted)" }}>DCT 주파수 스펙트럼·노이즈 패턴에서 AI 생성 징후 탐지. 위치 특정 결과의 신뢰도가 크게 저하될 수 있습니다.</p>
          </div>
        </motion.div>
      )}

      {/* Map */}
      {result.latitude && result.longitude && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}
          className="rounded-xl border overflow-hidden" style={{ borderColor: "var(--border)" }}>
          <div className="px-4 py-2.5 border-b flex items-center justify-between"
            style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
            <span className="text-xs font-medium" style={{ color: "var(--success)" }}>지도</span>
            <span className="font-mono text-xs" style={{ color: "var(--muted)" }}>
              {result.latitude.toFixed(4)}, {result.longitude.toFixed(4)}
            </span>
          </div>
          <MapView
            latitude={result.latitude}
            longitude={result.longitude}
            location={result.location || ""}
            confidence={result.confidence || 0}
          />
        </motion.div>
      )}

      {/* Feedback + download */}
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-1.5">
          <span className="text-xs" style={{ color: "var(--muted)" }}>결과가 맞나요?</span>
          {feedbackSent === null ? (
            <>
              <button onClick={async () => { try { await submitFeedback(result.job_id, true); setFeedbackSent(true); } catch {} }}
                className="text-xs px-2.5 py-1 rounded border transition-colors hover:bg-green-500/10"
                style={{ borderColor: "rgba(34,197,94,0.3)", color: "var(--success)" }}>
                맞음
              </button>
              <button onClick={async () => { try { await submitFeedback(result.job_id, false); setFeedbackSent(false); } catch {} }}
                className="text-xs px-2.5 py-1 rounded border transition-colors hover:bg-red-500/10"
                style={{ borderColor: "rgba(239,68,68,0.3)", color: "var(--danger)" }}>
                틀림
              </button>
            </>
          ) : (
            <span className="text-xs" style={{ color: feedbackSent ? "var(--success)" : "var(--danger)" }}>
              {feedbackSent ? "정확하다고 표시됨" : "부정확하다고 표시됨"}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          {(["markdown","json","pdf"] as const).map((fmt) => (
            <button key={fmt} onClick={async () => { try { await downloadReport(result.job_id, fmt); } catch {} }}
              className="text-xs px-2.5 py-1 rounded border transition-all hover:text-white"
              style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
              .{fmt === "markdown" ? "md" : fmt}
            </button>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b" style={{ borderColor: "var(--border)" }}>
        {(["result","evidence","hypothesis","reasoning"] as Tab[]).map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className="px-4 py-2.5 text-sm transition-all relative"
            style={{ color: tab === t ? "var(--accent)" : "var(--muted)" }}>
            {tab === t && (
              <motion.div layoutId="tab-indicator"
                className="absolute bottom-0 left-0 right-0 h-0.5 rounded-t-sm"
                style={{ background: "var(--accent)" }} />
            )}
            {t === "result" ? "핵심 정보" : t === "evidence" ? "증거" : t === "hypothesis" ? "가설" : "보고서"}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <AnimatePresence mode="wait">
        {tab === "result" && (
          <motion.div key="intel" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            {result.evidence_chain && result.evidence_chain.length > 0 && (
              <div className="rounded-xl border overflow-hidden" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
                <div className="px-4 py-2.5 border-b" style={{ borderColor: "var(--border)" }}>
                  <span className="text-sm font-medium text-white">주요 근거</span>
                </div>
                <div className="p-4 space-y-3">
                  {result.evidence_chain
                    .filter((e) => e.confidence_level === "HIGH" && !e.is_contradiction)
                    .slice(0, 5)
                    .map((ev, i) => (
                      <motion.div key={ev.id} initial={{ opacity: 0, x: -4 }} animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.06 }}
                        className="flex items-start gap-3">
                        <span className="text-xs px-1.5 py-0.5 rounded mt-0.5 shrink-0 font-medium"
                          style={{ background: "rgba(91,141,239,0.1)", color: "var(--accent)" }}>
                          {SOURCE_LABEL[ev.source] || ev.source}
                        </span>
                        <div className="text-sm" style={{ color: "var(--muted)" }}>{ev.description}</div>
                      </motion.div>
                    ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {tab === "evidence" && result.evidence_chain && (
          <motion.div key="evidence" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            <EvidenceChain chain={result.evidence_chain} />
          </motion.div>
        )}

        {tab === "hypothesis" && result.hypothesis_tree && (
          <motion.div key="hyp" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            <HypothesisTreeView tree={result.hypothesis_tree} />
          </motion.div>
        )}

        {tab === "reasoning" && result.final_reasoning && (
          <motion.div key="reasoning" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="rounded-xl border overflow-hidden" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
            <div className="px-4 py-2.5 border-b" style={{ borderColor: "var(--border)" }}>
              <span className="text-sm font-medium text-white">최종 보고서</span>
            </div>
            <div className="p-4">
              <p className="text-sm leading-relaxed whitespace-pre-wrap" style={{ color: "var(--muted)" }}>{result.final_reasoning}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function EvidenceChain({ chain }: { chain: EvidenceItem[] }) {
  return (
    <div className="rounded-xl border overflow-hidden" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
      <div className="px-4 py-2.5 border-b flex items-center justify-between" style={{ borderColor: "var(--border)" }}>
        <span className="text-sm font-medium text-white">증거 체인</span>
        <span className="text-xs" style={{ color: "var(--muted)" }}>{chain.length}개</span>
      </div>
      <div className="p-4 space-y-2 max-h-80 overflow-y-auto">
        {chain.map((ev, i) => (
          <motion.div key={ev.id}
            initial={{ opacity: 0, x: -4 }} animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.02 }}
            className={`flex items-start gap-3 py-2 border-b last:border-0 ${ev.is_contradiction ? "opacity-40" : ""}`}
            style={{ borderColor: "var(--border)" }}>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1 flex-wrap">
                <span className="text-[10px] font-medium px-1.5 py-0.5 rounded"
                  style={{ color: LEVEL_COLOR[ev.confidence_level] || "var(--muted)", background: LEVEL_BG[ev.confidence_level] || "transparent" }}>
                  {ev.confidence_level === "HIGH" ? "높음" : ev.confidence_level === "MED" ? "중간" : "낮음"}
                </span>
                {ev.is_contradiction && (
                  <span className="text-[10px] border rounded px-1.5 py-0.5"
                    style={{ borderColor: "rgba(239,68,68,0.3)", color: "var(--danger)" }}>모순</span>
                )}
                <span className="text-xs" style={{ color: "var(--dim)" }}>
                  {SOURCE_LABEL[ev.source] || ev.source}
                </span>
              </div>
              <div className="text-sm" style={{ color: "var(--muted)" }}>{ev.description}</div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

"use client";
import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AnalysisResult } from "@/lib/api";

interface Props {
  result: AnalysisResult | null;
  jobId: string;
}

const TOOL_LABEL: Record<string, string> = {
  exif_extract:          "EXIF 포렌식 분석",
  ocr_extract:           "OCR 텍스트 추출",
  object_detect:         "인프라·객체 탐지",
  geoclip_embed:         "AI 지리 임베딩",
  reverse_image_search:  "역방향 이미지 검색",
  naver_place_search:    "네이버 플레이스 검색",
  kakao_place_search:    "카카오맵 검색",
  sun_analysis:          "태양·달 역산",
  vpr_compare:           "시각 장소 매칭",
  web_search:            "웹 OSINT 검색",
  search_naver_blog:     "네이버 블로그 탐색",
  osm_poi_search:        "OSM POI 검색",
  street_view_fetch:     "스트리트뷰 조회",
  deep_crawl_url:        "URL 딥 크롤",
  knowledge_graph_query: "지식 그래프 조회",
  receipt_parse:         "영수증·문서 파싱",
  brand_locate:          "브랜드 위치 추적",
  barcode_lookup:        "바코드 제조국 조회",
  interior_osint:        "실내 통합 OSINT",
  auto_chain:            "자동 단서 체인",
  biz_reg_lookup:        "사업자등록 조회",
  phone_lookup:          "전화번호 역추적",
  crawl_social:          "SNS 위치 크롤",
  reverse_chain:         "역방향 URL 체인",
  korea_analyze:         "한국 위치 특화 분석",
  juso_lookup:           "도로명주소 지오코딩",
  roadview_check:        "네이버 로드뷰 검증",
  license_plate_lookup:  "번호판 지역 확정",
  naver_news_search:     "네이버 뉴스 역추적",
  naver_local_search:    "네이버 로컬 검색",
  kakao_local_search:    "카카오 로컬 검색",
  flickr_search:         "Flickr 지오태그",
  news_image_search:     "뉴스 이미지 역추적",
  osint_fuse:            "OSINT 융합 판정",
  transit_lookup:        "대중교통 DB 매칭",
  weather_cross_check:   "날씨·계절 교차검증",
  skyline_match:         "스카이라인 매칭",
  clova_ocr:             "CLOVA OCR",
  shadow_analysis:       "그림자 방위각 분석",
};

interface LogEntry { step: number; tool: string; label: string; ts: number; }

const PIPELINE_STAGES = [
  { id: "stage0",      label: "이미지 검증",    sub: "ELA·해시" },
  { id: "stage1",      label: "EXIF 포렌식",    sub: "GPS·메타데이터" },
  { id: "stage3",      label: "OCR·GIS",        sub: "텍스트·지명" },
  { id: "stage4",      label: "비전 AI",        sub: "CLIP·YOLO" },
  { id: "stage5",      label: "AI 임베딩",      sub: "GeoCLIP·VPR" },
  { id: "stage6",      label: "물리 분석",      sub: "태양·DEM" },
  { id: "stage2",      label: "역방향 검색",    sub: "Google·Yandex" },
  { id: "investigate", label: "AI 수사관",      sub: "OSINT 추론" },
  { id: "stage7",      label: "최종 판정",      sub: "앙상블" },
] as const;

type StageStatus = "waiting" | "running" | "done" | "skipped";
type StageMap = Record<string, StageStatus>;

export default function InvestigationProgress({ result, jobId }: Props) {
  const [log, setLog] = useState<LogEntry[]>([]);
  const [stages, setStages] = useState<StageMap>({});
  const logRef = useRef<HTMLDivElement>(null);
  const prevStepRef = useRef(0);
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef(Date.now());

  const msg = (result as any)?.message || "";
  const tree = result?.hypothesis_tree;
  const maxConf = tree?.max_confidence ?? 0;
  const evidenceCount = tree?.evidence_count ?? 0;
  const step = (result as any)?.step || log.length;

  const activeHyps = tree?.hypotheses
    ?.filter((h: any) => !h.is_rejected)
    .sort((a: any, b: any) => b.probability - a.probability)
    .slice(0, 5) ?? [];

  useEffect(() => {
    const iv = setInterval(() => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)), 500);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    const ps = (result as any)?.pipeline_stage;
    if (ps?.id && ps?.status) {
      setStages(prev => ({ ...prev, [ps.id]: ps.status as StageStatus }));
    }
  }, [result]);

  useEffect(() => {
    const m = msg.match(/\[Step (\d+)\] (\w+) 실행 중/);
    if (m) {
      const s = parseInt(m[1]);
      if (s !== prevStepRef.current) {
        prevStepRef.current = s;
        setLog(prev => [...prev, { step: s, tool: m[2], label: TOOL_LABEL[m[2]] || m[2], ts: Date.now() }]);
      }
    }
  }, [msg]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  const confColor = maxConf >= 0.7 ? "var(--success)" : maxConf >= 0.4 ? "var(--warning)" : maxConf > 0 ? "var(--danger)" : "var(--dim)";
  const doneCount = Object.values(stages).filter(s => s === "done").length;

  return (
    <div className="space-y-4">

      {/* Status card */}
      <div className="rounded-xl border p-4" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full animate-pulse-soft" style={{ background: "var(--accent)" }} />
            <span className="text-sm font-medium text-white">분석 진행 중</span>
          </div>
          <div className="flex items-center gap-4 text-xs" style={{ color: "var(--muted)" }}>
            <span>{step}단계</span>
            <span>{evidenceCount}개 단서</span>
            <span className="font-mono">{elapsed}s</span>
          </div>
        </div>

        {/* Confidence */}
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-full flex items-center justify-center border-2 shrink-0 font-mono text-xs font-bold"
            style={{ borderColor: confColor, color: confColor }}>
            {maxConf > 0 ? `${(maxConf * 100).toFixed(0)}%` : "—"}
          </div>
          <div className="flex-1">
            {log[log.length - 1] && (
              <div className="text-sm text-white mb-2">{log[log.length - 1].label}</div>
            )}
            {maxConf > 0 && (
              <div className="h-1.5 rounded-full overflow-hidden relative" style={{ background: "var(--border)" }}>
                <motion.div className="h-full rounded-full relative overflow-hidden progress-shimmer"
                  style={{ background: confColor }}
                  animate={{ width: `${maxConf * 100}%` }}
                  transition={{ duration: 0.7, ease: "easeOut" }} />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Pipeline stages */}
      <div className="rounded-xl border overflow-hidden" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
        <div className="px-4 py-3 border-b flex items-center justify-between" style={{ borderColor: "var(--border)" }}>
          <span className="text-sm font-medium text-white">분석 단계</span>
          <span className="text-xs" style={{ color: "var(--muted)" }}>{doneCount} / {PIPELINE_STAGES.length}</span>
        </div>
        <div className="p-3 grid grid-cols-3 gap-2">
          {PIPELINE_STAGES.map((st) => {
            const s = stages[st.id] ?? "waiting";
            const isRunning = s === "running";
            const isDone = s === "done";
            const isSkipped = s === "skipped";
            return (
              <div key={st.id} className="rounded-lg p-2.5 border transition-all duration-300"
                style={{
                  borderColor: isDone ? "rgba(34,197,94,0.3)" : isRunning ? "rgba(91,141,239,0.4)" : "var(--border)",
                  background: isDone ? "rgba(34,197,94,0.04)" : isRunning ? "rgba(91,141,239,0.06)" : "transparent",
                  opacity: isSkipped ? 0.4 : 1,
                }}>
                <div className="flex items-center gap-1.5 mb-1">
                  {isDone ? (
                    <span className="text-xs" style={{ color: "var(--success)" }}>✓</span>
                  ) : isRunning ? (
                    <svg className="w-3 h-3 animate-spin shrink-0" style={{ color: "var(--accent)" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                  ) : (
                    <span className="w-3 h-3 rounded-full border" style={{ borderColor: "var(--dim)" }} />
                  )}
                </div>
                <div className="text-xs font-medium leading-tight"
                  style={{ color: isDone ? "var(--success)" : isRunning ? "var(--accent)" : "var(--muted)" }}>
                  {st.label}
                </div>
                <div className="text-[10px] leading-tight mt-0.5" style={{ color: "var(--dim)" }}>{st.sub}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Hypothesis candidates */}
      <div className="rounded-xl border" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
        <div className="px-4 py-3 border-b flex items-center justify-between" style={{ borderColor: "var(--border)" }}>
          <span className="text-sm font-medium text-white">후보 위치</span>
          <span className="text-xs" style={{ color: "var(--muted)" }}>{activeHyps.length}개</span>
        </div>
        <div className="p-4 space-y-3 min-h-[80px]">
          {activeHyps.length === 0 ? (
            <div className="flex items-center justify-center h-10">
              <span className="text-sm animate-pulse-soft" style={{ color: "var(--dim)" }}>계산 중...</span>
            </div>
          ) : (
            <AnimatePresence mode="popLayout">
              {activeHyps.map((h: any, i: number) => {
                const pct = h.probability * 100;
                const isTop = i === 0;
                const barColor = isTop ? "var(--accent)" : "var(--dim)";
                return (
                  <motion.div key={h.id} layout initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.04 }}>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs w-4 shrink-0 font-mono" style={{ color: "var(--muted)" }}>{i + 1}</span>
                      <span className={`flex-1 truncate text-sm ${isTop ? "text-white font-medium" : ""}`}
                        style={{ color: isTop ? undefined : "var(--muted)" }}>
                        {h.location}
                      </span>
                      <span className="text-xs font-mono font-medium shrink-0" style={{ color: isTop ? "var(--accent)" : "var(--muted)" }}>
                        {pct.toFixed(1)}%
                      </span>
                    </div>
                    <div className="ml-6 h-1 rounded-full overflow-hidden" style={{ background: "var(--border)" }}>
                      <motion.div className="h-full rounded-full" style={{ background: barColor }}
                        animate={{ width: `${pct}%` }}
                        transition={{ duration: 0.5, ease: "easeOut" }} />
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          )}
        </div>
      </div>

      {/* Operation log */}
      <div className="rounded-xl border overflow-hidden" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
        <div className="px-4 py-3 border-b flex items-center justify-between" style={{ borderColor: "var(--border)" }}>
          <span className="text-sm font-medium text-white">실행 로그</span>
          <span className="text-xs" style={{ color: "var(--muted)" }}>{log.length}개</span>
        </div>
        <div ref={logRef} className="p-3 space-y-1.5 max-h-48 overflow-y-auto">
          {log.length === 0 ? (
            <div className="text-sm animate-pulse-soft py-2" style={{ color: "var(--dim)" }}>초기화 중...</div>
          ) : (
            log.map((entry) => (
              <motion.div key={`${entry.step}-${entry.ts}`}
                initial={{ opacity: 0, x: -4 }} animate={{ opacity: 1, x: 0 }}
                className="flex items-center gap-2.5 text-xs">
                <span className="font-mono shrink-0" style={{ color: "var(--dim)" }}>{String(entry.step).padStart(2, "0")}</span>
                <span className="flex-1 truncate" style={{ color: "var(--muted)" }}>{entry.label}</span>
              </motion.div>
            ))
          )}
          <div className="text-xs" style={{ color: "var(--dim)" }}>
            <span className="animate-pulse-soft">▸</span>
          </div>
        </div>
      </div>
    </div>
  );
}

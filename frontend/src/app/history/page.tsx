"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import { getHistory, AnalysisResult } from "@/lib/api";
import { useAuth } from "@/hooks/useAuth";

const CONF_COLOR: Record<string, string> = {
  HIGH: "text-[#10B981]",
  MEDIUM: "text-[#F59E0B]",
  LOW: "text-[#F43F5E]",
  UNKNOWN: "text-[#64748B]",
};

export default function HistoryPage() {
  const { user, loading: authLoading } = useAuth();
  const router = useRouter();
  const [history, setHistory] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [page, setPage] = useState(0);
  const LIMIT = 10;

  useEffect(() => {
    if (!authLoading && !user) {
      router.push("/");
    }
  }, [authLoading, user, router]);

  useEffect(() => {
    if (!user) return;
    setLoading(true);
    getHistory(LIMIT, page * LIMIT)
      .then((rows) => { setHistory(rows); setError(""); })
      .catch(() => setError("히스토리를 불러오는 데 실패했습니다"))
      .finally(() => setLoading(false));
  }, [user, page]);

  if (authLoading || !user) return null;

  return (
    <div className="min-h-screen" style={{ background: "var(--bg)" }}>
      {/* 헤더 */}
      <header className="sticky top-0 z-50 border-b px-5 h-14 flex items-center justify-between"
              style={{ borderColor: "var(--border)", background: "rgba(11,12,19,0.96)", backdropFilter: "blur(12px)" }}>
        <div className="flex items-center gap-3">
          <Link href="/" className="text-lg font-bold tracking-wider text-white">EXXAS</Link>
          <span className="text-sm" style={{ color: "var(--muted)" }}>/ 분석 기록</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm" style={{ color: "var(--muted)" }}>{user.name}</span>
          <Link href="/" className="text-sm px-3 py-1.5 rounded-lg border transition-colors hover:text-white"
            style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
            새 분석
          </Link>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-10">
        <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">분석 기록</h1>
          <p className="text-sm text-[#64748B]">
            {user.plan === "free" ? "7일" : user.plan === "pro" ? "1년" : "무제한"} 보관 ·&nbsp;
            이번 달 {user.monthly_usage}/{user.monthly_limit}회 사용
          </p>
        </motion.div>

        {error && (
          <div className="mb-6 p-4 rounded-lg border border-[#F43F5E]/30 bg-[#F43F5E]/5 text-[#F43F5E] text-sm">
            {error}
          </div>
        )}

        {loading ? (
          <div className="flex flex-col gap-2">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-16 rounded-lg bg-[#111827] animate-pulse" />
            ))}
          </div>
        ) : history.length === 0 ? (
          <div className="text-center py-20 text-[#475569]">
            <p className="text-lg mb-2">분석 기록이 없습니다</p>
            <Link href="/" className="text-sm text-[#0EA5E9] hover:underline">첫 수사 시작하기</Link>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {history.map((item, i) => (
              <motion.div
                key={item.job_id}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.04 }}
                className="flex items-center gap-4 p-4 rounded-lg border border-[#1E2D45] bg-[#0D1220]
                  hover:border-[#0EA5E9]/30 hover:bg-[#0EA5E9]/3 transition-all group"
              >
                {/* 상태 점 */}
                <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
                  item.status === "completed" ? "bg-[#10B981]" :
                  item.status === "failed" ? "bg-[#F43F5E]" : "bg-[#F59E0B] animate-pulse"
                }`} />

                {/* 위치 */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white font-medium truncate">
                    {item.location || (item.status === "failed" ? "분석 실패" : "분석 중...")}
                  </p>
                  {item.address && item.address !== item.location && (
                    <p className="text-xs text-[#64748B] truncate mt-0.5">{item.address}</p>
                  )}
                  <p className="text-xs text-[#475569] font-mono mt-0.5">
                    {item.created_at
                      ? new Date(item.created_at).toLocaleString("ko-KR", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" })
                      : `ID: ${item.job_id.slice(0, 8)}`}
                    {item.total_steps ? ` · ${item.total_steps}단계` : ""}
                    {item.elapsed_seconds ? ` · ${item.elapsed_seconds.toFixed(1)}s` : ""}
                  </p>
                </div>

                {/* 신뢰도 */}
                {item.confidence !== undefined && item.confidence_label && (
                  <div className="text-right flex-shrink-0">
                    <p className={`text-sm font-bold font-mono ${CONF_COLOR[item.confidence_label] ?? "text-[#64748B]"}`}>
                      {(item.confidence * 100).toFixed(0)}%
                    </p>
                    <p className="text-xs text-[#475569]">{item.confidence_label}</p>
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        )}

        {/* 페이지네이션 */}
        {!loading && history.length > 0 && (
          <div className="flex items-center justify-center gap-3 mt-8">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-4 py-2 text-xs border border-[#1E2D45] rounded text-[#64748B]
                hover:text-white hover:border-[#0EA5E9]/40 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              이전
            </button>
            <span className="text-xs text-[#475569] font-mono">{page + 1} 페이지</span>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={history.length < LIMIT}
              className="px-4 py-2 text-xs border border-[#1E2D45] rounded text-[#64748B]
                hover:text-white hover:border-[#0EA5E9]/40 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              다음
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

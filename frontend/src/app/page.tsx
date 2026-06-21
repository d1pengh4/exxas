"use client";
import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { startAnalysis, streamAnalysis, getAnalysisResult, pollAnalysis, AnalysisResult } from "@/lib/api";
import AnalysisResultView from "@/components/AnalysisResultView";
import InvestigationProgress from "@/components/InvestigationProgress";
import AuthModal from "@/components/AuthModal";
import { useAuth } from "@/hooks/useAuth";

type Status = "idle" | "uploading" | "analyzing" | "done" | "error";

export default function HomePage() {
  const { user, loading: authLoading, refresh, logout } = useAuth();
  const [authOpen, setAuthOpen] = useState(false);
  const [authTab, setAuthTab] = useState<"login" | "register">("login");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [progress, setProgress] = useState<AnalysisResult | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [jobId, setJobId] = useState("");

  const onDrop = useCallback((accepted: File[]) => {
    const f = accepted[0];
    if (!f) return;
    setFile(f);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(f));
    setResult(null); setProgress(null); setStatus("idle"); setError("");
  }, [preview]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp", ".heic"] },
    maxFiles: 1, maxSize: 50 * 1024 * 1024,
  });

  const handleAnalyze = async () => {
    if (!file) return;
    setStatus("uploading"); setError("");
    try {
      const job = await startAnalysis(file);
      setJobId(job.job_id); setStatus("analyzing");
      let closed = false;
      const close = streamAnalysis(
        job.job_id,
        (r) => setProgress(r),
        async (r) => { closed = true; setResult(r); setStatus("done"); close(); if (user) refresh(); },
        async () => {
          if (closed) return;
          try {
            const final = await pollAnalysis(job.job_id, (r) => setProgress(r));
            setResult(final); setStatus("done");
            if (final.error) setError(final.error);
            if (user) refresh();
          } catch {
            try {
              const final = await getAnalysisResult(job.job_id);
              setResult(final); setStatus("done");
              if (user) refresh();
            } catch {
              setStatus("error"); setError("분석 결과를 가져오지 못했습니다. 잠시 후 재시도하세요.");
            }
          }
        },
      );
    } catch (e: any) {
      setError(e?.response?.data?.detail || e.message || "오류가 발생했습니다");
      setStatus("error");
    }
  };

  const handleReset = () => {
    setFile(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null); setStatus("idle"); setResult(null); setProgress(null); setError(""); setJobId("");
  };

  const usageLeft = user ? Math.max(0, user.monthly_limit - user.monthly_usage) : null;
  const isAnalyzing = status === "analyzing" || status === "uploading";

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "var(--bg)" }}>

      {/* Header */}
      <header className="sticky top-0 z-50 border-b" style={{ borderColor: "var(--border)", background: "rgba(11,12,19,0.96)", backdropFilter: "blur(12px)" }}>
        <div className="max-w-6xl mx-auto px-5 h-14 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <span className="text-lg font-bold tracking-wider text-white">EXXAS</span>
            <nav className="hidden md:flex items-center gap-4 text-sm" style={{ color: "var(--muted)" }}>
              <Link href="/history" className="hover:text-white transition-colors">분석 기록</Link>
              <Link href="/plans" className="hover:text-white transition-colors">플랜</Link>
            </nav>
          </div>

          <div className="flex items-center gap-2">
            {!authLoading && (
              user ? (
                <div className="flex items-center gap-2">
                  <span className="text-xs px-2.5 py-1 rounded-full border font-medium"
                    style={{ borderColor: "var(--accent)", color: "var(--accent)", background: "rgba(91,141,239,0.08)" }}>
                    {user.plan.toUpperCase()}
                    {usageLeft !== null && <span style={{ color: "var(--muted)", marginLeft: 4 }}>· {usageLeft}회</span>}
                  </span>
                  <div className="relative group">
                    <button className="text-sm px-3 py-1.5 rounded-lg transition-colors hover:bg-white/5" style={{ color: "var(--muted)" }}>
                      {user.name}
                    </button>
                    <div className="absolute right-0 top-full mt-1 w-28 rounded-lg border shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50"
                      style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
                      <button onClick={logout} className="w-full text-left px-3 py-2 text-sm transition-colors hover:text-red-400" style={{ color: "var(--muted)" }}>
                        로그아웃
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <button onClick={() => { setAuthTab("login"); setAuthOpen(true); }}
                    className="text-sm px-3 py-1.5 rounded-lg transition-colors hover:bg-white/5" style={{ color: "var(--muted)" }}>
                    로그인
                  </button>
                  <button onClick={() => { setAuthTab("register"); setAuthOpen(true); }}
                    className="text-sm px-4 py-1.5 rounded-lg font-medium text-white transition-all hover:opacity-90"
                    style={{ background: "var(--accent)" }}>
                    회원가입
                  </button>
                </div>
              )
            )}
            {(result || isAnalyzing) && (
              <button onClick={handleReset}
                className="text-sm px-3 py-1.5 rounded-lg border transition-all hover:bg-red-500/10"
                style={{ borderColor: "rgba(239,68,68,0.3)", color: "var(--danger)" }}>
                새 분석
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1">
        <AnimatePresence mode="wait">

          {/* Idle: Upload */}
          {status === "idle" && (
            <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="max-w-xl mx-auto px-5 py-16">

              <div className="text-center mb-10">
                <motion.h1 initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}
                  className="text-4xl font-bold text-white mb-3">
                  사진으로 위치 찾기
                </motion.h1>
                <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.15 }}
                  className="text-base leading-relaxed" style={{ color: "var(--muted)" }}>
                  사진 한 장으로 대한민국 촬영지를 특정합니다<br />
                  AI·OSINT·역방향 검색을 조합해 GPS 좌표로 증명합니다
                </motion.p>
              </div>

              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
                {/* Drop zone */}
                <div {...getRootProps()} className="rounded-xl cursor-pointer transition-all duration-200 overflow-hidden"
                  style={{
                    border: `2px dashed ${isDragActive ? "var(--accent)" : "var(--border)"}`,
                    background: isDragActive ? "rgba(91,141,239,0.05)" : "var(--surface)",
                    minHeight: 200,
                  }}>
                  <input {...getInputProps()} />
                  {preview ? (
                    <div>
                      <div className="relative group">
                        <img src={preview} alt="preview" className="w-full max-h-64 object-contain" style={{ background: "#08090f" }} />
                        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                          <span className="text-sm text-white/80">클릭하여 교체</span>
                        </div>
                      </div>
                      <div className="px-4 py-2.5 border-t flex items-center gap-2" style={{ borderColor: "var(--border)" }}>
                        <svg className="w-3.5 h-3.5 shrink-0" style={{ color: "var(--muted)" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14" />
                        </svg>
                        <span className="text-sm truncate flex-1" style={{ color: "var(--muted)" }}>{file?.name}</span>
                        <span className="text-xs" style={{ color: "var(--dim)" }}>{((file?.size || 0) / 1024 / 1024).toFixed(2)} MB</span>
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-14 px-8 text-center gap-3">
                      <div className="w-12 h-12 rounded-xl flex items-center justify-center" style={{ background: "rgba(91,141,239,0.1)" }}>
                        <svg className="w-6 h-6" style={{ color: "var(--accent)" }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                        </svg>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-white mb-1">이미지를 드래그하거나 클릭하세요</p>
                        <p className="text-xs" style={{ color: "var(--dim)" }}>JPEG · PNG · WEBP · 최대 50MB</p>
                      </div>
                    </div>
                  )}
                </div>

                <AnimatePresence>
                  {preview && (
                    <motion.button initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                      onClick={handleAnalyze}
                      className="w-full mt-3 py-3 rounded-xl text-sm font-semibold text-white transition-all hover:opacity-90 active:scale-[0.99]"
                      style={{ background: "var(--accent)" }}>
                      위치 분석 시작
                    </motion.button>
                  )}
                  {!!error && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                      className="mt-3 p-3 rounded-lg border text-sm"
                      style={{ borderColor: "rgba(239,68,68,0.3)", background: "rgba(239,68,68,0.05)", color: "var(--danger)" }}>
                      {error}
                    </motion.div>
                  )}
                </AnimatePresence>

                {!preview && (
                  <div className="grid grid-cols-4 gap-2 mt-5">
                    {[
                      { label: "8단계 파이프라인" },
                      { label: "AI 지리 임베딩" },
                      { label: "역방향 이미지 검색" },
                      { label: "OSINT 10종 도구" },
                    ].map((c) => (
                      <div key={c.label} className="rounded-lg p-3 text-center border" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
                        <div className="text-xs" style={{ color: "var(--muted)" }}>{c.label}</div>
                      </div>
                    ))}
                  </div>
                )}
              </motion.div>
            </motion.div>
          )}

          {/* Analyzing */}
          {isAnalyzing && (
            <motion.div key="analyzing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              className="h-[calc(100vh-57px)] flex flex-col">

              {/* Sub-header */}
              <div className="border-b px-5 py-2.5 flex items-center gap-3" style={{ borderColor: "var(--border)", background: "rgba(17,18,25,0.8)" }}>
                <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse-soft" />
                <span className="text-sm font-medium text-white">분석 중</span>
                <span className="text-xs ml-1" style={{ color: "var(--muted)" }}>ID: {jobId.slice(0, 12)}</span>
                <div className="ml-auto flex items-center gap-1.5 text-xs" style={{ color: "var(--dim)" }}>
                  {["EXIF","OCR","YOLO","CLIP","REV","GEO","PHY","ENS"].map((s) => (
                    <span key={s} className="px-1.5 py-0.5 rounded text-[10px]"
                      style={{ background: "var(--border)", color: "var(--muted)" }}>{s}</span>
                  ))}
                </div>
              </div>

              <div className="flex-1 grid grid-cols-[40%_60%] overflow-hidden">
                {/* Image */}
                <div className="border-r flex flex-col" style={{ borderColor: "var(--border)" }}>
                  <div className="px-4 py-2.5 border-b flex items-center gap-2 text-xs" style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
                    <span className="flex-1 truncate">{file?.name}</span>
                    <span style={{ color: "var(--dim)" }}>{((file?.size || 0)/1024/1024).toFixed(2)}MB</span>
                  </div>
                  <div className="flex-1 relative overflow-hidden" style={{ background: "#08090f" }}>
                    {preview && <img src={preview} alt="target" className="w-full h-full object-contain" />}
                    <div className="absolute bottom-4 left-0 right-0 flex justify-center pointer-events-none">
                      <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs"
                        style={{ background: "rgba(11,12,19,0.85)", border: "1px solid var(--border)", color: "var(--muted)" }}>
                        <svg className="w-3 h-3 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                        분석 중
                      </div>
                    </div>
                  </div>
                </div>

                {/* Progress */}
                <div className="overflow-y-auto p-5">
                  <InvestigationProgress result={progress} jobId={jobId} />
                </div>
              </div>
            </motion.div>
          )}

          {/* Error */}
          {status === "error" && (
            <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="max-w-md mx-auto px-5 py-20 text-center">
              <div className="rounded-xl p-8 border" style={{ background: "var(--surface)", borderColor: "rgba(239,68,68,0.2)" }}>
                <div className="text-4xl mb-4">⚠️</div>
                <div className="text-base font-medium text-white mb-2">분석 실패</div>
                <div className="text-sm mb-6" style={{ color: "var(--muted)" }}>{error || "분석 중 오류가 발생했습니다"}</div>
                <button onClick={handleReset}
                  className="text-sm px-5 py-2 rounded-lg border transition-all hover:bg-red-500/10"
                  style={{ borderColor: "rgba(239,68,68,0.4)", color: "var(--danger)" }}>
                  다시 시도
                </button>
              </div>
            </motion.div>
          )}

          {/* Done */}
          {status === "done" && result && (
            <motion.div key="done" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="max-w-5xl mx-auto px-5 py-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                <div className="rounded-xl overflow-hidden border" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
                  <div className="px-4 py-2.5 border-b flex items-center justify-between" style={{ borderColor: "var(--border)" }}>
                    <span className="text-sm" style={{ color: "var(--muted)" }}>{file?.name}</span>
                    <span className="text-xs font-medium" style={{ color: "var(--success)" }}>분석 완료</span>
                  </div>
                  {preview && (
                    <img src={preview} alt="result" className="w-full max-h-80 object-contain" style={{ background: "#08090f" }} />
                  )}
                  <div className="px-4 py-2 text-xs" style={{ color: "var(--dim)", borderTop: "1px solid var(--border)" }}>
                    {((file?.size || 0)/1024/1024).toFixed(2)} MB
                  </div>
                </div>
                <div>
                  <AnalysisResultView result={result} imagePreview={preview} />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      {status === "idle" && (
        <footer className="border-t py-4 px-5" style={{ borderColor: "var(--border)", background: "rgba(11,12,19,0.8)" }}>
          <div className="max-w-xl mx-auto text-center text-xs" style={{ color: "var(--dim)" }}>
            EXXAS v2.0 · AI 기반 위치 수사 플랫폼
          </div>
        </footer>
      )}

      <AuthModal open={authOpen} onClose={() => setAuthOpen(false)}
        onSuccess={() => { setAuthOpen(false); refresh(); }} defaultTab={authTab} />
    </div>
  );
}

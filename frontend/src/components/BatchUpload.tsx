"use client";
import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "@/lib/api";
import { UserInfo } from "@/lib/api";

interface BatchJob {
  job_id: string;
  filename: string;
  status: "queued" | "running" | "completed" | "failed";
  location?: string;
  confidence?: number;
  confidence_label?: string;
}

interface Props {
  user: UserInfo;
}

export default function BatchUpload({ user }: Props) {
  const maxFiles = user.plan === "expert" ? 50 : user.plan === "pro" ? 10 : 1;
  const [files, setFiles] = useState<File[]>([]);
  const [jobs, setJobs] = useState<BatchJob[]>([]);
  const [running, setRunning] = useState(false);
  const [batchId, setBatchId] = useState("");
  const [error, setError] = useState("");

  const onDrop = useCallback((accepted: File[]) => {
    setFiles((prev) => {
      const merged = [...prev, ...accepted];
      return merged.slice(0, maxFiles);
    });
    setError("");
  }, [maxFiles]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    maxFiles,
    maxSize: 50 * 1024 * 1024,
  });

  const removeFile = (i: number) => setFiles((f) => f.filter((_, idx) => idx !== i));

  const handleBatch = async () => {
    if (!files.length) return;
    setRunning(true);
    setError("");
    setJobs([]);

    try {
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      const { data } = await api.post("/api/v1/analyze/batch", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setBatchId(data.batch_id);
      setJobs(data.job_ids.map((id: string, i: number) => ({
        job_id: id,
        filename: files[i]?.name ?? id,
        status: "queued",
      })));

      // 폴링으로 결과 업데이트
      const interval = setInterval(async () => {
        try {
          const { data: batchData } = await api.get(`/api/v1/analyze/batch/${data.batch_id}`);
          setJobs(batchData.jobs.map((j: any, i: number) => ({
            job_id: j.job_id,
            filename: files[i]?.name ?? j.job_id,
            status: j.status,
            location: j.location,
            confidence: j.confidence,
            confidence_label: j.confidence_label,
          })));
          if (batchData.pending === 0) {
            clearInterval(interval);
            setRunning(false);
          }
        } catch {
          clearInterval(interval);
          setRunning(false);
        }
      }, 2500);
    } catch (e: any) {
      setError(e?.response?.data?.detail || "배치 분석 실패");
      setRunning(false);
    }
  };

  const CONF_COLOR: Record<string, string> = {
    HIGH: "text-[#10B981]", MEDIUM: "text-[#F59E0B]", LOW: "text-[#F43F5E]", UNKNOWN: "text-[#64748B]",
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold text-[#94A3B8] uppercase tracking-widest">배치 분석</h2>
        <span className="text-xs text-[#475569] font-mono">최대 {maxFiles}장</span>
      </div>

      {/* 드롭존 */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all
          ${isDragActive ? "border-[#0EA5E9] bg-[#0EA5E9]/5" : "border-[#1E2D45] hover:border-[#0EA5E9]/30"}`}
      >
        <input {...getInputProps()} />
        <p className="text-sm text-[#64748B]">
          {isDragActive ? "여기에 놓으세요" : `이미지 ${maxFiles}장까지 드래그하거나 클릭`}
        </p>
      </div>

      {/* 파일 목록 */}
      {files.length > 0 && !jobs.length && (
        <div className="flex flex-col gap-1.5">
          {files.map((f, i) => (
            <div key={i} className="flex items-center gap-3 p-2.5 bg-[#111827] rounded-lg border border-[#1E2D45]">
              <span className="flex-1 text-xs text-[#94A3B8] truncate font-mono">{f.name}</span>
              <span className="text-xs text-[#475569]">{(f.size / 1024).toFixed(0)}KB</span>
              <button onClick={() => removeFile(i)} className="text-[#475569] hover:text-[#F43F5E] text-xs transition-colors">✕</button>
            </div>
          ))}
        </div>
      )}

      {/* 실행 버튼 */}
      {files.length > 0 && !jobs.length && (
        <button
          onClick={handleBatch}
          disabled={running}
          className="py-2.5 rounded-lg text-sm font-bold tracking-[2px] uppercase
            bg-[#0EA5E9] hover:bg-[#38BDF8] text-black disabled:opacity-50 transition-colors"
        >
          {running ? "분석 중..." : `${files.length}장 일괄 수사`}
        </button>
      )}

      {error && <p className="text-xs text-[#F43F5E]">{error}</p>}

      {/* 결과 목록 */}
      <AnimatePresence>
        {jobs.length > 0 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-1.5">
            {jobs.map((job, i) => (
              <div key={job.job_id} className="flex items-center gap-3 p-2.5 bg-[#0D1220] rounded-lg border border-[#1E2D45]">
                <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
                  job.status === "completed" ? "bg-[#10B981]" :
                  job.status === "failed" ? "bg-[#F43F5E]" : "bg-[#F59E0B] animate-pulse"
                }`} />
                <span className="flex-1 text-xs text-[#64748B] truncate font-mono">{job.filename}</span>
                {job.status === "completed" && job.location && (
                  <span className="text-xs text-[#94A3B8] truncate max-w-[160px]">{job.location}</span>
                )}
                {job.confidence !== undefined && job.confidence_label && (
                  <span className={`text-xs font-bold font-mono ${CONF_COLOR[job.confidence_label] ?? "text-[#64748B]"}`}>
                    {(job.confidence * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            ))}
            {running && (
              <p className="text-xs text-[#475569] text-center font-mono animate-pulse">분석 진행 중...</p>
            )}
            {!running && jobs.length > 0 && (
              <button
                onClick={() => { setFiles([]); setJobs([]); setBatchId(""); }}
                className="text-xs text-[#64748B] hover:text-[#94A3B8] text-center mt-1 transition-colors"
              >
                초기화
              </button>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

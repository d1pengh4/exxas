import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_URL,
  timeout: 60000,
});

// 토큰 자동 주입
api.interceptors.request.use((config) => {
  const token = typeof window !== "undefined" ? localStorage.getItem("exxas_token") : null;
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// ── 분석 API ──────────────────────────────────────────────

export interface AnalysisJob {
  job_id: string;
  status: string;
  message: string;
}

export interface AnalysisResult {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  location?: string;
  latitude?: number;
  longitude?: number;
  address?: string;
  confidence?: number;
  confidence_label?: "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN";
  exploration_mode?: string;
  total_steps?: number;
  elapsed_seconds?: number;
  evidence_chain?: EvidenceItem[];
  hypothesis_tree?: HypothesisTree;
  final_reasoning?: string;
  hallucination_check_passed?: boolean;
  image_manipulation_suspected?: boolean;
  ai_generated_suspected?: boolean;
  created_at?: string;
  error?: string;
}

export interface EvidenceItem {
  id: string;
  source: string;
  description: string;
  confidence_level: "HIGH" | "MED" | "LOW" | "SPEC";
  is_contradiction: boolean;
}

export interface HypothesisTree {
  hypotheses: HypothesisNode[];
  evidence_count: number;
  max_confidence: number;
  step_count: number;
}

export interface HypothesisNode {
  id: string;
  location: string;
  probability: number;
  level: number;
  is_rejected: boolean;
  supporting_evidence_count: number;
  contradicting_evidence_count: number;
}

export async function startAnalysis(file: File): Promise<AnalysisJob> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<AnalysisJob>("/api/v1/analyze", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function getAnalysisResult(jobId: string): Promise<AnalysisResult> {
  const { data } = await api.get<AnalysisResult>(`/api/v1/analyze/${jobId}`);
  return data;
}

export function streamAnalysis(
  jobId: string,
  onUpdate: (result: AnalysisResult) => void,
  onDone: (result: AnalysisResult) => void,
  onError: (error: string) => void
): () => void {
  const evtSource = new EventSource(`${API_URL}/api/v1/analyze/${jobId}/stream`);

  evtSource.onmessage = (e) => {
    try {
      const data: AnalysisResult = JSON.parse(e.data);
      if (data.status === "completed" || data.status === "failed") {
        onDone(data);
        evtSource.close();
      } else {
        onUpdate(data);
      }
    } catch {
      // ignore parse errors
    }
  };

  evtSource.onerror = () => {
    onError("스트리밍 연결 오류");
    evtSource.close();
  };

  return () => evtSource.close();
}

// ── 폴링 헬퍼 ─────────────────────────────────────────────
export async function pollAnalysis(
  jobId: string,
  onUpdate: (result: AnalysisResult) => void,
  intervalMs = 2000
): Promise<AnalysisResult> {
  return new Promise((resolve, reject) => {
    const interval = setInterval(async () => {
      try {
        const result = await getAnalysisResult(jobId);
        onUpdate(result);
        if (result.status === "completed" || result.status === "failed") {
          clearInterval(interval);
          resolve(result);
        }
      } catch (e) {
        clearInterval(interval);
        reject(e);
      }
    }, intervalMs);
  });
}

// ── 인증 API ──────────────────────────────────────────────

export interface AuthToken {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface UserInfo {
  id: number;
  email: string;
  name: string;
  plan: string;
  monthly_usage: number;
  monthly_limit: number;
  can_analyze: boolean;
}

export async function register(email: string, password: string, name?: string): Promise<AuthToken> {
  const { data } = await api.post<AuthToken>("/api/v1/auth/register", { email, password, name });
  if (typeof window !== "undefined") {
    localStorage.setItem("exxas_token", data.access_token);
  }
  return data;
}

export async function login(email: string, password: string): Promise<AuthToken> {
  const form = new URLSearchParams({ username: email, password });
  const { data } = await api.post<AuthToken>("/api/v1/auth/token", form.toString(), {
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
  });
  if (typeof window !== "undefined") {
    localStorage.setItem("exxas_token", data.access_token);
  }
  return data;
}

export function logout(): void {
  if (typeof window !== "undefined") {
    localStorage.removeItem("exxas_token");
  }
}

export async function getMe(): Promise<UserInfo> {
  const { data } = await api.get<UserInfo>("/api/v1/auth/me");
  return data;
}

// ── 피드백 API ─────────────────────────────────────────────

export async function submitFeedback(
  jobId: string,
  isCorrect: boolean,
  actualLocation?: string
): Promise<void> {
  await api.post(`/api/v1/analyze/${jobId}/feedback`, {
    is_correct: isCorrect,
    actual_location: actualLocation || "",
  });
}

// ── 리포트 다운로드 ────────────────────────────────────────

export async function downloadReport(
  jobId: string,
  format: "markdown" | "json" | "pdf" = "markdown"
): Promise<void> {
  const mimeMap = { markdown: "text/markdown", json: "application/json", pdf: "application/pdf" };
  const extMap = { markdown: "md", json: "json", pdf: "pdf" };
  const response = await api.get(`/api/v1/analyze/${jobId}/report`, {
    params: { format },
    responseType: "blob",
  });
  const blob = new Blob([response.data], { type: mimeMap[format] });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `exxas_report_${jobId.slice(0, 8)}.${extMap[format]}`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── 플랜 정보 ──────────────────────────────────────────────

export interface PlanFeatures {
  plan: string;
  monthly_limit: number;
  monthly_usage: number;
  features: Record<string, boolean | number>;
}

export async function getPlanFeatures(): Promise<PlanFeatures> {
  const { data } = await api.get<PlanFeatures>("/api/v1/plan/features");
  return data;
}

// ── 분석 히스토리 ──────────────────────────────────────────

export async function getHistory(limit = 10, offset = 0): Promise<AnalysisResult[]> {
  const { data } = await api.get<AnalysisResult[]>("/api/v1/analyses/history", {
    params: { limit, offset },
  });
  return data;
}

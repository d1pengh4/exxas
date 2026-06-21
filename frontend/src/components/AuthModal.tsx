"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { register, login } from "@/lib/api";

interface Props {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
  defaultTab?: "login" | "register";
}

const PLAN_LABELS: Record<string, string> = {
  free: "Free",
  pro: "Pro",
  expert: "Expert",
};

export default function AuthModal({ open, onClose, onSuccess, defaultTab = "login" }: Props) {
  const [tab, setTab] = useState<"login" | "register">(defaultTab);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const reset = () => {
    setEmail(""); setPassword(""); setName(""); setError(""); setLoading(false);
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) return;
    setLoading(true);
    setError("");
    try {
      if (tab === "register") {
        await register(email, password, name || undefined);
      } else {
        await login(email, password);
      }
      reset();
      onSuccess();
    } catch (err: any) {
      setError(err?.response?.data?.detail || "오류가 발생했습니다");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
        >
          {/* 배경 */}
          <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={handleClose} />

          <motion.div
            className="relative w-full max-w-sm mx-4 rounded-xl border border-[#1E2D45] bg-[#0D1220] overflow-hidden"
            initial={{ scale: 0.95, y: 12 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.95, y: 12 }}
          >
            {/* 상단 탭 */}
            <div className="flex border-b border-[#1E2D45]">
              {(["login", "register"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => { setTab(t); setError(""); }}
                  className={`flex-1 py-3.5 text-sm font-bold tracking-wider uppercase transition-colors
                    ${tab === t ? "text-[#0EA5E9] border-b-2 border-[#0EA5E9] -mb-px bg-[#0EA5E9]/5" : "text-[#64748B] hover:text-[#94A3B8]"}`}
                >
                  {t === "login" ? "로그인" : "회원가입"}
                </button>
              ))}
            </div>

            <form onSubmit={handleSubmit} className="p-6 flex flex-col gap-4">
              {tab === "register" && (
                <div className="flex flex-col gap-1.5">
                  <label className="text-xs text-[#64748B] uppercase tracking-widest font-bold">이름 (선택)</label>
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="홍길동"
                    className="w-full bg-[#111827] border border-[#1E2D45] rounded-lg px-4 py-2.5 text-sm text-white
                      placeholder:text-[#475569] focus:outline-none focus:border-[#0EA5E9] transition-colors"
                  />
                </div>
              )}

              <div className="flex flex-col gap-1.5">
                <label className="text-xs text-[#64748B] uppercase tracking-widest font-bold">이메일</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="user@example.com"
                  required
                  className="w-full bg-[#111827] border border-[#1E2D45] rounded-lg px-4 py-2.5 text-sm text-white
                    placeholder:text-[#475569] focus:outline-none focus:border-[#0EA5E9] transition-colors"
                />
              </div>

              <div className="flex flex-col gap-1.5">
                <label className="text-xs text-[#64748B] uppercase tracking-widest font-bold">비밀번호</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  required
                  className="w-full bg-[#111827] border border-[#1E2D45] rounded-lg px-4 py-2.5 text-sm text-white
                    placeholder:text-[#475569] focus:outline-none focus:border-[#0EA5E9] transition-colors"
                />
              </div>

              {error && (
                <p className="text-xs text-[#F43F5E] bg-[#F43F5E]/10 border border-[#F43F5E]/20 rounded-lg px-3 py-2">
                  {error}
                </p>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full py-3 rounded-lg font-bold tracking-[3px] uppercase text-sm bg-[#0EA5E9]
                  hover:bg-[#38BDF8] text-black transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "처리 중..." : tab === "login" ? "로그인" : "가입하기"}
              </button>

              {tab === "register" && (
                <p className="text-xs text-[#64748B] text-center leading-relaxed">
                  가입 시 Free 플랜으로 시작 — 월 5회 분석 무료
                </p>
              )}
            </form>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

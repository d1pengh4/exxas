"use client";
import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { useAuth } from "@/hooks/useAuth";
import AuthModal from "@/components/AuthModal";

const PLANS = [
  {
    id: "free",
    name: "Free",
    price: "₩0",
    period: "무료",
    color: "#64748B",
    features: [
      { label: "월간 분석", value: "5회" },
      { label: "분석 레이어", value: "EXIF + 기본 임베딩" },
      { label: "OSINT 역추적", value: false },
      { label: "조작 탐지", value: false },
      { label: "히스토리", value: "7일" },
      { label: "배치 업로드", value: false },
      { label: "리포트 다운로드", value: false },
      { label: "API 접근", value: false },
    ],
  },
  {
    id: "pro",
    name: "Pro",
    price: "₩9,900",
    period: "/ 월",
    color: "#0EA5E9",
    featured: true,
    features: [
      { label: "월간 분석", value: "100회" },
      { label: "분석 레이어", value: "전체 7개 레이어" },
      { label: "OSINT 역추적", value: false },
      { label: "조작 탐지", value: "기본 ELA + PRNU" },
      { label: "히스토리", value: "1년" },
      { label: "배치 업로드", value: "최대 10장" },
      { label: "리포트 다운로드", value: ".md / .json" },
      { label: "API 접근", value: false },
    ],
  },
  {
    id: "expert",
    name: "Expert",
    price: "₩29,900",
    period: "/ 월",
    color: "#F59E0B",
    features: [
      { label: "월간 분석", value: "무제한" },
      { label: "분석 레이어", value: "전체 7개 + 풀 OSINT" },
      { label: "OSINT 역추적", value: "5종 + Wayback + 블로그" },
      { label: "조작 탐지", value: "고급 포렌식 전체" },
      { label: "히스토리", value: "무제한" },
      { label: "배치 업로드", value: "최대 50장" },
      { label: "리포트 다운로드", value: ".md / .json" },
      { label: "API 접근", value: "월 500회 포함" },
    ],
  },
];

const B2B_PLANS = [
  { name: "Standard", price: "₩99,000", calls: "월 5,000회", target: "스타트업, 중소 미디어" },
  { name: "Professional", price: "₩490,000", calls: "월 50,000회", target: "중견 기업, 언론사" },
  { name: "Enterprise", price: "협의", calls: "무제한", target: "수사기관, 대기업, 연구기관" },
];

export default function PlansPage() {
  const { user, loading } = useAuth();
  const [authOpen, setAuthOpen] = useState(false);
  const [upgradeMsg, setUpgradeMsg] = useState("");

  const handleUpgrade = (planId: string) => {
    if (!user) {
      setAuthOpen(true);
      return;
    }
    if (user.plan === planId) return;
    // 실제 결제 연동 전 — 관리자 문의 안내
    setUpgradeMsg(`${planId.toUpperCase()} 플랜 업그레이드 요청이 접수되었습니다. 빠른 시일 내 이메일(${user.email})로 안내 드리겠습니다.`);
  };

  return (
    <div className="min-h-screen" style={{ background: "var(--bg)" }}>
      <header className="sticky top-0 z-50 border-b px-5 h-14 flex items-center justify-between"
              style={{ borderColor: "var(--border)", background: "rgba(11,12,19,0.96)", backdropFilter: "blur(12px)" }}>
        <Link href="/" className="text-lg font-bold tracking-wider text-white">EXXAS</Link>
        <div className="flex items-center gap-3">
          {user ? (
            <span className="text-xs px-2.5 py-1 rounded-full border font-medium"
              style={{ borderColor: "var(--accent)", color: "var(--accent)", background: "rgba(91,141,239,0.08)" }}>
              현재 {user.plan}
            </span>
          ) : (
            <button onClick={() => setAuthOpen(true)}
              className="text-sm px-4 py-1.5 rounded-lg font-medium text-white transition-all hover:opacity-90"
              style={{ background: "var(--accent)" }}>
              무료 가입
            </button>
          )}
          <Link href="/" className="text-sm px-3 py-1.5 rounded-lg border transition-colors hover:text-white"
            style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
            분석 시작
          </Link>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-16">
        <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-14">
          <div className="text-xs font-semibold uppercase tracking-widest mb-3" style={{ color: "var(--accent)" }}>PRICING</div>
          <h1 className="text-4xl font-bold text-white mb-4">플랜 & 요금</h1>
          <p className="text-[#64748B] text-sm max-w-lg mx-auto">
            모든 플랜은 월간 자동 갱신. 언제든지 해지 가능.
          </p>
        </motion.div>

        {upgradeMsg && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="mb-8 p-4 rounded-lg border border-[#0EA5E9]/30 bg-[#0EA5E9]/5 text-[#0EA5E9] text-sm text-center">
            {upgradeMsg}
          </motion.div>
        )}

        {/* 구독 플랜 */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-16">
          {PLANS.map((plan, i) => {
            const isCurrent = user?.plan === plan.id;
            const isLower = user && (
              (user.plan === "expert") ||
              (user.plan === "pro" && plan.id === "free")
            );
            return (
              <motion.div
                key={plan.id}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                className={`rounded-xl border overflow-hidden flex flex-col ${
                  plan.featured
                    ? "border-[#0EA5E9] shadow-[0_0_30px_rgba(14,165,233,0.12)]"
                    : "border-[#1E2D45]"
                } bg-[#0D1220]`}
              >
                {plan.featured && (
                  <div className="text-center py-1.5 text-xs font-bold tracking-widest uppercase bg-[#0EA5E9] text-black">
                    추천
                  </div>
                )}
                <div className="p-5 border-b border-[#1E2D45]">
                  <div className="text-xs font-bold uppercase tracking-[3px] mb-2" style={{ color: plan.color }}>
                    {plan.name}
                  </div>
                  <div className="text-4xl font-bold" style={{ color: plan.color }}>
                    {plan.price}
                  </div>
                  <div className="text-xs text-[#64748B] mt-1">{plan.period}</div>
                </div>

                <div className="p-5 flex-1">
                  {plan.features.map((f) => (
                    <div key={f.label} className="flex items-center justify-between py-2 border-b border-[#1E2D45] last:border-0 text-sm">
                      <span className="text-[#64748B]">{f.label}</span>
                      {f.value === false ? (
                        <span className="text-[#2D3F55] font-mono">—</span>
                      ) : (
                        <span className="text-[#E2E8F0] font-medium">{f.value}</span>
                      )}
                    </div>
                  ))}
                </div>

                <div className="p-5 pt-0">
                  {isCurrent ? (
                    <div className="w-full py-2.5 rounded-lg text-center text-xs font-bold tracking-widest uppercase border border-[#1E2D45] text-[#64748B]">
                      현재 플랜
                    </div>
                  ) : isLower ? (
                    <div className="w-full py-2.5 rounded-lg text-center text-xs text-[#2D3F55]">
                      다운그레이드 불가
                    </div>
                  ) : (
                    <button
                      onClick={() => handleUpgrade(plan.id)}
                      style={{ background: plan.id === "free" ? "transparent" : plan.color }}
                      className={`w-full py-2.5 rounded-lg text-xs font-bold tracking-widest uppercase transition-all
                        ${plan.id === "free"
                          ? "border border-[#1E2D45] text-[#64748B] hover:text-white"
                          : "text-black hover:opacity-90"
                        }`}
                    >
                      {plan.id === "free" ? "무료 시작" : `${plan.name} 업그레이드`}
                    </button>
                  )}
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* B2B API */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}>
          <div className="text-center mb-8">
            <div className="text-xs font-semibold uppercase tracking-widest mb-2" style={{ color: "var(--warning)" }}>B2B API</div>
            <h2 className="text-3xl font-bold text-white">API 라이선스</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {B2B_PLANS.map((p, i) => (
              <div key={p.name} className="border border-[#1E2D45] rounded-xl p-5 bg-[#0D1220]">
                <div className="text-xl font-bold mb-1" style={{ color: "var(--warning)" }}>{p.name}</div>
                <div className="text-2xl font-bold text-white mb-1">{p.price}</div>
                <div className="text-xs text-[#64748B] mb-3">{p.calls}</div>
                <div className="text-xs text-[#475569]">{p.target}</div>
                <a href="mailto:api@exxas.app"
                  className="mt-4 block w-full py-2 text-center text-xs font-bold tracking-wider uppercase rounded-lg
                    border border-[#F59E0B]/30 text-[#F59E0B] hover:bg-[#F59E0B]/10 transition-colors">
                  문의하기
                </a>
              </div>
            ))}
          </div>
        </motion.div>

        {/* FAQ */}
        <div className="mt-16 border-t border-[#1E2D45] pt-12">
          <h3 className="text-2xl font-bold text-white text-center mb-8">자주 묻는 질문</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-3xl mx-auto">
            {[
              { q: "무료 플랜은 얼마나 사용할 수 있나요?", a: "매월 5회 분석이 무료입니다. 매달 1일 자동으로 초기화됩니다." },
              { q: "결제는 어떻게 하나요?", a: "신용카드/체크카드 결제를 지원합니다. 현재 결제 시스템 구축 중으로, 이메일로 문의해 주세요." },
              { q: "분석 이미지는 외부에 공유되나요?", a: "이미지는 암호화된 상태로 24시간 내 자동 삭제됩니다. 역방향 검색 시 이미지 해시값만 전송됩니다." },
              { q: "API 호출 형식이 궁금합니다.", a: "REST API 및 JSON 응답을 지원합니다. Expert 플랜 가입 후 /docs에서 전체 명세를 확인하세요." },
            ].map((item) => (
              <div key={item.q} className="border border-[#1E2D45] rounded-lg p-4 bg-[#0D1220]">
                <div className="text-sm font-bold text-white mb-2">{item.q}</div>
                <div className="text-sm text-[#64748B] leading-relaxed">{item.a}</div>
              </div>
            ))}
          </div>
        </div>
      </main>

      <AuthModal open={authOpen} onClose={() => setAuthOpen(false)}
        onSuccess={() => setAuthOpen(false)} defaultTab="register" />
    </div>
  );
}

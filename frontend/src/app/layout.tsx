import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EXXAS — 사진 기반 위치 수사 플랫폼",
  description: "단일 이미지로 촬영 위치를 GPS 좌표까지 역추적하는 AI 기반 OSINT 플랫폼",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Noto+Sans+KR:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500;600&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-[#04080F] text-[#C8D8EC] antialiased">{children}</body>
    </html>
  );
}

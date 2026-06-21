/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        bg: "#080C14",
        surface: "#0D1220",
        surface2: "#111827",
        border: "#1E2D45",
        accent: "#0EA5E9",
        accent2: "#F43F5E",
        accent3: "#10B981",
        accent4: "#F59E0B",
        "text-dim": "#94A3B8",
        "text-muted": "#64748B",
      },
      fontFamily: {
        mono: ["JetBrains Mono", "monospace"],
      },
      animation: {
        pulse: "pulse 2s infinite",
        "spin-slow": "spin 3s linear infinite",
      },
    },
  },
  plugins: [],
};

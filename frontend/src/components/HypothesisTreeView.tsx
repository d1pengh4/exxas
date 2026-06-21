"use client";
import { motion } from "framer-motion";
import { HypothesisTree, HypothesisNode } from "@/lib/api";

interface Props {
  tree: HypothesisTree;
}

const LEVEL_COLORS = ["#0EA5E9", "#10B981", "#F59E0B", "#A78BFA", "#F43F5E"];

export default function HypothesisTreeView({ tree }: Props) {
  const active = tree.hypotheses
    .filter((h) => !h.is_rejected)
    .sort((a, b) => b.probability - a.probability);

  const rejected = tree.hypotheses.filter((h) => h.is_rejected);

  return (
    <div className="border border-[#1E2D45] rounded-lg p-4 bg-[#0D1220]">
      <div className="flex items-center justify-between mb-4">
        <div className="text-xs text-[#0EA5E9] font-bold uppercase tracking-widest">
          가설 트리
        </div>
        <div className="text-xs text-[#64748B] font-mono">
          {tree.step_count}단계 · 단서 {tree.evidence_count}개
        </div>
      </div>

      {/* 활성 가설 */}
      <div className="space-y-2">
        {active.map((h, i) => (
          <HypothesisRow key={h.id} node={h} rank={i} />
        ))}
      </div>

      {/* 기각된 가설 */}
      {rejected.length > 0 && (
        <div className="mt-3 pt-3 border-t border-[#1E2D45]">
          <div className="text-xs text-[#64748B] mb-2">기각 ({rejected.length}개)</div>
          <div className="flex flex-wrap gap-2">
            {rejected.map((h) => (
              <span key={h.id} className="text-xs text-[#475569] line-through">
                {h.location}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function HypothesisRow({ node, rank }: { node: HypothesisNode; rank: number }) {
  const pct = (node.probability * 100).toFixed(1);
  const color = LEVEL_COLORS[Math.min(rank, LEVEL_COLORS.length - 1)];
  const indent = node.level * 16;

  return (
    <div className="flex items-center gap-2" style={{ paddingLeft: indent }}>
      {node.level > 0 && (
        <span className="text-[#1E2D45] text-xs">└─</span>
      )}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-sm text-[#E2E8F0] truncate">{node.location}</span>
          {node.supporting_evidence_count > 0 && (
            <span className="text-xs text-[#10B981]">+{node.supporting_evidence_count}</span>
          )}
          {node.contradicting_evidence_count > 0 && (
            <span className="text-xs text-[#F43F5E]">-{node.contradicting_evidence_count}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <div className="flex-1 h-1 bg-[#1E2D45] rounded overflow-hidden">
            <motion.div
              className="h-full rounded"
              style={{ background: color }}
              initial={{ width: 0 }}
              animate={{ width: `${node.probability * 100}%` }}
              transition={{ duration: 0.6, ease: "easeOut" }}
            />
          </div>
          <span className="text-xs font-mono w-12 text-right" style={{ color }}>
            {pct}%
          </span>
        </div>
      </div>
    </div>
  );
}

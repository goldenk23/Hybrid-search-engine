"use client"

import { useState } from "react"
import { ChevronDown } from "lucide-react"
import type { SearchMode, SearchResultItem } from "@/lib/search"

interface ResultCardProps {
  result: SearchResultItem
  mode: SearchMode
  showBody: boolean
}

function fmt(n: number | null | undefined, digits = 4): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—"
  return Number(n).toFixed(digits)
}

export function ResultCard({ result, mode, showBody }: ResultCardProps) {
  const [expanded, setExpanded] = useState(false)
  const snippet = result.snippet ?? result.body ?? ""

  return (
    <article className="group rounded-xl border border-border bg-card p-5 transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-lg hover:shadow-black/20">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <h3 className="text-lg font-semibold text-foreground transition-colors group-hover:text-primary">
          {result.title || "Untitled"}
        </h3>
        {result.category && (
          <span className="rounded-full border border-border bg-secondary px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
            {result.category}
          </span>
        )}
      </div>

      {snippet && (
        <p className="mt-2 line-clamp-3 text-sm leading-relaxed text-muted-foreground">{snippet}</p>
      )}

      <ScoreRow result={result} mode={mode} fmt={fmt} />

      {showBody && result.body && (
        <div className="mt-3 border-t border-border pt-3">
          <button
            type="button"
            onClick={() => setExpanded((e) => !e)}
            aria-expanded={expanded}
            className="flex items-center gap-1.5 text-xs font-medium text-primary"
          >
            <ChevronDown
              className={"size-3.5 transition-transform " + (expanded ? "rotate-180" : "")}
              aria-hidden="true"
            />
            {expanded ? "Hide full passage" : "Show full passage"}
          </button>
          {expanded && (
            <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-foreground/90">
              {result.body}
            </p>
          )}
        </div>
      )}
    </article>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <span className="whitespace-nowrap">
      <span className="text-muted-foreground/70">{label}:</span>{" "}
      <span className="font-mono text-foreground/80">{value}</span>
    </span>
  )
}

function ScoreRow({
  result,
  mode,
  fmt,
}: {
  result: SearchResultItem
  mode: SearchMode
  fmt: (n: number | null | undefined, d?: number) => string
}) {
  const hasRanks = result.bm25_rank != null || result.vector_rank != null

  return (
    <div className="mt-3 flex flex-col gap-1.5 text-xs text-muted-foreground">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        {mode === "keyword" && <Metric label="Score" value={fmt(result.score)} />}

        {(mode === "hybrid" || mode === "reranked") && (
          <>
            <Metric label="RRF" value={fmt(result.rrf_score)} />
            <Metric label="BM25" value={fmt(result.bm25_score)} />
            <Metric label="Vector" value={fmt(result.vector_score)} />
            {mode === "reranked" && (
              <Metric label="Cross-encoder" value={fmt(result.cross_encoder_score)} />
            )}
          </>
        )}
      </div>

      {(mode === "hybrid" || mode === "reranked") && hasRanks && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          {result.bm25_rank != null && (
            <Metric label="BM25 rank" value={`#${result.bm25_rank}`} />
          )}
          {result.vector_rank != null && (
            <Metric label="Vector rank" value={`#${result.vector_rank}`} />
          )}
        </div>
      )}
    </div>
  )
}

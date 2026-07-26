"use client"

import type { SearchMode } from "@/lib/search"

const MODES: { id: SearchMode; label: string }[] = [
  { id: "keyword", label: "Keyword" },
  { id: "hybrid", label: "Hybrid" },
  { id: "reranked", label: "Reranked" },
]

interface ModeSelectorProps {
  mode: SearchMode
  onChange: (mode: SearchMode) => void
}

export function ModeSelector({ mode, onChange }: ModeSelectorProps) {
  return (
    <div
      role="tablist"
      aria-label="Search mode"
      className="inline-flex w-fit items-center gap-1 rounded-lg border border-border bg-card p-1"
    >
      {MODES.map((m) => {
        const active = m.id === mode
        return (
          <button
            key={m.id}
            role="tab"
            aria-selected={active}
            type="button"
            onClick={() => onChange(m.id)}
            className={
              "rounded-md px-4 py-1.5 text-sm font-medium transition-colors " +
              (active
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:text-foreground")
            }
          >
            {m.label}
          </button>
        )
      })}
    </div>
  )
}

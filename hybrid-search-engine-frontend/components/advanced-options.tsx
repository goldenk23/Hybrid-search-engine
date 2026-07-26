"use client"

import { useState } from "react"
import { ChevronDown, SlidersHorizontal } from "lucide-react"
import type { SearchMode, SearchOptions } from "@/lib/search"

interface AdvancedOptionsProps {
  mode: SearchMode
  options: SearchOptions
  onChange: (options: SearchOptions) => void
}

export function AdvancedOptions({ mode, options, onChange }: AdvancedOptionsProps) {
  const [open, setOpen] = useState(false)

  if (mode === "keyword") return null

  function set<K extends keyof SearchOptions>(key: K, value: SearchOptions[K]) {
    onChange({ ...options, [key]: value })
  }

  return (
    <div className="rounded-xl border border-border bg-card">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium text-foreground"
      >
        <span className="flex items-center gap-2">
          <SlidersHorizontal className="size-4 text-muted-foreground" aria-hidden="true" />
          Advanced Options
        </span>
        <ChevronDown
          className={"size-4 text-muted-foreground transition-transform " + (open ? "rotate-180" : "")}
          aria-hidden="true"
        />
      </button>

      {open && (
        <div className="grid grid-cols-1 gap-5 border-t border-border px-4 py-4 sm:grid-cols-2">
          <SliderField
            label="Results"
            hint={`top_k · ${options.top_k}`}
            min={1}
            max={50}
            step={1}
            value={options.top_k}
            onChange={(v) =>
              onChange({
                ...options,
                top_k: v,
                candidates_k: Math.max(options.candidates_k, v),
              })
            }
          />
          <SliderField
            label="BM25 Weight"
            hint={`bm25_weight · ${options.bm25_weight.toFixed(1)}`}
            min={0}
            max={2}
            step={0.1}
            value={options.bm25_weight}
            onChange={(v) => set("bm25_weight", v)}
          />
          <SliderField
            label="Vector Weight"
            hint={`vector_weight · ${options.vector_weight.toFixed(1)}`}
            min={0}
            max={2}
            step={0.1}
            value={options.vector_weight}
            onChange={(v) => set("vector_weight", v)}
          />
          <NumberField
            label="RRF k (rank constant)"
            hint="rrf_k"
            value={options.rrf_k}
            min={1}
            onChange={(v) => set("rrf_k", v)}
          />

          {mode === "reranked" && (
            <NumberField
              label="Reranker Candidates"
              hint="candidates_k"
              value={options.candidates_k}
              min={options.top_k}
              max={500}
              onChange={(v) => set("candidates_k", v)}
            />
          )}

          <ToggleField
            label="Show Full Passage"
            hint="include_body"
            checked={options.include_body}
            onChange={(v) => set("include_body", v)}
          />
        </div>
      )}
    </div>
  )
}

function FieldLabel({ label, hint }: { label: string; hint: string }) {
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-sm font-medium text-foreground">{label}</span>
      <span className="font-mono text-xs text-muted-foreground">{hint}</span>
    </div>
  )
}

function SliderField({
  label,
  hint,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string
  hint: string
  min: number
  max: number
  step: number
  value: number
  onChange: (v: number) => void
}) {
  return (
    <label className="flex flex-col gap-2">
      <FieldLabel label={label} hint={hint} />
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-secondary accent-primary"
      />
    </label>
  )
}

function NumberField({
  label,
  hint,
  value,
  min,
  max,
  onChange,
}: {
  label: string
  hint: string
  value: number
  min?: number
  max?: number
  onChange: (v: number) => void
}) {
  return (
    <label className="flex flex-col gap-2">
      <FieldLabel label={label} hint={hint} />
      <input
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={(e) => {
          const next = e.currentTarget.valueAsNumber
          if (Number.isNaN(next)) return
          onChange(Math.min(max ?? Infinity, Math.max(min ?? -Infinity, next)))
        }}
        className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-primary focus:ring-2 focus:ring-primary/30"
      />
    </label>
  )
}

function ToggleField({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string
  hint: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <div className="flex-1">
        <FieldLabel label={label} hint={hint} />
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={() => onChange(!checked)}
        className={
          "relative h-6 w-11 shrink-0 rounded-full transition-colors " +
          (checked ? "bg-primary" : "bg-secondary")
        }
      >
        <span
          className={
            "absolute top-0.5 size-5 rounded-full bg-white transition-transform " +
            (checked ? "translate-x-5" : "translate-x-0.5")
          }
        />
      </button>
    </div>
  )
}

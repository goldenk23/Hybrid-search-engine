"use client"

import type React from "react"
import { Search } from "lucide-react"

interface SearchBarProps {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  loading: boolean
}

export function SearchBar({ value, onChange, onSubmit, loading }: SearchBarProps) {
  const validLength = value.trim().length >= 3 && value.trim().length <= 256
  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key !== "Enter") return
    if (e.nativeEvent.isComposing || e.keyCode === 229) return
    e.preventDefault()
    onSubmit()
  }

  return (
    <div className="flex w-full items-center gap-2 rounded-xl border border-border bg-card px-4 py-2 transition-colors focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/30">
      <Search className="size-5 shrink-0 text-muted-foreground" aria-hidden="true" />
      <input
        type="text"
        minLength={3}
        maxLength={256}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Search anything…"
        aria-label="Search query"
        className="w-full bg-transparent py-2 text-base text-foreground outline-none placeholder:text-muted-foreground"
      />
      <button
        type="button"
        onClick={onSubmit}
        disabled={loading || !validLength}
        className="shrink-0 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {loading ? "Searching…" : "Search"}
      </button>
    </div>
  )
}

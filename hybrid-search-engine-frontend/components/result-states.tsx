"use client"

import { Info, SearchX, AlertCircle, Clock } from "lucide-react"

export function MetaBar({
  count,
  latencyMs,
  correctedQuery,
}: {
  count: number
  latencyMs: number | null
  correctedQuery?: string | null
}) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
      <p className="text-sm text-muted-foreground">
        {count} {count === 1 ? "result" : "results"}
      </p>
      {latencyMs != null && (
        <span className="inline-flex w-fit items-center gap-1.5 rounded-full border border-border bg-card px-2.5 py-1 text-xs font-medium text-muted-foreground">
          <Clock className="size-3.5" aria-hidden="true" />
          {latencyMs} ms
        </span>
      )}
      {correctedQuery && (
        <p className="inline-flex items-center gap-1.5 text-sm text-muted-foreground sm:order-last sm:w-full sm:basis-full">
          <Info className="size-4 text-primary" aria-hidden="true" />
          Showing results for: <span className="font-semibold text-foreground">{correctedQuery}</span>
        </p>
      )}
    </div>
  )
}

export function SkeletonCards() {
  return (
    <div className="flex flex-col gap-4" aria-hidden="true">
      {[0, 1, 2].map((i) => (
        <div key={i} className="rounded-xl border border-border bg-card p-5">
          <div className="shimmer h-5 w-1/2 rounded" />
          <div className="mt-3 shimmer h-3 w-full rounded" />
          <div className="mt-2 shimmer h-3 w-11/12 rounded" />
          <div className="mt-2 shimmer h-3 w-4/6 rounded" />
          <div className="mt-4 flex gap-4">
            <div className="shimmer h-3 w-20 rounded" />
            <div className="shimmer h-3 w-20 rounded" />
            <div className="shimmer h-3 w-20 rounded" />
          </div>
        </div>
      ))}
    </div>
  )
}

export function EmptyState({ query }: { query: string }) {
  return (
    <div className="flex flex-col items-center gap-3 py-16 text-center">
      <SearchX className="size-10 text-muted-foreground" aria-hidden="true" />
      <p className="text-sm text-muted-foreground">
        No results found for <span className="font-medium text-foreground">{`"${query}"`}</span>
      </p>
    </div>
  )
}

type BannerKind = "error" | "warning" | "info"

export function Banner({ kind, message }: { kind: BannerKind; message: string }) {
  const styles: Record<BannerKind, string> = {
    error: "border-destructive/40 bg-destructive/10 text-destructive",
    warning: "border-warning/40 bg-warning/10 text-warning",
    info: "border-border bg-secondary text-muted-foreground",
  }
  return (
    <div
      role="alert"
      className={"flex items-center gap-2.5 rounded-xl border px-4 py-3 text-sm font-medium " + styles[kind]}
    >
      <AlertCircle className="size-4 shrink-0" aria-hidden="true" />
      <span>{message}</span>
    </div>
  )
}

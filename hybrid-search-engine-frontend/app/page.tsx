"use client"

import { useEffect, useRef, useState } from "react"
import { Search } from "lucide-react"
import {
  DEFAULT_OPTIONS,
  runSearch,
  SearchApiError,
  type SearchMode,
  type SearchOptions,
  type SearchResponse,
} from "@/lib/search"
import { SearchBar } from "@/components/search-bar"
import { ModeSelector } from "@/components/mode-selector"
import { AdvancedOptions } from "@/components/advanced-options"
import { ResultCard } from "@/components/result-card"
import { MetaBar, SkeletonCards, EmptyState, Banner } from "@/components/result-states"

interface BannerState {
  kind: "error" | "warning" | "info"
  message: string
}

export default function Page() {
  const [query, setQuery] = useState("")
  const [mode, setMode] = useState<SearchMode>("keyword")
  const [options, setOptions] = useState<SearchOptions>(DEFAULT_OPTIONS)

  const [loading, setLoading] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [response, setResponse] = useState<SearchResponse | null>(null)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const [banner, setBanner] = useState<BannerState | null>(null)
  const [lastQuery, setLastQuery] = useState("")
  // Snapshot of the mode used for the currently displayed results.
  const [resultMode, setResultMode] = useState<SearchMode>("keyword")
  const requestController = useRef<AbortController | null>(null)

  useEffect(() => () => requestController.current?.abort(), [])

  async function search(searchMode: SearchMode = mode) {
    const trimmed = query.trim()

    if (trimmed.length < 3 || trimmed.length > 256) {
      requestController.current?.abort()
      requestController.current = null
      setLoading(false)
      setHasSearched(true)
      setResponse(null)
      setLatencyMs(null)
      setBanner({
        kind: "error",
        message: "Search queries must be between 3 and 256 characters.",
      })
      return
    }

    requestController.current?.abort()
    const controller = new AbortController()
    requestController.current = controller

    setLoading(true)
    setBanner(null)
    setHasSearched(true)
    setLastQuery(trimmed)
    setResultMode(searchMode)

    try {
      const { data, latencyMs } = await runSearch(
        searchMode,
        trimmed,
        options,
        controller.signal,
      )
      if (requestController.current !== controller) return

      setResponse(data)
      setLatencyMs(data.latency_ms ?? latencyMs)
    } catch (err) {
      if (controller.signal.aborted || requestController.current !== controller) return

      setResponse(null)
      setLatencyMs(null)
      if (err instanceof SearchApiError) {
        if (err.status === 429) {
          setBanner({ kind: "warning", message: "Reranker is busy — try again in a moment." })
        } else if (err.status === 503) {
          setBanner({ kind: "info", message: "Reranker is not enabled on this server." })
        } else {
          setBanner({ kind: "error", message: err.message })
        }
      } else {
        setBanner({ kind: "error", message: "Something went wrong. Please try again." })
      }
    } finally {
      if (requestController.current === controller) {
        requestController.current = null
        setLoading(false)
      }
    }
  }

  function handleModeChange(nextMode: SearchMode) {
    if (nextMode === mode) return
    setMode(nextMode)
    if (hasSearched) void search(nextMode)
  }

  const results = response?.results ?? []
  const returnedCount = response?.returned_count ?? results.length

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-3xl flex-col px-4 sm:px-6">
      <div
        className={
          "flex w-full flex-col gap-5 transition-all " +
          (hasSearched ? "pt-8" : "flex-1 justify-center pb-24")
        }
      >
        <header className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2.5">
            <span className="flex size-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Search className="size-5" aria-hidden="true" />
            </span>
            <h1 className="text-xl font-semibold tracking-tight text-foreground">
              Hybrid Search Engine
            </h1>
          </div>
          <p className="text-sm text-muted-foreground">BM25 · Vector · RRF Fusion</p>
        </header>

        <SearchBar value={query} onChange={setQuery} onSubmit={() => void search()} loading={loading} />

        <div className="flex flex-col gap-4">
          <ModeSelector mode={mode} onChange={handleModeChange} />
          <AdvancedOptions mode={mode} options={options} onChange={setOptions} />
        </div>
      </div>

      {(hasSearched || banner) && (
        <section className="flex flex-col gap-4 py-6" aria-live="polite">
          {banner && <Banner kind={banner.kind} message={banner.message} />}

          {loading && <SkeletonCards />}

          {!loading && !banner && response && (
            <>
              <MetaBar
                count={returnedCount}
                latencyMs={latencyMs}
                correctedQuery={response.corrected_query}
              />
              {results.length > 0 ? (
                <div className="flex flex-col gap-4">
                  {results.map((r, i) => (
                    <ResultCard
                      key={r.id ?? i}
                      result={r}
                      mode={resultMode}
                      showBody={options.include_body}
                    />
                  ))}
                </div>
              ) : (
                <EmptyState query={lastQuery} />
              )}
            </>
          )}
        </section>
      )}
    </main>
  )
}

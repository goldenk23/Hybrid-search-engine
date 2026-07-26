export type SearchMode = "keyword" | "hybrid" | "reranked"

export interface SearchResultItem {
  id?: string | number
  title: string
  category?: string | null
  snippet?: string | null
  body?: string | null
  url?: string | null
  // keyword
  score?: number | null
  // hybrid / reranked
  rrf_score?: number | null
  bm25_score?: number | null
  vector_score?: number | null
  bm25_rank?: number | null
  vector_rank?: number | null
  // reranked only
  cross_encoder_score?: number | null
}

export interface SearchResponse {
  query: string
  results: SearchResultItem[]
  returned_count: number
  latency_ms: number
  corrected_query: string | null
}

export interface SearchOptions {
  top_k: number
  bm25_weight: number
  vector_weight: number
  rrf_k: number
  include_body: boolean
  candidates_k: number
}

export const DEFAULT_OPTIONS: SearchOptions = {
  top_k: 10,
  bm25_weight: 1.0,
  vector_weight: 1.0,
  rrf_k: 60,
  include_body: false,
  candidates_k: 100,
}

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000"

export class SearchApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
    this.name = "SearchApiError"
  }
}

function buildUrl(mode: SearchMode, query: string, options: SearchOptions): string {
  const params = new URLSearchParams()
  params.set("q", query)
  params.set("top_k", String(options.top_k))
  params.set("include_body", String(options.include_body))

  if (mode === "hybrid" || mode === "reranked") {
    params.set("bm25_weight", String(options.bm25_weight))
    params.set("vector_weight", String(options.vector_weight))
    params.set("rrf_k", String(options.rrf_k))
  }
  if (mode === "reranked") {
    params.set("candidates_k", String(options.candidates_k))
  }

  const path =
    mode === "keyword"
      ? "/search"
      : mode === "hybrid"
        ? "/hybrid-search"
        : "/hybrid-search/rerank"

  return `${API_BASE_URL}${path}?${params.toString()}`
}

export async function runSearch(
  mode: SearchMode,
  query: string,
  options: SearchOptions,
  signal?: AbortSignal,
): Promise<{ data: SearchResponse; latencyMs: number }> {
  const url = buildUrl(mode, query, options)
  const start = performance.now()

  let res: Response
  try {
    res = await fetch(url, {
      cache: "no-store",
      headers: { Accept: "application/json" },
      signal,
    })
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err
    throw new SearchApiError(0, "Could not reach the search API. Is the server running?")
  }

  const latencyMs = Math.round(performance.now() - start)

  if (!res.ok) {
    let message = `Request failed with status ${res.status}`
    try {
      const errBody = await res.json()
      if (typeof errBody?.detail === "string") {
        message = errBody.detail
      } else if (Array.isArray(errBody?.detail)) {
        const details = errBody.detail
          .map((item: { msg?: unknown }) => item?.msg)
          .filter((msg: unknown): msg is string => typeof msg === "string")
        if (details.length) message = details.join("; ")
      } else if (typeof errBody?.message === "string") {
        message = errBody.message
      }
    } catch {
      // ignore parse failure
    }
    throw new SearchApiError(res.status, message)
  }

  const data = (await res.json()) as SearchResponse
  return { data, latencyMs }
}

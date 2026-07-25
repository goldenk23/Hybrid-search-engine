"""
Pydantic response models for the search API.

Design notes:
- body is str | None = None in all result models.  Routes set it only when
  include_body=True is requested; otherwise the client gets a snippet.
  Smaller payloads = faster responses and lower bandwidth cost.
- returned_count replaces the misleading 'total' field (which suggested the
  total number of matching documents in the index, not the number returned).
"""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str


# ------------------------------------------------------------------ BM25

class SearchResult(BaseModel):
    id: str
    title: str
    body: str | None = None      # None unless include_body=True
    category: str
    score: float
    snippet: str | None = None


class SearchResponse(BaseModel):
    query: str
    corrected_query: str | None = None
    returned_count: int          # was 'total' — renamed to be unambiguous
    latency_ms: int
    results: list[SearchResult]


# ------------------------------------------------------------------ Hybrid RRF

class HybridSearchResult(BaseModel):
    id: str
    title: str
    body: str | None = None
    category: str | None = None
    rrf_score: float
    bm25_score: float
    vector_score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    snippet: str | None = None


class HybridSearchResponse(BaseModel):
    query: str
    corrected_query: str | None = None
    returned_count: int
    latency_ms: int
    results: list[HybridSearchResult]


# ------------------------------------------------------------------ Reranked

class RerankedSearchResult(BaseModel):
    id: str
    title: str
    body: str | None = None
    category: str | None = None
    rrf_score: float
    bm25_score: float
    vector_score: float
    cross_encoder_score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    snippet: str | None = None


class RerankedSearchResponse(BaseModel):
    query: str
    corrected_query: str | None = None
    returned_count: int
    latency_ms: int
    results: list[RerankedSearchResult]

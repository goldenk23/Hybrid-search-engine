"""
Search API routes.

Services are loaded once at startup and stored in app.state.services.
Each handler pulls them out via get_services() — no lazy globals, no
repeated model loading, no risk of two handlers loading different instances.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.api.models import (
    HybridSearchResponse,
    HybridSearchResult,
    RerankedSearchResponse,
    RerankedSearchResult,
    SearchResponse,
    SearchResult,
)
from src.config import RESULTS_PER_PAGE
from src.indexing.preprocessing import generate_snippet

if TYPE_CHECKING:
    from src.api.main import SearchServices

router = APIRouter(tags=["search"])


# ------------------------------------------------------------------ dependency

def get_services(request: Request) -> SearchServices:
    """Extract the shared SearchServices from app.state.

    FastAPI calls this automatically for every route that declares it as a
    Depends — no route needs to touch request.app.state directly.
    """
    return request.app.state.services


# ------------------------------------------------------------------ helpers

def _correct(spell, query: str) -> tuple[str, str | None]:
    """Return (search_query, corrected_query_or_None)."""
    try:
        corrected = spell.correct_query(query)
    except Exception:  # noqa: BLE001 — spell corrector may raise anything; fall back silently
        corrected = query
    if corrected and corrected != query:
        return corrected, corrected
    return query, None


def _body_or_none(text: str, include: bool) -> str | None:
    return text if include else None


# ------------------------------------------------------------------ /search

@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=3, max_length=256,
                   description="Search query"),
    top_k: int = Query(default=RESULTS_PER_PAGE, ge=1, le=100),
    include_body: bool = Query(default=False,
                               description="Include full passage body in results"),
    svc: SearchServices = Depends(get_services),  # noqa: B008 — FastAPI DI pattern
) -> SearchResponse:
    query_text  = q.strip()
    search_q, corrected = _correct(svc.spell, query_text)

    t0 = perf_counter()
    raw = svc.bm25.search(query=search_q, top_k=top_k)
    latency_ms = int((perf_counter() - t0) * 1000)

    results = [
        SearchResult(
            id=str(r["id"]),
            title=r["title"] or "",
            body=_body_or_none(r["body"] or "", include_body),
            category=r["category"],
            score=r["score"],
            snippet=generate_snippet(r["body"] or "", search_q),
        )
        for r in raw
    ]
    return SearchResponse(
        query=query_text,
        corrected_query=corrected,
        returned_count=len(results),
        latency_ms=latency_ms,
        results=results,
    )


# ------------------------------------------------------------------ /hybrid-search

@router.get("/hybrid-search", response_model=HybridSearchResponse)
def hybrid_search(
    q: str = Query(..., min_length=3, max_length=256),
    top_k: int = Query(default=RESULTS_PER_PAGE, ge=1, le=100),
    bm25_weight: float = Query(default=1.0, ge=0.0, le=100.0),
    vector_weight: float = Query(default=1.0, ge=0.0, le=100.0),
    rrf_k: int = Query(default=60, ge=1),
    include_body: bool = Query(default=False),
    svc: SearchServices = Depends(get_services),  # noqa: B008 — FastAPI DI pattern
) -> HybridSearchResponse:
    query_text  = q.strip()
    search_q, corrected = _correct(svc.spell, query_text)

    t0 = perf_counter()
    raw = svc.hybrid.search(
        query=search_q, top_k=top_k,
        bm25_weight=bm25_weight, vector_weight=vector_weight, rrf_k=rrf_k,
    )
    latency_ms = int((perf_counter() - t0) * 1000)

    results = [
        HybridSearchResult(
            id=str(r["id"]),
            title=r.get("title", "") or "",
            body=_body_or_none(r.get("body", "") or "", include_body),
            category=r.get("category"),
            rrf_score=float(r["rrf_score"]),
            bm25_score=float(r["bm25_score"]),
            vector_score=float(r["vector_score"]),
            bm25_rank=r.get("bm25_rank"),
            vector_rank=r.get("vector_rank"),
            snippet=generate_snippet(r.get("body", "") or "", search_q),
        )
        for r in raw
    ]
    return HybridSearchResponse(
        query=query_text,
        corrected_query=corrected,
        returned_count=len(results),
        latency_ms=latency_ms,
        results=results,
    )


# ------------------------------------------------------------------ /hybrid-search/rerank

@router.get("/hybrid-search/rerank", response_model=RerankedSearchResponse)
def hybrid_search_rerank(
    q: str = Query(..., min_length=3, max_length=256),
    top_k: int = Query(default=RESULTS_PER_PAGE, ge=1, le=50),
    candidates_k: int = Query(default=100, ge=1, le=500),
    bm25_weight: float = Query(default=1.0, ge=0.0, le=100.0),
    vector_weight: float = Query(default=1.0, ge=0.0, le=100.0),
    rrf_k: int = Query(default=60, ge=1),
    include_body: bool = Query(default=False),
    svc: SearchServices = Depends(get_services),  # noqa: B008 — FastAPI DI pattern
) -> RerankedSearchResponse:
    # Guard: you cannot return more results than you fetched as candidates.
    if candidates_k < top_k:
        raise HTTPException(
            status_code=422,
            detail=f"candidates_k ({candidates_k}) must be >= top_k ({top_k})",
        )

    # 503 when reranking is disabled at startup.
    if svc.reranker is None:
        raise HTTPException(
            status_code=503,
            detail="Reranking is disabled. Set ENABLE_RERANKER=true and restart.",
        )

    # 429 when a rerank is already in progress (semaphore = 1 slot).
    if not svc.rerank_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Reranker is busy. Retry in a moment.",
        )

    try:
        query_text  = q.strip()
        search_q, corrected = _correct(svc.spell, query_text)

        t0 = perf_counter()
        candidates = svc.hybrid.search(
            query=search_q, top_k=candidates_k,
            bm25_weight=bm25_weight, vector_weight=vector_weight, rrf_k=rrf_k,
        )
        reranked = svc.reranker.rerank(
            query=search_q, candidates=candidates,
            top_k=top_k, max_candidates=candidates_k,
        )
        latency_ms = int((perf_counter() - t0) * 1000)
    finally:
        svc.rerank_slots.release()

    results = [
        RerankedSearchResult(
            id=str(r["id"]),
            title=r.get("title", "") or "",
            body=_body_or_none(r.get("body", "") or "", include_body),
            category=r.get("category"),
            rrf_score=float(r["rrf_score"]),
            bm25_score=float(r["bm25_score"]),
            vector_score=float(r["vector_score"]),
            cross_encoder_score=float(r["cross_encoder_score"]),
            bm25_rank=r.get("bm25_rank"),
            vector_rank=r.get("vector_rank"),
            snippet=generate_snippet(r.get("body", "") or "", search_q),
        )
        for r in reranked[:top_k]
    ]
    return RerankedSearchResponse(
        query=query_text,
        corrected_query=corrected,
        returned_count=len(results),
        latency_ms=latency_ms,
        results=results,
    )

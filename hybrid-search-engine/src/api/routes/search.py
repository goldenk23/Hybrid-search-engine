"""
Search API routes.

This module exposes BM25 search through HTTP
"""

from time import perf_counter

from fastapi import APIRouter, HTTPException, Query

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
from src.query.spell_check import SpellCorrector
from src.search.bm25 import BM25Search
from src.search.cross_encoder_reranker import CrossEncoderReranker
from src.search.hybrid_search import HybridSearchEngine

spell_corrector = SpellCorrector()
spell_corrector.load_default_dictionary()

router = APIRouter(tags=["search"])

_bm25: BM25Search | None = None
_hybrid_engine: HybridSearchEngine | None = None
_reranker: CrossEncoderReranker | None = None


def get_bm25() -> BM25Search:
    """Create the BM25 engine only when the first BM25 request arrives."""
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Search()
    return _bm25


def get_hybrid_engine() -> HybridSearchEngine:
    """Create the hybrid engine only when vector search is needed."""
    global _hybrid_engine
    if _hybrid_engine is None:
        _hybrid_engine = HybridSearchEngine()
    return _hybrid_engine


def get_reranker() -> CrossEncoderReranker:
    """Reuse one lazy-loading cross-encoder reranker across requests."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=3, description="The search query"),
    top_k: int = Query(
        default=RESULTS_PER_PAGE,
        ge=1,
        le=100,
        description="Number of results to return",
    ),
) -> SearchResponse:
    """Search docmuments using BM25 search"""
    
    query_text = q.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Search query can not be empty")
    
    # Attempt spell correction
    try:
        corrected_query = spell_corrector.correct_query(query_text)
    except Exception:
        corrected_query = query_text
    
    # Use corrected query for search, but only if it's different
    search_query = corrected_query if corrected_query and corrected_query != query_text else query_text
    
    start_time = perf_counter()
    raw_results = get_bm25().search(query=search_query, top_k=top_k)
    latency_ms = int((perf_counter() - start_time) * 1000)

    results = [
        SearchResult(
            id=str(result["id"]),
            title=result["title"] or "",
            body=result["body"] or "",
            category=result["category"],
            score=result["score"],
            snippet=generate_snippet(result["body"] or "", search_query),                
        )
        for result in raw_results
    ]
    
    return SearchResponse(
        query=query_text,
        corrected_query=corrected_query if corrected_query != query_text else None,
        total=len(results),
        latency_ms=latency_ms,
        results=results,
    
    )
    
@router.get("/hybrid-search", response_model=HybridSearchResponse)
def hybrid_search(
    q: str = Query(..., min_length=3, description="The search query"),
    top_k: int = Query(default=RESULTS_PER_PAGE, ge=1, le=100),
    bm25_weight: float = Query(
        default=1.0,
        ge=0.0,
        description="BM25 contribution weight for RRF fusion",
    ),
    vector_weight: float = Query(
        default=1.0,
        ge=0.0,
        description="Vector contribution weight for RRF fusion",
    ),
    rrf_k: int = Query(
        default=60,
        ge=1,
        description="RRF rank constant; larger values smooth rank differences",
    ),
) -> HybridSearchResponse:
    """ Search documents using BM25 + vector search + weighted RRF fusion """
    query_text = q.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Search query can not be empty")
    
    # Attempt spell correction
    try:
        corrected_query = spell_corrector.correct_query(query_text)
    except Exception:
        corrected_query = query_text
    
    # Use corrected query for search, but only if it's different
    search_query = corrected_query if corrected_query and corrected_query != query_text else query_text
    
    start_time = perf_counter()
    raw_results = get_hybrid_engine().search(
        query=search_query,
        top_k=top_k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        rrf_k=rrf_k,
    )
    latency_ms = int((perf_counter() - start_time) * 1000)
    
    results = [
        HybridSearchResult(
            id=str(result["id"]),
            title=result.get("title", "") or "",
            body=result.get("body", "") or "",
            category=result.get("category"),
            rrf_score=float(result["rrf_score"]),
            bm25_score=float(result["bm25_score"]),
            vector_score=float(result["vector_score"]),
            bm25_rank=result.get("bm25_rank"),
            vector_rank=result.get("vector_rank"),
            snippet=generate_snippet(result.get("body", "") or "", search_query),
        )
        for result in raw_results
    ]

    return HybridSearchResponse(
        query=query_text,
        corrected_query=corrected_query if corrected_query != query_text else None,
        total=len(results),
        latency_ms=latency_ms,
        results=results,
    )
    
    
@router.get("/hybrid-search/rerank", response_model=RerankedSearchResponse)
def hybrid_search_rerank(
    q: str = Query(..., min_length=3, description="The search query"),
    top_k: int = Query(
        default=RESULTS_PER_PAGE,
        ge=1,
        le=50,
        description="Number of final results to return after reranking",
    ),
    candidates_k: int = Query(
        default=100,
        ge=1,
        le=100,
        description="Number of candidates to retrieve before reranking",
    ),
    bm25_weight: float = Query(
        default=1.0,
        ge=0.0,
        description="BM25 contribution weight for RRF candidate retrieval",
    ),
    vector_weight: float = Query(
        default=1.0,
        ge=0.0,
        description="Vector contribution weight for RRF candidate retrieval",
    ),
    rrf_k: int = Query(
        default=60,
        ge=1,
        description="RRF rank constant for candidate retrieval",
    ),
) -> RerankedSearchResponse:
    """ Search documents using weighted RRF candidate retrieval + cross-encoder reranking """
    query_text = q.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Search query can not be empty")

    # Attempt spell correction
    try:
        corrected_query = spell_corrector.correct_query(query_text)
    except Exception:
        corrected_query = query_text

    # Use corrected query for search, but only if it's different
    search_query = (
        corrected_query
        if corrected_query and corrected_query != query_text
        else query_text
    )

    start_time = perf_counter()

    # ---------------- INTENTIONAL BUG ----------------
    # BUG: Retrieves only top_k candidates instead of candidates_k.
    # The reranker never sees enough candidates, reducing retrieval quality.
    hybrid_candidates = get_hybrid_engine().search(
        query=search_query,
        top_k=top_k,  # BUG: should be candidates_k
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        rrf_k=rrf_k,
    )
    # -------------------------------------------------

    reranked_raw_results = get_reranker().rerank(
        query=search_query,
        candidates=hybrid_candidates,
        top_k=top_k,
        max_candidates=candidates_k,
    )

    latency_ms = int((perf_counter() - start_time) * 1000)

    reranked_results = [
        RerankedSearchResult(
            id=str(result["id"]),
            title=result.get("title", "") or "",
            body=result.get("body", "") or "",
            category=result.get("category"),
            rrf_score=float(result["rrf_score"]),
            bm25_score=float(result["bm25_score"]),
            vector_score=float(result["vector_score"]),
            cross_encoder_score=float(result["cross_encoder_score"]),
            bm25_rank=result.get("bm25_rank"),
            vector_rank=result.get("vector_rank"),
            snippet=generate_snippet(result.get("body", "") or "", search_query),
        )
        for result in reranked_raw_results[:top_k]
    ]

    return RerankedSearchResponse(
        query=query_text,
        corrected_query=corrected_query if corrected_query != query_text else None,
        total=len(reranked_results),
        latency_ms=latency_ms,
        results=reranked_results,
    )

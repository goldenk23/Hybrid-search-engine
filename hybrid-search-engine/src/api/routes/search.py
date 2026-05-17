"""
Search API routes.

This module exposes BM25 search through HTTP.
"""

from time import perf_counter

from fastapi import APIRouter, HTTPException, Query

from src.api.models import SearchResult, SearchResponse, HybridSearchResult, HybridSearchResponse
from src.config import RESULTS_PER_PAGE
from src.indexing.preprocessing import generate_snippet
from src.search.bm25 import BM25Search
from src.search.hybrid_search import HybridSearchEngine
from src.query.spell_check import SpellCorrector

spell_corrector = SpellCorrector()
spell_corrector.load_default_dictionary()

router = APIRouter(tags=["search"])
bm25 = BM25Search()
hybrid_engine = HybridSearchEngine()

@router.get("/search", response_model=SearchResponse)
def search(
    q: str=Query(..., min_length=3, description="The search query"),
    top_k: int=Query(default=RESULTS_PER_PAGE, ge=1, le=100, description="Number of results to return")            
)-> SearchResponse:
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
    raw_results = bm25.search(query=search_query, top_k=top_k)
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
    
@router.get("/hybrid-search", response_model = HybridSearchResponse)
def hybrid_search(
    q: str=Query(..., min_length=3, description="The search query"),
    top_k: int=Query(default=RESULTS_PER_PAGE, ge=1, le=100),
) -> HybridSearchResponse:
    """ Search documents using BM25 + vetor search + RRF fusion """
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
    raw_results = hybrid_engine.search(query=search_query, top_k=top_k)
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
        total=len(results),
        latency_ms=latency_ms,
        results=results,
    )
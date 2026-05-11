"""
Search API routes.

This module exposes BM25 search through HTTP.
"""

from time import perf_counter

from fastapi import APIRouter, HTTPException, Query

from src.api.models import SearchResult, SearchResponse
from src.config import RESULTS_PER_PAGE
from src.indexing.preprocessing import generate_snippet
from src.search.bm25 import BM25Search

router = APIRouter(tags=["search"])
bm25 = BM25Search()

@router.get("/search", response_model=SearchResponse)
def search(
    q: str=Query(..., min_length=3, description="The search query"),
    top_k: int=Query(default=RESULTS_PER_PAGE, ge=1, le=100, description="Number of results to return")            
)-> SearchResponse:
    """Search docmuments using BM25 search"""
    
    query_text = q.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Search query can not be empty")
    start_time = perf_counter()
    raw_results = bm25.search(query=query_text, top_k=top_k)
    latency_ms = int((perf_counter() - start_time) * 1000)

    results = [
        SearchResult(
            id=str(result["id"]),
            title=result["title"] or "",
            body=result["body"] or "",
            category=result["category"],
            score=result["score"],
            snippet=generate_snippet(result["body"] or "", query_text)                
        )
        for result in raw_results
    ]
    
    return SearchResponse(
        query=query_text,
        total=len(results),
        latency_ms=latency_ms,
        results=results,
    
    )
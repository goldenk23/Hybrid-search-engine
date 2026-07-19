""" 
This module combines the vector search and BM25 keyword search results using Reciprocal Rank Fusion (RRF) to produce a final ranked list of documents for a given query.
"""

from typing import Any

from src.config import BM25_TOP_K, VECTOR_TOP_K, RERANK_TOP_K
from src.search.bm25 import BM25Search
from src.search.fusion import reciprocal_rank_fusion
from src.search.vector import VectorSearch

class HybridSearchEngine:
    
    def __init__(
        self,
        bm25_search: BM25Search ,
        vector_search: VectorSearch | None = None,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        rrf_k: int = 60,
    ):
        self.bm25_search = bm25_search or BM25Search()
        self.vector_search = vector_search or VectorSearch()
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        
    def search(
        self,
        query: str,
        top_k: int = RERANK_TOP_K,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
        rrf_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run hybrid search:
        
        steps:
        1. Run BM25 search to get top BM25_TOP_K results
        2. Run vector semantic search to get top VECTOR_TOP_K_RESULTS
        3. Combine results using Reciprocal Rank Fusion (RRF) to get final top_k results
        
        """
        bm25_results = self.bm25_search.search(query, top_k = BM25_TOP_K)
        vector_results = self.vector_search.search(query, top_k = VECTOR_TOP_K)
        
        return reciprocal_rank_fusion(
            bm25_results = bm25_results,
            vector_results = vector_results,
            top_k = top_k,
            k = rrf_k if rrf_k is not None else self.rrf_k,
            bm25_weight = bm25_weight if bm25_weight is not None else self.bm25_weight,
            vector_weight = vector_weight if vector_weight is not None else self.vector_weight,
        )

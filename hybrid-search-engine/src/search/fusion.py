""" 
Reciprocal Rank Fusion (RRF) implementation for combining search results from BM25 and vctor search
"""

from typing import Any

def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    k: int = 60,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    
    # type annotations
    rrf_score: dict[str, float] = {}
    bm25_score: dict[str, float] = {}
    bm25_rank: dict[str, int] = {}
    vector_score: dict[str, float] = {}
    vector_rank: dict[str, int] = {}
    documents: dict[str, dict[str, Any]] = {}
    
    # process BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        
        bm25_score[doc_id] = result["score"]
        bm25_rank[doc_id] = rank # Rank is position in BM25 results (1-based)
        documents[doc_id] = result
        rrf_score[doc_id] = rrf_score.get(doc_id, 0) + 1 / (k + rank) # RRF contribution from BM25{ rrf_score.get(doc_id, 0): this is the current RRF score for the document, defaulting to 0 if it hasn't been seen before. + 1 / (k + rank): this adds the RRF contribution from the BM25 result, where k is a constant and rank is the position of the document in the BM25 results. The higher the rank (i.e., the lower the position), the larger the contribution to the RRF score.
        
    # process vector results 
    for rank, result in enumerate(vector_results, start = 1):
        doc_id = result["id"]
        
        vector_score[doc_id] = result["score"]
        vector_rank[doc_id] = rank # Rank is position in vector results (1-based)
        documents.setdefault(doc_id, result)# if bm25 already added this doc_id, we keep the existing document info (title, body, category). If not, we add the vector result's document info.
        rrf_score[doc_id] = rrf_score.get(doc_id, 0) + 1 / (k + rank) # RRF contribution from vector search
    
    sorted_docs_ids=sorted(
        rrf_score.keys(),
        key=lambda doc_id: rrf_score[doc_id],
        reverse=True
    )[:top_k] # Sort document IDs by RRF score in descending order and take the top_k results
    fused_results = []
    for doc_id in sorted_docs_ids:
        document = documents.get(doc_id, {})
        
        
        fused_results.append(
            {
                "id": doc_id,
                "title": document.get("title", ""),
                "body": document.get("body", ""),
                "category": document.get("category", ""),
                "rrf_score": rrf_score[doc_id],
                "bm25_score": bm25_score.get(doc_id, 0.0),
                "vector_score": vector_score.get(doc_id, 0.0),
                "bm25_rank": bm25_rank.get(doc_id),
                "vector_rank": vector_rank.get(doc_id),
            }
        )
    
    return fused_results

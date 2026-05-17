import sys
import os
from pathlib import Path
""" 
usage: python scripts/test_cross_encoder_reranker.py
"""

# Add src to python path so modules can be found
sys.path.append(str(Path(__file__).parent.parent))

from src.search.hybrid_search import HybridSearchEngine
from src.search.cross_encoder_reranker import CrossEncoderReranker

def main():
    query = "explain difference between nuclear fusion and fission"
    
    # Init search engine
    print(f"Initializing HybridSearchEngine...")
    hybrid_engine = HybridSearchEngine()
    
    # Init reranker
    print(f"Initializing CrossEncoderReranker...")
    reranker = CrossEncoderReranker()
    
    print(f"\nSearching for: '{query}'")
    print("-" * 50)
    
    # 1. Get hybrid results
    print("Running hybrid search...")
    # Fetch top 50 to rerank
    hybrid_results = hybrid_engine.search(query=query, top_k=50)
    
    print(f"Retrieved {len(hybrid_results)} results from hybrid search.")
    
    if not hybrid_results:
        print("No results found. Cannot test reranking.")
        return
        
    # 2. Rerank results
    print("Running cross-encoder reranking...")
    reranked_results = reranker.rerank(
        query=query,
        candidates=hybrid_results,
        top_k=5
    )
    
    print(f"\nTop 5 Results after Reranking:")
    print("=" * 100)
    
    for i, res in enumerate(reranked_results, 1):
        doc_id = res.get('id', 'N/A')
        title = res.get('title', 'N/A')
        body = res.get('body', 'N/A')
        score = res.get('cross_encoder_score', 0.0)
        
        print(f"\n[Rank {i}]")
        print(f"Doc ID: {doc_id}")
        print(f"Score: {score:.4f}")
        print(f"Title: {title}")
        print(f"Body:\n{body}")
        print("-" * 100)
    
if __name__ == "__main__":
    main()
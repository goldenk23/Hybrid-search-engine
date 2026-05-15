# scripts/test_hybrid.py
""" 
Hybrid search test - combines BM25 and vector search with reciprocal rank fusion.

Usage:
    python scripts/test_hybrid_search.py
"""
import sys
from pathlib import Path
from tabulate import tabulate

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.search.hybrid_search import HybridSearchEngine


def main() -> None:
    """Run hybrid search and display results in tabulated format."""
    
    pipeline = HybridSearchEngine()
    
    query = "what is quantum entanglement?"
    print(f"\n{'='*100}")
    print(f"HYBRID SEARCH RESULTS")
    print(f"Query: {query}")
    print(f"{'='*100}\n")
    
    results = pipeline.search(query, top_k=5)
    
    # Create summary table
    summary_data = []
    for rank, result in enumerate(results, 1):
        summary_data.append([
            rank,
            result['id'],
            f"{result['rrf_score']:.6f}",
            f"{result['bm25_score']:.6f}",
            f"{result['vector_score']:.6f}",
            result['title'][:60] + "..." if len(result['title']) > 60 else result['title'],
        ])
    
    headers = ["Rank", "Doc ID", "RRF Score", "BM25 Score", "Vector Score", "Title"]
    print(tabulate(summary_data, headers=headers, tablefmt="grid"))
    
    # Print detailed results
    print(f"\n{'='*100}")
    print("DETAILED RESULTS")
    print(f"{'='*100}\n")
    
    for rank, result in enumerate(results, 1):
        print(f"RESULT #{rank}")
        print(f"{'-'*100}")
        print(f"Document ID:    {result['id']}")
        print(f"Title:          {result['title']}")
        print(f"RRF Score:      {result['rrf_score']:.6f}")
        print(f"BM25 Score:     {result['bm25_score']:.6f}")
        print(f"Vector Score:   {result['vector_score']:.6f}")
        print(f"\nBody:")
        body = result.get('body', 'N/A')
        print(f"{body}")
        print(f"\n")
    
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()

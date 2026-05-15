"""
usge:
python scripts/test_fusion.py
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.search.fusion import reciprocal_rank_fusion


bm25_results = [
    {"id": "A", "title": "BM25 A", "body": "A body", "category": "test", "score": 12.0},
    {"id": "B", "title": "BM25 B", "body": "B body", "category": "test", "score": 8.0},
    {"id": "C", "title": "BM25 C", "body": "C body", "category": "test", "score": 5.0},
]

vector_results = [
    {"id": "C", "score": 0.92},
    {"id": "D", "score": 0.88},
    {"id": "A", "score": 0.70},
]

results = reciprocal_rank_fusion(
    bm25_results=bm25_results,
    vector_results=vector_results,
    top_k=4,
)

for result in results:
    print(result)

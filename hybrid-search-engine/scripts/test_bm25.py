import shutil
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.search.bm25 import BM25Search

temp_index_dir = Path(tempfile.gettempdir()) / "bm25_test_index"
if temp_index_dir.exists():
    shutil.rmtree(temp_index_dir)

documents = [
    {
        "id": "1",
        "title": "Python FastAPI Tutorial",
        "body": "FastAPI is a modern Python web framework for building APIs.",
        "category": "programming",
    },
    {
        "id": "2",
        "title": "PostgreSQL Indexing Guide",
        "body": "PostgreSQL supports B-tree, GIN, and full text indexes.",
        "category": "database",
    },
    {
        "id": "3",
        "title": "Neural Search Systems",
        "body": "Hybrid search combines BM25 keyword search with vector embeddings.",
        "category": "search",
    },
]

engine = BM25Search(index_path=temp_index_dir)
engine.add_documents(documents)

results = engine.search("Python api", top_k=2)
for result in results:
    print(result["score"], result["title"])

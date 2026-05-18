import sqlite3
from pathlib import Path

import faiss
import tantivy


def count_bm25(index_path: Path) -> int:
    index = tantivy.Index.open(str(index_path))
    return index.searcher().num_docs


def count_faiss(index_path: Path) -> int:
    index = faiss.read_index(str(index_path))
    return index.ntotal


def count_docstore(docstore_path: Path) -> int:
    with sqlite3.connect(docstore_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


print("bm25:", count_bm25(Path("data/indexes/bm25_compact")))
print("faiss:", count_faiss(Path("data/indexes/vector.faiss")))
print("docstore:", count_docstore(Path("data/docstore.sqlite")))
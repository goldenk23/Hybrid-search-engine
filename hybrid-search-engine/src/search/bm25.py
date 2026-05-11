"""
BM25 keyword search using Tantivy.
"""

from pathlib import Path
from typing import Any

import tantivy

from src.config import INDEX_DIR


class BM25Search:
    """BM25 search engine using Tantivy."""

    def __init__(self, index_path: Path | None = None):
        self.index_path = index_path or INDEX_DIR / "bm25"

        self.schema = (
            tantivy.SchemaBuilder()
            .add_text_field("id", stored=True)
            .add_text_field("title", stored=True)
            .add_text_field("body", stored=True)
            .add_text_field("category", stored=True)
            .build()
        )

        if self.index_path.exists():
            self.index = tantivy.Index.open(str(self.index_path))
        else:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.index = tantivy.Index(self.schema, path=str(self.index_path))

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add documents to the BM25 index."""
        writer = self.index.writer()

        for document in documents:
            writer.add_document(
                tantivy.Document(
                    id=str(document["id"]),
                    title=document.get("title", ""),
                    body=document.get("body", ""),
                    category=document.get("category", ""),
                )
            )

        writer.commit()
        self.index.reload()

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search the BM25 index and return ranked results."""
        searcher = self.index.searcher()
        query_obj = self.index.parse_query(query)
        hits = searcher.search(query_obj, top_k).hits

        results = []
        for score, doc_address in hits:
            retrieved_doc = searcher.doc(doc_address)
            results.append(
                {
                    "id": retrieved_doc.get_first("id"),
                    "title": retrieved_doc.get_first("title"),
                    "body": retrieved_doc.get_first("body"),
                    "category": retrieved_doc.get_first("category"),
                    "score": score,
                }
            )

        return results

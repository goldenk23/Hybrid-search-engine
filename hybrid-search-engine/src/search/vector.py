"""
Vector semantic search using Sentence Transformer and FAISS.
"""
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_REVISION, VECTOR_INDEX_PATH
from src.database.docstore import SQLiteDocstore


class VectorSearch:
    def __init__(
        self,
        index_path: Path | None = None,
        *,
        docstore: SQLiteDocstore | None = None,
        model: SentenceTransformer | None = None,
    ):
        self.index_path = index_path or VECTOR_INDEX_PATH

        # Accept an injected docstore (e.g. a temp one from a test) or create
        # the default production store.  Tests pass their own throwaway instance
        # here so they never touch data/docstore.sqlite.
        self.docstore = docstore or SQLiteDocstore()
        self.docstore.init()

        # Accept an injected model so tests can pass a lightweight fake instead
        # of downloading and loading the full SentenceTransformer.
        # Pin the same revision used at index time (index_vectors.py) so query
        # embeddings always match the indexed vectors. Empty config -> None ->
        # huggingface resolves latest, but pinning a hash makes it reproducible.
        self.model = model or SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            revision=EMBEDDING_MODEL_REVISION or None,
        )

        # The FAISS index is loaded lazily (on first search) or built explicitly.
        # IDs are stored inside the index itself via IndexIDMap2 — no sidecar file.
        self.index: faiss.Index | None = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        """
        Convert texts into L2-normalized float32 embedding vectors.

        Normalization makes each vector have magnitude 1, so inner-product
        search in FAISS is equivalent to cosine similarity: vectors pointing
        in the same direction (same meaning) score close to 1.0; orthogonal
        vectors (different meanings) score close to 0.0.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,  # progress bars belong in batch jobs, not web requests
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        """
        Build a FAISS index from a list of document dicts.

        Each dict must have 'id'; 'title' and 'body' are used for embedding.

        ID system: IndexIDMap2 wraps IndexFlatIP and stores the real MS MARCO
        passage ID (e.g. 12345) alongside each vector inside FAISS itself.
        There is no separate doc_ids list or .npy sidecar file — one source of
        truth means no possible mismatch between the two.
        """
        # Fix: "document" (singular) for each item; the old code used "documents"
        # for both the list parameter and the loop variable — a shadow that caused
        # .get() to be called on the list object instead of each dict.
        texts = [
            f"{document.get('title', '')} {document.get('body', '')}"
            for document in documents
        ]

        embeddings = self._encode(texts)

        # Extract numeric IDs as int64 — FAISS IndexIDMap2 requires int64.
        ids = np.asarray(
            [int(document["id"]) for document in documents], dtype=np.int64
        )

        # IndexFlatIP: exact inner-product (cosine) search over all vectors.
        # IndexIDMap2: wrapper that attaches our real passage IDs to each vector
        # so FAISS returns them directly instead of positional 0,1,2… integers.
        base = faiss.IndexFlatIP(embeddings.shape[1])
        self.index = faiss.IndexIDMap2(base)
        self.index.add_with_ids(embeddings, ids)

        self.docstore.upsert_documents(documents)
        self.save()

    def save(self) -> None:
        """Persist the FAISS index to disk."""
        if self.index is None:
            raise ValueError("Cannot save index: index is not built yet.")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        # The old np.save(doc_ids_path, ...) line is intentionally gone.
        # IDs live inside the IndexIDMap2 — no sidecar .npy file needed.

    def load(self) -> None:
        """Load the FAISS index from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Vector index not found: {self.index_path}")

        self.index = faiss.read_index(str(self.index_path))
        # The old .npy sidecar load block is intentionally gone.
        # IDs are stored inside the IndexIDMap2 and come back from .search()
        # directly — no parallel list to keep in sync.

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return the top_k semantically closest documents for a query."""
        if self.index is None:
            self.load()

        query_embedding = self._encode([query])

        # scores shape: (1, top_k)  — similarity scores for our single query
        # faiss_ids shape: (1, top_k) — the real passage IDs stored in IndexIDMap2
        scores, faiss_ids = self.index.search(query_embedding, top_k)

        doc_ids: list[str] = []
        scores_by_id: dict[str, float] = {}

        for similarity, faiss_id in zip(scores[0], faiss_ids[0]):
            if faiss_id == -1:
                # FAISS pads with -1 when fewer than top_k results exist.
                continue
            doc_id = str(int(faiss_id))  # int() drops the int64 type; str() for dict key
            doc_ids.append(doc_id)
            scores_by_id[doc_id] = float(similarity)

        docs_by_id = self.docstore.get_documents_by_ids(doc_ids)

        results = []
        for doc_id in doc_ids:
            document = docs_by_id.get(doc_id)
            if document is None:
                continue
            results.append(
                {
                    "id": doc_id,
                    "title": document.get("title", ""),
                    "body": document.get("body", ""),
                    "category": document.get("category", ""),
                    "score": scores_by_id[doc_id],
                }
            )
        return results

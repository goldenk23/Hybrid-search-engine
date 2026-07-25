"""
Hermetic tests for VectorSearch.

Uses a fake model that returns deterministic unit vectors so:
  - no SentenceTransformer download
  - no GPU/CPU model load
  - tests run in milliseconds
"""
import numpy as np
import pytest
import faiss

from src.database.docstore import SQLiteDocstore
from src.search.vector import VectorSearch


# ------------------------------------------------------------------ fake model

class FakeModel:
    """Returns a deterministic unit vector for each text.

    Each distinct text gets a unique basis vector so cosine similarity
    between identical texts is 1.0 and between different texts is 0.0.
    This makes search results completely predictable.
    """
    DIM = 8  # small enough to be fast

    def encode(self, texts, **kwargs):
        vectors = []
        for i, text in enumerate(texts):
            # Deterministic: hash the text to pick one dimension to set to 1.
            dim_index = hash(text) % self.DIM
            v = np.zeros(self.DIM, dtype="float32")
            v[dim_index] = 1.0
            vectors.append(v)
        return np.array(vectors, dtype="float32")


DOCS = [
    {"id": "10", "title": "Alpha", "body": "alpha content", "category": "test"},
    {"id": "20", "title": "Beta",  "body": "beta content",  "category": "test"},
]


@pytest.fixture
def vector(tmp_path):
    store = SQLiteDocstore(tmp_path / "docstore.sqlite")
    engine = VectorSearch(
        index_path=tmp_path / "test.faiss",
        docstore=store,
        model=FakeModel(),
    )
    return engine


# ------------------------------------------------------------------ build_index

def test_build_index_creates_faiss_index(vector):
    vector.build_index(DOCS)
    assert vector.index is not None
    assert vector.index.ntotal == 2


def test_ids_stored_inside_faiss(vector):
    """IDs must live inside IndexIDMap2, not a sidecar file."""
    vector.build_index(DOCS)
    # faiss.vector_to_array extracts the id_map from IndexIDMap2.
    stored = faiss.vector_to_array(vector.index.id_map).tolist()
    assert sorted(stored) == [10, 20]


def test_build_index_saves_to_disk(vector, tmp_path):
    vector.build_index(DOCS)
    assert (tmp_path / "test.faiss").exists()


def test_no_npy_sidecar_created(vector, tmp_path):
    vector.build_index(DOCS)
    assert not (tmp_path / "vector_doc_ids.npy").exists()


def test_documents_written_to_docstore(vector):
    vector.build_index(DOCS)
    result = vector.docstore.get_documents_by_ids(["10", "20"])
    assert "10" in result
    assert "20" in result


# ------------------------------------------------------------------ save / load round-trip

def test_save_and_load_round_trip(vector, tmp_path):
    vector.build_index(DOCS)
    vector.index = None   # clear in-memory index

    vector.load()
    assert vector.index is not None
    assert vector.index.ntotal == 2


def test_load_raises_on_missing_file(tmp_path):
    store = SQLiteDocstore(tmp_path / "docstore.sqlite")
    engine = VectorSearch(
        index_path=tmp_path / "nonexistent.faiss",
        docstore=store,
        model=FakeModel(),
    )
    with pytest.raises(FileNotFoundError):
        engine.load()


# ------------------------------------------------------------------ search

def test_search_returns_results(vector):
    vector.build_index(DOCS)
    # Query with the exact text of doc "10" — fake model gives it score 1.0.
    results = vector.search("alpha content", top_k=2)
    assert len(results) > 0


def test_search_result_ids_are_strings(vector):
    vector.build_index(DOCS)
    results = vector.search("alpha content", top_k=2)
    for r in results:
        assert isinstance(r["id"], str)


def test_search_result_has_required_fields(vector):
    vector.build_index(DOCS)
    results = vector.search("alpha content", top_k=1)
    assert len(results) == 1
    for field in ("id", "title", "body", "category", "score"):
        assert field in results[0]

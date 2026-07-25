"""
API tests using fake search services — no real indexes required.

Every test injects a fake retriever via monkeypatch so the test:
  - never reads from data/
  - never loads a model
  - runs in milliseconds
"""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

# ------------------------------------------------------------------ fakes

class FakeBM25:
    def search(self, query: str, top_k: int = 10):
        return [
            {
                "id": "1",
                "title": "Python Tutorial",
                "body": "Python is a programming language used to build APIs.",
                "category": "test",
                "score": 1.0,
            }
        ][:top_k]


class FakeHybridEngine:
    def search(self, query: str, top_k: int = 10, **kwargs):
        return [
            {
                "id": "1",
                "title": "Python Tutorial",
                "body": "Python builds fast APIs.",
                "category": "test",
                "rrf_score": 0.5,
                "bm25_score": 0.3,
                "vector_score": 0.7,
                "bm25_rank": 1,
                "vector_rank": 1,
            }
        ][:top_k]


class FakeReranker:
    def rerank(self, query: str, candidates, top_k: int = 10, **kwargs):
        return [
            {**c, "cross_encoder_score": 0.9}
            for c in candidates[:top_k]
        ]


# ------------------------------------------------------------------ fixtures

@pytest.fixture(autouse=True)
def inject_fakes(monkeypatch):
    """Replace all lazy-loading globals in search.py with fakes before each test."""
    monkeypatch.setattr("src.api.routes.search.get_bm25", lambda: FakeBM25())
    monkeypatch.setattr("src.api.routes.search.get_hybrid_engine", lambda: FakeHybridEngine())
    monkeypatch.setattr("src.api.routes.search.get_reranker", lambda: FakeReranker())


# ------------------------------------------------------------------ health

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "hybrid-search-engine"}


# ------------------------------------------------------------------ /search

def test_search_rejects_too_short_query():
    response = client.get("/search?q=py")
    assert response.status_code == 422


def test_search_response_has_expected_shape():
    response = client.get("/search?q=python")
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "total" in data
    assert "latency_ms" in data
    assert "results" in data
    assert data["query"] == "python"
    assert isinstance(data["results"], list)


def test_search_returns_fake_result():
    response = client.get("/search?q=python")
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["id"] == "1"


def test_search_top_k_respected():
    response = client.get("/search?q=python&top_k=1")
    assert response.status_code == 200
    assert len(response.json()["results"]) <= 1


# ------------------------------------------------------------------ /hybrid-search

def test_hybrid_search_response_shape():
    response = client.get("/hybrid-search?q=python")
    assert response.status_code == 200
    data = response.json()
    assert "rrf_score" in data["results"][0]
    assert "bm25_score" in data["results"][0]
    assert "vector_score" in data["results"][0]


def test_hybrid_search_rejects_short_query():
    response = client.get("/hybrid-search?q=py")
    assert response.status_code == 422


# ------------------------------------------------------------------ /hybrid-search/rerank

def test_rerank_response_shape():
    response = client.get("/hybrid-search/rerank?q=python")
    assert response.status_code == 200
    data = response.json()
    assert "cross_encoder_score" in data["results"][0]


def test_rerank_rejects_short_query():
    response = client.get("/hybrid-search/rerank?q=py")
    assert response.status_code == 422

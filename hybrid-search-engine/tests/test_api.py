"""
API tests using fake search services — no real indexes required.

Tests call create_app(lambda: fake_services) so the lifespan loads fakes
instead of real models.  Fully offline, no data/ access.
"""

import pytest
from threading import BoundedSemaphore
from dataclasses import dataclass
from fastapi.testclient import TestClient

from src.api.main import create_app, SearchServices


# ------------------------------------------------------------------ fakes

class FakeBM25:
    def search(self, query: str, top_k: int = 10):
        return [
            {"id": "1", "title": "Python Tutorial",
             "body": "Python is a language.", "category": "test", "score": 1.0},
        ][:top_k]

    def committed_document_count(self):
        return 1


class FakeVectorSearch:
    index = type("I", (), {"ntotal": 1})()

    def load(self): pass
    def search(self, query, top_k=10):
        return []


class FakeHybridEngine:
    def search(self, query: str, top_k: int = 10, **kwargs):
        return [
            {"id": "1", "title": "Python Tutorial", "body": "Python builds APIs.",
             "category": "test", "rrf_score": 0.5, "bm25_score": 0.3,
             "vector_score": 0.7, "bm25_rank": 1, "vector_rank": 1},
        ][:top_k]


class FakeReranker:
    def rerank(self, query, candidates, top_k=10, **kwargs):
        return [{**c, "cross_encoder_score": 0.9} for c in candidates[:top_k]]


class FakeSpell:
    def load_default_dictionary(self): pass
    def correct_query(self, q): return q


def _fake_services() -> SearchServices:
    return SearchServices(
        spell=FakeSpell(),
        bm25=FakeBM25(),
        hybrid=FakeHybridEngine(),
        reranker=FakeReranker(),
        rerank_slots=BoundedSemaphore(1),
    )


def _no_reranker_services() -> SearchServices:
    svc = _fake_services()
    svc.reranker = None
    return svc


# ------------------------------------------------------------------ clients

@pytest.fixture
def client():
    app = create_app(loader=_fake_services)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_no_reranker():
    app = create_app(loader=_no_reranker_services)
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------ health

def test_health_live(client):
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_health_ready(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_health_legacy(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "hybrid-search-engine"}


# ------------------------------------------------------------------ /search

def test_search_rejects_short_query(client):
    assert client.get("/search?q=py").status_code == 422


def test_search_rejects_long_query(client):
    assert client.get(f"/search?q={'x' * 257}").status_code == 422


def test_search_response_shape(client):
    r = client.get("/search?q=python")
    assert r.status_code == 200
    data = r.json()
    assert "query" in data
    assert "returned_count" in data   # renamed from 'total'
    assert "latency_ms" in data
    assert "results" in data
    assert data["query"] == "python"


def test_search_body_absent_by_default(client):
    r = client.get("/search?q=python")
    assert r.json()["results"][0]["body"] is None


def test_search_body_present_when_requested(client):
    r = client.get("/search?q=python&include_body=true")
    assert r.json()["results"][0]["body"] is not None


def test_search_top_k_respected(client):
    r = client.get("/search?q=python&top_k=1")
    assert len(r.json()["results"]) <= 1


# ------------------------------------------------------------------ /hybrid-search

def test_hybrid_search_shape(client):
    r = client.get("/hybrid-search?q=python")
    assert r.status_code == 200
    result = r.json()["results"][0]
    for field in ("rrf_score", "bm25_score", "vector_score"):
        assert field in result


def test_hybrid_search_rejects_short_query(client):
    assert client.get("/hybrid-search?q=py").status_code == 422


def test_hybrid_returned_count(client):
    r = client.get("/hybrid-search?q=python")
    assert "returned_count" in r.json()


# ------------------------------------------------------------------ /hybrid-search/rerank

def test_rerank_shape(client):
    r = client.get("/hybrid-search/rerank?q=python")
    assert r.status_code == 200
    assert "cross_encoder_score" in r.json()["results"][0]


def test_rerank_503_when_disabled(client_no_reranker):
    r = client_no_reranker.get("/hybrid-search/rerank?q=python")
    assert r.status_code == 503


def test_rerank_candidates_k_less_than_top_k_422(client):
    r = client.get("/hybrid-search/rerank?q=python&top_k=10&candidates_k=5")
    assert r.status_code == 422


def test_rerank_returned_count(client):
    r = client.get("/hybrid-search/rerank?q=python")
    assert "returned_count" in r.json()

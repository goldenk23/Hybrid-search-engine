from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "hybrid-search-engine",
    }


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

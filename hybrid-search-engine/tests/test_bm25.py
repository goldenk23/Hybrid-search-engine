import pytest
from src.search.bm25 import BM25Search


# fixture is a special function in pytest that sets up a test environment (reusable across multiple tests)
@pytest.fixture
def bm25_with_test_data(tmp_path):
    # tmp_path is a special fixture provided by pytest.
    # It creates a temporary directory (folder) for your test.
    engine = BM25Search(index_path=tmp_path / "bm25_test_index")

    documents = [
        {
            "id": "1",
            "title": "Python Programming Tutorial",
            "body": "Learn Python programming from scratch using variables loops and functions.",
            "category": "programming",
        },
        {
            "id": "2",
            "title": "JavaScript Web Development",
            "body": "Build modern web applications using JavaScript React and Node.",
            "category": "web",
        },
        {
            "id": "3",
            "title": "Python Data Science Guide",
            "body": "Use Python for data analysis with pandas numpy and matplotlib.",
            "category": "data",
        },
    ]
    engine.add_documents(documents)
    return engine


def test_search_returns_relevant_results(bm25_with_test_data):
    results = bm25_with_test_data.search("python", top_k=5)

    assert len(results) == 2
    assert results[0]["id"] == "1"
    assert results[1]["id"] == "3"


def test_search_results_are_sorted_by_score(bm25_with_test_data):
    results = bm25_with_test_data.search("python programming", top_k=5)
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_top_k(bm25_with_test_data):
    results = bm25_with_test_data.search("python", top_k=1)
    assert len(results) <= 1


def test_exact_title_match_rank_first(bm25_with_test_data):
    results = bm25_with_test_data.search("python programming Tutorial", top_k=5)

    assert len(results) >= 1
    assert results[0]["id"] == "1"

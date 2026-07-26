"""
Hermetic tests for reciprocal_rank_fusion().

All expected values computed by hand — no real indexes, no models.
"""
import pytest

from src.search.fusion import reciprocal_rank_fusion

# ------------------------------------------------------------------ helpers

def make_bm25(ids):
    return [{"id": i, "title": f"T{i}", "body": f"B{i}",
             "category": "t", "score": float(10 - rank)}
            for rank, i in enumerate(ids)]


def make_vector(ids):
    return [{"id": i, "score": 1.0 - rank * 0.1}
            for rank, i in enumerate(ids)]


# ------------------------------------------------------------------ basic correctness

def test_rrf_scores_decrease_monotonically():
    bm25 = make_bm25(["A", "B", "C"])
    vec  = make_vector(["A", "B", "C"])
    results = reciprocal_rank_fusion(bm25, vec, k=60, top_k=3)
    scores = [r["rrf_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_respected():
    bm25 = make_bm25(["A", "B", "C", "D"])
    vec  = make_vector(["A", "B", "C", "D"])
    results = reciprocal_rank_fusion(bm25, vec, top_k=2)
    assert len(results) == 2


def test_doc_appearing_in_both_lists_scores_higher():
    # "A" appears in both; "Z" only in BM25 at rank 1.
    # "A" should beat "Z" once its vector contribution is counted.
    bm25 = make_bm25(["Z", "A"])
    vec  = make_vector(["A"])
    results = reciprocal_rank_fusion(bm25, vec, k=60, top_k=2)
    ids = [r["id"] for r in results]
    assert ids[0] == "A"


def test_rrf_score_formula():
    # Manually compute: bm25_weight=1, vector_weight=1, k=60
    # "A" is rank 1 in bm25 and rank 1 in vector
    # expected score = 1/(60+1) + 1/(60+1) = 2/61
    bm25 = [{"id": "A", "title": "", "body": "", "category": "", "score": 1.0}]
    vec  = [{"id": "A", "score": 1.0}]
    results = reciprocal_rank_fusion(bm25, vec, k=60, top_k=1)
    expected = 2 / 61
    assert abs(results[0]["rrf_score"] - expected) < 1e-9


# ------------------------------------------------------------------ weights

def test_zero_bm25_weight_excludes_bm25_contribution():
    bm25 = make_bm25(["A", "B"])
    vec  = make_vector(["B", "A"])
    # With bm25_weight=0, only vector rank counts.
    # "B" is rank 1 in vector → should win.
    results = reciprocal_rank_fusion(bm25, vec, k=60, top_k=2,
                                     bm25_weight=0.0, vector_weight=1.0)
    assert results[0]["id"] == "B"


def test_zero_vector_weight_excludes_vector_contribution():
    bm25 = make_bm25(["A", "B"])
    vec  = make_vector(["B", "A"])
    results = reciprocal_rank_fusion(bm25, vec, k=60, top_k=2,
                                     bm25_weight=1.0, vector_weight=0.0)
    assert results[0]["id"] == "A"


def test_negative_weight_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(make_bm25(["A"]), make_vector(["A"]),
                               bm25_weight=-1.0)


# ------------------------------------------------------------------ edge cases

def test_empty_bm25_results():
    results = reciprocal_rank_fusion([], make_vector(["A", "B"]), top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "A"   # rank 1 in vector wins


def test_empty_vector_results():
    results = reciprocal_rank_fusion(make_bm25(["A", "B"]), [], top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "A"


def test_both_empty_returns_empty():
    assert reciprocal_rank_fusion([], [], top_k=5) == []


def test_top_k_larger_than_results():
    bm25 = make_bm25(["A"])
    vec  = make_vector(["A"])
    results = reciprocal_rank_fusion(bm25, vec, top_k=100)
    assert len(results) == 1


# ------------------------------------------------------------------ output fields

def test_result_has_all_fields():
    bm25 = make_bm25(["A"])
    vec  = make_vector(["A"])
    result = reciprocal_rank_fusion(bm25, vec, top_k=1)[0]
    for field in ("id", "title", "body", "category", "rrf_score",
                  "bm25_score", "vector_score", "bm25_rank", "vector_rank"):
        assert field in result, f"Missing field: {field}"


def test_doc_only_in_bm25_has_zero_vector_score():
    bm25 = make_bm25(["X"])
    vec  = make_vector(["Y"])
    results = reciprocal_rank_fusion(bm25, vec, top_k=2)
    x = next(r for r in results if r["id"] == "X")
    assert x["vector_score"] == 0.0
    assert x["vector_rank"] is None

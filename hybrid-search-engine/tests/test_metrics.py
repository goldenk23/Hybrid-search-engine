"""
Tests for src/evaluation/metrics.py.

Every expected value is computed by hand — no reference implementation,
no rounding shortcuts. If the formula changes, these tests break.
"""

import math

import pytest

from src.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k

# ================================================================== ndcg_at_k

class TestNdcgAtK:

    def test_perfect_ranking(self):
        # Both relevant docs at the very top → NDCG = 1.0
        assert ndcg_at_k(["A", "B", "C"], {"A", "B"}, k=3) == pytest.approx(1.0)

    def test_hand_computed_value(self):
        # ids=["A","B","C"], relevant={"A","C"}, k=3
        # DCG  = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 1/log2(3) ≈ 1.6309
        # NDCG = 1.5 / IDCG
        dcg  = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected = dcg / idcg
        assert ndcg_at_k(["A", "B", "C"], {"A", "C"}, k=3) == pytest.approx(expected)

    def test_no_relevant_in_results(self):
        assert ndcg_at_k(["X", "Y"], {"A", "B"}, k=2) == 0.0

    def test_relevant_below_k(self):
        # Relevant doc at rank 3 but k=2 — should not be counted
        assert ndcg_at_k(["X", "Y", "A"], {"A"}, k=2) == 0.0

    def test_empty_relevant_set(self):
        assert ndcg_at_k(["A", "B"], set(), k=2) == 0.0

    def test_empty_ids(self):
        assert ndcg_at_k([], {"A"}, k=10) == 0.0

    def test_k_zero(self):
        assert ndcg_at_k(["A"], {"A"}, k=0) == 0.0

    def test_k_larger_than_ids(self):
        # k=100 but only 2 results — should not crash
        1.0 / math.log2(2)
        1.0 / math.log2(2)
        assert ndcg_at_k(["A", "X"], {"A"}, k=100) == pytest.approx(1.0)

    def test_single_relevant_at_rank_1(self):
        # DCG = 1/log2(2) = 1.0, IDCG = 1.0 → NDCG = 1.0
        assert ndcg_at_k(["A"], {"A"}, k=1) == pytest.approx(1.0)

    def test_single_relevant_at_rank_2(self):
        # DCG = 1/log2(3), IDCG = 1/log2(2) = 1.0
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert ndcg_at_k(["X", "A"], {"A"}, k=2) == pytest.approx(expected)


# =================================================================== mrr_at_k

class TestMrrAtK:

    def test_first_hit_at_rank_1(self):
        assert mrr_at_k(["A", "B", "C"], {"A"}, k=3) == pytest.approx(1.0)

    def test_first_hit_at_rank_2(self):
        # Hand-computed: 1/2 = 0.5
        assert mrr_at_k(["X", "A", "C"], {"A"}, k=3) == pytest.approx(0.5)

    def test_first_hit_at_rank_3(self):
        assert mrr_at_k(["X", "Y", "A"], {"A"}, k=3) == pytest.approx(1.0 / 3)

    def test_no_hit_returns_zero(self):
        assert mrr_at_k(["X", "Y"], {"A"}, k=2) == 0.0

    def test_hit_beyond_k_not_counted(self):
        # A is at rank 3 but k=2
        assert mrr_at_k(["X", "Y", "A"], {"A"}, k=2) == 0.0

    def test_multiple_relevant_uses_first(self):
        # A at rank 1, B at rank 3 — should return 1.0 (rank 1 hit)
        assert mrr_at_k(["A", "X", "B"], {"A", "B"}, k=3) == pytest.approx(1.0)

    def test_empty_ids(self):
        assert mrr_at_k([], {"A"}, k=10) == 0.0

    def test_empty_relevant(self):
        assert mrr_at_k(["A", "B"], set(), k=2) == 0.0


# ================================================================= recall_at_k

class TestRecallAtK:

    def test_all_relevant_found(self):
        assert recall_at_k(["A", "B", "C"], {"A", "B"}, k=3) == pytest.approx(1.0)

    def test_hand_computed_half(self):
        # ids=["A","B","C","D"], relevant={"A","E"}, k=3
        # found: A (yes), E (no) → 1/2 = 0.5
        assert recall_at_k(["A", "B", "C", "D"], {"A", "E"}, k=3) == pytest.approx(0.5)

    def test_nothing_found(self):
        assert recall_at_k(["X", "Y"], {"A", "B"}, k=2) == 0.0

    def test_relevant_below_cutoff_not_counted(self):
        # A is at index 3 (rank 4) but k=3
        assert recall_at_k(["X", "Y", "Z", "A"], {"A"}, k=3) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["A", "B"], set(), k=2) == 0.0

    def test_empty_ids(self):
        assert recall_at_k([], {"A"}, k=10) == 0.0

    def test_k_zero(self):
        assert recall_at_k(["A"], {"A"}, k=0) == 0.0

    def test_k_larger_than_ids(self):
        assert recall_at_k(["A"], {"A", "B"}, k=100) == pytest.approx(0.5)

    def test_full_recall(self):
        assert recall_at_k(["A", "B", "C"], {"A", "B", "C"}, k=3) == pytest.approx(1.0)

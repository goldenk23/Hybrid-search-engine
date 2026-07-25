"""
Retrieval evaluation metrics — single authoritative copy.

Every script that reports NDCG, MRR, or Recall imports from here.
Two copies that drift apart produce untrustworthy benchmark numbers.

Functions
---------
ndcg_at_k   — quality of the top-k ranking (rewards relevant docs near the top)
mrr_at_k    — position of the first relevant result
recall_at_k — fraction of all relevant docs found in top k
"""

from __future__ import annotations

import math


def ndcg_at_k(ids: list[str], relevant: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k.

    Measures how well the top-k results are *ordered* — a relevant document
    at rank 1 is worth more than one at rank k.

    DCG formula:  sum of  1 / log2(rank + 1)  for each relevant doc in top k
                  (binary relevance: a doc is either relevant or not)
    IDCG:         the best possible DCG — all relevant docs at the top
    NDCG:         DCG / IDCG  →  range [0, 1],  1.0 = perfect ranking

    Args:
        ids:      ranked list of retrieved document IDs (position 0 = rank 1)
        relevant: set of document IDs that are relevant for this query
        k:        depth to evaluate at (e.g. 10)

    Returns:
        NDCG@k in [0, 1].
    """
    if not relevant or k <= 0:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(ids[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal: all relevant docs placed at ranks 1, 2, … min(|relevant|, k)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return 0.0 if idcg == 0.0 else dcg / idcg


def mrr_at_k(ids: list[str], relevant: set[str], k: int) -> float:
    """Mean Reciprocal Rank at k (single-query component).

    Returns 1/rank of the *first* relevant document in the top-k list.
    If no relevant document appears in the top k, returns 0.

    To get MRR over a set of queries, average the per-query values:
        mrr = sum(mrr_at_k(...) for each query) / num_queries

    Args:
        ids:      ranked list of retrieved document IDs
        relevant: set of relevant document IDs
        k:        depth to evaluate at

    Returns:
        1/rank of the first hit, or 0.0 if no hit in top k.
    """
    for rank, doc_id in enumerate(ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ids: list[str], relevant: set[str], k: int) -> float:
    """Recall at k.

    Fraction of all relevant documents that appear anywhere in the top-k list.
    A high Recall@100 means later stages (e.g. a reranker) have a chance to
    surface the good results — if a relevant doc isn't in the top 100, no
    reranker can recover it.

    Args:
        ids:      ranked list of retrieved document IDs
        relevant: set of relevant document IDs
        k:        depth to evaluate at (e.g. 100)

    Returns:
        |retrieved ∩ relevant| / |relevant|  in [0, 1], or 0.0 if relevant is empty.
    """
    if not relevant or k <= 0:
        return 0.0
    return len(set(ids[:k]) & relevant) / len(relevant)

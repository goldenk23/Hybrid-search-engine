"""
Search Pipeline Overview:

1. BM25 Search (Word-based Retrieval)
   - Mechanism: Keyword-based matching.
   - Trade-off: Fast, but susceptible to vocabulary mismatch (misses synonyms).

2. Vector Search (Bi-Encoder)
   - Mechanism: Maps queries and documents into the same vector space independently; calculates relevance via Cosine Similarity.
   - Trade-off: Extremely fast due to pre-computed embeddings, but lacks fine-grained interaction between query and document terms.

3. Reranking (Cross-Encoder)
   - Mechanism: Feeds query and document simultaneously as a single input pair (e.g., [CLS] Query [SEP] Document [SEP]), allowing full attention across both.
   - Trade-off: Significantly more accurate via deep semantic interactions, but computationally expensive as scores cannot be pre-computed.
"""

from typing import Any

class CrossEncoderReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL_NAME):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
        max_candidates: int = 50,
    ) -> list[dict[str, Any]]:
        """Rerank candidate documents using query-document relevance scores."""
        if not candidates:
            return []

        candidates_to_score = candidates[:max_candidates]
        pairs = [
            [query, f"{candidate.get('title', '')} {candidate.get('body', '')}"[:2000]]
            for candidate in candidates_to_score
        ]
        scores = self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=True,
        )

        reranked = [
            {**candidate, "cross_encoder_score": float(score)}
            for candidate, score in zip(candidates_to_score, scores)
        ]

        return sorted(
            reranked,
            key=lambda result: result["cross_encoder_score"],
            reverse=True,
        )[:top_k]

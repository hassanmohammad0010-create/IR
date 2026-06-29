from __future__ import annotations

import logging

import numpy as np

from src.indexing.embedding_index import EmbeddingIndex

logger = logging.getLogger(__name__)


class EmbeddingRetriever:

    def __init__(self, embedding_index: EmbeddingIndex):
        self.index = embedding_index

    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:

        if not query.strip():
            return []

        if self.index.matrix is None or len(self.index.doc_ids) == 0:
            logger.warning("EmbeddingRetriever: index is empty.")
            return []

        query_vec = self.index.encode_query(query)  # shape (D,)

        scores: np.ndarray = self.index.matrix @ query_vec  # shape (N,)

        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            part = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = part[np.argsort(scores[part])[::-1]]

        results: list[tuple[str, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            results.append((self.index.doc_ids[idx], score))

        logger.debug(
            f"EmbeddingRetriever: query='{query[:60]}' "
            f"top_score={results[0][1]:.4f}"
            if results
            else f"EmbeddingRetriever: no results for query='{query[:60]}'"
        )
        return results

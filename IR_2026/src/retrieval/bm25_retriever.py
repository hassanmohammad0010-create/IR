import logging

import numpy as np

from src.indexing.bm25_index import BM25Index, DEFAULT_K1, DEFAULT_B

logger = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, bm25_index: BM25Index):
        self.index = bm25_index

    def search(
        self,
        query: str,
        top_k: int = 10,
        k1: float | None = None,
        b: float | None = None,
    ) -> list[tuple[str, float]]:

        query_tokens = query.split()
        if not query_tokens:
            return []

        scorer = self.index.get_scorer(k1=k1, b=b)

        scores: np.ndarray = scorer.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1]
        results: list[tuple[str, float]] = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            score = float(scores[idx])
            results.append((self.index.doc_ids[idx], score))

        return results

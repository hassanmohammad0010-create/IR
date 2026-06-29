from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

RRF_K = 60  # standard constant for RRF


class ParallelHybrid:

    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        retriever_a,
        retriever_b,
        top_k: int = 10,
        fusion: str = "rrf",
        alpha: float = 0.5,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
        pool_size: int = 100,
    ) -> list[tuple[str, float]]:
        results_a = self._retrieve(retriever_a, query, pool_size, bm25_k1, bm25_b)
        results_b = self._retrieve(retriever_b, query, pool_size, bm25_k1, bm25_b)

        logger.debug(
            f"ParallelHybrid | fusion={fusion} | "
            f"A={len(results_a)} hits | B={len(results_b)} hits"
        )

        if fusion == "rrf":
            fused = self._rrf(results_a, results_b)
        elif fusion == "weighted":
            fused = self._weighted(results_a, results_b, alpha=alpha)
        else:
            raise ValueError(
                f"Unknown fusion method '{fusion}'. Use 'rrf' or 'weighted'."
            )

        return fused[:top_k]

    # ------------------------------------------------------------------

    @staticmethod
    def _rrf(
        results_a: list[tuple[str, float]],
        results_b: list[tuple[str, float]],
        k: int = RRF_K,
    ) -> list[tuple[str, float]]:

        scores: dict[str, float] = defaultdict(float)

        for rank, (doc_id, _) in enumerate(results_a, start=1):
            scores[doc_id] += 1.0 / (k + rank)

        for rank, (doc_id, _) in enumerate(results_b, start=1):
            scores[doc_id] += 1.0 / (k + rank)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _weighted(
        results_a: list[tuple[str, float]],
        results_b: list[tuple[str, float]],
        alpha: float = 0.5,
    ) -> list[tuple[str, float]]:

        def normalise(results: list[tuple[str, float]]) -> dict[str, float]:
            if not results:
                return {}
            scores = [s for _, s in results]
            min_s = min(scores)
            max_s = max(scores)
            rng = max_s - min_s if max_s != min_s else 1.0
            return {doc_id: (s - min_s) / rng for doc_id, s in results}

        norm_a = normalise(results_a)
        norm_b = normalise(results_b)

        all_docs = set(norm_a) | set(norm_b)
        combined: dict[str, float] = {}
        for doc_id in all_docs:
            sa = norm_a.get(doc_id, 0.0)
            sb = norm_b.get(doc_id, 0.0)
            combined[doc_id] = alpha * sa + (1.0 - alpha) * sb

        return sorted(combined.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------

    @staticmethod
    def _retrieve(
        retriever,
        query: str,
        top_k: int,
        bm25_k1: float | None,
        bm25_b: float | None,
    ) -> list[tuple[str, float]]:
        from src.retrieval.bm25_retriever import BM25Retriever

        if isinstance(retriever, BM25Retriever) and (
            bm25_k1 is not None or bm25_b is not None
        ):
            return retriever.search(query, top_k=top_k, k1=bm25_k1, b=bm25_b)
        return retriever.search(query, top_k=top_k)

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SerialHybrid:

    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        retriever_a,
        retriever_b,
        top_k: int = 10,
        pool_size: int = 200,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
    ) -> list[tuple[str, float]]:
        candidates = self._retrieve(retriever_a, query, pool_size, bm25_k1, bm25_b)
        candidate_ids = [doc_id for doc_id, _ in candidates]

        logger.debug(f"SerialHybrid | stage-1 returned {len(candidate_ids)} candidates")

        if not candidate_ids:
            return []

        reranked = self._rerank(retriever_b, query, candidate_ids, top_k)

        logger.debug(f"SerialHybrid | stage-2 returned {len(reranked)} results")

        return reranked

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

    @staticmethod
    def _rerank(
        retriever_b,
        query: str,
        candidate_ids: list[str],
        top_k: int,
    ) -> list[tuple[str, float]]:

        from src.retrieval.embedding_retriever import EmbeddingRetriever
        from src.retrieval.tfidf_retriever import TFIDFRetriever

        if isinstance(retriever_b, EmbeddingRetriever):
            return SerialHybrid._rerank_embedding(
                retriever_b, query, candidate_ids, top_k
            )
        if isinstance(retriever_b, TFIDFRetriever):
            return SerialHybrid._rerank_tfidf(retriever_b, query, candidate_ids, top_k)

        all_results = retriever_b.search(query, top_k=len(candidate_ids) * 2)
        cand_set = set(candidate_ids)
        filtered = [(d, s) for d, s in all_results if d in cand_set]
        return filtered[:top_k]

    @staticmethod
    def _rerank_embedding(retriever_b, query, candidate_ids, top_k):
        index = retriever_b.index
        query_vec = index.encode_query(query)

        id_to_pos = {doc_id: i for i, doc_id in enumerate(index.doc_ids)}
        rows = [id_to_pos[d] for d in candidate_ids if d in id_to_pos]

        if not rows:
            return []

        sub_matrix = index.matrix[rows]
        scores = sub_matrix @ query_vec
        order = np.argsort(scores)[::-1]

        results = []
        for idx in order[:top_k]:
            doc_id = index.doc_ids[rows[idx]]
            results.append((doc_id, float(scores[idx])))
        return results

    @staticmethod
    def _rerank_tfidf(retriever_b, query, candidate_ids, top_k):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        index = retriever_b.index
        query_vec = index.vectorizer.transform([query])
        id_to_pos = {doc_id: i for i, doc_id in enumerate(index.doc_ids)}
        rows = [id_to_pos[d] for d in candidate_ids if d in id_to_pos]

        if not rows:
            return []

        sub_matrix = index.matrix[rows]
        sims = cosine_similarity(query_vec, sub_matrix).flatten()
        order = np.argsort(sims)[::-1]

        results = []
        for idx in order[:top_k]:
            doc_id = index.doc_ids[rows[idx]]
            results.append((doc_id, float(sims[idx])))
        return results

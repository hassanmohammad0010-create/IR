from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


class EvaluationRunner:

    def __init__(self, retrieval_service, evaluation_service):
        self.retrieval_service = retrieval_service
        self.evaluation_service = evaluation_service

    def run(
        self,
        dataset: str,
        model: str,
        queries: list,
        qrels: dict,
        top_k: int = 100,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
        metrics: set[str] | None = None,
        enable_refinement: bool = False,
        # hybrid params
        hybrid_mode: str = "parallel",
        hybrid_model_a: str = "bm25",
        hybrid_model_b: str = "embedding",
        hybrid_fusion: str = "rrf",
        hybrid_alpha: float = 0.5,
        hybrid_pool: int = 200,
    ) -> dict:

        results: dict = {}
        total = len(queries)
        is_hybrid = model == "hybrid"

        for i, query in enumerate(queries, start=1):
            qid = query["query_id"]
            text = query["text"]

            if i % 10 == 0 or i == 1:
                logger.info(f"Evaluating query {i}/{total}: {qid}")

            if is_hybrid:
                retrieved = self.retrieval_service.hybrid_search(
                    dataset=dataset,
                    query=text,
                    mode=hybrid_mode,
                    model_a=hybrid_model_a,
                    model_b=hybrid_model_b,
                    top_k=top_k,
                    pool_size=hybrid_pool,
                    fusion=hybrid_fusion,
                    alpha=hybrid_alpha,
                    bm25_k1=bm25_k1,
                    bm25_b=bm25_b,
                    enable_refinement=enable_refinement,
                )
            else:
                retrieved = self.retrieval_service.search(
                    dataset=dataset,
                    query=text,
                    model=model,
                    top_k=top_k,
                    bm25_k1=bm25_k1,
                    bm25_b=bm25_b,
                    enable_refinement=enable_refinement,
                )

            results[qid] = self.evaluation_service.results_to_pytrec_format(retrieved)

        return self.evaluation_service.evaluate(
            qrels=qrels, results=results, metrics=metrics
        )

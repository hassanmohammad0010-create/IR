from __future__ import annotations

import logging
from typing import Any

from src.retrieval.hybrid.hybrid_service import HybridService

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("bm25", "tfidf", "inverted", "embedding")
SUPPORTED_MODES = ("parallel", "serial")
SUPPORTED_FUSIONS = ("rrf", "weighted")


class RetrievalService:

    def __init__(
        self,
        datasets: dict,
        preprocessing_service=None,
        refinement_service=None,
    ):
        self.datasets = datasets
        self.preprocessing = preprocessing_service
        self.refinement = refinement_service
        self._hybrid = HybridService()

    # ------------------------------------------------------------------

    def search(
        self,
        dataset: str,
        query: str,
        model: str = "bm25",
        top_k: int = 10,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
        enable_refinement: bool = False,
    ) -> list[dict[str, Any]]:

        self._validate_dataset(dataset)
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model '{model}'. Supported: {SUPPORTED_MODELS}")
        if model not in self.datasets[dataset]:
            raise ValueError(f"Model '{model}' not loaded for dataset '{dataset}'.")

        query = self._apply_pipeline(query, enable_refinement)
        if not query:
            return []

        dataset_obj = self.datasets[dataset]
        retriever = dataset_obj[model]
        doc_repo = dataset_obj["doc_repo"]

        if model == "bm25" and (bm25_k1 is not None or bm25_b is not None):
            raw = retriever.search(query, top_k=top_k, k1=bm25_k1, b=bm25_b)
        else:
            raw = retriever.search(query, top_k=top_k)

        return self._enrich(raw, doc_repo)

    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        dataset: str,
        query: str,
        mode: str = "parallel",
        model_a: str = "bm25",
        model_b: str = "embedding",
        top_k: int = 10,
        pool_size: int = 200,
        fusion: str = "rrf",
        alpha: float = 0.5,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
        enable_refinement: bool = False,
    ) -> list[dict[str, Any]]:

        self._validate_dataset(dataset)

        query = self._apply_pipeline(query, enable_refinement)
        if not query:
            return []

        dataset_obj = self.datasets[dataset]
        doc_repo = dataset_obj["doc_repo"]

        raw = self._hybrid.search(
            dataset_obj=dataset_obj,
            query=query,
            mode=mode,
            model_a=model_a,
            model_b=model_b,
            top_k=top_k,
            pool_size=pool_size,
            fusion=fusion,
            alpha=alpha,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )

        return self._enrich(raw, doc_repo)

    # ------------------------------------------------------------------

    def pipeline(
        self,
        query: str,
        enable_refinement: bool = False,
    ) -> dict:

        result: dict = {}

        preprocessed_query = query
        if self.preprocessing:
            prep = self.preprocessing.pipeline(query=query)
            if "tokenizer" in prep and isinstance(prep["tokenizer"], list):
                prep["tokenizer"] = [str(t) for t in prep["tokenizer"]]
            if "lemmatizer" in prep and isinstance(prep["lemmatizer"], list):
                prep["lemmatizer"] = [str(t) for t in prep["lemmatizer"]]
            result.update(prep)
            if prep.get("lemmatizer"):
                preprocessed_query = " ".join(prep["lemmatizer"])
            elif prep.get("normalizer"):
                preprocessed_query = prep["normalizer"]

        if self.refinement is not None:
            refinement_info = self.refinement.explain(
                preprocessed_query, enabled=enable_refinement
            )
            result["refinement"] = refinement_info

        return result

    # ------------------------------------------------------------------

    def available_datasets(self) -> list[str]:
        return list(self.datasets.keys())

    def available_models(self) -> list[str]:
        return list(SUPPORTED_MODELS)

    def dataset_meta(self, dataset: str) -> dict:
        self._validate_dataset(dataset)
        return self.datasets[dataset].get("meta", {})

    # ------------------------------------------------------------------

    def _apply_pipeline(self, query: str, enable_refinement: bool) -> str:

        if self.preprocessing:
            query = self.preprocessing.preprocess_query(query)

        if enable_refinement and self.refinement is not None:
            query = self.refinement.refine(query, enabled=True)

        if not query.strip():
            return ""
        return query

    @staticmethod
    def _enrich(
        results: list[tuple[str, float]],
        document_repository,
    ) -> list[dict[str, Any]]:
        enriched: list[dict] = []
        rank = 1
        for doc_id, score in results:
            doc = document_repository.get_document(doc_id)
            if doc is None:
                logger.warning(f"doc_id '{doc_id}' not found — skipped.")
                continue
            enriched.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": round(float(score), 6),
                    "text": doc.get("text", ""),
                }
            )
            rank += 1
        return enriched

    def _validate_dataset(self, dataset: str) -> None:
        if dataset not in self.datasets:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Available: {list(self.datasets.keys())}"
            )

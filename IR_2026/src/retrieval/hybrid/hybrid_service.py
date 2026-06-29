from __future__ import annotations

import logging
from typing import Any

from src.retrieval.hybrid.hybrid_parallel import ParallelHybrid
from src.retrieval.hybrid.hybrid_serial import SerialHybrid

logger = logging.getLogger(__name__)

SUPPORTED_MODES = ("parallel", "serial")
SUPPORTED_MODELS = ("bm25", "tfidf", "inverted", "embedding")
SUPPORTED_FUSIONS = ("rrf", "weighted")


class HybridService:

    def __init__(self):
        self._parallel = ParallelHybrid()
        self._serial = SerialHybrid()

    # ------------------------------------------------------------------

    def search(
        self,
        dataset_obj: dict,
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
    ) -> list[tuple[str, float]]:

        self._validate(dataset_obj, mode, model_a, model_b, fusion)

        retriever_a = dataset_obj[model_a]
        retriever_b = dataset_obj[model_b]

        if mode == "parallel":
            raw = self._parallel.search(
                query=query,
                retriever_a=retriever_a,
                retriever_b=retriever_b,
                top_k=top_k,
                fusion=fusion,
                alpha=alpha,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
                pool_size=pool_size,
            )
        else:
            raw = self._serial.search(
                query=query,
                retriever_a=retriever_a,
                retriever_b=retriever_b,
                top_k=top_k,
                pool_size=pool_size,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
            )

        return raw

    # ------------------------------------------------------------------
    def _validate(
        self,
        dataset_obj: dict,
        mode: str,
        model_a: str,
        model_b: str,
        fusion: str,
    ) -> None:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Use: {SUPPORTED_MODES}")
        if fusion not in SUPPORTED_FUSIONS:
            raise ValueError(f"Unknown fusion '{fusion}'. Use: {SUPPORTED_FUSIONS}")
        for m in (model_a, model_b):
            if m not in SUPPORTED_MODELS:
                raise ValueError(f"Unknown model '{m}'. Use: {SUPPORTED_MODELS}")
            if m not in dataset_obj:
                raise ValueError(
                    f"Model '{m}' not loaded for this dataset. "
                    "Run the build script first."
                )
        if model_a == model_b:
            raise ValueError("model_a and model_b must be different.")

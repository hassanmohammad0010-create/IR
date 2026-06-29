from __future__ import annotations

import json
import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from src.common.pkl_io import pkl_load, pkl_dump

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH = 64


class EmbeddingIndex:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.doc_ids: list[str] = []
        self.matrix: np.ndarray | None = None
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading sentence-transformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _normalise(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vecs / norms).astype(np.float32)

    # ------------------------------------------------------------------

    def build(
        self,
        documents_path: str,
        batch_size: int = DEFAULT_BATCH,
        limit: int | None = None,
    ) -> int:

        texts: list[str] = []
        self.doc_ids = []

        with open(documents_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                self.doc_ids.append(str(doc["doc_id"]))
                texts.append(doc.get("text", ""))

        logger.info(
            f"EmbeddingIndex: encoding {len(texts):,} docs "
            f"with model '{self.model_name}' …"
        )
        model = self._get_model()
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalise ourselves
        )
        self.matrix = self._normalise(vecs)
        logger.info(
            f"EmbeddingIndex built: {len(self.doc_ids):,} docs, "
            f"dim={self.matrix.shape[1]}"
        )
        return len(self.doc_ids)

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        pkl_dump(
            {
                "model_name": self.model_name,
                "doc_ids": self.doc_ids,
                "matrix": self.matrix,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "EmbeddingIndex":
        data = pkl_load(path)
        obj = cls(model_name=data.get("model_name", DEFAULT_MODEL))
        obj.doc_ids = data["doc_ids"]
        obj.matrix = data["matrix"].astype(np.float32)
        return obj

    # ------------------------------------------------------------------

    def encode_query(self, text: str) -> np.ndarray:
        model = self._get_model()
        vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=False)
        return self._normalise(vec)[0]  # shape (D,)

    # ------------------------------------------------------------------

    def vocabulary_size(self) -> int:
        return self.matrix.shape[1] if self.matrix is not None else 0

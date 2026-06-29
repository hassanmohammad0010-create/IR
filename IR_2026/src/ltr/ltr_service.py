from __future__ import annotations

import logging
from typing import Any
from src.common.pkl_io import pkl_load, pkl_dump

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "bm25_score",
    "tfidf_score",
    "embedding_score",
    "bm25_recip_rank",
    "tfidf_recip_rank",
    "embedding_recip_rank",
]

DEFAULT_POOL_SIZE = 200


class LTRModel:

    def __init__(self):
        self.scaler: StandardScaler | None = None
        self.clf: LogisticRegression | None = None
        self.feature_names: list[str] = FEATURE_NAMES
        self.trained_on: dict[str, Any] = {}

    @property
    def is_trained(self) -> bool:
        return self.clf is not None and self.scaler is not None

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # qrels are usually mostly-negative
        )
        self.clf.fit(X_scaled, y)

        coefs = dict(zip(self.feature_names, self.clf.coef_[0].tolist()))
        self.trained_on = {
            "n_samples": int(X.shape[0]),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "coefficients": coefs,
            "intercept": float(self.clf.intercept_[0]),
        }
        return self.trained_on

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("LTRModel: not trained yet.")
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)[:, 1]

    def save(self, path: str) -> None:
        pkl_dump(
            {
                "scaler": self.scaler,
                "clf": self.clf,
                "feature_names": self.feature_names,
                "trained_on": self.trained_on,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "LTRModel":
        data = pkl_load(path)
        obj = cls()
        obj.scaler = data["scaler"]
        obj.clf = data["clf"]
        obj.feature_names = data.get("feature_names", FEATURE_NAMES)
        obj.trained_on = data.get("trained_on", {})
        return obj


class LTRService:

    def __init__(self):
        self._models: dict[str, LTRModel] = {}

    # ------------------------------------------------------------------

    def register(self, dataset_name: str, model: LTRModel) -> None:
        self._models[dataset_name] = model

    def available_datasets(self) -> list[str]:
        return list(self._models.keys())

    def is_ready(self, dataset_name: str) -> bool:
        m = self._models.get(dataset_name)
        return m is not None and m.is_trained

    # ------------------------------------------------------------------

    def train(
        self,
        dataset_name: str,
        dataset_obj: dict,
        queries: list[dict],
        qrels: dict[str, dict[str, int]],
        pool_size: int = DEFAULT_POOL_SIZE,
        relevance_threshold: int = 1,
    ) -> dict[str, Any]:

        rows: list[list[float]] = []
        labels: list[int] = []

        for query in queries:
            qid = query["query_id"]
            text = query["text"]
            judged = qrels.get(qid)
            if not judged:
                continue

            features_by_doc = self._candidate_features(dataset_obj, text, pool_size)
            if not features_by_doc:
                continue

            for doc_id, feats in features_by_doc.items():
                relevance = judged.get(doc_id, 0)
                label = 1 if relevance >= relevance_threshold else 0
                rows.append(feats)
                labels.append(label)

        if not rows:
            raise ValueError(
                f"LTRService: no training rows produced for dataset "
                f"'{dataset_name}'. Check that queries/qrels are loaded."
            )

        X = np.array(rows, dtype=np.float64)
        y = np.array(labels, dtype=np.int32)

        model = LTRModel()
        stats = model.fit(X, y)
        self.register(dataset_name, model)

        logger.info(
            f"LTRService: trained '{dataset_name}' — "
            f"{stats['n_samples']} samples "
            f"({stats['n_positive']} pos / {stats['n_negative']} neg)"
        )
        return stats

    # ------------------------------------------------------------------

    def rerank(
        self,
        dataset_name: str,
        dataset_obj: dict,
        query: str,
        top_k: int = 10,
        pool_size: int = DEFAULT_POOL_SIZE,
    ) -> list[tuple[str, float]]:
        model = self._get_model(dataset_name)

        features_by_doc = self._candidate_features(dataset_obj, query, pool_size)
        if not features_by_doc:
            return []

        doc_ids = list(features_by_doc.keys())
        X = np.array([features_by_doc[d] for d in doc_ids], dtype=np.float64)

        scores = model.predict_proba(X)
        order = np.argsort(scores)[::-1][:top_k]

        return [(doc_ids[i], float(scores[i])) for i in order]

    def explain(self, dataset_name: str) -> dict[str, Any]:
        model = self._get_model(dataset_name)
        return {
            "dataset": dataset_name,
            "feature_names": model.feature_names,
            **model.trained_on,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _candidate_features(
        dataset_obj: dict,
        query: str,
        pool_size: int,
    ) -> dict[str, list[float]]:

        bm25 = dataset_obj.get("bm25")
        tfidf = dataset_obj.get("tfidf")
        embedding = dataset_obj.get("embedding")

        bm25_hits = bm25.search(query, top_k=pool_size) if bm25 else []
        tfidf_hits = tfidf.search(query, top_k=pool_size) if tfidf else []
        embedding_hits = embedding.search(query, top_k=pool_size) if embedding else []

        bm25_scores = {d: s for d, s in bm25_hits}
        tfidf_scores = {d: s for d, s in tfidf_hits}
        embedding_scores = {d: s for d, s in embedding_hits}

        bm25_ranks = {d: r + 1 for r, (d, _) in enumerate(bm25_hits)}
        tfidf_ranks = {d: r + 1 for r, (d, _) in enumerate(tfidf_hits)}
        embedding_ranks = {d: r + 1 for r, (d, _) in enumerate(embedding_hits)}

        all_doc_ids = set(bm25_scores) | set(tfidf_scores) | set(embedding_scores)

        features: dict[str, list[float]] = {}
        for doc_id in all_doc_ids:
            features[doc_id] = [
                bm25_scores.get(doc_id, 0.0),
                tfidf_scores.get(doc_id, 0.0),
                embedding_scores.get(doc_id, 0.0),
                1.0 / bm25_ranks[doc_id] if doc_id in bm25_ranks else 0.0,
                1.0 / tfidf_ranks[doc_id] if doc_id in tfidf_ranks else 0.0,
                1.0 / embedding_ranks[doc_id] if doc_id in embedding_ranks else 0.0,
            ]
        return features

    # ------------------------------------------------------------------

    def _get_model(self, dataset_name: str) -> LTRModel:
        model = self._models.get(dataset_name)
        if model is None or not model.is_trained:
            raise ValueError(
                f"No trained LTR model for dataset '{dataset_name}'. "
                f"Available: {[k for k, v in self._models.items() if v.is_trained]}. "
                f"Run scripts/train_ltr_model.py first."
            )
        return model

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from src.common.pkl_io import pkl_load, pkl_dump

import numpy as np
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)

CLUSTER_COUNTS: dict[str, int] = {
    "msmarco": 500,
    "quora": 800,
}
DEFAULT_CLUSTER_COUNT = 300


class ClusteringIndex:

    def __init__(self, n_clusters: int = DEFAULT_CLUSTER_COUNT):
        self.n_clusters = n_clusters
        self.doc_ids: list[str] = []
        self.labels: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self._kmeans: MiniBatchKMeans | None = None

    # ------------------------------------------------------------------

    def build(
        self,
        doc_ids: list[str],
        matrix: np.ndarray,
        *,
        batch_size: int = 4096,
        random_state: int = 42,
        n_init: int = 5,
    ) -> None:

        if len(doc_ids) != matrix.shape[0]:
            raise ValueError(
                f"doc_ids length ({len(doc_ids)}) != matrix rows ({matrix.shape[0]})"
            )

        self.doc_ids = list(doc_ids)
        n = len(doc_ids)

        effective_k = min(self.n_clusters, n // 5)
        if effective_k != self.n_clusters:
            logger.warning(
                f"ClusteringIndex: capping n_clusters from {self.n_clusters} "
                f"to {effective_k} (dataset has only {n} docs)."
            )
            self.n_clusters = effective_k

        km = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=batch_size,
            n_init=n_init,
            random_state=random_state,
            verbose=0,
        )
        km.fit(matrix)

        self._kmeans = km
        self.labels = km.labels_.astype(np.int32)
        # L2-normalise centroids so we can use dot-product for nearest-centroid
        raw_centroids = km.cluster_centers_.astype(np.float32)
        norms = np.linalg.norm(raw_centroids, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.centroids = raw_centroids / norms

        inertia = km.inertia_
        logger.info(
            f"ClusteringIndex: done — "
            f"n_clusters={self.n_clusters}, inertia={inertia:.2f}"
        )

    # ------------------------------------------------------------------

    def nearest_clusters(self, query_vec: np.ndarray, n_probes: int = 3) -> list[int]:

        if self.centroids is None:
            raise RuntimeError("ClusteringIndex not built yet.")

        sims = self.centroids @ query_vec  # shape (K,)
        n_probes = min(n_probes, self.n_clusters)
        top = np.argsort(sims)[::-1][:n_probes]
        return top.tolist()

    def docs_in_clusters(self, cluster_ids: list[int]) -> list[str]:
        if self.labels is None:
            raise RuntimeError("ClusteringIndex not built yet.")
        mask = np.isin(self.labels, cluster_ids)
        return [self.doc_ids[i] for i in np.where(mask)[0]]

    # ------------------------------------------------------------------

    def cluster_info(self) -> list[dict]:

        if self.labels is None:
            return []
        unique, counts = np.unique(self.labels, return_counts=True)
        return [{"cluster_id": int(c), "size": int(s)} for c, s in zip(unique, counts)]

    def get_label(self, doc_id: str) -> int | None:
        try:
            idx = self.doc_ids.index(doc_id)
            return int(self.labels[idx])
        except ValueError:
            return None

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        pkl_dump(
            {
                "n_clusters": self.n_clusters,
                "doc_ids": self.doc_ids,
                "labels": self.labels,
                "centroids": self.centroids,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "ClusteringIndex":
        data = pkl_load(path)
        obj = cls(n_clusters=data["n_clusters"])
        obj.doc_ids = data["doc_ids"]
        obj.labels = data["labels"].astype(np.int32)
        obj.centroids = data["centroids"].astype(np.float32)
        return obj

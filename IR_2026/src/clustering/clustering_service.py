from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.clustering.clustering_index import ClusteringIndex

logger = logging.getLogger(__name__)


class ClusteringService:

    def __init__(self):
        self._indexes: dict[str, ClusteringIndex] = {}

    def register(self, dataset_name: str, index: ClusteringIndex) -> None:
        self._indexes[dataset_name] = index

    def available_datasets(self) -> list[str]:
        return list(self._indexes.keys())

    # ------------------------------------------------------------------

    def cluster_search(
        self,
        dataset_name: str,
        query_vec: np.ndarray,
        embedding_retriever,
        top_k: int = 10,
        n_probes: int = 3,
    ) -> list[dict[str, Any]]:

        index = self._get_index(dataset_name)
        emb_index = embedding_retriever.index

        cluster_ids = index.nearest_clusters(query_vec, n_probes=n_probes)
        candidate_ids = index.docs_in_clusters(cluster_ids)

        if not candidate_ids:
            return []

        id_to_pos = {doc_id: i for i, doc_id in enumerate(emb_index.doc_ids)}
        rows = [id_to_pos[d] for d in candidate_ids if d in id_to_pos]

        if not rows:
            return []

        rows_arr = np.array(rows, dtype=np.int64)
        sub_matrix = emb_index.matrix[rows_arr]  # (C, D)
        scores = sub_matrix @ query_vec  # (C,)

        k = min(top_k, len(rows))
        top_local = np.argsort(scores)[::-1][:k]

        results = []
        for local_idx in top_local:
            global_row = rows[local_idx]
            doc_id = emb_index.doc_ids[global_row]
            results.append(
                {
                    "doc_id": doc_id,
                    "score": float(scores[local_idx]),
                    "cluster_id": (
                        int(index.labels[index.doc_ids.index(doc_id)])
                        if doc_id in index.doc_ids
                        else -1
                    ),
                }
            )

        return results

    # ------------------------------------------------------------------

    def cluster_info(self, dataset_name: str) -> dict[str, Any]:
        index = self._get_index(dataset_name)
        sizes = [c["size"] for c in index.cluster_info()]
        return {
            "dataset": dataset_name,
            "n_clusters": index.n_clusters,
            "n_docs": len(index.doc_ids),
            "avg_cluster_size": round(float(np.mean(sizes)), 1) if sizes else 0,
            "min_cluster_size": int(min(sizes)) if sizes else 0,
            "max_cluster_size": int(max(sizes)) if sizes else 0,
            "clusters": index.cluster_info(),
        }

    def get_doc_cluster(self, dataset_name: str, doc_id: str) -> dict[str, Any]:
        index = self._get_index(dataset_name)
        label = index.get_label(doc_id)
        return {
            "dataset": dataset_name,
            "doc_id": doc_id,
            "cluster_id": label,
            "found": label is not None,
        }

    def cluster_neighbors(
        self,
        dataset_name: str,
        cluster_id: int,
        limit: int = 20,
    ) -> dict[str, Any]:
        index = self._get_index(dataset_name)
        doc_ids = index.docs_in_clusters([cluster_id])
        return {
            "dataset": dataset_name,
            "cluster_id": cluster_id,
            "total_docs": len(doc_ids),
            "sample_doc_ids": doc_ids[:limit],
        }

    # ------------------------------------------------------------------

    def _get_index(self, dataset_name: str) -> ClusteringIndex:
        if dataset_name not in self._indexes:
            raise ValueError(
                f"No ClusteringIndex for dataset '{dataset_name}'. "
                f"Available: {list(self._indexes.keys())}"
            )
        return self._indexes[dataset_name]

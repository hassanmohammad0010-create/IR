import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indexing.embedding_index import EmbeddingIndex
from src.clustering.clustering_index import ClusteringIndex, CLUSTER_COUNTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("build_cluster_index")

ROOT = Path(__file__).resolve().parents[1]

DATASET_CONFIG: dict[str, dict[str, str]] = {
    "msmarco": {
        "embedding": "data/indexes/msmarco/embedding_index.pkl",
        "cluster": "data/indexes/msmarco/cluster_index.pkl",
    },
    "quora": {
        "embedding": "data/indexes/quora/embedding_index.pkl",
        "cluster": "data/indexes/quora/cluster_index.pkl",
    },
}


def build_one(dataset_name: str, n_clusters: int | None) -> None:
    cfg = DATASET_CONFIG[dataset_name]
    emb_path = ROOT / cfg["embedding"]
    out_path = ROOT / cfg["cluster"]

    if not emb_path.exists():
        sys.exit(1)

    k = n_clusters or CLUSTER_COUNTS.get(dataset_name, 300)
    logger.info(f"[{dataset_name}] n_clusters={k}")

    logger.info(f"[{dataset_name}] Loading embedding index …")
    emb_idx = EmbeddingIndex.load(str(emb_path))

    cluster_idx = ClusteringIndex(n_clusters=k)
    cluster_idx.build(
        doc_ids=emb_idx.doc_ids,
        matrix=emb_idx.matrix,
        batch_size=4096,
        random_state=42,
        n_init=5,
    )

    cluster_idx.save(str(out_path))
    logger.info(f"[{dataset_name}] Cluster index saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ClusteringIndex")
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIG.keys()) + ["all"],
        default="all",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    targets = list(DATASET_CONFIG.keys()) if args.dataset == "all" else [args.dataset]

    for ds in targets:
        logger.info(f"=== Building cluster index for '{ds}' ===")
        build_one(ds, args.n_clusters)

    logger.info("All cluster indexes built successfully.")


if __name__ == "__main__":
    main()

"""
==============================================================
  Cluster Visualization — Quora & MS MARCO
  Produces a PCA 2-D scatter (like the reference image)
  for each pre-built cluster index.
  Usage:  python cluster_visualize.py
==============================================================
"""

import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from notebooks.helper.helper import banner, section, fmt_time, SEP
from indexing.embedding_index import EmbeddingIndex
from clustering.clustering_index import ClusteringIndex

DATASETS = {
    "quora": {
        "cluster_index": "data/indexes/quora/cluster_index.pkl.gz",
        "embedding_index": "data/indexes/quora/embedding_index.pkl.gz",
    },
    "msmarco": {
        "cluster_index": "data/indexes/msmarco/cluster_index.pkl.gz",
        "embedding_index": "data/indexes/msmarco/embedding_index.pkl.gz",
    },
}

SCATTER_SAMPLE = 200000  # docs to sample for PCA (keeps it fast)
MAX_SHOW_CLUSTERS = 5  # how many clusters to color distinctly


def plot_scatter(
    ds_name, cl_idx: ClusteringIndex, emb_matrix: np.ndarray, out_dir: Path
):
    import time

    t0 = time.time()

    labels = cl_idx.labels  # (N,)
    centroids = cl_idx.centroids  # (K, D)
    n_total = len(cl_idx.doc_ids)

    # random sample
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_total, size=min(SCATTER_SAMPLE, n_total), replace=False)
    sample_vecs = emb_matrix[sample_idx]
    sample_labels = labels[sample_idx]

    # pick top-N largest clusters
    unique, counts = np.unique(labels, return_counts=True)
    top_clusters = unique[np.argsort(counts)[::-1][:MAX_SHOW_CLUSTERS]]
    mask = np.isin(sample_labels, top_clusters)

    vecs_show = sample_vecs[mask]
    labels_show = sample_labels[mask]
    centroid_vecs = centroids[top_clusters]

    # PCA on docs + centroids together so they share the same projection
    all_vecs = np.vstack([vecs_show, centroid_vecs])
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(all_vecs)

    doc_proj = proj[: len(vecs_show)]
    centroid_proj = proj[len(vecs_show) :]

    # distinct colors — works on all matplotlib versions
    cmap = plt.colormaps["tab20"]
    colors = [cmap(i / MAX_SHOW_CLUSTERS) for i in range(MAX_SHOW_CLUSTERS)]
    color_map = {cid: colors[i] for i, cid in enumerate(top_clusters)}

    fig, ax = plt.subplots(figsize=(9, 7))

    # scatter each cluster's docs
    for i, cid in enumerate(top_clusters):
        pts = doc_proj[labels_show == cid]
        if len(pts) == 0:
            continue
        ax.scatter(
            pts[:, 0], pts[:, 1], s=18, alpha=0.55, color=color_map[cid], linewidths=0
        )

    # centroids as black filled circles — exactly like the reference image
    ax.scatter(
        centroid_proj[:, 0],
        centroid_proj[:, 1],
        s=130,
        color="black",
        zorder=5,
        label="Centroid",
    )

    ax.set_title(
        f"Document Clusters — {ds_name.upper()}\n"
        f"PCA 2-D  |  {len(vecs_show):,} sampled docs  |  "
        f"top-{MAX_SHOW_CLUSTERS} of {cl_idx.n_clusters} clusters",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel(f"PC-1  ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC-2  ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    path = out_dir / f"cluster_scatter_{ds_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved → {path.name}  ({fmt_time(time.time()-t0)})")


def main():
    import time

    out_dir = SCRIPT_DIR / "charts"
    out_dir.mkdir(exist_ok=True)

    banner("Cluster Scatter Visualization")
    total = time.time()

    for ds_name, cfg in DATASETS.items():
        section(f"Dataset: {ds_name.upper()}")

        print("  Loading embedding index …")
        t0 = time.time()
        emb_idx = EmbeddingIndex.load(str(ROOT / cfg["embedding_index"]))
        print(
            f"  ✔ Embedding loaded  ({fmt_time(time.time()-t0)})  "
            f"docs={len(emb_idx.doc_ids):,}  dim={emb_idx.matrix.shape[1]}"
        )

        print("  Loading cluster index …")
        t0 = time.time()
        cl_idx = ClusteringIndex.load(str(ROOT / cfg["cluster_index"]))
        print(
            f"  ✔ Cluster index loaded  ({fmt_time(time.time()-t0)})  "
            f"clusters={cl_idx.n_clusters}"
        )

        section(f"[{ds_name.upper()}] Generating scatter chart")
        plot_scatter(ds_name, cl_idx, emb_idx.matrix, out_dir)

    print(f"\n  Charts saved in: {out_dir}/")
    print(f"\n{SEP}\n  Total wall time: {fmt_time(time.time()-total)}\n{SEP}")


if __name__ == "__main__":
    main()

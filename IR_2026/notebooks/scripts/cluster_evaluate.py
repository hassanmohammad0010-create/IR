"""
==============================================================
  Cluster Evaluation — Quora & MS MARCO
  Evaluates cluster-based retrieval against qrels across
  different n_probes values.
  Usage:  python cluster_evaluate.py --query_count 100
==============================================================
"""

import argparse
import time
import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT       = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from notebooks.helper.helper import banner, section, fmt_time, load_queries, load_qrels, SEP
from indexing.embedding_index      import EmbeddingIndex
from retrieval.embedding_retriever import EmbeddingRetriever
from clustering.clustering_index   import ClusteringIndex
from clustering.clustering_service import ClusteringService
from evaluation.evaluation_service import EvaluationService

DATASETS = {
    "quora": {
        "cluster_index":   "data/indexes/quora/cluster_index.pkl.gz",
        "embedding_index": "data/indexes/quora/embedding_index.pkl.gz",
        "queries":         "datasets/dataset2/queries.jsonl",
        "qrels":           "datasets/dataset2/qrels.jsonl",
        "metrics":       {"map": "map", "precision": "P_10", "recall": "recall_100", "ndcg": "ndcg_cut_10"},
        "metric_labels": {"map": "MAP",  "precision": "P@10", "recall": "Recall@100", "ndcg": "nDCG@10"},
    },
    "msmarco": {
        "cluster_index":   "data/indexes/msmarco/cluster_index.pkl.gz",
        "embedding_index": "data/indexes/msmarco/embedding_index.pkl.gz",
        "queries":         "datasets/dataset3/queries.jsonl",
        "qrels":           "datasets/dataset3/qrels.jsonl",
        "metrics":       {"map": "map", "precision": "P_5", "recall": "recall_50", "ndcg": "ndcg_cut_10"},
        "metric_labels": {"map": "MAP",  "precision": "P@5", "recall": "Recall@50", "ndcg": "nDCG@10"},
    },
}

N_PROBES_LIST = [1, 3, 5, 10]

def evaluate(ds_name, cluster_svc, emb_ret, queries, qrels, cfg, n_probes):
    metric_keys    = cfg["metrics"]
    metric_labels  = cfg["metric_labels"]
    pytrec_metrics = set(metric_keys.values())
    eval_svc       = EvaluationService()
    results        = {}

    max_k = max(
        (int(v.split("_")[-1]) for v in pytrec_metrics if v.split("_")[-1].isdigit()),
        default=100,
    )

    t0 = time.time()
    for q in queries:
        query_vec = emb_ret.index.encode_query(q["text"])
        raw = cluster_svc.cluster_search(
            dataset_name=ds_name,
            query_vec=query_vec,
            embedding_retriever=emb_ret,
            top_k=max_k,
            n_probes=n_probes,
        )
        results[q["query_id"]] = {r["doc_id"]: float(r["score"]) for r in raw}

    output  = eval_svc.evaluate(qrels=qrels, results=results, metrics=pytrec_metrics)
    elapsed = time.time() - t0
    agg     = output.get("aggregate", {})

    print(f"\n  📊 [{ds_name.upper()}] n_probes={n_probes}")
    print(f"     Queries evaluated : {len(queries)}")
    print(f"     Evaluation time   : {fmt_time(elapsed)}")
    for key, pk in metric_keys.items():
        print(f"     {metric_labels[key]:<20}: {agg.get(pk, 0):.4f}")
    return agg


def main():
    parser = argparse.ArgumentParser(description="Cluster — evaluate against qrels")
    parser.add_argument("--query_count", type=int, default=100)
    args = parser.parse_args()

    banner(f"Clustering Evaluation  |  query_count={args.query_count}")
    total_start = time.time()
    summary: dict = {}
    cluster_svc   = ClusteringService()

    for ds_name, cfg in DATASETS.items():
        banner(f"Dataset: {ds_name.upper()}")

        print("  Loading embedding index …")
        t0      = time.time()
        emb_idx = EmbeddingIndex.load(str(ROOT / cfg["embedding_index"]))
        emb_ret = EmbeddingRetriever(emb_idx)
        print(f"  ✔ Embedding loaded  ({fmt_time(time.time()-t0)})  "
              f"docs={len(emb_idx.doc_ids):,}")

        print("  Loading cluster index …")
        t0     = time.time()
        cl_idx = ClusteringIndex.load(str(ROOT / cfg["cluster_index"]))
        cluster_svc.register(ds_name, cl_idx)
        print(f"  ✔ Cluster index loaded  ({fmt_time(time.time()-t0)})  "
              f"clusters={cl_idx.n_clusters}")

        queries = load_queries(ROOT, cfg["queries"], args.query_count)
        qrels   = load_qrels(ROOT, cfg["qrels"])

        section(f"[{ds_name.upper()}] Evaluation across n_probes")
        probe_results: dict = {}
        for n_probes in N_PROBES_LIST:
            agg = evaluate(ds_name, cluster_svc, emb_ret, queries, qrels, cfg, n_probes)
            probe_results[n_probes] = agg

        summary[ds_name] = {"cfg": cfg, "probe_results": probe_results}

    # final summary table
    banner("FINAL SUMMARY")
    for ds_name, data in summary.items():
        cfg           = data["cfg"]
        metric_keys   = cfg["metrics"]
        metric_labels = cfg["metric_labels"]
        col_headers   = list(metric_labels.values())
        col_w         = max(len(h) for h in col_headers) + 2

        print(f"\n  Dataset: {ds_name.upper()}")
        header = f"  {'n_probes':<12}" + "".join(f"{h:>{col_w}}" for h in col_headers)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for n_probes, agg in sorted(data["probe_results"].items()):
            row = f"  {n_probes:<12}"
            for key, pk in metric_keys.items():
                row += f"{agg.get(pk, 0):>{col_w}.4f}"
            print(row)

    print(f"\n{SEP}\n  Total wall time: {fmt_time(time.time()-total_start)}\n{SEP}")


if __name__ == "__main__":
    main()

import argparse
import time
import json
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from indexing.tfidf_index import TFIDFIndex
from retrieval.tfidf_retriever import TFIDFRetriever
from evaluation.evaluation_service import EvaluationService
from notebooks.helper.helper import (
    banner,
    section,
    fmt_time,
    load_queries,
    load_qrels,
    SEP,
)

DATASETS = {
    "quora": {
        "index": "data/indexes/quora/tfidf_index.pkl.gz",
        "queries": "datasets/dataset2/queries.jsonl",
        "qrels": "datasets/dataset2/qrels.jsonl",
        "metrics": {
            "map": "map",
            "precision": "P_10",
            "recall": "recall_200",
            "ndcg": "ndcg_cut_10",
        },
        "metric_labels": {
            "map": "MAP",
            "precision": "P@10",
            "recall": "Recall@200",
            "ndcg": "nDCG@10",
        },
    },
    "msmarco": {
        "index": "data/indexes/msmarco/tfidf_index.pkl.gz",
        "queries": "datasets/dataset3/queries.jsonl",
        "qrels": "datasets/dataset3/qrels.jsonl",
        "metrics": {
            "map": "map",
            "precision": "P_1",
            "recall": "recall_50",
            "ndcg": "ndcg_cut_1",
        },
        "metric_labels": {
            "map": "MAP",
            "precision": "P@1",
            "recall": "Recall@50",
            "ndcg": "nDCG@1",
        },
    },
}


def evaluate(ds_name, retriever, queries, qrels, cfg):
    metric_keys = cfg["metrics"]
    metric_labels = cfg["metric_labels"]
    pytrec_metrics = set(metric_keys.values())
    eval_svc = EvaluationService()
    results = {}

    max_k = max(
        (int(v.split("_")[-1]) for v in pytrec_metrics if v.split("_")[-1].isdigit()),
        default=100,
    )

    t0 = time.time()
    for q in queries:
        raw = retriever.search(q["text"], top_k=max_k)
        results[q["query_id"]] = {doc_id: float(score) for doc_id, score in raw}

    output = eval_svc.evaluate(qrels=qrels, results=results, metrics=pytrec_metrics)
    elapsed = time.time() - t0
    agg = output.get("aggregate", {})

    print(f"\n  📊 [{ds_name.upper()}] TF-IDF")
    print(f"     Queries evaluated : {len(queries)}")
    print(f"     Evaluation time   : {fmt_time(elapsed)}")
    for key, pytrec_key in metric_keys.items():
        print(f"     {metric_labels[key]:<20}: {agg.get(pytrec_key, 0):.4f}")
    return agg


def main():
    parser = argparse.ArgumentParser(description="TF-IDF — evaluate")
    parser.add_argument("--query_count", type=int, default=100)
    args = parser.parse_args()

    banner(f"TF-IDF Evaluation  |  query_count={args.query_count}")
    total_start = time.time()
    summary: dict = {}

    for ds_name, cfg in DATASETS.items():
        banner(f"Dataset: {ds_name.upper()}")

        index_path = ROOT / cfg["index"]
        t0 = time.time()
        idx = TFIDFIndex.load(str(index_path))
        print(f"  ✔ Documents     : {len(idx.doc_ids):,}")
        print(f"  ✔ Vocabulary    : {idx.vocabulary_size():,}")
        print(f"  ✔ Matrix shape  : {idx.matrix.shape}")

        retriever = TFIDFRetriever(idx)
        queries = load_queries(cfg["queries"], args.query_count)
        qrels = load_qrels(cfg["qrels"])

        section(f"[{ds_name.upper()}] Evaluation")
        agg = evaluate(ds_name, retriever, queries, qrels, cfg)
        summary[ds_name] = {"cfg": cfg, "agg": agg}

    banner("FINAL SUMMARY")
    for ds_name, data in summary.items():
        cfg = data["cfg"]
        metric_labels = cfg["metric_labels"]
        metric_keys = cfg["metrics"]
        col_headers = list(metric_labels.values())
        col_w = max(len(h) for h in col_headers) + 2

        print(f"\n  Dataset: {ds_name.upper()}")
        header = f"  {'Model':<20}" + "".join(f"{h:>{col_w}}" for h in col_headers)
        print(header)
        print("  " + "-" * (len(header) - 2))
        row = f"  {'TF-IDF':<20}"
        for key, pytrec_key in metric_keys.items():
            row += f"{data['agg'].get(pytrec_key, 0):>{col_w}.4f}"
        print(row)

    print(f"\n{SEP}\n  Total wall time: {fmt_time(time.time()-total_start)}\n{SEP}")


if __name__ == "__main__":
    main()

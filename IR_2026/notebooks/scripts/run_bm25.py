import argparse
import time
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from indexing.bm25_index import BM25Index
from retrieval.bm25_retriever import BM25Retriever
from evaluation.evaluation_service import EvaluationService
from notebooks.helper.helper import (
    banner,
    section,
    fmt_time,
    load_queries,
    load_qrels,
    SEP,
)

# ── dataset config ────────────────────────────────────────────────────────────
DATASETS = {
    "quora": {
        "index": "data/indexes/quora/bm25_index.pkl.gz",
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
        "index": "data/indexes/msmarco/bm25_index.pkl.gz",
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

BM25_PRESETS = [
    {"k1": 1.2, "b": 0.75, "label": "Default (k1=1.2, b=0.75)"},
    {"k1": 1.5, "b": 0.75, "label": "High k1  (k1=1.5, b=0.75)"},
    {"k1": 1.2, "b": 0.50, "label": "Low b    (k1=1.2, b=0.50)"},
]


def evaluate(ds_name, retriever, queries, qrels, cfg, k1, b, label):
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
        raw = retriever.search(q["text"], top_k=max_k, k1=k1, b=b)
        results[q["query_id"]] = {doc_id: float(score) for doc_id, score in raw}

    output = eval_svc.evaluate(qrels=qrels, results=results, metrics=pytrec_metrics)
    elapsed = time.time() - t0
    agg = output.get("aggregate", {})

    print(f"\n  📊 [{ds_name.upper()}] {label}")
    print(f"     Queries evaluated : {len(queries)}")
    print(f"     Evaluation time   : {fmt_time(elapsed)}")
    for key, pytrec_key in metric_keys.items():
        print(f"     {metric_labels[key]:<20}: {agg.get(pytrec_key, 0):.4f}")
    return agg


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BM25 — evaluate")
    parser.add_argument("--query_count", type=int, default=100)
    args = parser.parse_args()

    banner(f"BM25 Evaluation  |  query_count={args.query_count}")
    total_start = time.time()
    summary: dict = {}

    for ds_name, cfg in DATASETS.items():
        banner(f"Dataset: {ds_name.upper()}")

        index_path = ROOT / cfg["index"]
        t0 = time.time()
        idx = BM25Index.load(str(index_path))
        print(f"  ✔ Documents     : {len(idx.doc_ids):,}")
        print(f"  ✔ Vocabulary    : {idx.vocabulary_size():,}")

        retriever = BM25Retriever(idx)
        queries = load_queries(cfg["queries"], args.query_count)
        qrels = load_qrels(cfg["qrels"])
        summary[ds_name] = {"cfg": cfg, "presets": {}}

        section(f"[{ds_name.upper()}] Evaluation across BM25 parameter presets")
        for preset in BM25_PRESETS:
            agg = evaluate(
                ds_name,
                retriever,
                queries,
                qrels,
                cfg,
                k1=preset["k1"],
                b=preset["b"],
                label=preset["label"],
            )
            summary[ds_name]["presets"][preset["label"]] = agg

    banner("FINAL SUMMARY")
    for ds_name, data in summary.items():
        cfg = data["cfg"]
        metric_labels = cfg["metric_labels"]
        metric_keys = cfg["metrics"]
        col_headers = list(metric_labels.values())
        col_w = max(len(h) for h in col_headers) + 2

        print(f"\n  Dataset: {ds_name.upper()}")
        header = f"  {'Preset':<30}" + "".join(f"{h:>{col_w}}" for h in col_headers)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for label, agg in data["presets"].items():
            row = f"  {label:<30}"
            for key, pytrec_key in metric_keys.items():
                row += f"{agg.get(pytrec_key, 0):>{col_w}.4f}"
            print(row)

    print(f"\n{SEP}\n  Total wall time: {fmt_time(time.time()-total_start)}\n{SEP}")


if __name__ == "__main__":
    main()

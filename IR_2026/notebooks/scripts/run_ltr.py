import argparse
import time
import json
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from notebooks.helper.helper import (
    banner,
    section,
    fmt_time,
    load_queries,
    load_qrels,
    SEP,
)
from indexing.bm25_index import BM25Index
from indexing.tfidf_index import TFIDFIndex
from indexing.embedding_index import EmbeddingIndex
from retrieval.bm25_retriever import BM25Retriever
from retrieval.tfidf_retriever import TFIDFRetriever
from retrieval.embedding_retriever import EmbeddingRetriever
from ltr.ltr_service import LTRService, LTRModel
from evaluation.evaluation_service import EvaluationService

DATASETS = {
    "quora": {
        "bm25_index": "data/indexes/quora/bm25_index.pkl.gz",
        "tfidf_index": "data/indexes/quora/tfidf_index.pkl.gz",
        "embedding_index": "data/indexes/quora/embedding_index.pkl.gz",
        "ltr_model": "data/indexes/quora/ltr_model.pkl.gz",
        "queries": "datasets/dataset2/queries.jsonl",
        "qrels": "datasets/dataset2/qrels.jsonl",
        "metrics": {
            "map": "map",
            "precision": "P_10",
            "recall": "recall_100",
            "ndcg": "ndcg_cut_10",
        },
        "metric_labels": {
            "map": "MAP",
            "precision": "P@10",
            "recall": "Recall@100",
            "ndcg": "nDCG@10",
        },
    },
    "msmarco": {
        "bm25_index": "data/indexes/msmarco/bm25_index.pkl.gz",
        "tfidf_index": "data/indexes/msmarco/tfidf_index.pkl.gz",
        "embedding_index": "data/indexes/msmarco/embedding_index.pkl.gz",
        "ltr_model": "data/indexes/msmarco/ltr_model.pkl.gz",
        "queries": "datasets/dataset3/queries.jsonl",
        "qrels": "datasets/dataset3/qrels.jsonl",
        "metrics": {
            "map": "map",
            "precision": "P_5",
            "recall": "recall_50",
            "ndcg": "ndcg_cut_10",
        },
        "metric_labels": {
            "map": "MAP",
            "precision": "P@5",
            "recall": "Recall@50",
            "ndcg": "nDCG@10",
        },
    },
}


def _run_eval(retriever_fn, queries, qrels, pytrec_metrics, max_k):
    """Generic eval loop — retriever_fn(query_text) -> list[(doc_id, score)]"""
    eval_svc = EvaluationService()
    results = {}
    t0 = time.time()
    for q in queries:
        raw = retriever_fn(q["text"], max_k)
        results[q["query_id"]] = {doc_id: float(score) for doc_id, score in raw}
    output = eval_svc.evaluate(qrels=qrels, results=results, metrics=pytrec_metrics)
    return output.get("aggregate", {}), time.time() - t0


def evaluate_bm25(ds_name, dataset_obj, queries, qrels, cfg):
    metric_keys = cfg["metrics"]
    pytrec_metrics = set(metric_keys.values())
    max_k = max(
        (int(v.split("_")[-1]) for v in pytrec_metrics if v.split("_")[-1].isdigit()),
        default=100,
    )
    retriever = dataset_obj["bm25"]
    agg, elapsed = _run_eval(
        lambda text, k: retriever.search(text, top_k=k),
        queries,
        qrels,
        pytrec_metrics,
        max_k,
    )
    print(f"\n  📊 [{ds_name.upper()}] BM25 Baseline")
    print(f"     Evaluation time : {fmt_time(elapsed)}")
    for key, pk in metric_keys.items():
        print(f"     {cfg['metric_labels'][key]:<20}: {agg.get(pk, 0):.4f}")
    return agg


def evaluate_ltr(ds_name, cfg, dataset_obj, ltr_svc, queries, qrels, pool_size):
    metric_keys = cfg["metrics"]
    pytrec_metrics = set(metric_keys.values())
    max_k = max(
        (int(v.split("_")[-1]) for v in pytrec_metrics if v.split("_")[-1].isdigit()),
        default=100,
    )

    def rerank_fn(text, k):
        raw = ltr_svc.rerank(
            dataset_name=ds_name,
            dataset_obj=dataset_obj,
            query=text,
            top_k=k,
            pool_size=pool_size,
        )
        return raw  # already list[(doc_id, score)]

    agg, elapsed = _run_eval(rerank_fn, queries, qrels, pytrec_metrics, max_k)
    print(f"\n  📊 [{ds_name.upper()}] LTR Reranking")
    print(f"     Evaluation time : {fmt_time(elapsed)}")
    for key, pk in metric_keys.items():
        print(f"     {cfg['metric_labels'][key]:<20}: {agg.get(pk, 0):.4f}")
    return agg


# ── charts ────────────────────────────────────────────────────────────────────
def plot_feature_importance(ds_name, model: LTRModel, out_dir: Path):
    coefs = model.trained_on.get("coefficients", {})
    if not coefs:
        print(f"  ⚠  No coefficients found for {ds_name}, skipping chart.")
        return
    feats = list(coefs.keys())
    values = list(coefs.values())
    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(feats, values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(
        f"LTR Feature Coefficients — {ds_name.upper()}", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Coefficient value  (blue = positive, red = negative)")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, values):
        offset = 0.003 if val >= 0 else -0.003
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
        )
    plt.tight_layout()
    path = out_dir / f"ltr_feature_importance_{ds_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Chart saved → {path.name}")


def plot_ltr_vs_baseline(summary: dict, out_dir: Path):
    datasets = list(summary.keys())
    first_cfg = list(summary.values())[0]["cfg"]
    metric_keys = first_cfg["metrics"]
    metric_labels = first_cfg["metric_labels"]
    pytrec_keys = list(metric_keys.values())
    disp_names = list(metric_labels.values())

    x = np.arange(len(pytrec_keys))
    width = 0.28
    colors = {"BM25 Baseline": "#4C72B0", "LTR Reranking": "#DD8452"}

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(8 * len(datasets), 5), sharey=False
    )
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, datasets):
        data = summary[ds_name]
        bvals = [data["baseline"].get(pk, 0) for pk in pytrec_keys]
        lvals = [data["ltr_agg"].get(pk, 0) for pk in pytrec_keys]

        ax.bar(
            x - width / 2,
            bvals,
            width,
            label="BM25 Baseline",
            color=colors["BM25 Baseline"],
        )
        ax.bar(
            x + width / 2,
            lvals,
            width,
            label="LTR Reranking",
            color=colors["LTR Reranking"],
        )

        # delta annotations above LTR bars
        for xi, (b, l) in enumerate(zip(bvals, lvals)):
            delta = l - b
            color = "#228B22" if delta >= 0 else "#CC0000"
            ax.text(
                xi + width / 2,
                l + 0.008,
                f"{delta:+.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

        ax.set_title(
            f"{ds_name.upper()} — LTR vs BM25 Baseline", fontsize=12, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(disp_names)
        ax.set_ylabel("Score")
        ax.set_ylim(0, min(max(bvals + lvals) * 1.25, 1.0))
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "ltr_vs_baseline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Chart saved → {path.name}")


def plot_training_info(ds_name, model: LTRModel, out_dir: Path):
    n_pos = model.trained_on.get("n_positive", 0)
    n_neg = model.trained_on.get("n_negative", 0)
    coefs = model.trained_on.get("coefficients", {})

    if not coefs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"LTR Model Info — {ds_name.upper()}", fontsize=13, fontweight="bold")

    # pie — label balance
    axes[0].pie(
        [n_pos, n_neg],
        labels=[f"Relevant\n({n_pos:,})", f"Non-relevant\n({n_neg:,})"],
        colors=["#55A868", "#C44E52"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[0].set_title("Training Label Distribution")

    # bar — absolute coefficients
    feats = list(coefs.keys())
    values = [abs(v) for v in coefs.values()]
    axes[1].bar(feats, values, color="#4C72B0", edgecolor="white")
    axes[1].set_title("Feature Importance  |coefficient|")
    axes[1].set_ylabel("|Coefficient|")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"ltr_model_info_{ds_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Chart saved → {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LTR — evaluate pre-trained model")
    parser.add_argument("--query_count", type=int, default=100)
    parser.add_argument("--pool_size", type=int, default=200)
    args = parser.parse_args()

    out_dir = SCRIPT_DIR / "charts"
    out_dir.mkdir(exist_ok=True)

    banner(
        f"LTR Evaluation  |  query_count={args.query_count}  pool_size={args.pool_size}"
    )
    total_start = time.time()
    ltr_svc = LTRService()
    summary: dict = {}

    for ds_name, cfg in DATASETS.items():
        banner(f"Dataset: {ds_name.upper()}")

        # ── load all indexes ──
        section(f"[{ds_name.upper()}] Loading indexes")
        t0 = time.time()
        bm25_idx = BM25Index.load(str(ROOT / cfg["bm25_index"]))
        print(
            f"  ✔ BM25      loaded  ({fmt_time(time.time()-t0)})  docs={len(bm25_idx.doc_ids):,}"
        )

        t0 = time.time()
        tfidf_idx = TFIDFIndex.load(str(ROOT / cfg["tfidf_index"]))
        print(f"  ✔ TF-IDF    loaded  ({fmt_time(time.time()-t0)})")

        t0 = time.time()
        emb_idx = EmbeddingIndex.load(str(ROOT / cfg["embedding_index"]))
        print(
            f"  ✔ Embedding loaded  ({fmt_time(time.time()-t0)})  shape={emb_idx.matrix.shape}"
        )

        dataset_obj = {
            "bm25": BM25Retriever(bm25_idx),
            "tfidf": TFIDFRetriever(tfidf_idx),
            "embedding": EmbeddingRetriever(emb_idx),
        }

        # ── load pre-trained LTR model ──
        section(f"[{ds_name.upper()}] Loading pre-trained LTR model")
        t0 = time.time()
        model = LTRModel.load(str(ROOT / cfg["ltr_model"]))
        ltr_svc.register(ds_name, model)
        print(f"  ✔ LTR model loaded  ({fmt_time(time.time()-t0)})")
        print(f"  ✔ Training samples  : {model.trained_on.get('n_samples', '?'):,}")
        print(f"  ✔ Positive labels   : {model.trained_on.get('n_positive', '?'):,}")
        print(f"  ✔ Negative labels   : {model.trained_on.get('n_negative', '?'):,}")
        print(f"\n  Feature coefficients:")
        for feat, coef in model.trained_on.get("coefficients", {}).items():
            bar = "█" * max(1, int(abs(coef) * 25))
            sign = "+" if coef >= 0 else "-"
            print(f"    {feat:<28} {sign}{abs(coef):.4f}  {bar}")

        # ── charts from pre-trained model ──
        section(f"[{ds_name.upper()}] Model Charts")
        plot_feature_importance(ds_name, model, out_dir)
        plot_training_info(ds_name, model, out_dir)

        # ── load eval data ──
        queries = load_queries(cfg["queries"], args.query_count)
        qrels = load_qrels(cfg["qrels"])

        # ── baseline ──
        section(f"[{ds_name.upper()}] BM25 Baseline Evaluation")
        baseline_agg = evaluate_bm25(ds_name, dataset_obj, queries, qrels, cfg)

        # ── ltr reranking ──
        section(f"[{ds_name.upper()}] LTR Reranking Evaluation")
        ltr_agg = evaluate_ltr(
            ds_name, cfg, dataset_obj, ltr_svc, queries, qrels, args.pool_size
        )

        summary[ds_name] = {
            "cfg": cfg,
            "baseline": baseline_agg,
            "ltr_agg": ltr_agg,
        }

    # cross-dataset chart
    section("LTR vs Baseline Chart")
    plot_ltr_vs_baseline(summary, out_dir)

    # final summary table
    banner("FINAL SUMMARY")
    for ds_name, data in summary.items():
        cfg = data["cfg"]
        metric_keys = cfg["metrics"]
        metric_labels = cfg["metric_labels"]
        col_headers = list(metric_labels.values())
        col_w = max(len(h) for h in col_headers) + 2

        print(f"\n  Dataset: {ds_name.upper()}")
        header = f"  {'Model':<22}" + "".join(f"{h:>{col_w}}" for h in col_headers)
        print(header)
        print("  " + "-" * (len(header) - 2))

        for label, agg in [
            ("BM25 Baseline", data["baseline"]),
            ("LTR Reranking", data["ltr_agg"]),
        ]:
            row = f"  {label:<22}"
            for key, pk in metric_keys.items():
                row += f"{agg.get(pk, 0):>{col_w}.4f}"
            print(row)

        # Δ row
        row = f"  {'Δ (LTR − BM25)':<22}"
        for key, pk in metric_keys.items():
            delta = data["ltr_agg"].get(pk, 0) - data["baseline"].get(pk, 0)
            sign = "+" if delta >= 0 else ""
            row += f"{sign}{delta:{col_w-1}.4f}"
        print(row)

    print(f"\n  Charts saved in: {out_dir}/")
    print(f"\n{SEP}\n  Total wall time: {fmt_time(time.time()-total_start)}\n{SEP}")


if __name__ == "__main__":
    main()

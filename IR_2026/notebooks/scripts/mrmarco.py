print("🚀 Script started", flush=True)
import sys
import os
import time
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

print(f"📁 ROOT: {ROOT}", flush=True)

start_global = time.time()

from src.config.datasets import load_datasets
from src.config.evaluation_datasets import load_evaluation_datasets

from src.preprocessing.preprocessing_service import PreprocessingService
from src.query_refinement.refinement_service import RefinementService
from src.retrieval.retrieval_service import RetrievalService

from src.evaluation.evaluation_service import EvaluationService
from src.evaluation.evaluation_runner import EvaluationRunner

print("✅ Imports completed", flush=True)

print("⚙️ Initializing system...", flush=True)

preprocessor = PreprocessingService()
refinement_service = RefinementService()

datasets = load_datasets()
evaluation_datasets = load_evaluation_datasets()

retrieval_service = RetrievalService(
    datasets=datasets,
    preprocessing_service=preprocessor,
    refinement_service=refinement_service,
)

evaluation_service = EvaluationService()

evaluation_runner = EvaluationRunner(
    retrieval_service=retrieval_service,
    evaluation_service=evaluation_service,
)

print("✅ System ready", flush=True)

MODELS = ["bm25", "tfidf", "inverted", "embedding"]
DATASETS = ["quora"]  # add msmarco if needed
QUERY_LIMIT = 2

MODEL_CONFIG = {
    "bm25": {"k1": 1.2, "b": 0.75},
    "tfidf": {},
    "inverted": {},
    "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
}


def run_eval():
    all_results = {}

    for dataset_name in DATASETS:

        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}")
        print("=" * 100)

        dataset = evaluation_datasets[dataset_name]
        queries = dataset["queries"][:QUERY_LIMIT]
        qrels = dataset["qrels"]

        all_results[dataset_name] = {}

        model_ranking = []

        for model in MODELS:

            print("\n" + "-" * 80)
            print(f"MODEL: {model}")

            params = MODEL_CONFIG.get(model, {})

            print(f"PARAMS: {params}")

            start = time.time()

            output = evaluation_runner.run(
                dataset=dataset_name,
                model=model,
                queries=queries,
                qrels=qrels,
                top_k=100,
                metrics={"map", "recall_10", "P_10", "ndcg_cut_10"},
                enable_refinement=False,
            )

            exec_time = time.time() - start
            agg = output["aggregate"]

            all_results[dataset_name][model] = {
                "metrics": agg,
                "time_sec": exec_time,
                "params": params,
            }

            print(f"TIME       : {exec_time:.2f}s")
            print(f"MAP        : {agg['map']:.4f}")
            print(f"Recall@10  : {agg['recall_10']:.4f}")
            print(f"P@10       : {agg['P_10']:.4f}")
            print(f"NDCG@10    : {agg['ndcg_cut_10']:.4f}")

            model_ranking.append((model, agg["ndcg_cut_10"], exec_time))

        model_ranking.sort(key=lambda x: x[1], reverse=True)

        print("\n" + "=" * 100)
        print("MODEL COMPARISON (RANKED BY NDCG@10)")
        print("=" * 100)

        for i, (model, ndcg, t) in enumerate(model_ranking, 1):
            print(f"{i}. {model:10} | NDCG@10={ndcg:.4f} | TIME={t:.2f}s")

        best = model_ranking[0][0]
        print(f"\n🥇 BEST MODEL: {best}")

    print("\n🏁 TOTAL TIME:", round(time.time() - start_global, 2), "seconds")

    with open("evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("💾 Saved: evaluation_results.json")


if __name__ == "__main__":
    run_eval()

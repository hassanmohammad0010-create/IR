from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_ltr_model")

from src.config.datasets import load_datasets
from src.config.evaluation_datasets import load_evaluation_datasets
from src.ltr.ltr_service import LTRService

LTR_MODEL_PATHS: dict[str, str] = {
    "msmarco": "data/indexes/msmarco/ltr_model.pkl",
    "quora": "data/indexes/quora/ltr_model.pkl",
}


def train_one(
    dataset_name: str,
    datasets: dict,
    evaluation_datasets: dict,
    max_queries: int,
    pool_size: int,
) -> None:
    if dataset_name not in datasets:
        logger.error(f"Dataset '{dataset_name}' not loaded — skipping.")
        return
    if dataset_name not in evaluation_datasets:
        logger.error(
            f"No queries/qrels loaded for '{dataset_name}' — skipping. "
            "LTR training needs evaluation data for this dataset."
        )
        return

    rel_path = LTR_MODEL_PATHS.get(dataset_name)
    if rel_path is None:
        logger.warning(
            f"No save path configured for '{dataset_name}' in "
            f"LTR_MODEL_PATHS — add one before training."
        )
        return

    eval_data = evaluation_datasets[dataset_name]
    service = LTRService()

    logger.info(f"Training LTR model for '{dataset_name}' …")
    stats = service.train(
        dataset_name=dataset_name,
        dataset_obj=datasets[dataset_name],
        queries=eval_data["queries"][:max_queries],
        qrels=eval_data["qrels"],
        pool_size=pool_size,
    )
    logger.info(
        f"  ✓ '{dataset_name}' trained — "
        f"{stats['n_samples']} samples "
        f"({stats['n_positive']} pos / {stats['n_negative']} neg)"
    )

    out_path = ROOT / rel_path
    model = service._models[dataset_name]
    model.save(str(out_path))
    logger.info(f"  ✓ saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the offline LTR model.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--pool-size", type=int, default=150)
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Pass --dataset NAME or --all")

    logger.info("Loading datasets …")
    datasets = load_datasets()
    logger.info("Loading evaluation datasets …")
    evaluation_datasets = load_evaluation_datasets()

    targets = list(LTR_MODEL_PATHS.keys()) if args.all else [args.dataset]

    for name in targets:
        train_one(
            dataset_name=name,
            datasets=datasets,
            evaluation_datasets=evaluation_datasets,
            max_queries=args.max_queries,
            pool_size=args.pool_size,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()

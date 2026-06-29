import json
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

EVALUATION_CONFIG: dict[str, dict[str, str]] = {
    # "lotte": {
    #     "queries": "datasets/dataset1/queries.jsonl",
    #     "qrels": "datasets/dataset1/qrels.jsonl",
    # },
    "quora": {
        "queries": "datasets/dataset2/queries.jsonl",
        "qrels": "datasets/dataset2/qrels.jsonl",
    },
    "msmarco": {
        "queries": "datasets/dataset3/queries.jsonl",
        "qrels": "datasets/dataset3/qrels.jsonl",
    },
    # "trec": {
    #     "queries": "datasets/dataset4/queries.jsonl",
    #     "qrels": "datasets/dataset4/qrels.jsonl",
    # },
    # "beir": {
    #     "queries": "datasets/dataset5/queries.jsonl",
    #     "qrels": "datasets/dataset5/qrels.jsonl",
    # },
}


def load_evaluation_datasets() -> dict:

    result = {}

    for name, paths in EVALUATION_CONFIG.items():
        logger.info(f"Loading evaluation data: '{name}' ...")
        try:
            queries = _load_queries(paths["queries"])
            qrels = _load_qrels(paths["qrels"])
            result[name] = {"queries": queries, "qrels": qrels}
            logger.info(
                f"  ✓ '{name}'  queries={len(queries):,}  "
                f"qrel_topics={len(qrels):,}"
            )
        except FileNotFoundError as e:
            logger.error(str(e))
            raise

    return result


def _resolve(rel: str) -> Path:
    p = ROOT / rel
    if not p.exists():
        raise FileNotFoundError(f"Evaluation file not found: {p}")
    return p


def _load_queries(rel_path: str) -> list[dict]:

    queries = []
    with open(_resolve(rel_path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Normalise field names across dataset variants
            query_id = str(obj.get("query_id") or obj.get("id") or obj.get("qid", ""))
            text = str(obj.get("text") or obj.get("query", ""))

            if query_id and text:
                queries.append({"query_id": query_id, "text": text})

    return queries


def _load_qrels(rel_path: str) -> dict[str, dict[str, int]]:

    qrels: dict[str, dict[str, int]] = defaultdict(dict)

    with open(_resolve(rel_path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            query_id = str(obj.get("query_id") or obj.get("qid", ""))
            doc_id = str(obj.get("doc_id") or obj.get("docid", ""))
            relevance = int(obj.get("relevance") or obj.get("rel", 0))

            if query_id and doc_id:
                qrels[query_id][doc_id] = relevance

    return dict(qrels)

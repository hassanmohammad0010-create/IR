import json
import logging

import pytrec_eval

logger = logging.getLogger(__name__)


class EvaluationService:
    # METRICS = {
    #     "map",
    #     "recall_100",
    #     "P_10",
    #     "P_2",
    #     "ndcg_cut_10",
    # }

    def evaluate(self, qrels: dict, results: dict, metrics: set[str]) -> dict:

        qrels_int = {
            qid: {did: int(rel) for did, rel in docs.items()}
            for qid, docs in qrels.items()
        }

        shared = set(qrels_int) & set(results)

        if not shared:
            logger.warning(
                "No shared query IDs between qrels and results. "
                "Check that query_id formats match between queries.jsonl and qrels.jsonl."
            )
            return {"per_query": {}, "aggregate": {m: 0.0 for m in self.metrics}}

        filtered_qrels = {qid: qrels_int[qid] for qid in shared}
        filtered_results = {qid: results[qid] for qid in shared}

        evaluator = pytrec_eval.RelevanceEvaluator(filtered_qrels, metrics)
        per_query_scores = evaluator.evaluate(filtered_results)

        aggregate = {}
        for metric in metrics:
            values = [
                scores[metric]
                for scores in per_query_scores.values()
                if metric in scores
            ]
            aggregate[metric] = round(sum(values) / len(values), 4) if values else 0.0

        return {"per_query": per_query_scores, "aggregate": aggregate}

    @staticmethod
    def results_to_pytrec_format(results: list) -> dict:

        return {item["doc_id"]: float(item["score"]) for item in results}

    @staticmethod
    def load_qrels_jsonl(qrels_path: str) -> dict:

        qrels: dict[str, dict[str, int]] = {}

        with open(qrels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                qid = str(obj.get("query_id") or obj.get("qid", ""))
                did = str(obj.get("doc_id") or obj.get("docid", ""))
                rel = int(obj.get("relevance") or obj.get("rel", 0))
                if qid and did:
                    qrels.setdefault(qid, {})[did] = rel

        return qrels

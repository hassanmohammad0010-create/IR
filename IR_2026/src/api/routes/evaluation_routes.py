import asyncio
import logging

from fastapi import APIRouter

from src.api.schemas import EvaluationRequest
from src.api.dependencies.container import evaluation_runner, evaluation_datasets

router = APIRouter(tags=["Evaluation"])
logger = logging.getLogger(__name__)


@router.post("/evaluate")
async def evaluate(request: EvaluationRequest):

    dataset_data = evaluation_datasets.get(request.dataset)
    if dataset_data is None:
        return {"error": f"Unknown dataset: {request.dataset!r}"}

    queries = dataset_data["queries"][: request.query_count]
    qrels = dataset_data["qrels"]
    p_k = max(1, min(request.p_k, 100))
    ndcg_k = max(1, min(request.ndcg_k, 100))
    recall_k = max(1, min(request.recall_k, 1000))

    effective_top_k = max(request.top_k, recall_k)

    metrics = {"map", f"P_{p_k}", f"recall_{recall_k}", f"ndcg_cut_{ndcg_k}"}

    loop = asyncio.get_running_loop()
    output = await loop.run_in_executor(
        None,
        evaluation_runner.run,
        request.dataset,
        request.model,
        queries,
        qrels,
        effective_top_k,
        request.bm25_k1,
        request.bm25_b,
        metrics,
        request.enable_refinement,
        # hybrid params
        request.hybrid_mode,
        request.hybrid_model_a,
        request.hybrid_model_b,
        request.hybrid_fusion,
        request.hybrid_alpha,
        request.hybrid_pool,
    )

    return output

import asyncio
import logging

from fastapi import APIRouter

from src.api.schemas import LTRRerankRequest, LTRTrainRequest
from src.api.dependencies.container import (
    ltr_service,
    datasets,
    evaluation_datasets,
)

router = APIRouter(tags=["LTR"])
logger = logging.getLogger(__name__)


@router.post("/ltr/rerank")
async def ltr_rerank(request: LTRRerankRequest):

    if request.dataset not in datasets:
        return {"error": f"Unknown dataset '{request.dataset}'"}

    dataset_obj = datasets[request.dataset]

    if not ltr_service.is_ready(request.dataset):
        return {
            "error": (
                f"No trained LTR model for dataset '{request.dataset}'. "
                f"Call POST /ltr/train first (or run scripts/train_ltr_model.py)."
            )
        }

    loop = asyncio.get_running_loop()

    def _run():
        raw = ltr_service.rerank(
            dataset_name=request.dataset,
            dataset_obj=dataset_obj,
            query=request.query,
            top_k=request.top_k,
            pool_size=request.pool_size,
        )
        doc_repo = dataset_obj["doc_repo"]
        enriched = []
        for rank, (doc_id, score) in enumerate(raw, start=1):
            doc = doc_repo.get_document(doc_id)
            enriched.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": round(score, 6),
                    "text": doc["text"] if doc else "",
                }
            )
        return enriched

    results = await loop.run_in_executor(None, _run)
    return {"query": request.query, "dataset": request.dataset, "results": results}


@router.post("/ltr/train")
async def ltr_train(request: LTRTrainRequest):

    if request.dataset not in datasets:
        return {"error": f"Unknown dataset '{request.dataset}'"}
    if request.dataset not in evaluation_datasets:
        return {
            "error": (
                f"No queries/qrels loaded for dataset '{request.dataset}'. "
                f"LTR training requires evaluation data for this dataset."
            )
        }

    dataset_obj = datasets[request.dataset]
    eval_data = evaluation_datasets[request.dataset]

    loop = asyncio.get_running_loop()

    def _run():
        return ltr_service.train(
            dataset_name=request.dataset,
            dataset_obj=dataset_obj,
            queries=eval_data["queries"][: request.max_queries],
            qrels=eval_data["qrels"],
            pool_size=request.pool_size,
        )

    try:
        stats = await loop.run_in_executor(None, _run)
    except ValueError as e:
        return {"error": str(e)}

    return {"dataset": request.dataset, "status": "trained", **stats}


@router.get("/ltr/status/{dataset}")
async def ltr_status(dataset: str):
    return {
        "dataset": dataset,
        "ready": ltr_service.is_ready(dataset),
        "available_datasets": ltr_service.available_datasets(),
    }


@router.get("/ltr/explain/{dataset}")
async def ltr_explain(dataset: str):
    try:
        return ltr_service.explain(dataset)
    except ValueError as e:
        return {"error": str(e)}

import asyncio
import logging

from fastapi import APIRouter

from src.api.schemas import QueryRequest, HybridRequest
from src.api.dependencies.container import retrieval_service, refinement_service

router = APIRouter(tags=["Search"])
logger = logging.getLogger(__name__)


@router.post("/search")
async def search(request: QueryRequest):

    loop = asyncio.get_running_loop()

    results = await loop.run_in_executor(
        None,
        lambda: retrieval_service.search(
            dataset=request.dataset,
            query=request.query,
            model=request.model,
            top_k=request.top_k,
            bm25_k1=request.bm25_k1,
            bm25_b=request.bm25_b,
            enable_refinement=request.enable_refinement,
        ),
    )

    pipeline = await loop.run_in_executor(
        None,
        lambda: retrieval_service.pipeline(
            query=request.query,
            enable_refinement=request.enable_refinement,
        ),
    )
    suggestions = await loop.run_in_executor(
        None, lambda: refinement_service.get_suggestions(request.query)
    )

    print({"results": results, "pipeline": pipeline, "suggestions": suggestions})
    return {"results": results, "pipeline": pipeline, "suggestions": suggestions}


@router.post("/hybrid-search")
async def hybrid_search(request: HybridRequest):
    logger.info(
        f"Hybrid | query='{request.query}' mode={request.mode} "
        f"A={request.model_a} B={request.model_b} fusion={request.fusion} "
        f"alpha={request.alpha} dataset={request.dataset} "
        f"refinement={request.enable_refinement}"
    )
    loop = asyncio.get_running_loop()

    results = await loop.run_in_executor(
        None,
        lambda: retrieval_service.hybrid_search(
            dataset=request.dataset,
            query=request.query,
            mode=request.mode,
            model_a=request.model_a,
            model_b=request.model_b,
            top_k=request.top_k,
            pool_size=request.pool_size,
            fusion=request.fusion,
            alpha=request.alpha,
            bm25_k1=request.bm25_k1,
            bm25_b=request.bm25_b,
            enable_refinement=request.enable_refinement,
        ),
    )

    pipeline = await loop.run_in_executor(
        None,
        lambda: retrieval_service.pipeline(
            query=request.query,
            enable_refinement=request.enable_refinement,
        ),
    )
    suggestions = await loop.run_in_executor(
        None, lambda: refinement_service.get_suggestions(request.query)
    )
    return {"results": results, "pipeline": pipeline, "suggestions": suggestions}

import asyncio
import logging

from fastapi import APIRouter

from src.api.schemas import ClusterSearchRequest
from src.api.dependencies.container import clustering_service, datasets

router = APIRouter(tags=["Clustering"])
logger = logging.getLogger(__name__)


@router.post("/cluster/search")
async def cluster_search(request: ClusterSearchRequest):

    if request.dataset not in datasets:
        return {"error": f"Unknown dataset '{request.dataset}'"}

    dataset_obj = datasets[request.dataset]
    embedding_retriever = dataset_obj.get("embedding")
    if embedding_retriever is None:
        return {"error": (f"Dataset '{request.dataset}' has no embedding index. ")}

    loop = asyncio.get_running_loop()

    def _run():
        query_vec = embedding_retriever.index.encode_query(request.query)
        raw = clustering_service.cluster_search(
            dataset_name=request.dataset,
            query_vec=query_vec,
            embedding_retriever=embedding_retriever,
            top_k=request.top_k,
            n_probes=request.n_probes,
        )
        doc_repo = dataset_obj["doc_repo"]
        enriched = []
        for rank, item in enumerate(raw, start=1):
            doc = doc_repo.get_document(item["doc_id"])
            enriched.append(
                {
                    "rank": rank,
                    "doc_id": item["doc_id"],
                    "score": round(item["score"], 6),
                    "cluster_id": item.get("cluster_id", -1),
                    "text": doc["text"] if doc else "",
                }
            )
        return enriched

    results = await loop.run_in_executor(None, _run)
    return {"query": request.query, "dataset": request.dataset, "results": results}


@router.get("/cluster/info/{dataset}")
async def cluster_info(dataset: str):
    try:
        info = clustering_service.cluster_info(dataset)
        return info
    except ValueError as e:
        return {"error": str(e)}


@router.get("/cluster/doc/{dataset}/{doc_id}")
async def doc_cluster(dataset: str, doc_id: str):
    try:
        return clustering_service.get_doc_cluster(dataset, doc_id)
    except ValueError as e:
        return {"error": str(e)}


@router.get("/cluster/neighbors/{dataset}/{cluster_id}")
async def cluster_neighbors(dataset: str, cluster_id: int, limit: int = 20):
    try:
        return clustering_service.cluster_neighbors(dataset, cluster_id, limit=limit)
    except ValueError as e:
        return {"error": str(e)}

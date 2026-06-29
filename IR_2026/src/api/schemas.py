from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model: str = "bm25"  # "bm25" | "tfidf" | "inverted" | "embedding"
    dataset: str = "lotte"
    top_k: int = Field(default=10, ge=1, le=1000)

    bm25_k1: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    bm25_b: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    enable_refinement: bool = Field(
        default=False,
    )


class HybridRequest(BaseModel):
    query: str = Field(..., min_length=1)
    dataset: str = "lotte"

    mode: str = Field(default="parallel", description="'parallel' or 'serial'")
    model_a: str = Field(default="bm25", description="First model")
    model_b: str = Field(default="embedding", description="Second model")

    top_k: int = Field(default=10, ge=1, le=1000)
    pool_size: int = Field(
        default=200,
        ge=10,
        le=1000,
    )

    fusion: str = Field(default="rrf")
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    bm25_k1: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    bm25_b: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    enable_refinement: bool = Field(default=False)


class EvaluationRequest(BaseModel):
    dataset: str = "lotte"
    model: str = "bm25"

    # hybrid params (used when model == "hybrid")
    hybrid_mode: str = Field(default="parallel")
    hybrid_model_a: str = Field(default="bm25")
    hybrid_model_b: str = Field(default="embedding")
    hybrid_fusion: str = Field(default="rrf")
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    hybrid_pool: int = Field(default=200, ge=10, le=1000)

    query_count: int = Field(default=100, ge=1)
    top_k: int = Field(default=100, ge=1, le=1000)
    p_k: int = Field(default=10, ge=1, le=100)
    ndcg_k: int = Field(default=10, ge=1, le=100)
    recall_k: int = Field(
        default=150,
        ge=1,
        le=1000,
    )

    bm25_k1: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    bm25_b: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    enable_refinement: bool = Field(default=False)


# ── RAG (online, via OpenRouter — independent service) ──────────────────────


class RAGRequest(BaseModel):
    question: str = Field(..., min_length=1)
    dataset: str = Field(default="quora")
    model: str = Field(
        default="embedding",
    )
    top_k: int = Field(default=10, ge=1, le=100)
    max_passages: int = Field(default=10, ge=1, le=20)

    bm25_k1: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    bm25_b: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    enable_refinement: bool = Field(default=False)

    # Hybrid params — used when model == "hybrid"
    hybrid_mode: Optional[str] = Field(default=None)
    hybrid_model_a: str = Field(default="bm25")
    hybrid_model_b: str = Field(default="embedding")
    hybrid_fusion: str = Field(default="rrf")
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    hybrid_pool: int = Field(default=200, ge=10, le=1000)


class RAGStreamRequest(BaseModel):
    question: str = Field(..., min_length=1)
    dataset: str = Field(default="quora")
    model: str = Field(default="embedding")
    top_k: int = Field(default=10, ge=1, le=100)
    max_passages: int = Field(default=10, ge=1, le=20)
    bm25_k1: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    bm25_b: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    enable_refinement: bool = Field(default=False)


class RAGHealthResponse(BaseModel):

    available: bool
    reason: str
    model: str


# ── Clustering ────────────────────────────────────────────────────────────


class ClusterSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    dataset: str = Field(default="quora")
    top_k: int = Field(default=10, ge=1, le=200)
    n_probes: int = Field(
        default=3,
        ge=1,
        le=20,
    )


# ── LTR (Learning-to-Rank — independent offline service) ───────────────────


class LTRRerankRequest(BaseModel):

    query: str = Field(..., min_length=1)
    dataset: str = Field(default="quora")
    top_k: int = Field(default=10, ge=1, le=200)
    pool_size: int = Field(
        default=200,
        ge=10,
        le=1000,
    )


class LTRTrainRequest(BaseModel):
    dataset: str = Field(default="quora")
    max_queries: int = Field(
        default=200,
        ge=1,
        le=5000,
    )
    pool_size: int = Field(default=200, ge=10, le=1000)


class LTRStatusResponse(BaseModel):
    dataset: str
    ready: bool
    available_datasets: list[str]


# ── Response models ───────────────────────────────────────────────────────


class SearchResult(BaseModel):
    rank: int
    doc_id: str
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    dataset: str
    model: str
    total_results: int
    results: list[SearchResult]


class BM25PresetInfo(BaseModel):
    key: str
    k1: float
    b: float
    label: str


class BM25PresetsResponse(BaseModel):
    presets: list[BM25PresetInfo]
    description: str

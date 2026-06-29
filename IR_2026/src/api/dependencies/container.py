import logging
from pathlib import Path

from src.config.datasets import load_datasets
from src.config.evaluation_datasets import load_evaluation_datasets

from src.preprocessing.preprocessing_service import PreprocessingService
from src.query_refinement.refinement_service import RefinementService
from src.query_refinement.suggestion_service import QuerySuggestionService
from src.retrieval.retrieval_service import RetrievalService

from src.clustering.clustering_index import ClusteringIndex, CLUSTER_COUNTS
from src.clustering.clustering_service import ClusteringService

from src.rag.rag_service import RAGService

from src.ltr.ltr_service import LTRService, LTRModel

from src.evaluation.evaluation_service import EvaluationService
from src.evaluation.evaluation_runner import EvaluationRunner

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]

# ── Preprocessing ──────────────────────────────────────────────────────────

logger.info("Initialising PreprocessingService …")
preprocessor = PreprocessingService()

# ── Query Refinement ───────────────────────────────────────────────────────

suggestions_service = QuerySuggestionService()

logger.info("Initialising RefinementService …")
refinement_service = RefinementService(suggestion_service=suggestions_service)

# ── Datasets (indexes + retrievers) ───────────────────────────────────────

logger.info("Loading datasets …")
datasets = load_datasets()

logger.info("Loading evaluation datasets …")
evaluation_datasets = load_evaluation_datasets()

# ── Retrieval service ──────────────────────────────────────────────────────

retrieval_service = RetrievalService(
    datasets=datasets,
    preprocessing_service=preprocessor,
    refinement_service=refinement_service,
)

# ── Clustering service ─────────────────────────────────────────────────────

logger.info("Initialising ClusteringService …")
clustering_service = ClusteringService()


_CLUSTER_INDEX_PATHS: dict[str, str] = {
    "msmarco": "data/indexes/msmarco/cluster_index.pkl.gz",
    "quora": "data/indexes/quora/cluster_index.pkl.gz",
}


for _ds_name, _rel_path in _CLUSTER_INDEX_PATHS.items():
    _abs_path = ROOT / _rel_path
    if _abs_path.exists():
        try:
            _idx = ClusteringIndex.load(str(_abs_path))
            clustering_service.register(_ds_name, _idx)
        except Exception as _exc:
            logger.warning(
                f"ClusteringService: could not load index for '{_ds_name}': {_exc}"
            )
    else:
        logger.info(
            f"ClusteringService: no index found for '{_ds_name}' at {_abs_path}. "
            f"Run scripts/build_cluster_index.py to create it."
        )

# ── RAG service (independent — online via OpenRouter) ──────────────────────
# Selected explicitly from the UI. No coupling to LTR or any other
# service; if OpenRouter is unreachable, /rag returns an explanatory
# message and the user picks a different service from the UI themselves.

logger.info("Initialising RAGService …")
rag_service = RAGService(retrieval_service=retrieval_service)

# ── LTR service (independent — offline Logistic Regression re-ranker) ──────
# Selected explicitly from the UI, exactly like Clustering. No coupling
# to RAG; this is not a fallback that gets triggered automatically.

logger.info("Initialising LTRService …")
ltr_service = LTRService()
_LTR_MODEL_PATHS: dict[str, str] = {
    "msmarco": "data/indexes/msmarco/ltr_model.pkl.gz",
    "quora": "data/indexes/quora/ltr_model.pkl.gz",
}
for _ds_name, _rel_path in _LTR_MODEL_PATHS.items():
    _abs_path = ROOT / _rel_path
    if _abs_path.exists():
        try:
            _model = LTRModel.load(str(_abs_path))
            ltr_service.register(_ds_name, _model)
            logger.info(f"LTRService: loaded trained model for '{_ds_name}'.")
        except Exception as _exc:
            logger.warning(f"LTRService: could not load model for '{_ds_name}': {_exc}")
    else:
        logger.info(
            f"LTRService: no trained model found for '{_ds_name}' at {_abs_path}. "
            f"Call POST /ltr/train or run scripts/train_ltr_model.py to create it."
        )

# ── Evaluation ─────────────────────────────────────────────────────────────

evaluation_service = EvaluationService()

evaluation_runner = EvaluationRunner(
    retrieval_service=retrieval_service,
    evaluation_service=evaluation_service,
)

logger.info("Container initialised — all services ready.")

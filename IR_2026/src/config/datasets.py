from __future__ import annotations

import logging
from pathlib import Path

from src.common.document_repository import DocumentRepository
from src.indexing.bm25_index import BM25Index
from src.indexing.tfidf_index import TFIDFIndex
from src.indexing.inverted_index import InvertedIndex
from src.indexing.embedding_index import EmbeddingIndex
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.tfidf_retriever import TFIDFRetriever
from src.retrieval.inverted_retriever import InvertedRetriever
from src.retrieval.embedding_retriever import EmbeddingRetriever

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

DATASET_CONFIG: dict[str, dict[str, str]] = {
    "quora": {
        "bm25": "data/indexes/quora/bm25_index.pkl.gz",
        "tfidf": "data/indexes/quora/tfidf_index.pkl.gz",
        "inverted": "data/indexes/quora/inverted_index.pkl.gz",
        "embedding": "data/indexes/quora/embedding_index.pkl.gz",
    },
    "msmarco": {
        "bm25": "data/indexes/msmarco/bm25_index.pkl.gz",
        "tfidf": "data/indexes/msmarco/tfidf_index.pkl.gz",
        "inverted": "data/indexes/msmarco/inverted_index.pkl.gz",
        "embedding": "data/indexes/msmarco/embedding_index.pkl.gz",
    },
}
# DATASET_CONFIG: dict[str, dict[str, str]] = {
#     "quora": {
#         "bm25": "data/indexes/quora/bm25_index.pkl",
#         "tfidf": "data/indexes/quora/tfidf_index.pkl",
#         "inverted": "data/indexes/quora/inverted_index.pkl",
#         "embedding": "data/indexes/quora/embedding_index.pkl",
#     },
#     "msmarco": {
#         "bm25": "data/indexes/msmarco/bm25_index.pkl",
#         "tfidf": "data/indexes/msmarco/tfidf_index.pkl",
#         "inverted": "data/indexes/msmarco/inverted_index.pkl",
#         "embedding": "data/indexes/msmarco/embedding_index.pkl",
#     },
#     # "lotte": {
#     #     "bm25": "data/indexes/lotte/lotte_bm25.pkl",
#     #     "tfidf": "data/indexes/lotte/lotte_tfidf.pkl",
#     #     "inverted": "data/indexes/lotte/lotte_inverted.pkl",
#     #     "embedding": "data/indexes/lotte/embedding_index.pkl",  # ← NEW
#     # },
#     # "trec": {
#     #     "bm25": "data/indexes/trec/bm25_index.pkl",
#     #     "tfidf": "data/indexes/trec/tfidf_index.pkl",
#     #     "inverted": "data/indexes/trec/inverted_index.pkl",
#     #     "embedding": "data/indexes/trec/embedding_index.pkl",
#     # },
#     # "beir": {
#     #     "bm25": "data/indexes/beir/bm25_index.pkl",
#     #     "tfidf": "data/indexes/beir/tfidf_index.pkl",
#     #     "inverted": "data/indexes/beir/inverted_index.pkl",
#     #     # "embedding": "data/indexes/beir/embedding_index.pkl",
#     # },
# }


def load_datasets() -> dict:
    datasets: dict = {}
    for name, paths in DATASET_CONFIG.items():
        logger.info(f"Loading dataset: '{name}' …")
        try:
            datasets[name] = _load_single(name, paths)
            logger.info(
                f"  ✓ '{name}' ready "
                f"({datasets[name]['meta']['total_docs']:,} docs)"
            )
        except FileNotFoundError as e:
            logger.error(str(e))
            raise
    return datasets


def _resolve(rel: str, optional: bool = False) -> Path | None:
    p = ROOT / rel
    if not p.exists():
        if optional:
            return None
        raise FileNotFoundError(f"File not found: {p}\n")
    return p


def _load_single(name: str, paths: dict) -> dict:
    # ── Document repository (PostgreSQL) ───────────────────────────────
    doc_repo = DocumentRepository(dataset_name=name)  # ← changed
    total_docs = doc_repo.count()

    logger.info(f"  [{name}] loading BM25 index …")
    bm25_idx = BM25Index.load(str(_resolve(paths["bm25"])))
    bm25 = BM25Retriever(bm25_idx)

    logger.info(f"  [{name}] loading TF-IDF index …")
    tfidf_idx = TFIDFIndex.load(str(_resolve(paths["tfidf"])))
    tfidf = TFIDFRetriever(tfidf_idx)

    logger.info(f"  [{name}] loading inverted index …")
    inv_idx = InvertedIndex.load(str(_resolve(paths["inverted"])))
    inverted = InvertedRetriever(inv_idx, total_docs=total_docs)

    embedding_retriever = None
    emb_path = _resolve(paths.get("embedding", ""), optional=True)
    if emb_path:
        try:
            emb_idx = EmbeddingIndex.load(str(emb_path))
            embedding_retriever = EmbeddingRetriever(emb_idx)
        except Exception as exc:
            logger.warning(f"  [{name}] could not load embedding index: {exc}")

    dataset: dict = {
        "bm25": bm25,
        "tfidf": tfidf,
        "inverted": inverted,
        "doc_repo": doc_repo,
        "meta": {"name": name, "total_docs": total_docs},
    }
    if embedding_retriever is not None:
        dataset["embedding"] = embedding_retriever

    return dataset

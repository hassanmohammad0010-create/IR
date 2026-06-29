import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.preprocessing_service import PreprocessingService
from src.indexing.inverted_index import InvertedIndex
from src.indexing.bm25_index import BM25Index
from src.indexing.tfidf_index import TFIDFIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

DATASET_DIR = ROOT / "datasets" / "dataset3"

RAW_DOCS = DATASET_DIR / "documents.jsonl"
PROCESSED = ROOT / "data" / "processed" / "msmarco_processed.jsonl"

INDEX_DIR = ROOT / "data" / "indexes" / "msmarco"

INVERTED_PATH = INDEX_DIR / "inverted_index.pkl"
BM25_PATH = INDEX_DIR / "bm25_index.pkl"
TFIDF_PATH = INDEX_DIR / "tfidf_index.pkl"

QRELS_PATH = DATASET_DIR / "qrels.jsonl"
QUERIES_PATH = DATASET_DIR / "queries.jsonl"

BATCH_SIZE = 512
BM25_K1 = 1.2
BM25_B = 0.75


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser.parse_args()


def preprocess(args):
    if args.skip_preprocess and PROCESSED.exists():
        logger.info("[SKIP] preprocessing")
        return

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[1/4] Preprocessing MS MARCO corpus")

    svc = PreprocessingService()

    t0 = time.time()

    count = svc.process_collection(
        str(RAW_DOCS),
        str(PROCESSED),
        batch_size=BATCH_SIZE,
        limit=args.limit,
    )

    logger.info(f"processed docs: {count:,} in {time.time()-t0:.1f}s")


def build_inverted(args):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("[2/4] Building Inverted Index")

    idx = InvertedIndex()

    t0 = time.time()

    count = idx.build(
        str(PROCESSED),
        limit=args.limit,
    )

    idx.save(str(INVERTED_PATH))

    logger.info(f"docs={count:,} time={time.time()-t0:.1f}s")


def build_bm25(args):
    logger.info("[3/4] Building BM25 Index")

    idx = BM25Index(
        k1=BM25_K1,
        b=BM25_B,
    )

    t0 = time.time()

    count = idx.build(
        str(PROCESSED),
        limit=args.limit,
    )

    idx.save(str(BM25_PATH))

    logger.info(f"docs={count:,} time={time.time()-t0:.1f}s")


def build_tfidf(args):
    logger.info("[4/4] Building TF-IDF Index")

    idx = TFIDFIndex(
        sublinear_tf=True,
    )

    t0 = time.time()

    count = idx.build(
        str(PROCESSED),
        limit=args.limit,
    )

    idx.save(str(TFIDF_PATH))

    logger.info(f"docs={count:,} time={time.time()-t0:.1f}s")


def sanity_check_qrels():

    logger.info("[SANITY] Checking qrels ↔ queries alignment")

    import json

    qrels_queries = set()

    for line in open(QRELS_PATH, "r", encoding="utf-8"):
        r = json.loads(line)

        if int(r["relevance"]) > 0:
            qrels_queries.add(r["query_id"])

    queries = set()

    for line in open(QUERIES_PATH, "r", encoding="utf-8"):
        r = json.loads(line)
        queries.add(r["query_id"])

    covered = len(qrels_queries & queries)

    logger.info(f"queries in qrels: {len(qrels_queries)}")
    logger.info(f"total queries: {len(queries)}")
    logger.info(f"coverage: {covered/len(queries):.4f}")


def main():

    args = parse_args()

    logger.info("Starting MS MARCO build pipeline...")

    start = time.time()

    preprocess(args)
    build_inverted(args)
    build_bm25(args)
    build_tfidf(args)

    sanity_check_qrels()

    logger.info(f"Build completed in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()

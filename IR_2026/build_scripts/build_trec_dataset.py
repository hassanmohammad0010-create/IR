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

RAW_JSONL = ROOT / "datasets" / "dataset4" / "documents.jsonl"
PROCESSED_JSONL = ROOT / "data" / "processed" / "trec_processed.jsonl"

INDEX_DIR = ROOT / "data" / "indexes" / "trec"

INVERTED_PATH = INDEX_DIR / "inverted_index.pkl"
BM25_PATH = INDEX_DIR / "bm25_index.pkl"
TFIDF_PATH = INDEX_DIR / "tfidf_index.pkl"

BATCH_SIZE = 512
BM25_K1 = 1.5
BM25_B = 0.75


def parse_args() -> argparse.Namespace:
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


def step_preprocess(args: argparse.Namespace) -> None:

    if args.skip_preprocess and PROCESSED_JSONL.exists():
        logger.info("[SKIP] preprocessing")
        return

    if not RAW_JSONL.exists():
        logger.error(f"Dataset not found: {RAW_JSONL}")
        sys.exit(1)

    PROCESSED_JSONL.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[1/4] Preprocessing")

    svc = PreprocessingService()

    t0 = time.time()

    count = svc.process_collection(
        str(RAW_JSONL),
        str(PROCESSED_JSONL),
        batch_size=BATCH_SIZE,
        limit=args.limit,
    )

    logger.info(f"      processed {count:,} docs in {time.time() - t0:.1f}s")


def step_inverted(args: argparse.Namespace) -> None:

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("[2/4] Building Inverted Index")

    idx = InvertedIndex()

    t0 = time.time()

    count = idx.build(
        str(PROCESSED_JSONL),
        limit=args.limit,
    )

    idx.save(str(INVERTED_PATH))

    logger.info(
        f"      docs={count:,} vocab={idx.vocabulary_size():,} "
        f"time={time.time() - t0:.1f}s"
    )


def step_bm25(args: argparse.Namespace) -> None:

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("[3/4] Building BM25 Index")

    idx = BM25Index(
        k1=BM25_K1,
        b=BM25_B,
    )

    t0 = time.time()

    count = idx.build(
        str(PROCESSED_JSONL),
        limit=args.limit,
    )

    idx.save(str(BM25_PATH))

    logger.info(
        f"      docs={count:,} vocab={idx.vocabulary_size():,} "
        f"time={time.time() - t0:.1f}s"
    )


def step_tfidf(args: argparse.Namespace) -> None:

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("[4/4] Building TF-IDF Index")

    idx = TFIDFIndex(
        sublinear_tf=True,
    )

    t0 = time.time()

    count = idx.build(
        str(PROCESSED_JSONL),
        limit=args.limit,
    )

    idx.save(str(TFIDF_PATH))

    logger.info(
        f"      docs={count:,} vocab={idx.vocabulary_size():,} "
        f"time={time.time() - t0:.1f}s"
    )


def main() -> None:

    args = parse_args()

    logger.info("Starting build pipeline...")

    total_start = time.time()

    step_preprocess(args)
    step_inverted(args)
    step_bm25(args)
    step_tfidf(args)

    logger.info(f"Build completed in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()

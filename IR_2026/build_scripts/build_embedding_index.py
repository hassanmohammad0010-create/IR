import argparse
import gc
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indexing.embedding_index import DEFAULT_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a dense EmbeddingIndex (chunked, low-memory)."
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Sentences per encoding batch (lower = less RAM). Default: 32",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=5000,
        help="Docs to encode before freeing memory. Default: 5000",
    )
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset  : {args.dataset}")
    logger.info(f"Input    : {input_path}")
    logger.info(f"Output   : {output_path}")
    logger.info(f"Model    : {args.model}")
    logger.info(f"Batch    : {args.batch}")
    logger.info(f"Chunk    : {args.chunk} docs")
    logger.info(f"Limit    : {args.limit or 'none (all docs)'}")

    # ── 1. Read all doc_ids and texts ────────────────────────────────
    logger.info("Reading corpus ...")
    doc_ids: list[str] = []
    texts: list[str] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_ids.append(str(doc["doc_id"]))
            texts.append(doc.get("text", ""))

    total = len(doc_ids)
    logger.info(f"Loaded {total:,} documents.")

    # ── 2. Load model once ───────────────────────────────────────────
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading model: {args.model} ...")
    model = SentenceTransformer(args.model)
    logger.info("Model loaded. Starting encoding ...")

    # ── 3. Encode in chunks ──────────────────────────────────────────
    all_vecs: list[np.ndarray] = []
    t0 = time.time()

    for start in range(0, total, args.chunk):
        end = min(start + args.chunk, total)
        chunk_text = texts[start:end]

        logger.info(f"Encoding docs {start+1:,} - {end:,} / {total:,} ...")
        vecs = model.encode(
            chunk_text,
            batch_size=args.batch,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        # L2-normalise so dot product == cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vecs = (vecs / norms).astype(np.float32)

        all_vecs.append(vecs)
        del chunk_text, norms
        gc.collect()

        elapsed = time.time() - t0
        rate = end / elapsed
        eta = (total - end) / rate if rate > 0 else 0
        logger.info(
            f"  Progress: {end:,}/{total:,} | "
            f"{rate:.0f} docs/s | "
            f"ETA {eta/60:.1f} min"
        )

    # ── 4. Stack into single matrix ──────────────────────────────────
    logger.info("Stacking all vectors into final matrix ...")
    matrix = np.vstack(all_vecs)
    del all_vecs, texts
    gc.collect()

    elapsed = time.time() - t0
    logger.info(
        f"Encoded {total:,} docs in {elapsed:.1f}s  "
        f"({total/elapsed:.0f} docs/s)  "
        f"shape={matrix.shape}"
    )

    # ── 5. Save ──────────────────────────────────────────────────────
    logger.info(f"Saving to {output_path} ...")
    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "model_name": args.model,
                "doc_ids": doc_ids,
                "matrix": matrix,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved — {size_mb:.0f} MB")
    logger.info("Done.")


if __name__ == "__main__":
    main()

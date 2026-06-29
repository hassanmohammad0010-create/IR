from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import psycopg2
import psycopg2.extras

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.db import DB_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASETS: dict[str, str] = {
    "quora": "datasets/dataset2/documents.jsonl",
    "msmarco": "datasets/dataset3/documents.jsonl",
    # "beir":  "datasets/dataset5/documents.jsonl",
}

BATCH_SIZE = 2_000


def connect() -> psycopg2.extensions.connection:
    logger.info(
        f"Connecting to PostgreSQL "
        f"{DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}"
        f"/{DB_CONFIG['dbname']} …"
    )
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn


def create_table(cur, table: str) -> None:
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            doc_id  TEXT PRIMARY KEY,
            text    TEXT NOT NULL
        );
    """)
    logger.info(f"Table '{table}' ensured.")


def ingest(conn, table: str, jsonl_path: Path) -> int:
    total = 0
    batch: list[tuple[str, str]] = []

    with conn.cursor() as cur:
        create_table(cur, table)
        conn.commit()

        logger.info(f"Ingesting '{jsonl_path}' → '{table}' …")
        t0 = time.time()

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"  line {line_no}: invalid JSON — skipped")
                    continue

                doc_id = str(
                    obj.get("doc_id") or obj.get("id") or obj.get("_id") or line_no
                )
                text = str(
                    obj.get("text") or obj.get("body") or obj.get("contents") or ""
                )

                batch.append((doc_id, text))

                if len(batch) >= BATCH_SIZE:
                    _flush(cur, table, batch)
                    total += len(batch)
                    batch = []
                    elapsed = time.time() - t0
                    logger.info(
                        f"  {total:,} rows  ({elapsed:.1f}s)  "
                        f"~{total / elapsed:,.0f} rows/s"
                    )

        if batch:
            _flush(cur, table, batch)
            total += len(batch)

        conn.commit()

    elapsed = time.time() - t0
    logger.info(
        f"Done — {total:,} rows ingested into '{table}' "
        f"in {elapsed:.1f}s ({total/elapsed:,.0f} rows/s)"
    )
    return total


def _flush(cur, table: str, batch: list[tuple[str, str]]) -> None:
    psycopg2.extras.execute_values(
        cur,
        f"""
        INSERT INTO {table} (doc_id, text)
        VALUES %s
        ON CONFLICT (doc_id) DO NOTHING
        """,
        batch,
        template="(%s, %s)",
        page_size=BATCH_SIZE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest IR dataset documents into PostgreSQL."
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        required=True,
    )
    args = parser.parse_args()

    targets: dict[str, str] = (
        DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    )

    conn = connect()
    try:
        for name, rel_path in targets.items():
            jsonl_path = ROOT / rel_path
            if not jsonl_path.exists():
                logger.error(f"File not found: {jsonl_path}  — skipping '{name}'")
                continue
            table = f"documents_{name}"
            ingest(conn, table, jsonl_path)
    finally:
        conn.close()

    logger.info("All done.")


if __name__ == "__main__":
    main()

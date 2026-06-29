import json
import logging
import threading
from src.common.pkl_io import pkl_load, pkl_dump

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

DEFAULT_K1: float = 1.5
DEFAULT_B: float = 0.75


class BM25Index:

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.tokenized_docs: list[list[str]] = []

        self._scorer_cache: dict[tuple[float, float], BM25Okapi] = {}
        self._cache_lock = threading.Lock()

    def build(self, documents_path: str, limit: int | None = None) -> int:

        self.doc_ids = []
        self.tokenized_docs = []
        self._scorer_cache.clear()

        with open(documents_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                self.doc_ids.append(str(doc["doc_id"]))
                self.tokenized_docs.append(doc["text"].split())

        self._build_scorer(self.k1, self.b)
        logger.info(
            f"BM25Index built: {len(self.doc_ids):,} docs, "
            f"default k1={self.k1}, b={self.b}"
        )
        return len(self.doc_ids)

    def get_scorer(self, k1: float | None = None, b: float | None = None) -> BM25Okapi:

        k1 = round(float(k1 if k1 is not None else self.k1), 4)
        b = round(float(b if b is not None else self.b), 4)
        key = (k1, b)

        with self._cache_lock:
            if key not in self._scorer_cache:
                logger.info(f"BM25Index: building scorer for k1={k1}, b={b} …")
                self._build_scorer(k1, b)
            return self._scorer_cache[key]

    def _build_scorer(self, k1: float, b: float) -> None:
        scorer = BM25Okapi(self.tokenized_docs, k1=k1, b=b)
        self._scorer_cache[(k1, b)] = scorer

    @property
    def bm25(self) -> BM25Okapi:
        return self.get_scorer(self.k1, self.b)

    def vocabulary_size(self) -> int:
        scorer = self.get_scorer()
        return len(scorer.idf) if scorer else 0

    def get_doc_tokens(self, doc_id: str) -> list[str]:
        try:
            idx = self.doc_ids.index(doc_id)
            return self.tokenized_docs[idx]
        except ValueError:
            return []

    def save(self, path: str) -> None:
        pkl_dump(
            {
                "doc_ids": self.doc_ids,
                "tokenized_docs": self.tokenized_docs,
                "k1": self.k1,
                "b": self.b,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        data = pkl_load(path)
        obj = cls(k1=data.get("k1", DEFAULT_K1), b=data.get("b", DEFAULT_B))
        obj.doc_ids = data["doc_ids"]
        obj.tokenized_docs = data.get("tokenized_docs", [])
        obj._build_scorer(obj.k1, obj.b)
        return obj

import json
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix
from src.common.pkl_io import pkl_load, pkl_dump


class TFIDFIndex:
    def __init__(
        self,
        max_features: int | None = None,
        ngram_range: tuple[int, int] = (1, 1),
        sublinear_tf: bool = True,
    ):

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
        )
        self.matrix: spmatrix | None = None
        self.doc_ids: list[str] = []

    def build(self, documents_path: str, limit: int | None = None) -> int:

        texts: list[str] = []
        self.doc_ids = []

        with open(documents_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                self.doc_ids.append(doc["doc_id"])
                texts.append(doc["text"])

        self.matrix = self.vectorizer.fit_transform(texts)
        return len(self.doc_ids)

    def save(self, path: str) -> None:
        pkl_dump(
            {
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "doc_ids": self.doc_ids,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TFIDFIndex":
        data = pkl_load(path)
        obj = cls()
        obj.vectorizer = data["vectorizer"]
        obj.matrix = data["matrix"]
        obj.doc_ids = data["doc_ids"]
        return obj

    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_) if self.vectorizer else 0

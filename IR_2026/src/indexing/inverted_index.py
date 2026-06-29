import pickle
from collections import defaultdict
import json

from src.common.pkl_io import pkl_load, pkl_dump


class InvertedIndex:

    def __init__(self):
        self.index: defaultdict[str, set] = defaultdict(set)

    def add_document(self, doc_id: str, text: str) -> None:
        terms = text.split()
        for term in terms:
            self.index[term].add(doc_id)

    def build(self, documents_path: str, limit: int | None = None) -> int:

        count = 0
        with open(documents_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                self.add_document(doc["doc_id"], doc["text"])
                count += 1
        return count

    def save(self, path: str) -> None:
        pkl_dump({term: sorted(ids) for term, ids in self.index.items()}, path)

    @classmethod
    def load(cls, path: str) -> "InvertedIndex":
        data = pkl_load(path)
        obj = cls()
        obj.index = defaultdict(set, {term: set(ids) for term, ids in data.items()})
        return obj

    def get_posting_list(self, term: str) -> list[str]:
        return sorted(self.index.get(term, set()))

    def vocabulary_size(self) -> int:
        return len(self.index)

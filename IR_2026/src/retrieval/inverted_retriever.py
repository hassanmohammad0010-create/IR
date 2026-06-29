import math
from collections import defaultdict

from src.indexing.inverted_index import InvertedIndex


class InvertedRetriever:

    def __init__(self, index: InvertedIndex, total_docs: int):

        self.index = index
        self.total_docs = total_docs

    def _idf(self, term: str) -> float:
        df = len(self.index.get_posting_list(term))
        if df == 0:
            return 0.0
        return math.log(self.total_docs / df)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        terms = query.split()

        if not terms:
            return []

        scores: dict[str, float] = defaultdict(float)

        for term in terms:
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for doc_id in self.index.get_posting_list(term):
                scores[doc_id] += idf

        if not scores:
            return []

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.indexing.tfidf_index import TFIDFIndex


class TFIDFRetriever:

    def __init__(self, tfidf_index: TFIDFIndex):
        self.index = tfidf_index

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:

        if not query.strip():
            return []

        query_vector = self.index.vectorizer.transform([query])

        if query_vector.nnz == 0:
            return []

        similarities: np.ndarray = cosine_similarity(
            query_vector, self.index.matrix
        ).flatten()

        top_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            score = float(similarities[idx])
            results.append((self.index.doc_ids[idx], score))

        return results

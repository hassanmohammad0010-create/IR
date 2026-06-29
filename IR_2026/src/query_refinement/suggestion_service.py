from nltk.corpus import wordnet


class QuerySuggestionService:
    """
    Suggests alternative queries without modifying the original.
    These are shown in the UI as clickable suggestions, not injected into the query.
    """

    def suggest(self, query: str, max_suggestions: int = 5) -> list[str]:
        """
        Generate query suggestions based on:
        1. Synonym substitution (replace one word at a time)
        2. Partial query variations
        """
        suggestions = []
        words = query.lower().split()

        # Strategy 1: synonym substitution per word (swap, don't append)
        for i, word in enumerate(words):
            synonym = self._best_synonym(word)
            if synonym and synonym != word:
                new_query = words[:i] + [synonym] + words[i + 1 :]
                candidate = " ".join(new_query)
                if candidate != query:
                    suggestions.append(candidate)

        # Strategy 2: drop one word (for longer queries)
        if len(words) > 2:
            for i in range(len(words)):
                candidate = " ".join(words[:i] + words[i + 1 :])
                if candidate not in suggestions:
                    suggestions.append(candidate)

        return suggestions[:max_suggestions]

    @staticmethod
    def _best_synonym(word: str) -> str | None:
        synsets = wordnet.synsets(word)
        if not synsets:
            return None
        for lemma in synsets[0].lemmas():
            name = lemma.name().replace("_", " ").lower()
            if name != word:
                return name
        return None

from __future__ import annotations

import logging
import string
from functools import lru_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------


def _get_spell():
    try:
        from spellchecker import SpellChecker

        return SpellChecker()
    except ImportError:
        logger.warning(
            "pyspellchecker not installed. "
            "Spell correction disabled.  "
            "Install with: pip install pyspellchecker"
        )
        return None


def _get_wordnet():
    try:
        import nltk

        try:
            from nltk.corpus import wordnet as wn

            wn.synsets("test")
            return wn
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.corpus import wordnet as wn

            return wn
    except ImportError:
        logger.warning(
            "nltk not installed. "
            "Synonym expansion disabled.  "
            "Install with: pip install nltk"
        )
        return None


# ---------------------------------------------------------------------------


class RefinementService:

    def __init__(self, suggestion_service):
        self._spell = None  # lazy
        self._wn = None  # lazy
        self._suggestion_service = suggestion_service

    # ------------------------------------------------------------------

    def refine(self, query: str, enabled: bool = True) -> str:

        if not enabled or not query.strip():
            return query

        q = self._correct_spelling(query)
        # q = self._expand_synonyms(q)
        logger.debug(f"Refinement: '{query}' → '{q}'")
        return q

    def get_suggestions(self, query: str) -> list[str]:
        """Return alternative query suggestions (shown in UI, not injected)."""
        corrected = self._correct_spelling(query)
        return self._suggestion_service.suggest(corrected)

    def explain(self, query: str, enabled: bool = True) -> dict:

        if not enabled:
            return {
                "original": query,
                "enabled": False,
                "spell_corrected": query,
                "spell_changes": {},
                # "synonym_expanded": query,
                # "synonym_additions": {},
                "final": query,
            }

        spell_corrected, spell_changes = self._correct_spelling_with_diff(query)
        # synonym_expanded, synonym_additions = self._expand_synonyms_with_diff(
        #     spell_corrected
        # )

        return {
            "original": query,
            "enabled": True,
            "spell_corrected": spell_corrected,
            "spell_changes": spell_changes,
            # "synonym_expanded": synonym_expanded,
            # "synonym_additions": synonym_additions,
            # "final": synonym_expanded,
        }

    # ------------------------------------------------------------------

    def _correct_spelling(self, text: str) -> str:
        corrected, _ = self._correct_spelling_with_diff(text)
        return corrected

    def _correct_spelling_with_diff(self, text: str) -> tuple[str, dict[str, str]]:
        spell = self._load_spell()
        if spell is None:
            return text, {}

        words = text.split()
        changes: dict[str, str] = {}
        corrected_words: list[str] = []

        for word in words:
            stripped = word.strip(string.punctuation)
            if not stripped or not stripped.isalpha():
                corrected_words.append(word)
                continue

            correction = spell.correction(stripped.lower())
            if correction and correction != stripped.lower():
                new_word = word.replace(stripped, correction, 1)
                changes[word] = new_word
                corrected_words.append(new_word)
            else:
                corrected_words.append(word)

        return " ".join(corrected_words), changes

    # ------------------------------------------------------------------

    # def _expand_synonyms(self, text: str) -> str:
    #     expanded, _ = self._expand_synonyms_with_diff(text)
    #     return expanded

    # def _expand_synonyms_with_diff(self, text: str) -> tuple[str, dict[str, str]]:
    #     wn = self._load_wordnet()
    #     if wn is None:
    #         return text, {}

    #     words = text.split()
    #     additions: dict[str, str] = {}
    #     extra: list[str] = []

    #     for word in words:
    #         stripped = word.strip(string.punctuation).lower()
    #         if not stripped or not stripped.isalpha():
    #             continue

    #         synonym = self._best_synonym(stripped, wn)
    #         if synonym and synonym != stripped:
    #             additions[stripped] = synonym
    #             extra.append(synonym)

    #     if extra:
    #         expanded = text + " " + " ".join(extra)
    #     else:
    #         expanded = text

    #     return expanded, additions

    # @staticmethod
    # @lru_cache(maxsize=2048)
    # def _best_synonym(word: str, wn) -> str | None:

    #     synsets = wn.synsets(word)
    #     if not synsets:
    #         return None
    #     for lemma in synsets[0].lemmas():
    #         name = lemma.name().replace("_", " ").lower()
    #         if name != word:
    #             return name
    #     return None

    # ------------------------------------------------------------------

    def _load_spell(self):
        if self._spell is None:
            self._spell = _get_spell()
        return self._spell

    def _load_wordnet(self):
        if self._wn is None:
            self._wn = _get_wordnet()
        return self._wn

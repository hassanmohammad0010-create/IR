import json
import logging
from pathlib import Path

from src.preprocessing.nlp_singleton import _nlp
from src.preprocessing.normalizer import Normalizer
from src.preprocessing.tokenizer import Tokenizer
from src.preprocessing.lemmatizer import Lemmatizer

logger = logging.getLogger(__name__)


class PreprocessingService:

    def __init__(self):
        self._normalizer = Normalizer()
        self._tokenizer = Tokenizer()
        self._lemmatizer = Lemmatizer()
        logger.info("PreprocessingService ready.")

    def preprocess_text(self, text: str) -> str:
        cleaned = self._normalizer.normalize(text)
        if not cleaned:
            return ""
        tokens = self._tokenizer.tokenize(cleaned)
        lemmas = self._lemmatizer.lemmatize(tokens)  # Tokens → strings
        return " ".join(lemmas)

    def preprocess_query(self, query: str) -> str:
        return self.preprocess_text(query)

    def pipeline(self, query: str):
        normalizer = self._normalizer.normalize(query)
        tokenizer = self._tokenizer.tokenize(normalizer)
        lemmatizer = self._lemmatizer.lemmatize(tokenizer)
        return {
            "normalizer": normalizer,
            "tokenizer": tokenizer,
            "lemmatizer": lemmatizer,
        }

    def preprocess_texts_batch(
        self, texts: list[str], batch_size: int = 512
    ) -> list[str]:

        cleaned = [self._normalizer.normalize(t) for t in texts]
        results: list[str] = []

        for doc in _nlp.pipe(cleaned, batch_size=batch_size):
            tokens = self._tokenizer.tokenize(doc)  # Doc → filtered Tokens
            lemmas = self._lemmatizer.lemmatize(tokens)  # Tokens → strings
            results.append(" ".join(lemmas))

        return results

    def process_collection(
        self,
        input_path: str,
        output_path: str,
        batch_size: int = 512,
        limit: int | None = None,
    ) -> int:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        raw_docs: list[tuple[str, str]] = []
        with open(input_path, "r", encoding="utf-8") as fin:
            for line_num, line in enumerate(fin, start=1):
                if limit is not None and len(raw_docs) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON at line {line_num}")
                    continue
                doc_id = obj.get("doc_id") or obj.get("id") or str(line_num)
                raw_text = obj.get("text") or obj.get("body") or ""
                raw_docs.append((str(doc_id), raw_text))

        logger.info(f"Preprocessing {len(raw_docs):,} docs …")
        processed = self.preprocess_texts_batch(
            [t for _, t in raw_docs], batch_size=batch_size
        )

        count = 0
        with open(output_path, "w", encoding="utf-8") as fout:
            for (doc_id, _), text in zip(raw_docs, processed):
                fout.write(
                    json.dumps({"doc_id": doc_id, "text": text}, ensure_ascii=False)
                    + "\n"
                )
                count += 1

        logger.info(f"Done — {count:,} docs → {output_path}")
        return count

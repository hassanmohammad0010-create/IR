from spacy.tokens import Doc
from src.preprocessing.nlp_singleton import _nlp


class Tokenizer:

    @staticmethod
    def tokenize(text_or_doc: str | Doc) -> list:
        doc = _nlp(text_or_doc) if isinstance(text_or_doc, str) else text_or_doc
        return [
            token
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]

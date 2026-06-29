import re
from html import unescape

_CONTRACTIONS: dict[str, str] = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "shan't": "shall not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
}


class Normalizer:

    @staticmethod
    def normalize(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+\.\S+", " ", text)
        text = text.lower()

        for c, e in _CONTRACTIONS.items():
            text = text.replace(c, e)
        text = re.sub(r"[*_>#`~\-]{2,}", " ", text)
        return re.sub(r"\s+", " ", text).strip()

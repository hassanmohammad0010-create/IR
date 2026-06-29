import spacy
from spacy.lang.en import English


def _load() -> English:
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        return nlp
    except OSError:
        nlp = English()
        nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
        nlp.initialize()
        return nlp


_nlp: English = _load()

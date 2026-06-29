import gzip
import pickle
from pathlib import Path


def pkl_load(path: str):
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(path, "rb") as f:
        return pickle.load(f)


def pkl_dump(obj, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

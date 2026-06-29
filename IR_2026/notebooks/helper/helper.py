import json
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SEP = "=" * 65
SEP2 = "-" * 65


def banner(msg):
    print(f"\n{SEP}\n  {msg}\n{SEP}")


def section(msg):
    print(f"\n{SEP2}\n  {msg}\n{SEP2}")


def fmt_time(s):
    return f"{s:.2f}s" if s < 60 else f"{s/60:.1f}m"


def load_queries(path, limit):
    queries = []
    with open(ROOT / path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id") or obj.get("id") or obj.get("qid", ""))
            text = str(obj.get("text") or obj.get("query", ""))
            if qid and text:
                queries.append({"query_id": qid, "text": text})
            if len(queries) >= limit:
                break
    return queries


def load_qrels(path):
    qrels = defaultdict(dict)
    with open(ROOT / path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id") or obj.get("qid", ""))
            did = str(obj.get("doc_id") or obj.get("docid", ""))
            rel = int(obj.get("relevance") or obj.get("rel", 0))
            if qid and did:
                qrels[qid][did] = rel
    return dict(qrels)

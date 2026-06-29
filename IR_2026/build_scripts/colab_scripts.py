# download the lotte_writing_forum dataset
# !pip install -q ir-datasets tqdm
import ir_datasets
import json
from tqdm import tqdm

dataset = ir_datasets.load("lotte/writing/test/forum")

print("Dataset loaded successfully")

dataset = ir_datasets.load("lotte/writing/test/forum")

print("Queries:", dataset.queries_count())
print("Documents:", dataset.docs_count())
print("Has qrels:", dataset.has_qrels())
#
dataset = ir_datasets.load("lotte/writing/test/forum")

output_path = "/content/lotte_writing_forum.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for doc in tqdm(dataset.docs_iter(), total=dataset.docs_count()):
        f.write(json.dumps({"doc_id": doc.doc_id, "text": doc.text}) + "\n")

print("Done writing dataset:", output_path)
#
import ir_datasets
import json
from tqdm import tqdm

dataset = ir_datasets.load("lotte/writing/test/forum")

with open("/content/queries.jsonl", "w", encoding="utf-8") as f:
    for q in tqdm(dataset.queries_iter(), total=dataset.queries_count()):
        f.write(json.dumps({"query_id": q.query_id, "text": q.text}) + "\n")
#
import json
from tqdm import tqdm

dataset = ir_datasets.load("lotte/writing/test/forum")

with open("/content/qrels.jsonl", "w", encoding="utf-8") as f:
    for qrel in tqdm(dataset.qrels_iter()):
        f.write(
            json.dumps(
                {
                    "query_id": qrel.query_id,
                    "doc_id": qrel.doc_id,
                    "relevance": qrel.relevance,
                }
            )
            + "\n"
        )
# then
# !mkdir -p "/content/drive/MyDrive/IR_Datasets/lotte_writing_forum"
# !cp /content/lotte_writing_forum.jsonl \
# "/content/drive/MyDrive/IR_Datasets/lotte_writing_forum/"
# !cp /content/queries.jsonl \
# "/content/drive/MyDrive/IR_Datasets/lotte_writing_forum/"
# !cp /content/queries.jsonl \
# "/content/drive/MyDrive/IR_Datasets/lotte_writing_forum/"
# !cp /content/qrels.jsonl \
# "/content/drive/MyDrive/IR_Datasets/lotte_writing_forum/"
# !cd "/content/drive/MyDrive/IR_Datasets" && \
# zip -r lotte_writing_forum.zip lotte_writing_forum
#
import json

docs = sum(1 for _ in open("/content/lotte_writing_forum.jsonl", encoding="utf-8"))
queries = sum(1 for _ in open("/content/queries.jsonl", encoding="utf-8"))
qrels = sum(1 for _ in open("/content/qrels.jsonl", encoding="utf-8"))

print("Documents:", docs)
print("Queries:", queries)
print("Qrels:", qrels)

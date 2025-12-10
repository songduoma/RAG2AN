import pandas as pd
from pathlib import Path
import sys

# 確保能匯入專案模組
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from discriminator import get_retrieval_ctx

df = pd.read_csv("FakeNews_Task3_2022/eval.csv")
row = df.iloc[0]
example = {
    "title": row.get("title", ""),
    "description": row["text"],
    "date_publish": None,
    "url": "",
}

# mimic discriminator google query logic
query = example.get("title") or str(example.get("description", ""))[:50]
ctx = get_retrieval_ctx(example, prefix="", source="google")
print("Title:", example["title"])
print("Google query:", query)
print("RAG context:\n", ctx)

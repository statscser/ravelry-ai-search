# diagnose.py
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from hybrid_search import hybrid_search, reranked_search
from rag_chroma import parse_query

load_dotenv()

patterns   = json.loads(Path("data/patterns.json").read_text(encoding="utf-8"))
embeddings = np.load(Path("data/embeddings.npy"))
client     = OpenAI()

query  = "cable knit sweater"
intent = parse_query(query, client)
print(f"Intent: {intent.model_dump()}\n")

# hybrid top 20（reranker 的输入）
candidates = hybrid_search(
    query=intent.semantic_query,
    patterns=patterns,
    embeddings=embeddings,
    openai_client=client,
    top_k=20,
    intent=intent,
)

# reranked top 20
results = reranked_search(
    query=intent.semantic_query,
    patterns=patterns,
    embeddings=embeddings,
    openai_client=client,
    top_k=20,
    intent=intent,
)

print(f"{'#':<3} {'Name':<45} {'Cohere':>8} {'BM25':>6} {'Vec':>5} {'Categories'}")
print("-" * 100)
for i, p in enumerate(results, 1):
    cats = ", ".join(c["name"] for c in (p.get("pattern_categories") or []))
    attrs = " ".join(
        a["permalink"] for a in (p.get("pattern_attributes") or [])
        if a["permalink"] in {"cables", "lace", "stranded", "oversized"}
    )
    flag = "⚠️" if "cables" not in attrs and i <= 20 else "✅"
    print(
        f"{i:<3} {p['name']:<45} "
        f"{p.get('_cohere_score', 0):>8.4f} "
        f"{p.get('_bm25_rank', 0):>6} "
        f"{p.get('_vec_rank', 0):>5}  "
        f"{flag} {cats[:30]}"
    )
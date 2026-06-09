# eval_v0.py
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from rag_chroma import load_collection, parse_query, search

load_dotenv()

QUERIES = [
    "cozy winter sweater with cable knitting",
    "summer crochet top lightweight",
    "colorwork yoke pullover",
    "crochet granny square cardigan",
    "knitting lace shawl beginner friendly",
    "classic socks pattern rating above 4.5",
    "oversized slouchy beanie rating above 4.8",
    "free easy crochet baby blanket",
    "free knitting fingerless gloves pattern",
    "free knitting cardigan rating above 4.5",
]

def main():
    client = OpenAI()
    collection, patterns = load_collection()

    eval_results = []

    for query in QUERIES:
        print(f"Testing: {query}")
        intent = parse_query(query, client)
        results = search(
            query=intent.semantic_query,
            collection=collection,
            openai_client=client,
            patterns=patterns,
            top_k=5,
            craft=intent.craft,
            free_only=intent.free_only,
            min_rating=intent.min_rating,
        )

        eval_results.append({
            "query": query,
            "intent": intent.model_dump(),
            "results": [
                {
                    "name": p["name"],
                    "craft": (p.get("craft") or {}).get("name", ""),
                    "yarn_weight": p.get("yarn_weight_description", ""),
                    "rating": p.get("rating_average", 0),
                    "free": p.get("free", False),
                    "similarity": p.get("_similarity", 0),
                    "url": f"https://www.ravelry.com/patterns/library/{p['permalink']}",
                }
                for p in results
            ],
        })

    Path("eval_v0.json").write_text(
        json.dumps(eval_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Done. Results saved to eval_v0.json")

if __name__ == "__main__":
    main()
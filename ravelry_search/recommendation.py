"""
recommendation.py — One-sentence recommendations for search results.
Uses ThreadPoolExecutor for true I/O parallelism with OpenAI API calls.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = (
    "You are a knitting pattern search assistant. "
    "Given the user's search query and pattern details, "
    "explain in one sentence why this pattern matches what they're looking for. "
    "Be concise, under 30 words, in English."
)

# Only surface attributes that are meaningful to knitters
_USEFUL_ATTRS = {
    "seamless", "top-down", "bottom-up", "in-the-round", "worked-flat",
    "cables", "lace", "stranded", "intarsia", "other-colorwork",
    "mosaic", "textured", "ribbed", "reversible",
    "oversized", "fitted", "cropped", "sleeveless", "long-sleeve",
    "beginner-friendly", "circular-yoke", "raglan-sleeve",
}


def _user_message(query: str, pattern: dict) -> str:
    craft      = (pattern.get("craft") or {}).get("name", "")
    yarn       = pattern.get("yarn_weight_description") or ""
    categories = ", ".join(c["name"] for c in (pattern.get("pattern_categories") or []))
    attributes = ", ".join(
        a["permalink"]
        for a in (pattern.get("pattern_attributes") or [])
        if a.get("permalink") in _USEFUL_ATTRS
    )
    notes = (pattern.get("notes") or "")[:200]

    lines = [
        f"Query: {query}",
        f"Pattern: {pattern.get('name', '')}",
        f"Craft: {craft}",
        f"Yarn weight: {yarn}",
    ]
    if categories:
        lines.append(f"Categories: {categories}")
    if attributes:
        lines.append(f"Attributes: {attributes}")
    if notes:
        lines.append(f"Notes excerpt: {notes}")
    return "\n".join(lines)


def generate_recommendation(query: str, pattern: dict, client: OpenAI) -> str:
    """Generate a single recommendation. Raises on API error."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=80,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _user_message(query, pattern)},
        ],
    )
    return response.choices[0].message.content.strip()


def generate_recommendations_batch(
    query: str,
    patterns: list[dict],
    client: OpenAI,
    top_n: int = 5,
) -> list[str]:
    """
    Generate recommendations for the first top_n patterns in parallel.
    Uses ThreadPoolExecutor — works correctly inside Streamlit
    (no asyncio event loop conflicts).
    Returns a list of strings in the same order as input patterns.
    Falls back to empty string on individual failures.
    """
    targets = patterns[:top_n]
    results = [""] * len(targets)

    # Each thread gets its own OpenAI client to avoid shared state issues
    def _generate(idx: int, pattern: dict) -> tuple[int, str]:
        thread_client = OpenAI()  # lightweight — reuses connection pool
        try:
            rec = generate_recommendation(query, pattern, thread_client)
            return idx, rec
        except Exception as e:
            print(f"[recommendation] ERROR for pattern {pattern.get('name')}: {e}")
            return idx, ""

    with ThreadPoolExecutor(max_workers=min(top_n, 5)) as executor:
        futures = {
            executor.submit(_generate, i, p): i
            for i, p in enumerate(targets)
        }
        for future in as_completed(futures):
            idx, rec = future.result()  # never raises — errors handled inside _generate
            results[idx] = rec

    return results


# ── Demo ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import json
    import time
    from pathlib import Path
    import numpy as np
    from hybrid_search import reranked_search
    from rag_chroma import load_collection, parse_query

    patterns   = json.loads((Path(__file__).parent / "data" / "patterns.json").read_text(encoding="utf-8"))
    embeddings = np.load(Path(__file__).parent / "data" / "embeddings.npy")
    openai_client = OpenAI()
    load_collection()

    query  = "cable knit sweater"
    intent = parse_query(query, openai_client)
    results = reranked_search(
        query=intent.semantic_query,
        patterns=patterns,
        embeddings=embeddings,
        openai_client=openai_client,
        top_k=5,
        intent=intent,
    )

    print(f"Query: {query}\n")

    # Single
    print("── Single ─────────────────────────────────────────")
    rec = generate_recommendation(query, results[0], openai_client)
    print(f"  {results[0]['name']}\n  → {rec}\n")

    # Batch — measure wall time
    print("── Batch (top 5, parallel) ────────────────────────")
    t0   = time.perf_counter()
    recs = generate_recommendations_batch(query, results, openai_client, top_n=5)
    elapsed = time.perf_counter() - t0
    for p, r in zip(results, recs):
        print(f"  {p['name']}\n  → {r}")
    print(f"\n  Wall time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
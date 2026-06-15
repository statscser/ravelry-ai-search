"""
recommendation.py — One-sentence recommendations for search results.
"""

import anthropic
import contextvars
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI

from langfuse.decorators import observe, langfuse_context

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


@observe(as_type="generation", capture_input=False)
def generate_recommendation(query: str, pattern: dict, client: OpenAI) -> str:
    """Generate a single recommendation. Raises on API error."""
    langfuse_context.update_current_observation(
        input={"query": query, "pattern": pattern.get("name")}
    )
    anthropic_client = anthropic.Anthropic()
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=80,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_message(query, pattern)}],
    )
    try:
        langfuse_context.update_current_observation(
            usage={
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            }
        )
    except Exception:
        pass
    return response.content[0].text.strip()


def generate_recommendations_batch(
    query: str,
    patterns: list[dict],
    client: OpenAI,
    top_n: int = 5,
) -> list[str]:
    """
    Generate recommendations in parallel. contextvars.copy_context() propagates
    the Langfuse trace context into each thread so spans nest correctly.
    Falls back to empty string on individual failures.
    """
    sliced = patterns[:top_n]
    if not sliced:
        return []

    # Each task needs its own Context copy — a single Context cannot be entered
    # by more than one thread at a time. All copies are made here in the parent
    # thread so they all inherit the active Langfuse trace context.
    task_ctxs = [contextvars.copy_context() for _ in sliced]

    def _run(ctx, pattern):
        return ctx.run(generate_recommendation, query, pattern, client)

    with ThreadPoolExecutor(max_workers=len(sliced)) as executor:
        futures = [executor.submit(_run, ctx, p) for ctx, p in zip(task_ctxs, sliced)]

    results = []
    for f in futures:
        try:
            results.append(f.result())
        except Exception as e:
            print(f"[recommendation] ERROR: {e}")
            results.append("")
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
    load_collection()  # returns (collection, patterns, embeddings) — unused in demo

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
    print("── Batch (top 5, sequential) ───────────────────────")
    t0   = time.perf_counter()
    recs = generate_recommendations_batch(query, results, openai_client, top_n=5)
    elapsed = time.perf_counter() - t0
    for p, r in zip(results, recs):
        print(f"  {p['name']}\n  → {r}")
    print(f"\n  Wall time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
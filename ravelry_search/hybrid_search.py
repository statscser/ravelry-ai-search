"""
hybrid_search.py — BM25 + vector retrieval fused with RRF (k=60).
Usage: python hybrid_search.py
"""

import json
import time
from pathlib import Path

import cohere
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

from rag_chroma import PatternSearchIntent, parse_query
from langfuse.decorators import observe, langfuse_context

load_dotenv()

PATTERNS_PATH   = Path(__file__).parent / "data" / "patterns.json"
EMBEDDINGS_PATH = Path(__file__).parent / "data" / "embeddings.npy"


# ── Text / tokenisation ────────────────────────────────────────────────────────

def _bm25_text(p: dict) -> str:
    name       = p.get("name") or ""
    craft      = (p.get("craft") or {}).get("name", "")
    yarn_wt    = (p.get("yarn_weight") or {}).get("name", "")
    categories = " ".join(c["name"] for c in (p.get("pattern_categories") or []))
    attributes = " ".join(a["permalink"] for a in (p.get("pattern_attributes") or []))
    return f"{name} {craft} {yarn_wt} {categories} {attributes}"


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


# ── Metadata filter (mirrors rag_chroma._build_where) ─────────────────────────

def _passes_filter(p: dict, intent: PatternSearchIntent) -> bool:
    if intent.craft:
        if (p.get("craft") or {}).get("name", "") != intent.craft:
            return False
    if intent.free_only and not p.get("free", False):
        return False
    if intent.min_rating > 0:
        if (p.get("rating_average") or 0.0) < intent.min_rating:
            return False
    if intent.yarn_weight:
        yw = (p.get("yarn_weight") or {}).get("name", "")
        if yw.lower() != intent.yarn_weight.lower():
            return False
    if intent.needle_size_min:
        sizes = [n.get("metric") for n in (p.get("pattern_needle_sizes") or []) if n.get("metric")]
        if not sizes or max(sizes) < intent.needle_size_min:
            return False
    if intent.needle_size_max:
        sizes = [n.get("metric") for n in (p.get("pattern_needle_sizes") or []) if n.get("metric")]
        if not sizes or min(sizes) > intent.needle_size_max:
            return False
    return True


# ── RRF ───────────────────────────────────────────────────────────────────────

def _rrf_merge(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Combine multiple ranked lists; returns (local_idx, rrf_score) sorted desc."""
    scores: dict[int, float] = {}
    for ranked in rankings:
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Public API ─────────────────────────────────────────────────────────────────

@observe(as_type="span", capture_input=False, capture_output=False)
def hybrid_search(
    query: str,
    patterns: list[dict],
    embeddings: np.ndarray,
    openai_client: OpenAI,
    top_k: int = 20,
    chroma_collection=None,           # unused — kept for API compatibility
    intent: PatternSearchIntent = None,
) -> list[dict]:
    """
    BM25 + cosine-vector retrieval fused with RRF (k=60).

    query      : text used for both BM25 and the query embedding
    intent     : if provided, applies the same metadata pre-filter as rag_chroma.search()
    Returns top_k pattern dicts augmented with _rrf_score, _bm25_rank, _vec_rank, _vec_sim.
    """
    langfuse_context.update_current_observation(input={"query": query, "top_k": top_k})
    # ── 1. Metadata pre-filter ────────────────────────────────────────────────
    if intent is not None:
        local_indices = [i for i, p in enumerate(patterns) if _passes_filter(p, intent)]
    else:
        local_indices = list(range(len(patterns)))

    if not local_indices:
        return []

    sub_patterns   = [patterns[i] for i in local_indices]
    sub_embeddings = embeddings[local_indices]   # shape (N_filtered, D)

    # ── 2. BM25 ranking ───────────────────────────────────────────────────────
    corpus      = [_tokenize(_bm25_text(p)) for p in sub_patterns]
    bm25        = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_ranked = list(np.argsort(bm25_scores)[::-1])   # local indices, best first

    # ── 3. Vector ranking ─────────────────────────────────────────────────────
    q_emb = np.array(
        openai_client.embeddings.create(
            input=[query], model="text-embedding-3-small"
        ).data[0].embedding
    )
    denom      = np.linalg.norm(sub_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-9
    sims       = np.dot(sub_embeddings, q_emb) / denom
    vec_ranked = list(np.argsort(sims)[::-1])            # local indices, best first

    # ── 4. RRF fusion ─────────────────────────────────────────────────────────
    fused = _rrf_merge([bm25_ranked, vec_ranked], k=60)

    bm25_rank_of = {sub_idx: rank + 1 for rank, sub_idx in enumerate(bm25_ranked)}
    vec_rank_of  = {sub_idx: rank + 1 for rank, sub_idx in enumerate(vec_ranked)}

    results = []
    for sub_idx, rrf_score in fused[:top_k]:
        p = sub_patterns[sub_idx].copy()
        p["_rrf_score"] = round(rrf_score, 6)
        p["_bm25_rank"] = bm25_rank_of[sub_idx]
        p["_vec_rank"]  = vec_rank_of[sub_idx]
        p["_vec_sim"]   = round(float(sims[sub_idx]), 4)
        results.append(p)

    return results


# ── Cohere rerank ─────────────────────────────────────────────────────────────

@observe(as_type="span", capture_input=False, capture_output=False)
def reranked_search(
    query: str,
    patterns: list[dict],
    embeddings: np.ndarray,
    openai_client: OpenAI,
    top_k: int = 10,
    intent: PatternSearchIntent = None,
) -> list[dict]:
    """
    Retrieve top-20 candidates via hybrid_search, then rerank with
    Cohere rerank-multilingual-v3.0. COHERE_API_KEY must be in .env.
    Returns top_k results augmented with _cohere_score (and hybrid fields).
    """
    langfuse_context.update_current_observation(input={"query": query, "top_k": top_k})
    candidates = hybrid_search(
        query=query,
        patterns=patterns,
        embeddings=embeddings,
        openai_client=openai_client,
        top_k=20,
        intent=intent,
    )
    if not candidates:
        return []

    co = cohere.ClientV2()   # reads COHERE_API_KEY from env
    documents = [p.get("text_for_embedding") or "" for p in candidates]

    # retry with backoff for rate limit
    for attempt in range(3):
        try:
            response = co.rerank(
                model="rerank-multilingual-v3.0",
                query=query,
                documents=documents,
                top_n=top_k,
            )
            break
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

    results = []
    for hit in response.results:
        p = candidates[hit.index].copy()
        p["_cohere_score"] = round(hit.relevance_score, 6)
        results.append(p)

    return results


# ── Demo ───────────────────────────────────────────────────────────────────────

def main():
    patterns   = json.loads(PATTERNS_PATH.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_PATH)
    client     = OpenAI()

    test_queries = ["cable knit sweater", "lace shawl", "5mm needle hat"]

    for query in test_queries:
        print(f"\n{'='*68}")
        print(f"Query : {query}")

        intent = parse_query(query, client)
        print(
            f"Intent: semantic='{intent.semantic_query}'  craft={intent.craft}  "
            f"yarn={intent.yarn_weight}  needle_min={intent.needle_size_min}  "
            f"free={intent.free_only}"
        )

        # ── hybrid
        hybrid = hybrid_search(
            query=intent.semantic_query,
            patterns=patterns,
            embeddings=embeddings,
            openai_client=client,
            top_k=5,
            intent=intent,
        )
        print(f"\n  [hybrid top 5]")
        for i, p in enumerate(hybrid, 1):
            print(
                f"  {i}. {p['name']:<46}"
                f"  RRF={p['_rrf_score']:.5f}"
                f"  BM25={p['_bm25_rank']:>4}"
                f"  vec={p['_vec_rank']:>4}"
                f"  sim={p['_vec_sim']:.4f}"
            )

        # ── reranked
        reranked = reranked_search(
            query=intent.semantic_query,
            patterns=patterns,
            embeddings=embeddings,
            openai_client=client,
            top_k=5,
            intent=intent,
        )
        print(f"\n  [reranked top 5]")
        for i, p in enumerate(reranked, 1):
            print(
                f"  {i}. {p['name']:<46}"
                f"  cohere={p['_cohere_score']:.5f}"
                f"  BM25={p['_bm25_rank']:>4}"
                f"  vec={p['_vec_rank']:>4}"
            )


if __name__ == "__main__":
    main()

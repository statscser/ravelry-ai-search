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

from typing import Any, Optional

from rag_chroma import PatternSearchIntent, parse_query
from langfuse.decorators import observe, langfuse_context

from constants import (
    EMBEDDING_MODEL, RRF_K, RERANK_CANDIDATES, COHERE_RERANK_MODEL,
    MAX_RETRIES, RETRY_WAIT_BASE_S,
)

load_dotenv()

PATTERNS_PATH   = Path(__file__).parent / "data" / "patterns.json"
EMBEDDINGS_PATH = Path(__file__).parent / "data" / "embeddings.npy"


# ── Text / tokenisation ────────────────────────────────────────────────────────

def _bm25_text(p: dict) -> str:
    """Build a lightweight text representation of a pattern for BM25 indexing.

    Concatenates name, craft, yarn weight, categories, and attributes so that
    keyword search can match on structured fields without full pattern text.
    """
    name       = p.get("name") or ""
    craft      = (p.get("craft") or {}).get("name", "")
    yarn_wt    = (p.get("yarn_weight") or {}).get("name", "")
    categories = " ".join(c["name"] for c in (p.get("pattern_categories") or []))
    attributes = " ".join(a["permalink"] for a in (p.get("pattern_attributes") or []))
    return f"{name} {craft} {yarn_wt} {categories} {attributes}"


def _tokenize(text: str) -> list[str]:
    """Lowercase and whitespace-split text into BM25 tokens."""
    return text.lower().split()


# ── Metadata filter (mirrors rag_chroma._build_where) ─────────────────────────

def _passes_filter(p: dict, intent: PatternSearchIntent) -> bool:
    """Return True if pattern p satisfies all hard filters in intent.

    Mirrors the Chroma ``$where`` clause so the same logic applies to the
    in-memory BM25/vector pipeline without hitting the vector store.
    """
    if intent.craft:
        if (p.get("craft") or {}).get("name", "") != intent.craft:
            return False
    if intent.free_only and not p.get("free", False):
        return False
    if intent.min_rating > 0:
        if (p.get("rating_average") or 0.0) < intent.min_rating:
            return False
    if intent.yarn_weight:
        yarn_weight_name = (p.get("yarn_weight") or {}).get("name", "")
        if yarn_weight_name.lower() != intent.yarn_weight.lower():
            return False
    # Compute needle sizes once; used by both min and max checks below.
    needle_sizes = (
        [n.get("metric") for n in (p.get("pattern_needle_sizes") or []) if n.get("metric")]
        if intent.needle_size_min or intent.needle_size_max
        else []
    )
    if intent.needle_size_min:
        if not needle_sizes or max(needle_sizes) < intent.needle_size_min:
            return False
    if intent.needle_size_max:
        if not needle_sizes or min(needle_sizes) > intent.needle_size_max:
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

def _cohere_rerank_with_retry(
    cohere_client: cohere.ClientV2,
    query: str,
    documents: list[str],
    top_k: int,
) -> Any:
    """Call Cohere rerank with exponential backoff on 429 rate limits.

    Args:
        cohere_client: Initialised Cohere v2 client.
        query: Search query passed to the reranker.
        documents: Candidate document strings to rerank.
        top_k: Number of top results to return.

    Returns:
        Cohere RerankResponse whose ``.results`` list is ordered by relevance.

    Raises:
        Exception: Re-raises any non-429 error or a 429 after all retries exhausted.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return cohere_client.rerank(
                model=COHERE_RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=top_k,
            )
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                wait = RETRY_WAIT_BASE_S * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


@observe(as_type="span", capture_input=False, capture_output=False)
def hybrid_search(
    query: str,
    patterns: list[dict],
    embeddings: np.ndarray,
    openai_client: OpenAI,
    top_k: int = 20,
    chroma_collection=None,           # unused — kept for API compatibility
    intent: Optional[PatternSearchIntent] = None,
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
    query_embedding = np.array(
        openai_client.embeddings.create(
            input=[query], model=EMBEDDING_MODEL
        ).data[0].embedding
    )
    # Cosine similarity: dot(A, b) / (‖A‖ * ‖b‖); +1e-9 guards against zero norms.
    cosine_norms = np.linalg.norm(sub_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-9
    cosine_sims  = np.dot(sub_embeddings, query_embedding) / cosine_norms
    vec_ranked   = list(np.argsort(cosine_sims)[::-1])   # local indices, best first

    # ── 4. RRF fusion ─────────────────────────────────────────────────────────
    fused = _rrf_merge([bm25_ranked, vec_ranked], k=RRF_K)

    bm25_rank_of = {sub_idx: rank + 1 for rank, sub_idx in enumerate(bm25_ranked)}
    vec_rank_of  = {sub_idx: rank + 1 for rank, sub_idx in enumerate(vec_ranked)}

    results = []
    for sub_idx, rrf_score in fused[:top_k]:
        p = sub_patterns[sub_idx].copy()
        p["_rrf_score"] = round(rrf_score, 6)
        p["_bm25_rank"] = bm25_rank_of[sub_idx]
        p["_vec_rank"]  = vec_rank_of[sub_idx]
        p["_vec_sim"]   = round(float(cosine_sims[sub_idx]), 4)
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
    intent: Optional[PatternSearchIntent] = None,
) -> list[dict]:
    """Retrieve RERANK_CANDIDATES via hybrid_search, then rerank with Cohere.

    Args:
        query: Search query string.
        patterns: Full pattern corpus loaded from patterns.json.
        embeddings: Pre-computed numpy embedding matrix (shape N×D).
        openai_client: OpenAI client used for query embedding.
        top_k: Final number of results to return after reranking.
        intent: Optional parsed filters; passed through to hybrid_search.

    Returns:
        Up to top_k pattern dicts augmented with ``_cohere_score`` and hybrid fields.
    """
    langfuse_context.update_current_observation(input={"query": query, "top_k": top_k})
    candidates = hybrid_search(
        query=query,
        patterns=patterns,
        embeddings=embeddings,
        openai_client=openai_client,
        top_k=RERANK_CANDIDATES,
        intent=intent,
    )
    if not candidates:
        return []

    cohere_client = cohere.ClientV2()   # reads COHERE_API_KEY from env
    documents     = [p.get("text_for_embedding") or "" for p in candidates]
    response      = _cohere_rerank_with_retry(cohere_client, query, documents, top_k)

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

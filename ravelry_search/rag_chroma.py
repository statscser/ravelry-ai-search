import json
import os
from pathlib import Path

import anthropic
import chromadb
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

import instructor
from pydantic import BaseModel
from typing import Optional, Literal

from langfuse import observe, get_client

load_dotenv()

PATTERNS_PATH = Path(__file__).parent / "data" / "patterns.json"
EMBEDDINGS_PATH = Path(__file__).parent / "data" / "embeddings.npy"
COLLECTION_NAME = "ravelry_patterns"


def _safe(value, default=""):
    """Return value if not None, else default — Chroma rejects None in metadata."""
    return value if value is not None else default

class PatternSearchIntent(BaseModel):
    semantic_query: str
    craft: Optional[Literal["Knitting", "Crochet"]] = None
    yarn_weight: Optional[str] = None
    needle_size_min: Optional[float] = None
    needle_size_max: Optional[float] = None
    free_only: bool = False
    min_rating: float = 0.0
    exclude_fibers: list[str] = []
    include_fibers: list[str] = []
    categories: Optional[str] = None

SYSTEM_PROMPT = """
You are a knitting pattern search assistant. Parse the user's query into structured search intent.

Rules:
- semantic_query: style and aesthetic description. 
  IMPORTANT: if include_fibers is detected, append those fiber names to semantic_query 
  (e.g. "mohair cardigan" → semantic_query="cardigan mohair", include_fibers=["mohair"])
  Keep words like "lightweight", "oversized", "cozy", "chunky", "colorful".
  Remove needle size and yarn weight specs since those go in dedicated fields.
  Do NOT include excluded fibers in semantic_query.

- include_fibers: fibers the user WANTS. Also append these to semantic_query.
- exclude_fibers: fibers the user does NOT want (look for "no", "not", "without", 
  "avoid", "不用", "不要"). Do NOT add these to semantic_query.
- needle_size_min / needle_size_max: extract mm values
- yarn_weight: normalize to: Lace, Fingering, Sport, DK, Worsted, Bulky, Super Bulky
- craft: "crochet"/"hook"/"钩针" → Crochet, "knit"/"needle"/"棒针" → Knitting
- categories: Hat, Sweater, Cardigan, Sock, Shawl, Blanket, Mittens, Cowl, Top, Vest

Examples:
- "mohair cardigan needles > 5mm" 
  → semantic_query="cardigan mohair", include_fibers=["mohair"], needle_size_min=5.0
- "粗线粗针但轻盈不用马海毛的毛衣" 
  → semantic_query="lightweight chunky sweater", exclude_fibers=["mohair","马海毛"], yarn_weight="Bulky"
- "cotton linen summer top" 
  → semantic_query="summer top cotton linen", include_fibers=["cotton","linen"]
- "no mohair cozy sweater" 
  → semantic_query="cozy sweater", exclude_fibers=["mohair"]
"""

_parse_cache: dict[str, PatternSearchIntent] = {}


@observe(as_type="generation")
def parse_query(query: str, client: OpenAI) -> PatternSearchIntent:
    instructor_client = instructor.from_anthropic(anthropic.Anthropic())
    response = instructor_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        response_model=PatternSearchIntent,
        messages=[{"role": "user", "content": query}]
    )
    try:
        raw_usage = response._raw_response.usage
        get_client().update_current_generation(
            usage_details={"input": raw_usage.input_tokens, "output": raw_usage.output_tokens}
        )
    except Exception:
        pass
    return response

@observe(as_type="span", name="parse_query_cached")
def parse_query_cached(query: str, client: OpenAI) -> PatternSearchIntent:
    key = query.strip().lower()
    cache_hit = key in _parse_cache
    if not cache_hit:
        _parse_cache[key] = parse_query(query, client)
    try:
        get_client().update_current_span(metadata={"cache_hit": cache_hit})
    except Exception:
        pass
    return _parse_cache[key]

def build_metadata(pattern: dict) -> dict:
    """Extract filterable metadata fields from a pattern dict."""
    yarn_weight_name = (pattern.get("yarn_weight") or {}).get("name", "")
    needle_sizes_metric = [
        n["metric"] for n in (pattern.get("pattern_needle_sizes") or [])
        if n.get("metric")
    ]
    return {
        "name": _safe(pattern.get("name")),
        "craft": _safe(pattern.get("craft", {}) or {}).get("name", "") if pattern.get("craft") else "",
        "yarn_weight_description": _safe(pattern.get("yarn_weight_description")),
        "yarn_weight_name": yarn_weight_name.lower(),
        "needle_sizes": ", ".join(
            n["name"] for n in (pattern.get("pattern_needle_sizes") or [])
        ),
        "needle_size_min": min(needle_sizes_metric) if needle_sizes_metric else 0.0,
        "needle_size_max": max(needle_sizes_metric) if needle_sizes_metric else 0.0,
        "categories": ", ".join(
            c["name"] for c in (pattern.get("pattern_categories") or [])
        ),
        "attributes": ", ".join(
            a["permalink"] for a in (pattern.get("pattern_attributes") or [])
        ),
        "free": 1 if pattern.get("free") else 0,
        "rating_average": float(_safe(pattern.get("rating_average"), 0.0)),
        "permalink": _safe(pattern.get("permalink")),
    }

def _build_where(intent: PatternSearchIntent) -> dict | None:
    if intent is None:
        return None
    conditions = []
    if intent.craft:
        conditions.append({"craft": {"$eq": intent.craft}})
    if intent.free_only:
        conditions.append({"free": {"$eq": 1}})
    if intent.min_rating > 0:
        conditions.append({"rating_average": {"$gte": intent.min_rating}})
    if intent.yarn_weight:
        conditions.append({"yarn_weight_name": {"$eq": intent.yarn_weight.lower()}})
    if intent.needle_size_min:
        conditions.append({"needle_size_max": {"$gte": intent.needle_size_min}})
    if intent.needle_size_max:
        conditions.append({"needle_size_min": {"$lte": intent.needle_size_max}})
        
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}

def load_collection() -> tuple[chromadb.Collection, list[dict]]:
    """Load patterns and pre-computed embeddings into an in-memory Chroma collection.
    Returns both the collection and the original patterns list for full data access.
    """
    patterns = json.loads(PATTERNS_PATH.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_PATH)

    client = chromadb.Client()
    if COLLECTION_NAME in {c.name for c in client.list_collections()}:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids = [str(p["id"]) for p in patterns]
    documents = [p.get("text_for_embedding", "") for p in patterns]
    metadatas = [build_metadata(p) for p in patterns]
    embedding_list = embeddings.tolist()

    CHUNK = 500
    for start in range(0, len(ids), CHUNK):
        collection.add(
            ids=ids[start : start + CHUNK],
            documents=documents[start : start + CHUNK],
            metadatas=metadatas[start : start + CHUNK],
            embeddings=embedding_list[start : start + CHUNK],
        )

    print(f"Loaded {collection.count()} patterns into Chroma.")
    return collection, patterns


def search(
    query: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    patterns: list[dict],
    top_k: int = 5,
    intent: PatternSearchIntent = None,
) -> list[dict]:
    """
    Embed query with OpenAI, query Chroma with optional metadata filters.
    Returns list of full pattern dicts for the top_k results.
    """
    query_embedding = (
        openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small",
        )
        .data[0]
        .embedding
    )

    where = _build_where(intent)
    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances", "documents"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    # 用 id 查完整 pattern 数据
    id_to_pattern = {str(p["id"]): p for p in patterns}
    result_ids = results["ids"][0]
    distances = results["distances"][0]

    full_patterns = []
    for pid, dist in zip(result_ids, distances):
        pattern = id_to_pattern[pid].copy()
        pattern["_similarity"] = round(1 - dist, 4)
        full_patterns.append(pattern)

    return full_patterns


def main():
    openai_client = OpenAI()
    collection, patterns = load_collection()  # 解包 tuple

    query = input("\n搜索编织图解：").strip()

    intent = parse_query(query, openai_client)
    print(f"\n📋 解析意图：")
    print(f"   语义搜索：{intent.semantic_query}")
    print(f"   工艺：{intent.craft or '不限'}")
    print(f"   只看免费：{'是' if intent.free_only else '否'}")
    print(f"   最低评分：{intent.min_rating or '不限'}")

    results = search(
        query=intent.semantic_query,
        collection=collection,
        openai_client=openai_client,
        patterns=patterns,
        top_k=5,
        intent=intent,
    )

    print(f"\n--- 搜索结果（共 {len(results)} 条）---\n")
    for i, p in enumerate(results, 1):
        free_label = "免费" if p.get("free") else "付费"
        photo = (p.get("photos") or [{}])[0]
        print(f"{i}. {p['name']}")
        print(f"   {p.get('craft', {}).get('name', '')} · {p.get('yarn_weight_description', '')} · {free_label}")
        print(f"   评分：{p.get('rating_average', 0):.1f}  相似度：{p['_similarity']}")
        print(f"   图片：{photo.get('small_url', '无')}")
        print(f"   链接：https://www.ravelry.com/patterns/library/{p['permalink']}")
        print()

if __name__ == "__main__":
    main()

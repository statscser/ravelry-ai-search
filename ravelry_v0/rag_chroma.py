import json
import os
from pathlib import Path

import chromadb
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

import instructor
from pydantic import BaseModel
from typing import Optional, Literal

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
    free_only: bool = False
    min_rating: float = 0.0

def parse_query(query: str, client: OpenAI) -> PatternSearchIntent:
    instructor_client = instructor.from_openai(client)
    return instructor_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=500,
        response_model=PatternSearchIntent,
        messages=[{"role": "user", "content": query}]
    )

def build_metadata(pattern: dict) -> dict:
    """Extract filterable metadata fields from a pattern dict."""
    return {
        "name": _safe(pattern.get("name")),
        "craft": _safe(pattern.get("craft", {}) or {}).get("name", "") if pattern.get("craft") else "",
        "yarn_weight_description": _safe(pattern.get("yarn_weight_description")),
        "needle_sizes": ", ".join(
            n["name"] for n in (pattern.get("pattern_needle_sizes") or [])
        ),
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


def load_collection() -> chromadb.Collection:
    """Load patterns and pre-computed embeddings into an in-memory Chroma collection."""
    patterns = json.loads(PATTERNS_PATH.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_PATH)

    client = chromadb.Client()
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids = [str(p["id"]) for p in patterns]
    documents = [p.get("text_for_embedding", "") for p in patterns]
    metadatas = [build_metadata(p) for p in patterns]
    embedding_list = embeddings.tolist()

    # Chroma add() has a hard limit per call; batch in chunks of 500
    CHUNK = 500
    for start in range(0, len(ids), CHUNK):
        collection.add(
            ids=ids[start : start + CHUNK],
            documents=documents[start : start + CHUNK],
            metadatas=metadatas[start : start + CHUNK],
            embeddings=embedding_list[start : start + CHUNK],
        )

    print(f"Loaded {collection.count()} patterns into Chroma.")
    return collection


def _build_where(
    craft: str | None,
    free_only: bool,
    min_rating: float,
) -> dict | None:
    conditions = []
    if craft:
        conditions.append({"craft": {"$eq": craft}})
    if free_only:
        conditions.append({"free": {"$eq": 1}})
    if min_rating > 0:
        conditions.append({"rating_average": {"$gte": min_rating}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def search(
    query: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    top_k: int = 5,
    craft: str | None = None,
    free_only: bool = False,
    min_rating: float = 0.0,
) -> list[dict]:
    """
    Embed query with OpenAI, query Chroma with optional metadata filters.
    Returns list of metadata dicts for the top_k results.
    """
    query_embedding = (
        openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small",
        )
        .data[0]
        .embedding
    )

    where = _build_where(craft, free_only, min_rating)
    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances", "documents"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Attach cosine similarity (Chroma cosine space returns distance = 1 - similarity)
    for meta, dist in zip(metadatas, distances):
        meta["_similarity"] = round(1 - dist, 4)

    return metadatas


def main():
    openai_client = OpenAI()
    collection = load_collection()

    query = input("\n搜索编织图解：").strip()

    # 自动解析 query 里的过滤条件
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
        top_k=5,
        craft=intent.craft,
        free_only=intent.free_only,
        min_rating=intent.min_rating,
    )

    print(f"\n--- 搜索结果（共 {len(results)} 条）---\n")
    for i, meta in enumerate(results, 1):
        free_label = "免费" if meta["free"] else "付费"
        print(f"{i}. {meta['name']}")
        print(f"   {meta['craft']} · {meta['yarn_weight_description']} · {free_label}")
        print(f"   分类：{meta['categories']}")
        print(f"   评分：{meta['rating_average']:.1f}  相似度：{meta['_similarity']}")
        print(f"   链接：https://www.ravelry.com/patterns/library/{meta['permalink']}")
        print()

if __name__ == "__main__":
    main()

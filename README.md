# 🧶 Ravelry AI Search

Semantic search for Ravelry knitting patterns. Type natural language queries like *"cozy cable knit sweater in DK weight"* or *"free beginner crochet hat"* and get ranked results with AI-generated recommendations.

**Why this exists:** Ravelry's built-in search is keyword-only. It can't handle semantic queries, negative conditions ("no mohair"), cross-language search (Chinese → English patterns), or combined filters extracted from natural language.

---

## Demo

> *"粗线粗针但轻盈不用马海毛的毛衣"* → finds bulky knit sweaters, excludes mohair, ranks by relevance

> *"free DK cardigan rating above 4.5"* → applies free filter + rating filter + semantic search simultaneously

---

## Architecture

```
User query (natural language)
        │
        ▼
┌───────────────────────┐
│   Query Understanding  │  GPT-4o-mini + instructor + Pydantic
│   PatternSearchIntent  │  extracts: semantic_query, craft, yarn_weight,
│                        │  needle_size, free_only, min_rating,
│                        │  include/exclude_fibers, categories
└───────────┬───────────┘
            │ semantic_query + structured filters
            ▼
┌───────────────────────┐
│    Hybrid Search       │  BM25 (rank_bm25) + Vector (text-embedding-3-small)
│    BM25 + Vector + RRF │  fused with Reciprocal Rank Fusion (k=60)
│                        │  metadata pre-filter applied before retrieval
└───────────┬───────────┘
            │ top 20 candidates
            ▼
┌───────────────────────┐
│   Cohere Reranker      │  rerank-multilingual-v3.0 (Cross-Encoder)
│                        │  supports Chinese ↔ English cross-lingual reranking
└───────────┬───────────┘
            │ top 10 results (score ≥ 0.3)
            ▼
┌───────────────────────┐
│  GPT-4o-mini Recs      │  one-sentence recommendation for top 3 results
│  + Streamlit UI        │  sort by relevance / rating / favorites
└───────────────────────┘
```

**Two-layer storage design:**
- **Offline** (generated once): `patterns.json` + `embeddings.npy`
- **Runtime** (loaded at startup): Chroma in-memory collection with metadata for fast filtering

---

## Eval Results

Evaluated on a hand-annotated golden set of 20 queries across 4 categories (基础语义 / filter验证 / 细分特征 / 边缘案例).

| Version | Change | MRR@20 | P@10 | Notes |
|---------|--------|--------|------|-------|
| v0 baseline | Vector search only | 0.849 | 0.490 | Pure cosine similarity |
| v1 hybrid | + BM25 + RRF | 0.774 | 0.510 | BM25 improves recall, hurts ranking |
| v2 reranked | + Cohere reranker | 0.828 | **0.585** | **+9.5pp P@10 vs baseline** |

**Key insight:** BM25 increases candidate recall (finds more relevant patterns) but degrades ranking (exact keyword match ≠ semantic relevance). The Cohere Cross-Encoder restores ranking quality while keeping the recall gains — net result is higher precision across the board.

**Eval methodology:** Golden set annotated using three candidate sources (vector search + keyword search + structured field query) to avoid annotation bias. Binary relevance labels (relevant / not relevant), borderline excluded. Precision@10 as primary metric due to variable ground-truth set sizes.

---

## Key Technical Decisions

**Why Hybrid Search instead of pure vector?**
Exact terms like needle sizes (`5mm`), yarn weights (`DK`), and technique names (`cables`, `lace`) get diluted in dense embeddings. BM25 handles precise keyword matching; vector search handles semantic understanding. RRF combines both without requiring score normalization.

**Why Cohere Cross-Encoder instead of vector similarity for ranking?**
Bi-Encoders embed query and document independently — they can't model word-level interactions between the two. The Cross-Encoder sees query + document together, enabling it to understand combinations like "cable AND sweater" (not just either). In testing, "Sweater With Cables" ranked #131 in BM25 and was retrieved via vector search — the reranker correctly surfaced it to #2.

**Why PatternSearchIntent instead of raw query embedding?**
Negative conditions like "no mohair" are nearly invisible in query embeddings. Structured extraction puts `exclude_fibers` in a dedicated field, passed explicitly to the reranker query. Include fibers are appended to `semantic_query` so vector retrieval can find them.

---

## Limitations

| Issue | Root Cause | Plan |
|-------|-----------|------|
| Needle size filter edge cases | Low-quality candidates after size filtering | Migrate to pgvector for SQL range queries |
| `exclude_fibers` relies on semantics only | Fiber composition not in Ravelry pattern API (requires separate yarn API calls) | Batch-fetch yarn fiber data in v3 |
| Cohere score threshold may filter too many results | Fixed threshold (0.3) is query-agnostic | Dynamic threshold based on score distribution |
| Ravelry attribute annotations incomplete | Some patterns missing `cables` / `lace` attributes despite having them | Boost pattern name weight in `text_for_embedding` |
| Chroma `$contains` unsupported | In-memory Chroma version limitation | Migrate to pgvector for flexible metadata filtering |

---

## Setup

### 1. Install dependencies

```bash
cd ravelry_v0
pip install -r requirements.txt
# or with uv:
uv sync
```

### 2. Configure API keys

Create `ravelry_v0/.env`:

```
RAVELRY_USERNAME=your_ravelry_username
RAVELRY_PASSWORD=your_ravelry_api_password
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
```

Get Ravelry API credentials at `ravelry.com/pro/developer` (free, non-commercial use).

### 3. Generate data

**Step 1 — Collect patterns** (~8,000 patterns with notes + photos, ~40 min):

```bash
python collect_data.py
```

Writes `data/patterns.json`. Filters out patterns without notes or cover photos.
Each pattern includes a pre-built `text_for_embedding` field.

**Step 2 — Generate embeddings** (OpenAI `text-embedding-3-small`, ~$0.10):

```bash
python rag.py
```

Writes `data/embeddings.npy`. Skips if file already exists.

**Step 3 — (Optional) Rebuild text fields** after changing `data_processor.py`:

```bash
python rebuild_text.py
```

**Step 4 — (Optional) Run evaluation**:

```bash
python eval.py --version v2_reranked
```

Requires Cohere API key. Adds 7s sleep between queries for free-tier rate limit.
Results written to `data/eval_results_v2_reranked.json`.

### 4. Run the app

```bash
streamlit run app.py
```

---

## Cost & Latency

| Component | Cost per search | Latency |
|-----------|----------------|---------|
| Query parsing (GPT-4o-mini) | ~$0.0001 | ~0.5s |
| Query embedding | ~$0.000001 | ~0.2s |
| Hybrid search (BM25 + vector) | $0 | ~0.3s |
| Cohere rerank | $0 (free tier) | ~0.5s |
| Recommendations (GPT-4o-mini, top 3) | ~$0.0002 | ~1-2s (parallel) |
| **Total** | **~$0.0003** | **~5-10s** |

---

## Data Files

`data/golden_set.json` — manually annotated evaluation set (tracked in git).

All other `data/` files are excluded from git (too large) and must be generated locally:

| File | Size | Generated by |
|------|------|-------------|
| `data/patterns.json` | ~170 MB | `collect_data.py` |
| `data/embeddings.npy` | ~94 MB | `rag.py` |
| `data/eval_results_*.json` | ~4 KB each | `eval.py` |

---

## Stack

- **Retrieval:** OpenAI `text-embedding-3-small`, `rank_bm25`, Chroma (in-memory)
- **Reranking:** Cohere `rerank-multilingual-v3.0`
- **Query Understanding:** `instructor` + Pydantic + GPT-4o-mini
- **Recommendations:** GPT-4o-mini
- **UI:** Streamlit
- **Data:** Ravelry REST API (Basic Auth), 8,000 patterns across 12 categories
# 🧶 Ravelry AI Search

> **[🔍 Live Demo](https://ravelry-search-372988574601.us-central1.run.app)**
> *(First load takes ~30s — Cloud Run cold start while loading 30K patterns)*

A semantic search engine for knitting patterns, built to solve a real problem:
Ravelry has 1M+ patterns but keyword-only search. This project adds natural
language understanding, cross-lingual search, negative condition handling,
and a full eval framework — built and iterated end-to-end as a portfolio project.

**Why I built this:** Ravelry's search can't handle queries like
*"bulky sweater, no mohair"* or *"夏天钩针背心"*. This does.

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
│   Query Understanding  │  claude-haiku-4-5 + instructor + Pydantic
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
│  claude-haiku-4-5 Recs │  one-sentence recommendation for top 3 results
│  + Streamlit UI        │  sort by relevance / rating / favorites
└───────────────────────┘
```

**Two-layer storage design:**
- **Offline** (generated once): `patterns.json` + `embeddings.npy`
- **Runtime** (loaded at startup): Chroma in-memory collection with metadata for fast filtering

---

## Eval Results

Evaluated on a hand-annotated golden set of 20 queries across 4 categories (基础语义 / filter验证 / 细分特征 / 边缘案例).

| Version | Change | MRR@20 | P@10 | Recall@10 | Notes |
|---------|--------|--------|------|-----------|-------|
| v0 baseline | Vector search only | 0.849 | 0.490 | 0.303 | Pure cosine similarity |
| v1 hybrid | + BM25 + RRF | 0.774 | 0.510 | 0.278 | Precision ticked up, but recall and ranking both got worse |
| v2 reranked | + Cohere reranker | 0.828 | **0.585** | **0.338** | **+19.4% relative P@10, +11.6% relative recall vs baseline** |

**Key insight:** The hybrid-search hypothesis was that BM25 would boost recall on exact-term queries (needle sizes, yarn weights). The data said otherwise: Recall@10 actually *dropped* (0.303 → 0.278) and ranking quality fell (MRR@20 0.849 → 0.774) — BM25's keyword matches weren't the same patterns vector search was missing, so RRF fusion just reshuffled results rather than surfacing more relevant ones. The Cohere Cross-Encoder reranking pass fixed both problems at once: P@10 rose to 0.585 and Recall@10 rose to 0.338, the best of all three versions on both metrics.

**Eval methodology:** Golden set annotated using three candidate sources (vector search + keyword search + structured field query) to avoid annotation bias. Binary relevance labels (relevant / not relevant), borderline excluded. Precision@10 as primary metric due to variable ground-truth set sizes.

---

## Optimization Journey

The project went through three measurable iterations:

1. **v0 — baseline** (pure vector search): Established eval framework first —
   20-query golden set with binary relevance labels across 4 query categories.
   MRR@20=0.849, P@10=0.490, Recall@10=0.303.

2. **v1 — hybrid search**: Added BM25 + RRF fusion. Hypothesis: exact terms
   like "5mm needle" get diluted in embeddings. Result: precision ticked up
   (P@10 +2pp) but recall actually *dropped* (Recall@10 0.303 → 0.278) and
   ranking degraded (MRR@20 0.849 → 0.774) — BM25 keyword match ≠ semantic
   relevance.

3. **v2 — reranking**: Added Cohere Cross-Encoder on top of hybrid candidates.
   The reranker sees query + document together, nearly recovering baseline
   ranking quality (MRR@20 0.828) while lifting precision **and** recall
   together: P@10 0.585 (+19.4% relative vs baseline), Recall@10 0.338
   (+11.6% relative vs baseline).

This iterative, eval-driven approach — measure first, optimize second — is the
core engineering discipline this project was built to practice.

---

## Cost Optimization

| Change | Before | After | Saving |
|--------|--------|-------|--------|
| Query parsing model | gpt-4o-mini ($0.0001) | claude-haiku ($0.00003) | -70% |
| Recommendation model | gpt-4o-mini ($0.0002) | claude-haiku ($0.00006) | -70% |
| Exact query cache | no cache | cache hit = $0 | -100% on repeat queries |
| **Total per search** | **~$0.0003** | **~$0.00009** | **-70%** |

Switching decision was data-driven: ran haiku vs gpt-4o-mini on the 20-query golden set, both achieved 100% parsing accuracy, confirming haiku was safe to adopt.

---

## Production Features

- **Observability:** Full Langfuse tracing across parse → retrieve → rerank → recommend pipeline. Token usage and cost tracked per span. Admin dashboard shows daily search count, avg cost, P50/P95 latency, top queries.
- **Guardrails:** Keyword-based off-topic detection (no LLM cost). Three-tier fallback: relax filters → top-rated patterns → user guidance.
- **Caching:** Exact-match query cache for `parse_query` — repeated searches return instantly at $0 cost.
- **Memory optimization:** Profiled startup with `debug_memory.py`
  (psutil + background sampling thread). Found two OOM root causes:
  (1) `embeddings.tolist()` converting full 30K array at once (+1,769MB peak),
  fixed with batched processing (BATCH_SIZE=1000);
  (2) `embeddings.npy` loaded twice independently, fixed by returning embeddings
  from `load_collection()`. Peak RSS reduced from 4,137MB → 3,804MB.
- **Deployment:** Dockerized, deployed on Google Cloud Run. Auto-scales to zero when idle.

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
cd ravelry_search
pip install -r requirements.txt
# or with uv:
uv sync
```

### 2. Configure API keys

Create `ravelry_search/.env`:

```
RAVELRY_USERNAME=your_ravelry_username
RAVELRY_PASSWORD=your_ravelry_api_password
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Get Ravelry API credentials at `ravelry.com/pro/developer` (free, non-commercial use).

### 3. Generate data

**Step 1 — Collect patterns** (~29,000 patterns with notes + photos, ~3 hours):

```bash
python collect_data.py
```

Writes `data/patterns.json`. Filters out patterns without notes or cover photos.
Each pattern includes a pre-built `text_for_embedding` field.

**Step 2 — Generate embeddings** (OpenAI `text-embedding-3-small`, ~$0.30):

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

## Run with Docker

Data files are baked into the image. Build and run:

```bash
cd ravelry_search
docker compose up --build
```

Open http://localhost:8501.

---

## Cost & Latency

| Component | Cost per search | Latency |
|-----------|----------------|---------|
| Query parsing (claude-haiku) | ~$0.00003 | ~0.5s |
| Query embedding | ~$0.000001 | ~0.2s |
| Hybrid search (BM25 + vector) | $0 | ~0.3s |
| Cohere rerank | $0 (free tier) | ~0.5s |
| Recommendations (claude-haiku, top 3, parallel) | ~$0.00006 | ~1-2s |
| **Total** | **~$0.00009** | **~6-7s avg** |
| **Cache hit** | **$0** | **<0.1s** |

Note: 70% cost reduction vs original gpt-4o-mini implementation ($0.0003 → $0.00009) achieved by switching to claude-haiku and adding exact-match query caching.

---

## Data Files

`data/golden_set.json` — manually annotated evaluation set (tracked in git).

All other `data/` files are excluded from git (too large) and must be generated locally:

| File | Size | Generated by |
|------|------|-------------|
| `data/patterns.json` | ~600 MB | `collect_data.py` |
| `data/embeddings.npy` | ~320 MB | `rag.py` |
| `data/eval_results_*.json` | ~4 KB each | `eval.py` |

---

## Stack

- **Retrieval:** OpenAI `text-embedding-3-small`, `rank_bm25`, Chroma (in-memory)
- **Reranking:** Cohere `rerank-multilingual-v3.0`
- **Query Understanding:** `instructor` + Pydantic + `claude-haiku-4-5`
- **Recommendations:** `claude-haiku-4-5` (parallel, top 3)
- **Observability:** Langfuse (tracing, cost tracking, admin dashboard)
- **Deployment:** Docker, Google Cloud Run
- **UI:** Streamlit
- **Data:** Ravelry REST API (Basic Auth), ~29,000 patterns across 12 categories
- **Profiling:** psutil + custom `debug_memory.py` for startup memory analysis

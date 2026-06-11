# Ravelry AI Search — Design Decisions

This document records the key technical decisions made during development,
including the reasoning and trade-offs behind each choice.

---

## 1. Why Hybrid Search instead of pure vector search?

Pure vector search encodes queries and documents into dense embeddings, which
works well for semantic similarity but loses precision on exact terms. In
knitting search, users often specify exact technical attributes:

- Needle sizes: "5mm needle" → BM25 matches exactly; vector search dilutes this
- Yarn weights: "DK weight" → exact keyword matters
- Technique names: "cables", "lace" → BM25 finds these reliably

**Result:** BM25 + vector + RRF fusion (k=60) improved P@10 from 0.490 to
0.510 (+2pp) in the hybrid step. The Cohere reranker then restored ranking
quality while keeping the recall gains, bringing P@10 to 0.585 (+9.5pp vs
baseline).

The key insight: BM25 improves *recall* (finds more relevant patterns), but
degrades *ranking* (exact keyword match ≠ semantic relevance). The
Cross-Encoder reranker sees query + document together, restoring ranking
quality. Example: "Sweater With Cables" ranked #131 in BM25 alone — the
reranker correctly surfaced it to #2.

---

## 2. Why Cohere Cross-Encoder instead of LLM-based reranking?

Two options for reranking top-20 candidates:

**Option A: LLM reranker** — prompt a large model with all 20 candidates.
- Latency: ~5-10s (large context)
- Cost: ~$0.005 per search
- Quality: good but slow

**Option B: Cohere Cross-Encoder** — dedicated reranking model, sees
query + document jointly.
- Latency: ~0.5s
- Cost: $0.001/1000 calls (effectively free at demo scale)
- Quality: comparable to LLM reranker for this task
- Multilingual: rerank-multilingual-v3.0 handles Chinese ↔ English natively

**Decision:** Cohere reranker is 10-20x faster and ~5x cheaper. The
Cross-Encoder architecture captures word-level query-document interactions
that bi-encoders miss.

---

## 3. Why PatternSearchIntent (structured query understanding)?

Without query understanding, the raw user query goes directly to retrieval.
Structured extraction via PatternSearchIntent unlocks:

1. **Negative conditions**: "不用马海毛" → `exclude_fibers=["mohair"]`
   applied as post-retrieval filter. Without this, negations are nearly
   invisible in embeddings.
2. **Metadata pre-filtering**: `free_only`, `min_rating`, `yarn_weight`
   applied before retrieval, improving precision.
3. **Semantic query separation**: "free DK cardigan rating above 4.5" →
   `semantic_query="cardigan"` + structured filters. Cleaner embedding.
4. **Cross-lingual support**: "夏天钩针背心" → `semantic_query="summer
   crochet vest"` → matches English patterns correctly.

Quantified on 20-query golden set: v2 with query understanding achieved
**P@10 = 0.585** vs **0.490** for pure vector — **+19.4% relative improvement**.

---

## 4. How negative conditions like "no mohair" are handled

This is the most technically interesting part of the query understanding layer.

**The problem:** Negative conditions are nearly invisible in dense embeddings.
"bulky sweater no mohair" and "bulky sweater with mohair" produce very similar
embeddings — the negation gets lost.

**The solution — three-layer approach:**
User query: "粗线粗针但轻盈不用马海毛的毛衣"
↓
Layer 1: PatternSearchIntent extraction (claude-haiku + instructor)
semantic_query = "lightweight chunky sweater"  ← mohair excluded
exclude_fibers = ["mohair", "马海毛"]
yarn_weight = "Bulky"
↓
Layer 2: semantic_query sent to hybrid search
→ finds bulky, lightweight sweaters
→ mohair patterns may still appear in results
↓
Layer 3: Post-retrieval filter
→ patterns mentioning excluded fibers removed before reranking

**Limitation:** Fiber content comes from free-text `notes` field, not
structured metadata. Filter uses string matching, which catches most cases
but may miss implied fiber content.

**Future:** Batch-fetch yarn fiber data from Ravelry's yarn API and store
as structured metadata for exact filtering.

---

## 5. QPS capacity and scaling plan

**Current architecture (single Cloud Run instance):**

Each search takes ~6-7s end-to-end:
- parse_query (claude-haiku): ~0.5s
- hybrid search (BM25 + vector): ~0.3s
- Cohere rerank: ~0.5s
- recommendations × 3 (parallel): ~1-2s

Effective throughput: ~1-2 QPS with Streamlit's threading.

**Bottleneck:** Recommendation generation (3 parallel haiku calls).
Could be made optional or async to reduce perceived latency.

**Scaling to 1M patterns:**

| Component | Current | At 1M patterns |
|-----------|---------|----------------|
| Vector search | Chroma in-memory (29K) | Pinecone or pgvector |
| BM25 | rank_bm25 in-memory | Elasticsearch |
| Embeddings | numpy array at startup | ANN index (FAISS/ScaNN) |
| Reranker | Cohere API | Same — stateless |

The embedding storage is the first bottleneck: 1M patterns × 1536 dims ×
4 bytes = ~6GB, which won't fit in memory. pgvector on Cloud SQL with
IVFFlat index would handle this while keeping SQL-based metadata filtering.

---

## 6. Why keyword-based guardrails instead of LLM-based?

**Option A: LLM classifier** — prompt a model to classify if query is
knitting-related.
- Cost: ~$0.00003 per check (haiku)
- Latency: ~0.5s added to every search
- Overkill for this use case

**Option B: Keyword matching** — check against a list of ~50 knitting terms.
- Cost: $0
- Latency: <1ms
- Coverage: sufficient for demo scale

**Decision:** Keyword matching for off-topic detection. The domain vocabulary
is well-defined (needle, yarn, knit, crochet, etc.) and keyword matching
handles 95%+ of cases. LLM classifier would be appropriate at production
scale where adversarial inputs are a concern.

---

## 7. Why Langfuse instead of LangSmith?

**LangSmith** is tightly coupled to the LangChain ecosystem. Auto-tracing
only works for LangChain calls.

**Langfuse** uses `@observe()` decorator — framework-agnostic, works on any
Python function regardless of whether it uses LangChain, instructor, raw
OpenAI SDK, or Anthropic SDK.

Our pipeline is deliberately mixed: instructor for structured outputs,
Anthropic SDK for recommendations, Cohere SDK for reranking. LangSmith
would require manual instrumentation for all non-LangChain calls anyway.
Langfuse is open-source and self-hostable — relevant for enterprise
deployments where data privacy matters.

---

## 8. Why claude-haiku instead of gpt-4o-mini?

Decision was data-driven. Ran a controlled experiment on the 20-query
golden set comparing parsing accuracy:

| Model | Accuracy | Cost/query |
|-------|----------|------------|
| gpt-4o-mini | 100% | ~$0.0001 |
| claude-haiku | 100% | ~$0.00003 |

Both models achieved identical accuracy across all 20 queries including
edge cases (negative conditions, Chinese queries, multi-filter queries).
With equal quality, haiku's 70% cost advantage made the switch straightforward.

Applied to both `parse_query` and `generate_recommendation`.

---

## 9. Why exact cache instead of semantic cache?

**Semantic cache** embeds each new query, compares to cached query embeddings,
and returns cached results if similarity exceeds a threshold.
- Handles paraphrases ("free hat" ≈ "no-cost beanie")
- Requires extra embedding call per query
- Risk of false positives (different queries cached together)

**Exact cache** (current): normalize query → dict lookup, O(1).
- Zero cost, zero latency on cache hit
- No false positives
- Miss rate higher than semantic cache

**Decision:** Start simple. For a search tool, users tend to re-run the
exact same query (history sidebar re-searches). Exact cache handles this
perfectly. Semantic cache can be added later if miss rate analysis shows
significant paraphrase patterns.

---

## 10. Data quality filtering

Two hard filters applied during collection:

1. **Must have `notes`** — patterns without description text produce poor
   embeddings. The `text_for_embedding` field relies heavily on notes for
   semantic content.

2. **Must have cover photo** — patterns without photos have significantly
   lower engagement on Ravelry, indicating lower quality.

These filters remove ~10% of raw API results but substantially improve
embedding quality and user experience.

**text_for_embedding field selection:**
- Included: name, craft, yarn_weight, needle_sizes, categories, notes,
  useful attributes
- Excluded: noise attributes (chart, schematic, video-tutorial, etc.) —
  these describe format, not content, and dilute semantic signal

---

## 11. Why P@10 as primary eval metric instead of MRR?

**MRR (Mean Reciprocal Rank)** measures rank of the first relevant result.
Good for tasks where finding *one* good result matters.

**P@10 (Precision at 10)** measures what fraction of top-10 results are
relevant. Better for our use case because:

1. Users browse multiple results before choosing a pattern
2. Ground truth set sizes vary widely (5 to 48 relevant patterns per query)
   — MRR is unstable with large ground truth sets
3. P@10 directly measures "are the results I'm showing good?"

**Eval methodology note:** Golden set annotated using three candidate
sources (vector search + keyword search + structured field query) to avoid
annotation bias from any single retrieval method.

---

## 12. Why Chroma in-memory instead of persistent vector DB?

**Current:** Chroma in-memory, data baked into Docker image.
- Zero infrastructure cost
- Simple deployment (single container)
- Cold start: ~30s to load 29K patterns + embeddings
- Data update requires Docker rebuild

**Production alternative:** pgvector on Cloud SQL or Pinecone.
- Persistent, queryable without reload
- Supports SQL-based metadata filtering (fixes Chroma `$contains` limitation)
- Enables real-time data updates

**Decision:** In-memory is appropriate for demo scale (29K patterns, single
user). The cold start penalty is acceptable. Migrating to pgvector is the
natural next step when moving to production with multiple instances.
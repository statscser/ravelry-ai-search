"""
constants.py — Shared named constants for the ravelry_search pipeline.

Centralising magic values here means a model rename or threshold tweak
requires a single edit rather than hunting across multiple files.
"""

# ── Models ────────────────────────────────────────────────────────────────────

HAIKU_MODEL          = "claude-haiku-4-5-20251001"
EMBEDDING_MODEL      = "text-embedding-3-small"
COHERE_RERANK_MODEL  = "rerank-multilingual-v3.0"

# ── LLM generation limits ─────────────────────────────────────────────────────

PARSE_MAX_TOKENS     = 500   # max tokens for parse_query structured output
REC_MAX_TOKENS       = 80    # max tokens per recommendation sentence
NOTES_EXCERPT_CHARS  = 200   # characters of pattern notes included in rec prompt

# ── Data loading ──────────────────────────────────────────────────────────────

CHROMA_BATCH_SIZE    = 1000  # patterns per Chroma collection.add() call

# ── Search pipeline ───────────────────────────────────────────────────────────

RRF_K                = 60    # RRF smoothing constant; higher k = smaller rank differences
RERANK_CANDIDATES    = 20    # hybrid_search top-k fed into Cohere reranker
MIN_RESULTS          = 5     # minimum results to keep after score threshold filtering
SCORE_THRESHOLD      = 0.3   # minimum Cohere relevance score to retain a result
REC_TOP_N            = 3     # number of results that get AI-generated recommendations

# ── External API retry ────────────────────────────────────────────────────────

MAX_RETRIES          = 3     # Cohere rerank retry attempts on rate limit
RETRY_WAIT_BASE_S    = 10    # seconds; actual wait = RETRY_WAIT_BASE_S * attempt

# ── Guardrail fallback ────────────────────────────────────────────────────────

FALLBACK_MIN_RATING        = 4.5   # minimum rating_average for tier-2 fallback patterns
FALLBACK_MIN_RATING_COUNT  = 10    # minimum rating_count  for tier-2 fallback patterns

# ── Cost tracking ─────────────────────────────────────────────────────────────

COHERE_COST_PER_CALL = 0.001   # flat $ per Cohere rerank call (for admin dashboard)

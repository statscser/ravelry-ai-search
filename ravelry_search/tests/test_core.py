"""
tests/test_core.py — Unit tests for ravelry_search core logic.

Run from ravelry_search/:
    pytest tests/ -v
"""
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Pre-import env setup ───────────────────────────────────────────────────────
# Langfuse requires valid keys at import time; give it dummy values so the
# SDK initialises without errors.  The langfuse_context fixture below makes
# every .update_*() call a no-op during tests.
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "test-public-key")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "test-secret-key")
os.environ.setdefault("LANGFUSE_HOST", "http://localhost:19999")

import rag_chroma  # noqa: E402 — must come after env vars
from guardrails import is_knitting_related  # noqa: E402
from hybrid_search import _passes_filter, _rrf_merge  # noqa: E402
from rag_chroma import PatternSearchIntent, parse_query, parse_query_cached  # noqa: E402


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def silence_langfuse(monkeypatch):
    """Replace langfuse_context in every module with a silent MagicMock so
    no real Langfuse calls are made and no exceptions surface from tracing."""
    noop = MagicMock()
    monkeypatch.setattr("rag_chroma.langfuse_context", noop)
    monkeypatch.setattr("hybrid_search.langfuse_context", noop)


@pytest.fixture(autouse=True)
def clear_parse_cache():
    """Wipe the module-level parse cache before and after every test so
    cache-hit tests don't bleed into each other."""
    rag_chroma._parse_cache.clear()
    yield
    rag_chroma._parse_cache.clear()


# ── Small builders ────────────────────────────────────────────────────────────

def _intent(**kwargs) -> PatternSearchIntent:
    kwargs.setdefault("semantic_query", "test")
    return PatternSearchIntent(**kwargs)


def _pattern(**kwargs) -> dict:
    base = {
        "id": 1,
        "name": "Test Pattern",
        "craft": {"name": "Knitting"},
        "free": False,
        "rating_average": 4.0,
        "pattern_needle_sizes": [{"metric": 5.0, "name": "US 8"}],
        "pattern_categories": [],
        "pattern_attributes": [],
    }
    base.update(kwargs)
    return base


def _mock_lm(intent: PatternSearchIntent):
    """Return a context manager that stubs out the Anthropic + instructor
    calls so parse_query() returns `intent` without hitting the network."""
    fake_client = MagicMock()
    fake_client.messages.create.return_value = intent
    return (
        patch("anthropic.Anthropic"),                              # stop real SDK init
        patch("instructor.from_anthropic", return_value=fake_client),  # return fake client
    )


# ── guardrails.is_knitting_related ────────────────────────────────────────────

class TestIsKnittingRelated:
    def test_english_yarn_keyword(self):
        # "yarn" is in KNITTING_KEYWORDS
        assert is_knitting_related("chunky yarn sweater") is True

    def test_english_pattern_keyword(self):
        # "pattern" is in KNITTING_KEYWORDS
        assert is_knitting_related("free beginner pattern") is True

    def test_off_topic_returns_false(self):
        # No knitting words anywhere in the query
        assert is_knitting_related("how to cook pasta") is False

    def test_chinese_yarn_keyword(self):
        # "毛线" (yarn) is in the Chinese keyword set
        assert is_knitting_related("毛线") is True

    def test_chinese_knitting_keyword(self):
        # "钩针" (crochet hook) is in the Chinese keyword set
        assert is_knitting_related("钩针帽子") is True

    def test_empty_string(self):
        assert is_knitting_related("") is False

    def test_case_insensitive(self):
        # Keywords are lowercased before matching; "KNIT" should still match
        assert is_knitting_related("KNITTING") is True

    def test_partial_keyword_in_sentence(self):
        # "lace" appears mid-sentence
        assert is_knitting_related("delicate lace shawl pattern") is True


# ── hybrid_search._rrf_merge ─────────────────────────────────────────────────

class TestRRFMerge:
    def test_single_list_preserves_order(self):
        # With one ranked list the output order must match the input ranking
        result = _rrf_merge([[2, 0, 1]], k=60)
        assert [idx for idx, _ in result] == [2, 0, 1]

    def test_item_appearing_in_both_lists_scores_higher(self):
        # Index 0 is rank-1 in both lists → it must beat any item in only one
        result = _rrf_merge([[0, 1, 2], [0, 2, 1]], k=60)
        top_idx = result[0][0]
        assert top_idx == 0

    def test_rrf_score_formula_is_correct(self):
        # score(rank=1 in two lists, k=60) = 2 / (60 + 1)
        result = _rrf_merge([[0], [0]], k=60)
        assert abs(result[0][1] - 2 / 61) < 1e-9

    def test_smaller_k_increases_rank1_score(self):
        # 1/(k+1) grows as k shrinks → lower k gives higher scores to rank-1 items
        score_k1  = _rrf_merge([[0]], k=1)[0][1]
        score_k60 = _rrf_merge([[0]], k=60)[0][1]
        assert score_k1 > score_k60

    def test_disjoint_lists_same_rank_same_score(self):
        # Index 0 is rank-1 in list A, index 1 is rank-1 in list B → equal scores
        result = _rrf_merge([[0], [1]], k=60)
        scores = {idx: sc for idx, sc in result}
        assert abs(scores[0] - scores[1]) < 1e-9

    def test_empty_lists_return_empty(self):
        assert _rrf_merge([[], []], k=60) == []

    def test_output_sorted_descending(self):
        result = _rrf_merge([[0, 1, 2], [2, 1, 0]], k=60)
        scores = [sc for _, sc in result]
        assert scores == sorted(scores, reverse=True)


# ── hybrid_search._passes_filter ─────────────────────────────────────────────

class TestPassesFilter:
    def test_no_filters_accepts_any_pattern(self):
        # Empty intent (all defaults) → everything passes
        assert _passes_filter(_pattern(), _intent()) is True

    def test_free_only_blocks_paid(self):
        intent = _intent(free_only=True)
        assert _passes_filter(_pattern(free=False), intent) is False
        assert _passes_filter(_pattern(free=True),  intent) is True

    def test_min_rating_filter(self):
        intent = _intent(min_rating=4.5)
        assert _passes_filter(_pattern(rating_average=4.8), intent) is True
        assert _passes_filter(_pattern(rating_average=4.0), intent) is False

    def test_exact_min_rating_boundary_passes(self):
        # rating == min_rating exactly should pass (≥ comparison)
        intent = _intent(min_rating=4.5)
        assert _passes_filter(_pattern(rating_average=4.5), intent) is True

    def test_craft_filter_knitting(self):
        intent = _intent(craft="Knitting")
        assert _passes_filter(_pattern(craft={"name": "Knitting"}), intent) is True
        assert _passes_filter(_pattern(craft={"name": "Crochet"}),  intent) is False

    def test_craft_filter_crochet(self):
        intent = _intent(craft="Crochet")
        assert _passes_filter(_pattern(craft={"name": "Crochet"}),  intent) is True
        assert _passes_filter(_pattern(craft={"name": "Knitting"}), intent) is False

    def test_needle_size_min_filter(self):
        # needle_size_min=6: pattern must have max(needle_sizes) >= 6
        intent     = _intent(needle_size_min=6.0)
        big_needle = _pattern(pattern_needle_sizes=[{"metric": 8.0, "name": "US 11"}])
        sml_needle = _pattern(pattern_needle_sizes=[{"metric": 4.0, "name": "US 6"}])
        assert _passes_filter(big_needle, intent) is True
        assert _passes_filter(sml_needle, intent) is False

    def test_needle_size_max_filter(self):
        # needle_size_max=4: pattern must have min(needle_sizes) <= 4
        intent     = _intent(needle_size_max=4.0)
        sml_needle = _pattern(pattern_needle_sizes=[{"metric": 3.5, "name": "US 4"}])
        big_needle = _pattern(pattern_needle_sizes=[{"metric": 6.0, "name": "US 10"}])
        assert _passes_filter(sml_needle, intent) is True
        assert _passes_filter(big_needle, intent) is False

    def test_pattern_with_no_needle_sizes_fails_needle_filter(self):
        # A pattern listing no needle sizes cannot satisfy a size constraint
        intent = _intent(needle_size_min=5.0)
        assert _passes_filter(_pattern(pattern_needle_sizes=[]), intent) is False

    def test_combined_filters_all_must_pass(self):
        intent = _intent(free_only=True, min_rating=4.5, craft="Knitting")
        good   = _pattern(free=True, rating_average=4.8, craft={"name": "Knitting"})
        assert _passes_filter(good, intent) is True
        # Violating any single condition blocks the pattern
        assert _passes_filter({**good, "free": False}, intent) is False
        assert _passes_filter({**good, "rating_average": 3.0}, intent) is False
        assert _passes_filter({**good, "craft": {"name": "Crochet"}}, intent) is False


# ── rag_chroma.parse_query ────────────────────────────────────────────────────

class TestParseQuery:
    def test_basic_free_dk_cardigan(self):
        # "free DK cardigan" → LLM returns free_only=True, yarn_weight=DK
        expected = _intent(semantic_query="cardigan", yarn_weight="DK", free_only=True)
        p1, p2 = _mock_lm(expected)
        with p1, p2:
            result = parse_query("free DK cardigan", client=None)
        assert result.free_only is True
        assert result.yarn_weight == "DK"

    def test_rating_filter_extracted(self):
        # "free pattern rating above 4.5" → min_rating set, free_only set
        expected = _intent(semantic_query="pattern", free_only=True, min_rating=4.5)
        p1, p2 = _mock_lm(expected)
        with p1, p2:
            result = parse_query("free pattern rating above 4.5", client=None)
        assert result.free_only is True
        assert result.min_rating == 4.5

    def test_exclude_fibers_not_in_semantic_query(self):
        # "chunky sweater no mohair" → exclude_fibers=["mohair"],
        # and "mohair" must NOT appear in semantic_query per the system prompt rules
        expected = _intent(semantic_query="chunky sweater", exclude_fibers=["mohair"])
        p1, p2 = _mock_lm(expected)
        with p1, p2:
            result = parse_query("chunky sweater no mohair", client=None)
        assert "mohair" in result.exclude_fibers
        assert "mohair" not in result.semantic_query

    def test_chinese_query_parsed(self):
        # "免费蕾丝披肩" (free lace shawl) → free_only, categories=Shawl
        expected = _intent(semantic_query="lace shawl", free_only=True, categories="Shawl")
        p1, p2 = _mock_lm(expected)
        with p1, p2:
            result = parse_query("免费蕾丝披肩", client=None)
        assert result.free_only is True
        assert result.categories == "Shawl"

    def test_returns_pattern_search_intent_type(self):
        # parse_query must always return a PatternSearchIntent
        expected = _intent(semantic_query="hat")
        p1, p2 = _mock_lm(expected)
        with p1, p2:
            result = parse_query("hat", client=None)
        assert isinstance(result, PatternSearchIntent)


# ── rag_chroma.parse_query_cached ────────────────────────────────────────────

class TestParseQueryCached:
    def test_second_call_does_not_hit_llm(self):
        # Calling with the same query twice should invoke parse_query only once.
        fake = _intent(semantic_query="cardigan", free_only=True)
        with patch("rag_chroma.parse_query", return_value=fake) as mock_pq:
            r1 = parse_query_cached("free cardigan", client=None)
            r2 = parse_query_cached("free cardigan", client=None)
        assert mock_pq.call_count == 1, "LLM called more than once for an identical query"
        assert r1 is r2, "Cache should return the exact same object"

    def test_cache_key_normalises_case_and_whitespace(self):
        # "Free Cardigan" and "  free cardigan  " share the same cache key
        fake = _intent(semantic_query="cardigan")
        with patch("rag_chroma.parse_query", return_value=fake) as mock_pq:
            parse_query_cached("Free Cardigan", client=None)
            parse_query_cached("  free cardigan  ", client=None)
        assert mock_pq.call_count == 1

    def test_distinct_queries_each_call_llm(self):
        # Different queries must each trigger a separate LLM call.
        fake_a = _intent(semantic_query="cardigan")
        fake_b = _intent(semantic_query="hat")
        with patch("rag_chroma.parse_query", side_effect=[fake_a, fake_b]) as mock_pq:
            ra = parse_query_cached("cardigan", client=None)
            rb = parse_query_cached("hat", client=None)
        assert mock_pq.call_count == 2
        assert ra.semantic_query == "cardigan"
        assert rb.semantic_query == "hat"

    def test_cache_populated_after_first_call(self):
        # After one call, _parse_cache should contain the normalised key.
        fake = _intent(semantic_query="socks")
        with patch("rag_chroma.parse_query", return_value=fake):
            parse_query_cached("Socks", client=None)
        assert "socks" in rag_chroma._parse_cache

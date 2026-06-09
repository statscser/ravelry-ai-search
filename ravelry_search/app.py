import datetime
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from langfuse import observe, get_client, propagate_attributes

from hybrid_search import reranked_search
from rag_chroma import load_collection, parse_query
from recommendation import generate_recommendations_batch

load_dotenv()

st.set_page_config(page_title="Ravelry AI Search", page_icon="🧶", layout="wide")
st.title("🧶 Ravelry AI Search")


# ── Search pipeline (single Langfuse root trace) ──────────────────────────────

@observe(name="ravelry_search")
def run_search(
    query: str,
    patterns: list[dict],
    embeddings,
    openai_client: OpenAI,
    top_k: int = 20,
) -> tuple:
    """
    Full search pipeline under one Langfuse trace:
      parse_query → reranked_search → generate_recommendations_batch (top-3)

    Returns (intent, results, rec_map) where rec_map is {pattern_id: rec_text}.
    """
    # propagate_attributes sets the Langfuse trace name on all spans in this
    # context. @observe(name=...) only names the observation; trace_name is a
    # separate OTel attribute that the Langfuse dashboard filters on.
    with propagate_attributes(trace_name="ravelry_search"):
        intent = parse_query(query, openai_client)

        results = reranked_search(
            query=intent.semantic_query,
            patterns=patterns,
            embeddings=embeddings,
            openai_client=openai_client,
            top_k=top_k,
            intent=intent,
        )

        # Dynamic threshold: keep scores >= 0.3, guarantee at least 5 results
        MIN_RESULTS     = 5
        SCORE_THRESHOLD = 0.3
        filtered = [p for p in results if p.get("_cohere_score", 0) >= SCORE_THRESHOLD]
        results  = filtered if len(filtered) >= MIN_RESULTS else results[:MIN_RESULTS]

        recs    = generate_recommendations_batch(query=query, patterns=results[:3],
                                                 client=openai_client, top_n=3)
        rec_map = {p["id"]: rec for p, rec in zip(results[:3], recs)}

        try:
            get_client().update_current_span(
                metadata={
                    "cohere_model": "rerank-multilingual-v3.0",
                    "top_k": top_k,
                    "intent": {
                        "craft": intent.craft,
                        "free_only": intent.free_only,
                        "min_rating": intent.min_rating,
                        "yarn_weight": intent.yarn_weight,
                    },
                }
            )
        except Exception:
            pass

    return intent, results, rec_map


def _sort_results(results: list[dict], sort_option: str) -> list[dict]:
    if sort_option == "Rating":
        return sorted(results, key=lambda p: p.get("rating_average") or 0, reverse=True)
    if sort_option == "Favorites":
        return sorted(results, key=lambda p: p.get("favorites_count") or 0, reverse=True)
    return results  # "Relevance" keeps Cohere order


# ── Admin stats (Langfuse) ────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _fetch_admin_stats() -> dict:
    try:
        import os, requests
        from requests.auth import HTTPBasicAuth

        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        auth = HTTPBasicAuth(public_key, secret_key)

        today_utc = datetime.datetime.now(datetime.timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()

        resp = requests.get(
            f"{host}/api/public/traces",
            auth=auth,
            params={
                "name": "ravelry_search",
                "fromTimestamp": today_utc,
                "limit": 100,
            },
            timeout=10,
        )
        resp.raise_for_status()
        traces = resp.json().get("data", [])

        if not traces:
            return {"count": 0, "top_queries": []}

        latencies = [t["latency"] for t in traces if t.get("latency") is not None]
        llm_costs = [t["totalCost"] for t in traces if t.get("totalCost") is not None]

        queries = [
            t["input"] for t in traces
            if isinstance(t.get("input"), str) and t["input"]
        ]

        p50 = statistics.median(latencies) if latencies else None
        p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else (latencies[0] if latencies else None)

        COHERE_COST_PER_CALL = 0.001
        avg_total_cost = (sum(llm_costs) / len(llm_costs) + COHERE_COST_PER_CALL) if llm_costs else None

        return {
            "count": len(traces),
            "p50_latency": p50,
            "p95_latency": p95,
            "avg_cost": avg_total_cost,
            "top_queries": Counter(queries).most_common(5),
        }
    except Exception as exc:
        return {"error": str(exc)}

# ── Process-level resources (loaded once, shared across all sessions) ─────────

@st.cache_resource(show_spinner="Loading pattern index…")
def _load_resources():
    _collection, _patterns = load_collection()
    _embeddings = np.load(Path(__file__).parent / "data" / "embeddings.npy")
    return _collection, _patterns, _embeddings

_collection, _patterns, _embeddings = _load_resources()


# ── Per-session init ──────────────────────────────────────────────────────────

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI()

if "search_history" not in st.session_state:
    st.session_state.search_history = []


# ── Sidebar (always rendered; after search phase so history is current) ──────

with st.sidebar:
    if st.session_state.get("search_history"):
        st.markdown("### 🕐 Recent Searches")
        for past_query in st.session_state.search_history:
            if st.button(past_query, key=f"hist_{past_query}", use_container_width=True):
                st.session_state["prefill_query"] = past_query
                st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_search, tab_admin = st.tabs(["🔍 Search", "📊 Admin"])


# ══ Search tab ════════════════════════════════════════════════════════════════

with tab_search:

    # ── Search bar ────────────────────────────────────────────────────────────

    with st.form("search_form"):
        col_input, col_btn = st.columns([6, 1])
        with col_input:
            default_query = st.session_state.get("prefill_query", "")
            query = st.text_input(
                label="query",
                value=default_query,
                placeholder='e.g. "free beginner knitting hat with bulky yarn, rating above 4"',
                label_visibility="collapsed",
            )
        with col_btn:
            search_clicked = st.form_submit_button("Search", width="stretch", type="primary")

    # ── Search phase (runs only on form submit) ───────────────────────────────

    if search_clicked and query.strip():
        st.session_state.pop("prefill_query", None)  # clear prefill now that it's been submitted
        client = st.session_state.openai_client

        with st.spinner("Searching…"):
            intent, results, rec_map = run_search(
                query=query,
                patterns=_patterns,
                embeddings=_embeddings,
                openai_client=client,
            )
        # Flush after run_search returns so the root span is fully closed
        # before the OTel exporter drains the queue.
        get_client().flush()

        # Build filter summary string from the returned intent
        filters = []
        if intent.craft:           filters.append(intent.craft)
        if intent.yarn_weight:     filters.append(f"yarn: {intent.yarn_weight}")
        if intent.needle_size_min: filters.append(f"needle ≥ {intent.needle_size_min}mm")
        if intent.needle_size_max: filters.append(f"needle ≤ {intent.needle_size_max}mm")
        if intent.free_only:       filters.append("free only")
        if intent.min_rating > 0:  filters.append(f"rating ≥ {intent.min_rating}")
        if intent.exclude_fibers:  filters.append(f"exclude: {', '.join(intent.exclude_fibers)}")
        if intent.include_fibers:  filters.append(f"include: {', '.join(intent.include_fibers)}")
        if intent.categories:      filters.append(f"category: {intent.categories}")

        # Update search history — deduplicated, max 10
        history = st.session_state.search_history
        if query not in history:
            history.insert(0, query)
        st.session_state.search_history = history[:10]

        # Persist to session state — display phase reads from here
        st.session_state.search_results = results
        st.session_state.search_rec_map = rec_map
        st.session_state.search_caption = (
            f"LLM understanding：「{intent.semantic_query}」 — "
            + (" · ".join(filters) if filters else "no filters")
        )

    # ── Display phase (runs on every re-run, including sort radio clicks) ─────

    if "search_results" in st.session_state:
        results  = st.session_state.search_results
        rec_map  = st.session_state.search_rec_map   # {pattern_id: rec_text}

        st.caption(st.session_state.search_caption)

        if not results:
            st.warning("No results found. Try relaxing your filters.")
        else:
            # Sort control — purely visual, does not re-run the search
            sort_option = st.radio(
                "Sort by",
                options=["Relevance", "Rating", "Favorites"],
                horizontal=True,
                label_visibility="collapsed",
                key="sort_option",
            )
            results = _sort_results(results, sort_option)

            # Render result grid
            COLS = 4
            for global_idx, pattern in enumerate(results):
                if global_idx % COLS == 0:
                    cols = st.columns(COLS)
                col = cols[global_idx % COLS]

                with col:
                    # Photo
                    photos  = pattern.get("photos") or []
                    img_url = photos[0].get("small_url") if photos else None
                    if img_url:
                        st.markdown(
                            f'<img src="{img_url}" style="width:100%;height:300px;'
                            f'object-fit:cover;border-radius:8px">',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            "<div style='height:300px;background:#f0f0f0;border-radius:8px;"
                            "display:flex;align-items:center;justify-content:center;"
                            "color:#aaa;font-size:2rem'>🧶</div>",
                            unsafe_allow_html=True,
                        )

                    # Name + designer
                    author    = pattern.get("pattern_author") or {}
                    author_name = author.get("name", "")
                    author_slug = author.get("permalink", "")
                    if author_name and author_slug:
                        designer_md = (
                            f" <span style='font-weight:normal;font-size:0.85em;color:#666'>"
                            f"by [**{author_name}**](https://www.ravelry.com/designers/{author_slug})"
                            f"</span>"
                        )
                    else:
                        designer_md = ""
                    st.markdown(f"**{pattern['name']}**{designer_md}", unsafe_allow_html=True)

                    # Craft · yarn weight
                    craft_name = (pattern.get("craft") or {}).get("name", "")
                    yarn       = pattern.get("yarn_weight_description") or ""
                    st.markdown(f"{craft_name} · {yarn}" if yarn else craft_name)

                    # Rating + favorites
                    rating    = pattern.get("rating_average") or 0.0
                    count     = pattern.get("rating_count") or 0
                    favorites = pattern.get("favorites_count") or 0
                    rating_str = f"⭐ {rating:.1f} ({count})" if rating > 0 else "No ratings"
                    st.markdown(
                        f"{rating_str} &nbsp;"
                        f"<span style='color:#e05a5a'>♥</span> {favorites:,}",
                        unsafe_allow_html=True,
                    )

                    # Recommendation — shown whenever this pattern has one,
                    # regardless of its current position after sorting
                    rec = rec_map.get(pattern["id"], "")
                    if rec:
                        st.caption(f"✦ {rec}")

                    # Free / paid badge
                    if pattern.get("free"):
                        st.markdown(
                            "<span style='background:#d4edda;color:#155724;"
                            "padding:2px 8px;border-radius:4px;font-size:0.8rem'>Free</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        currency = pattern.get("currency_symbol") or ""
                        price    = pattern.get("price") or ""
                        st.markdown(
                            f"<span style='background:#f8d7da;color:#721c24;"
                            f"padding:2px 8px;border-radius:4px;font-size:0.8rem'>"
                            f"Paid {currency}{price}</span>",
                            unsafe_allow_html=True,
                        )

                    # Link
                    permalink = pattern.get("permalink", "")
                    st.markdown(
                        f"[ravelry.com/…/{permalink}]"
                        f"(https://www.ravelry.com/patterns/library/{permalink})"
                    )


# ══ Admin tab ═════════════════════════════════════════════════════════════════

with tab_admin:
    st.markdown("### 📊 Search Dashboard")
    st.caption(
        "Covers today's `ravelry_search` traces from Langfuse. "
        "Stats are cached for 60 s — click Refresh to update. "
        "Cost = LLM tokens (auto-calculated by Langfuse) + $0.001 flat per Cohere rerank call."
    )

    if st.button("🔄 Refresh"):
        _fetch_admin_stats.clear()
        st.rerun()

    with st.spinner("Loading stats…"):
        stats = _fetch_admin_stats()

    if "error" in stats:
        st.error(f"Could not reach Langfuse: {stats['error']}")
        st.info("Make sure LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST are set in .env")
    else:
        # ── KPI row ───────────────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Searches today", stats.get("count", 0))

        with col2:
            avg_cost = stats.get("avg_cost")
            st.metric(
                "Avg cost / search",
                f"${avg_cost:.4f}" if avg_cost is not None else "N/A",
                help="GPT-4o-mini: $0.15/1M input · $0.60/1M output · Cohere: $1/1000 calls",
            )

        with col3:
            p50 = stats.get("p50_latency")
            p95 = stats.get("p95_latency")
            if p50 is not None and p95 is not None:
                latency_str = f"{p50:.1f}s / {p95:.1f}s"
            else:
                latency_str = "N/A"
            st.metric("Latency  P50 / P95", latency_str)

        # ── Top queries ───────────────────────────────────────────────────────
        st.divider()
        st.markdown("#### Top 5 Queries Today")
        top_queries = stats.get("top_queries", [])
        if top_queries:
            for rank, (q, cnt) in enumerate(top_queries, 1):
                st.markdown(f"{rank}. **{q}** &nbsp; `{cnt}×`", unsafe_allow_html=True)
        else:
            st.info("No searches recorded today yet.")

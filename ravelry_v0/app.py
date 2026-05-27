from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from hybrid_search import reranked_search
from rag_chroma import load_collection, parse_query
from recommendation import generate_recommendations_batch

load_dotenv()

st.set_page_config(page_title="Ravelry AI Search", page_icon="🧶", layout="wide")
st.title("🧶 Ravelry AI Search")


# ── Cached recommendation generator ──────────────────────────────────────────
# _patterns and _client are prefixed with _ so Streamlit skips hashing them.
# Cache key is (query, top3_ids) — same search won't regenerate.

@st.cache_data(ttl=3600, show_spinner=False)
def cached_recommendations(
    query: str,
    top3_ids: tuple,
    _patterns: list,
    _client: OpenAI,
) -> dict[int, str]:
    """
    Generate recommendations for the given top-3 patterns (order-sensitive cache key).
    Returns a dict {pattern_id: recommendation_text}.
    """
    id_set    = set(top3_ids)
    top3      = [p for p in _patterns if p["id"] in id_set][:3]
    recs      = generate_recommendations_batch(query=query, patterns=top3,
                                               client=_client, top_n=len(top3))
    return {p["id"]: rec for p, rec in zip(top3, recs)}


def _sort_results(results: list[dict], sort_option: str) -> list[dict]:
    if sort_option == "Rating":
        return sorted(results, key=lambda p: p.get("rating_average") or 0, reverse=True)
    if sort_option == "Favorites":
        return sorted(results, key=lambda p: p.get("favorites_count") or 0, reverse=True)
    return results  # "Relevance" keeps Cohere order


# ── One-time init ─────────────────────────────────────────────────────────────

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI()

if "collection" not in st.session_state:
    with st.spinner("Loading pattern index…"):
        collection, patterns = load_collection()
        st.session_state.collection = collection
        st.session_state.patterns   = patterns

if "embeddings" not in st.session_state:
    st.session_state.embeddings = np.load(
        Path(__file__).parent / "data" / "embeddings.npy"
    )


# ── Search bar ────────────────────────────────────────────────────────────────

with st.form("search_form"):
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        query = st.text_input(
            label="query",
            placeholder='e.g. "free beginner knitting hat with bulky yarn, rating above 4"',
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.form_submit_button("Search", width="stretch", type="primary")


# ── Search phase (runs only on form submit) ───────────────────────────────────

if search_clicked and query.strip():
    client = st.session_state.openai_client

    with st.spinner("Thinking…"):
        intent = parse_query(query, client)

    # Build filter summary string
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

    with st.spinner("Searching…"):
        results = reranked_search(
            query=intent.semantic_query,
            patterns=st.session_state.patterns,
            embeddings=st.session_state.embeddings,
            openai_client=client,
            top_k=20,
            intent=intent,
        )

    # Dynamic threshold: keep scores >= 0.3, but guarantee at least 5 results
    MIN_RESULTS     = 5
    SCORE_THRESHOLD = 0.3
    filtered = [p for p in results if p.get("_cohere_score", 0) >= SCORE_THRESHOLD]
    results  = filtered if len(filtered) >= MIN_RESULTS else results[:MIN_RESULTS]

    # Sort by whatever the user had selected before this new search
    sort_option = st.session_state.get("sort_option", "Relevance")
    results = _sort_results(results, sort_option)

    # Generate recommendations for top-3 of the sorted results
    rec_map: dict[int, str] = {}
    if results:
        with st.spinner("Generating recommendations…"):
            top3_ids = tuple(p["id"] for p in results[:3])
            rec_map  = cached_recommendations(
                query, top3_ids,
                _patterns=st.session_state.patterns,
                _client=client,
            )

    # Persist to session state — display phase reads from here
    st.session_state.search_results = results
    st.session_state.search_rec_map = rec_map
    st.session_state.search_caption = (
        f"LLM understanding：「{intent.semantic_query}」 — "
        + (" · ".join(filters) if filters else "no filters")
    )


# ── Display phase (runs on every re-run, including sort radio clicks) ─────────

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

                # Name
                st.markdown(f"**{pattern['name']}**")

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
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag_chroma import load_collection, parse_query, search

load_dotenv()

st.set_page_config(page_title="Ravelry AI Search", page_icon="🧶", layout="wide")
st.title("🧶 Ravelry AI Search")

# --- one-time init -----------------------------------------------------------
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI()

if "collection" not in st.session_state:
    with st.spinner("Loading pattern index…"):
        collection, patterns = load_collection()
        st.session_state.collection = collection
        st.session_state.patterns = patterns

# --- search bar --------------------------------------------------------------
with st.form("search_form"):
    col_input, col_btn = st.columns([6, 1])
    with col_input:
        query = st.text_input(
            label="query",
            placeholder='e.g. "free beginner knitting hat with bulky yarn, rating above 4"',
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.form_submit_button("Search", width='stretch', type="primary")

# --- search & display --------------------------------------------------------
if search_clicked and query.strip():
    client: OpenAI = st.session_state.openai_client

    with st.spinner("Thinking…"):
        intent = parse_query(query, client)

    # show what the LLM understood
    filters = []
    if intent.craft:
        filters.append(intent.craft)
    if intent.free_only:
        filters.append("free only")
    if intent.min_rating > 0:
        filters.append(f"rating ≥ {intent.min_rating}")
    filter_str = " · ".join(filters) if filters else "no filters"
    st.caption(f"LLM understanding：「{intent.semantic_query}」 — {filter_str}")

    with st.spinner("Searching…"):
        results = search(
            query=intent.semantic_query,
            collection=st.session_state.collection,
            openai_client=client,
            patterns=st.session_state.patterns,
            top_k=8,
            craft=intent.craft,
            free_only=intent.free_only,
            min_rating=intent.min_rating,
        )

    if not results:
        st.warning("No results found. Try relaxing your filters.")
    else:
        COLS = 4
        for row_start in range(0, len(results), COLS):
            cols = st.columns(COLS)
            for col, pattern in zip(cols, results[row_start : row_start + COLS]):
                with col:
                    # photo
                    photos = pattern.get("photos") or []
                    img_url = photos[0].get("small_url") if photos else None
                    if img_url:
                        st.markdown(
                            f'<img src="{img_url}" style="width:100%;height:300px;object-fit:cover;border-radius:8px">',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            "<div style='height:160px;background:#f0f0f0;border-radius:8px;"
                            "display:flex;align-items:center;justify-content:center;"
                            "color:#aaa;font-size:2rem'>🧶</div>",
                            unsafe_allow_html=True,
                        )

                    # name
                    st.markdown(f"**{pattern['name']}**")

                    # craft · yarn weight
                    craft_name = (pattern.get("craft") or {}).get("name", "")
                    yarn = pattern.get("yarn_weight_description") or ""
                    st.markdown(f"{craft_name} · {yarn}" if yarn else craft_name)

                    # rating
                    rating = pattern.get("rating_average") or 0.0
                    count = pattern.get("rating_count") or 0
                    if rating > 0:
                        st.markdown(f"⭐ {rating:.1f} ({count} ratings)")
                    else:
                        st.markdown("No ratings yet")

                    # free / paid badge
                    if pattern.get("free"):
                        st.markdown(
                            "<span style='background:#d4edda;color:#155724;"
                            "padding:2px 8px;border-radius:4px;font-size:0.8rem'>Free</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        price = pattern.get("price") or ""
                        currency = pattern.get("currency_symbol") or ""
                        price = pattern.get("price") or ""
                        label = f"Paid {currency}{price}".strip()
                        st.markdown(
                            f"<span style='background:#f8d7da;color:#721c24;"
                            f"padding:2px 8px;border-radius:4px;font-size:0.8rem'>{label}</span>",
                            unsafe_allow_html=True,
                        )

                    # link
                    permalink = pattern.get("permalink", "")
                    url = f"https://www.ravelry.com/patterns/library/{permalink}"
                    st.markdown(f"[{url}]({url})")

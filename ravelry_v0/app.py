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

# --- one-time init -----------------------------------------------------------
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI()

if "collection" not in st.session_state:
    with st.spinner("Loading pattern index…"):
        collection, patterns = load_collection()
        st.session_state.collection = collection
        st.session_state.patterns = patterns

if "embeddings" not in st.session_state:
    st.session_state.embeddings = np.load(
        Path(__file__).parent / "data" / "embeddings.npy"
    )

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
        search_clicked = st.form_submit_button("Search", width="stretch", type="primary")

# --- search & display --------------------------------------------------------
if search_clicked and query.strip():
    client: OpenAI = st.session_state.openai_client

    with st.spinner("Thinking…"):
        intent = parse_query(query, client)

    # show what the LLM understood
    filters = []
    if intent.craft:
        filters.append(intent.craft)
    if intent.yarn_weight:
        filters.append(f"yarn: {intent.yarn_weight}")
    if intent.needle_size_min:
        filters.append(f"needle ≥ {intent.needle_size_min}mm")
    if intent.needle_size_max:
        filters.append(f"needle ≤ {intent.needle_size_max}mm")
    if intent.free_only:
        filters.append("free only")
    if intent.min_rating > 0:
        filters.append(f"rating ≥ {intent.min_rating}")
    if intent.exclude_fibers:
        filters.append(f"exclude: {', '.join(intent.exclude_fibers)}")
    if intent.include_fibers:
        filters.append(f"include: {', '.join(intent.include_fibers)}")
    if intent.categories:
        filters.append(f"category: {intent.categories}")
    filter_str = " · ".join(filters) if filters else "no filters"
    st.caption(f"LLM understanding：「{intent.semantic_query}」 — {filter_str}")

    with st.spinner("Searching…"):
        results = reranked_search(
            query=intent.semantic_query,
            patterns=st.session_state.patterns,
            embeddings=st.session_state.embeddings,
            openai_client=client,
            top_k=20,
            intent=intent,
        )

    results = [p for p in results if p.get("_cohere_score", 0) >= 0.3]

    if not results:
        st.warning("No results found. Try relaxing your filters.")
    else:
        # 并行生成 top 5 推荐理由
        with st.spinner("Generating recommendations…"):
            recommendations = generate_recommendations_batch(
                query=query,
                patterns=results,
                client=client,
                top_n=5,
            )
        # 补齐长度，让后面按 index 取值安全
        recommendations += [""] * max(0, len(results) - len(recommendations))

        COLS = 4
        for global_idx, pattern in enumerate(results):
            # 每行开头新建 4 列
            if global_idx % COLS == 0:
                cols = st.columns(COLS)
            col = cols[global_idx % COLS]

            with col:
                # 图片
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

                # 名称
                st.markdown(f"**{pattern['name']}**")

                # 工艺 · 线重
                craft_name = (pattern.get("craft") or {}).get("name", "")
                yarn       = pattern.get("yarn_weight_description") or ""
                st.markdown(f"{craft_name} · {yarn}" if yarn else craft_name)

                # 评分
                rating = pattern.get("rating_average") or 0.0
                count  = pattern.get("rating_count") or 0
                st.markdown(f"⭐ {rating:.1f} ({count} ratings)" if rating > 0 else "No ratings yet")

                # 推荐理由（只对前 5 个）
                if global_idx < 5 and recommendations[global_idx]:
                    st.caption(f"✦ {recommendations[global_idx]}")

                # 免费 / 付费标签
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

                # 链接
                permalink = pattern.get("permalink", "")
                st.markdown(f"[ravelry.com/…/{permalink}](https://www.ravelry.com/patterns/library/{permalink})")
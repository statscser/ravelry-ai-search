"""
golden_set_annotator.py
独立工具，不影响任何现有代码。
用法：streamlit run golden_set_annotator.py
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ── 路径 ──────────────────────────────────────────────────────────────────────
PATTERNS_PATH   = Path(__file__).parent / "data" / "patterns.json"
EMBEDDINGS_PATH = Path(__file__).parent / "data" / "embeddings.npy"
GOLDEN_SET_PATH = Path(__file__).parent / "data" / "golden_set.json"

load_dotenv()

# ── Sample queries（修改这里，不影响其他代码）────────────────────────────────
SAMPLE_QUERIES = [
    # 基础语义
    {"query": "cable knit sweater",                      "category": "基础语义"},
    {"query": "lace shawl",                              "category": "基础语义"},
    {"query": "fingering weight socks",                  "category": "基础语义"},
    {"query": "knitted throw blanket",                   "category": "基础语义"},
    {"query": "sleeveless knit vest",                    "category": "基础语义"},
    # filter 验证
    {"query": "free crochet beanie",                     "category": "filter验证"},
    {"query": "DK weight cardigan rating above 4.5",     "category": "filter验证"},
    {"query": "free fingering socks",                    "category": "filter验证"},
    {"query": "crochet sleeveless top free",             "category": "filter验证"},
    # 细分特征
    {"query": "oversized colorwork cardigan",            "category": "细分特征"},
    {"query": "stranded colorwork mittens",              "category": "细分特征"},
    {"query": "top down raglan sweater",                 "category": "细分特征"},
    {"query": "ribbed cowl",                             "category": "细分特征"},
    {"query": "fingerless gloves knitting",              "category": "细分特征"},
    # 多条件组合
    {"query": "bulky yarn chunky sweater seamless",      "category": "多条件组合"},
    {"query": "cotton summer sleeveless top free",       "category": "多条件组合"},
    {"query": "toddler knitting socks fingering",        "category": "多条件组合"},
    # 边缘案例
    {"query": "colorwork yoke pullover",                 "category": "边缘案例"},
    {"query": "hat with needle or hook >= 5mm",        "category": "边缘案例"},
    {"query": "crochet baby blanket beginner",           "category": "边缘案例"},
]

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def keyword_search(patterns: list[dict], query: str, top_n: int = 20) -> list[dict]:
    """关键词搜索：在 name / categories / attributes / notes 里匹配。"""
    keywords = [kw for kw in query.lower().split() if len(kw) > 2]
    scored = []
    for p in patterns:
        name       = (p.get("name") or "").lower()
        notes      = (p.get("notes") or "").lower()
        categories = " ".join(c["name"] for c in (p.get("pattern_categories") or [])).lower()
        attributes = " ".join(a["permalink"] for a in (p.get("pattern_attributes") or [])).lower()
        combined   = f"{name} {categories} {attributes} {notes}"
        score = 0
        for kw in keywords:
            if kw in name:        score += 4
            if kw in categories:  score += 3
            if kw in attributes:  score += 2
            if kw in notes:       score += 1
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_n]]


def structured_search(patterns: list[dict], query: str, top_n: int = 20) -> list[dict]:
    """
    结构化查询：根据 query 里检测到的关键词，
    对字段做精确匹配（yarn_weight、categories、craft、free、needle_size）。
    比向量检索更精确地找符合条件的 pattern。
    """
    q = query.lower()
    results = []

    # 检测条件
    yarn_weight_map = {
        "lace": "Lace", "fingering": "Fingering", "sport": "Sport",
        "dk": "DK", "worsted": "Worsted", "aran": "Aran",
        "bulky": "Bulky", "super bulky": "Super Bulky", "jumbo": "Jumbo",
    }
    detected_yarn = next((v for k, v in yarn_weight_map.items() if k in q), None)

    category_map = {
        "sweater": ["Pullover", "Cardigan"],
        "pullover": ["Pullover"],
        "cardigan": ["Cardigan"],
        "shawl": ["Shawl / Wrap"],
        "socks": ["Mid-calf", "Ankle"],
        "hat": ["Beanie, Toque", "Brimmed", "Earflap", "Cloche"],
        "beanie": ["Beanie, Toque"],
        "mittens": ["Mittens"],
        "cowl": ["Cowl"],
        "blanket": ["Baby Blanket", "Throw"],
        "vest": ["Vest", "Sleeveless Top"],
        "top": ["Sleeveless Top", "Tee"],
        "gloves": ["Gloves", "Fingerless Gloves/Mitts"],
        "fingerless": ["Fingerless Gloves/Mitts"],
        "scarf": ["Scarf"],
        "booties": ["Booties"],
    }
    detected_cats = []
    for kw, cats in category_map.items():
        if kw in q:
            detected_cats.extend(cats)
    detected_cats = list(set(detected_cats))

    craft = None
    if any(kw in q for kw in ["crochet", "hook"]):
        craft = "Crochet"
    elif any(kw in q for kw in ["knitting", "knit", "needle"]):
        craft = "Knitting"

    free_only = any(kw in q for kw in ["free", "免费"])

    # 检测针号（e.g. "5mm", "> 5mm", "needles > 5"）
    import re
    needle_min = None
    m = re.search(r'(?:needle[s]?\s*[>≥]\s*|>)\s*(\d+(?:\.\d+)?)\s*mm?', q)
    if m:
        needle_min = float(m.group(1))

    # 过滤
    for p in patterns:
        p_yarn  = (p.get("yarn_weight") or {}).get("name", "")
        p_cats  = [c["name"] for c in (p.get("pattern_categories") or [])]
        p_craft = (p.get("craft") or {}).get("name", "")
        p_free  = p.get("free", False)
        p_needles = [n.get("metric") for n in (p.get("pattern_needle_sizes") or []) if n.get("metric")]

        if detected_yarn and p_yarn != detected_yarn:
            continue
        if detected_cats and not any(c in p_cats for c in detected_cats):
            continue
        if craft and p_craft != craft:
            continue
        if free_only and not p_free:
            continue
        if needle_min and (not p_needles or max(p_needles) <= needle_min):
            continue

        results.append(p)

    # 按评分排序
    results.sort(key=lambda p: p.get("rating_average") or 0, reverse=True)
    return results[:top_n]


def vector_search(
    query: str,
    patterns: list[dict],
    embeddings: np.ndarray,
    client: OpenAI,
    top_n: int = 20,
) -> list[dict]:
    """向量检索。"""
    q_emb = np.array(
        client.embeddings.create(
            input=[query], model="text-embedding-3-small"
        ).data[0].embedding
    )
    sims = [cosine_similarity(q_emb, embeddings[i]) for i in range(len(embeddings))]
    top_idx = np.argsort(sims)[-top_n:][::-1]
    results = []
    for i in top_idx:
        p = patterns[i].copy()
        p["_similarity"] = round(sims[i], 4)
        results.append(p)
    return results


def merge_candidates(
    kw_results: list[dict],
    vec_results: list[dict],
    struct_results: list[dict],
) -> list[dict]:
    """
    合并三个来源，去重。
    优先级：向量（有相似度分）> 结构化 > 关键词。
    每个 pattern 标记来源方便查看。
    """
    seen: dict[int, dict] = {}

    for p in vec_results:
        pid = p["id"]
        if pid not in seen:
            seen[pid] = p.copy()
            seen[pid]["_sources"] = ["vector"]
        else:
            seen[pid]["_sources"].append("vector")
            seen[pid]["_similarity"] = p.get("_similarity")

    for p in struct_results:
        pid = p["id"]
        if pid not in seen:
            pp = p.copy()
            pp["_sources"] = ["structured"]
            pp.setdefault("_similarity", None)
            seen[pid] = pp
        else:
            if "structured" not in seen[pid].get("_sources", []):
                seen[pid]["_sources"].append("structured")

    for p in kw_results:
        pid = p["id"]
        if pid not in seen:
            pp = p.copy()
            pp["_sources"] = ["keyword"]
            pp.setdefault("_similarity", None)
            seen[pid] = pp
        else:
            if "keyword" not in seen[pid].get("_sources", []):
                seen[pid]["_sources"].append("keyword")

    # 排序：多来源 > 有相似度分的 > 其他
    def sort_key(p):
        source_count = len(p.get("_sources", []))
        sim = p.get("_similarity") or 0
        return (source_count, sim)

    return sorted(seen.values(), key=sort_key, reverse=True)


def load_golden_set() -> dict:
    if GOLDEN_SET_PATH.exists():
        return {
            item["query"]: item
            for item in json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
        }
    return {}


def save_golden_set(data: dict) -> None:
    GOLDEN_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_SET_PATH.write_text(
        json.dumps(list(data.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Golden Set Annotator 🧶",
    page_icon="🏷️",
    layout="wide",
)

st.markdown("""
<style>
.pattern-id { font-family:monospace; font-size:0.80rem; color:#888; }
.badge { padding:1px 6px; border-radius:4px; font-size:0.75rem;
         font-family:monospace; margin-right:2px; display:inline-block; }
.vec  { background:#eef7f6; color:#2d7a6e; }
.str  { background:#e8f0fe; color:#1a56a0; }
.kw   { background:#fff3e0; color:#b36a00; }
.free-badge { background:#d4edda; color:#155724; padding:1px 7px;
              border-radius:4px; font-size:0.75rem; font-weight:700; }
.cat-tag { background:#f0eae0; color:#7a5c2d; padding:1px 8px;
           border-radius:4px; font-size:0.75rem; margin-left:8px; }
.card-meta { line-height:1.2; margin:1px 0; font-size:0.87rem; color:#555; }
.card-row  { display:flex; align-items:center; gap:6px;
             font-size:0.80rem; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

st.title("🏷️ Golden Set Annotator")
st.caption(
    "三个来源合并候选：向量检索 · 结构化查询 · 关键词搜索。"
    " 勾选所有你认为相关的图解（不只是当前系统能找到的），保存即可。"
)

# ── 初始化 ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    patterns   = json.loads(PATTERNS_PATH.read_text(encoding="utf-8"))
    embeddings = np.load(EMBEDDINGS_PATH)
    client     = OpenAI()
    return patterns, embeddings, client

patterns, embeddings, openai_client = load_data()

if "golden_set" not in st.session_state:
    st.session_state.golden_set = load_golden_set()

if "candidates_cache" not in st.session_state:
    st.session_state.candidates_cache = {}

# ── 侧边栏 ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Query 列表")
    total     = len(SAMPLE_QUERIES)
    annotated = sum(
        1 for q in SAMPLE_QUERIES
        if q["query"] in st.session_state.golden_set
        and st.session_state.golden_set[q["query"]].get("relevant_pattern_ids")
    )
    st.progress(annotated / total, text=f"{annotated}/{total} 已标注")
    st.divider()

    selected_query = st.radio(
        "选择 query：",
        options=[q["query"] for q in SAMPLE_QUERIES],
        format_func=lambda q: (
            f"✅ {q}" if (
                q in st.session_state.golden_set
                and st.session_state.golden_set[q].get("relevant_pattern_ids")
            ) else f"⬜ {q}"
        ),
        label_visibility="collapsed",
    )

    st.divider()
    if st.button("💾 保存所有标注", type="primary", use_container_width=True):
        save_golden_set(st.session_state.golden_set)
        st.success(f"已保存到 {GOLDEN_SET_PATH.name}")

    st.markdown("**图例：**")
    st.markdown(
        "<span class='badge vec'>vector</span> 向量检索<br>"
        "<span class='badge str'>struct</span> 结构化查询<br>"
        "<span class='badge kw'>keyword</span> 关键词搜索<br>"
        "<span class='badge multi'>multi</span> 多来源命中",
        unsafe_allow_html=True,
    )

# ── 主区域 ────────────────────────────────────────────────────────────────────
current_q = next(q for q in SAMPLE_QUERIES if q["query"] == selected_query)

col_title, col_cat = st.columns([4, 1])
with col_title:
    st.markdown(f"## 🔍 `{selected_query}`")
with col_cat:
    st.markdown(
        f"<span class='cat-tag'>{current_q['category']}</span>",
        unsafe_allow_html=True,
    )

# 搜索候选（缓存避免重复调 API）
if selected_query not in st.session_state.candidates_cache:
    with st.spinner("搜索候选中（向量 + 结构化 + 关键词）…"):
        vec_res    = vector_search(selected_query, patterns, embeddings, openai_client, top_n=20)
        struct_res = structured_search(patterns, selected_query, top_n=20)
        kw_res     = keyword_search(patterns, selected_query, top_n=20)
        merged     = merge_candidates(kw_res, vec_res, struct_res)
        st.session_state.candidates_cache[selected_query] = {
            "candidates": merged,
            "vec_count":    len(vec_res),
            "struct_count": len(struct_res),
            "kw_count":     len(kw_res),
        }

cache      = st.session_state.candidates_cache[selected_query]
candidates = cache["candidates"]

# 统计来源
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("候选总数", len(candidates))
col_b.metric("向量检索", cache["vec_count"])
col_c.metric("结构化查询", cache["struct_count"])
col_d.metric("关键词搜索", cache["kw_count"])

# 已标注的 ids
existing     = st.session_state.golden_set.get(selected_query, {})
checked_ids  = set(existing.get("relevant_pattern_ids", []))
new_checked  = set(checked_ids)

st.markdown("---")
st.markdown(
    "勾选所有**真正相关**的图解（标准：如果你搜这个 query，这个结果会让你满意）。"
    f" 建议标注 **5-15 个**，太少 Recall 不稳定，太多标注负担大。"
)

# ── 候选网格 ──────────────────────────────────────────────────────────────────
COLS = 4
for row_start in range(0, len(candidates), COLS):
    cols = st.columns(COLS)
    for col, pattern in zip(cols, candidates[row_start : row_start + COLS]):
        pid     = pattern["id"]
        sources = pattern.get("_sources", [])
        is_multi = len(sources) > 1

        with col:
            # 图片
            photos  = pattern.get("photos") or []
            img_url = photos[0].get("medium_url") if photos else None
            border  = "2px solid #2d7a6e" if pid in new_checked else "1px solid #e0d8cc"
            if img_url:
                st.markdown(
                    f'<img src="{img_url}" style="width:100%;height:160px;'
                    f'object-fit:cover;border-radius:8px;border:{border};margin-top:14px">',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='height:160px;background:#f5f0ea;border-radius:8px;"
                    f"border:{border};display:flex;align-items:center;"
                    f"justify-content:center;color:#aaa;font-size:1.5rem;margin-top:14px'>🧶</div>",
                    unsafe_allow_html=True,
                )

            # 勾选框
            checked = st.checkbox(
                pattern["name"],
                value=pid in new_checked,
                key=f"cb_{selected_query}_{pid}",
            )
            if checked:
                new_checked.add(pid)
            else:
                new_checked.discard(pid)

            # 来源标签（每个来源单独一个 badge，不合并成 multi）
            src_label = {"vector": "vec", "structured": "str", "keyword": "kw"}
            src_name  = {"vector": "vec", "structured": "str", "keyword": "kw"}
            badge_html = ""
            for src in sources:
                cls   = src_label.get(src, "kw")
                label = src_name.get(src, src)[:3]
                badge_html += f"<span class='badge {cls}'>{label}</span>"
            sim = pattern.get("_similarity")
            if sim:
                badge_html += f"<span class='badge vec'>{sim:.3f}</span>"
            st.markdown(badge_html, unsafe_allow_html=True)

            # 详情（紧凑，单行）
            craft  = (pattern.get("craft") or {}).get("name", "")
            yarn   = pattern.get("yarn_weight_description") or ""
            rating = pattern.get("rating_average") or 0.0
            count  = pattern.get("rating_count") or 0
            rating_str = f"⭐ {rating:.1f} ({count})" if rating > 0 else "No ratings"
            free_badge = "<span class='free-badge'>FREE</span>" if pattern.get("free") else ""

            st.markdown(
                f"<div class='card-meta'>{craft} · {yarn}</div>"
                f"<div class='card-meta'>{rating_str} {free_badge}</div>",
                unsafe_allow_html=True,
            )

            # ID + 链接同一行
            permalink = pattern.get("permalink", "")
            url = f"https://www.ravelry.com/patterns/library/{permalink}"
            st.markdown(
                f"<div class='card-row'>"
                f"<span class='pattern-id'>ID: {pid}</span>"
                f"<a href='{url}' target='_blank' style='font-size:0.72rem;color:#2d7a6e;'>"
                f"ravelry ↗</a>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.divider()

# ── 保存 ──────────────────────────────────────────────────────────────────────
col_save, col_info = st.columns([2, 5])
with col_save:
    if st.button("💾 保存此 Query", type="primary", use_container_width=True):
        st.session_state.golden_set[selected_query] = {
            "query":                selected_query,
            "category":             current_q["category"],
            "relevant_pattern_ids": sorted(new_checked),
        }
        save_golden_set(st.session_state.golden_set)
        st.success(f"已保存 {len(new_checked)} 个相关图解 ✅")

with col_info:
    if new_checked:
        st.info(
            f"当前已选：**{len(new_checked)}** 个  |  "
            f"IDs: `{sorted(new_checked)}`"
        )
    else:
        st.warning("还没有选任何图解——真的没有相关结果也可以保存空列表。")

# ── 完整 JSON 预览 ────────────────────────────────────────────────────────────
with st.expander("📄 golden_set.json 预览"):
    st.json(list(st.session_state.golden_set.values()))
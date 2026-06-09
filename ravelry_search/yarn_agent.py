"""
yarn_agent.py — LangGraph yarn planning agent.
Given a user's yarn stash description, recommends knitting/crochet patterns that fit.

parse_yarn_input supports two formats:
  1. "我有 200g DK 线"              → extract grams/weight directly, compute yardage
  2. "我有 3 团 Lana Grossa Colors" → Ravelry yarn API lookup → LLM fallback if not found
"""

import json
import os
import sys
from pathlib import Path
from typing import Annotated, Literal, Optional

import instructor
import numpy as np
import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from openai import OpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# Ensure ravelry_v0 sibling modules are importable regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))
from hybrid_search import hybrid_search
from rag_chroma import PatternSearchIntent

load_dotenv()

# ── Module-level resources (loaded once) ───────────────────────────────────────

_PATTERNS_PATH = Path(__file__).parent / "data" / "patterns.json"
_EMBEDDINGS_PATH = Path(__file__).parent / "data" / "embeddings.npy"

_patterns: list[dict] = json.loads(_PATTERNS_PATH.read_text(encoding="utf-8"))
_embeddings: np.ndarray = np.load(_EMBEDDINGS_PATH)
_openai_client = OpenAI()
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Yards per gram by weight — used when Ravelry has no yardage data
_YARDS_PER_GRAM: dict[str, float] = {
    "Lace": 4.5, "Fingering": 3.5, "Sport": 3.0, "DK": 2.3,
    "Worsted": 2.0, "Aran": 1.8, "Bulky": 1.2, "Super Bulky": 0.7,
}


# ── State ──────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]
    yarn_grams: Optional[float]            # user's yarn weight in grams
    yarn_weight: Optional[str]             # "DK", "Bulky", etc.
    yarn_yardage: Optional[float]          # estimated yardage
    preferred_categories: list[str]        # Hat, Sweater, …
    candidates: list[dict]                 # patterns with enough yardage
    insufficient: list[dict]               # patterns that need more yardage
    search_query: Optional[str]            # query sent to hybrid_search
    human_feedback: Optional[str]          # "放宽" or "放弃"
    search_intent: Optional[dict]          # PatternSearchIntent.model_dump()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class YarnInfo(BaseModel):
    yarn_grams: float = Field(description="Total weight in grams")
    yarn_weight: str = Field(
        description="Yarn weight: Lace / Fingering / Sport / DK / Worsted / Aran / Bulky / Super Bulky"
    )
    yarn_yardage: float = Field(description="Total yards available")
    preferred_categories: list[str] = Field(
        default_factory=list,
        description="Project categories the user mentioned, e.g. Hat, Sweater, Shawl, Scarf",
    )
    needs_confirmation: bool = Field(
        default=False,
        description="True when data came from LLM estimation rather than direct input or Ravelry API",
    )
    confirmation_message: str = Field(
        default="",
        description="Warning to display to the user when needs_confirmation is True",
    )


class _ParsedInput(BaseModel):
    """LLM output for classifying user input as direct-grams or by-yarn-name."""
    input_type: Literal["direct", "by_name"] = Field(
        description="'direct' if user gave grams/weight; 'by_name' if user gave a yarn brand/name"
    )
    # direct format
    yarn_grams: Optional[float] = Field(None, description="Total grams stated by user")
    yarn_weight: Optional[str] = Field(None, description="Weight category stated by user")
    # by_name format
    skein_count: Optional[float] = Field(None, description="Number of skeins/balls")
    yarn_name: Optional[str] = Field(None, description="Full yarn brand + name to look up")
    # common
    preferred_categories: list[str] = Field(default_factory=list)


class _YarnEstimate(BaseModel):
    """LLM fallback: estimated per-skein properties when Ravelry lookup fails."""
    yarn_weight: str = Field(
        description="Estimated weight: Lace/Fingering/Sport/DK/Worsted/Aran/Bulky/Super Bulky"
    )
    grams_per_skein: float = Field(description="Estimated grams per skein")
    yardage_per_skein: float = Field(description="Estimated yards per skein")
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence in the estimate based on how well-known this yarn is"
    )


# ── parse_yarn_input helpers ──────────────────────────────────────────────────

def _classify_input(user_message: str) -> _ParsedInput:
    """Use gpt-4o-mini + instructor to classify input format and extract fields."""
    client = instructor.from_openai(OpenAI())
    return client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=250,
        response_model=_ParsedInput,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user's yarn description into one of two formats:\n"
                    "  'direct'  — user stated total grams and/or weight "
                    "(e.g. '200g DK 线', '50g lace', '300g Bulky 线')\n"
                    "  'by_name' — user mentioned a yarn brand/name + skein count "
                    "(e.g. '3 团 Lana Grossa Colors for You', '2 skeins Malabrigo Worsted')\n\n"
                    "For 'direct': extract yarn_grams (float) and yarn_weight normalized to "
                    "Lace/Fingering/Sport/DK/Worsted/Aran/Bulky/Super Bulky.\n"
                    "For 'by_name': extract skein_count (float, default 1) and yarn_name "
                    "(full brand + yarn name as a search string).\n"
                    "Also extract preferred_categories (Hat, Sweater, Shawl, Scarf, etc.) from both."
                ),
            },
            {"role": "user", "content": user_message},
        ],
    )


_CROCHET_WORDS = {"钩针", "crochet", "hook", "钩"}
_KNITTING_WORDS = {"棒针", "knitting", "needle", "棒"}


def _detect_craft(user_message: str) -> Optional[Literal["Knitting", "Crochet"]]:
    """Keyword-only craft detection — never infers from context."""
    msg = user_message.lower()
    if any(w in msg for w in _CROCHET_WORDS):
        return "Crochet"
    if any(w in msg for w in _KNITTING_WORDS):
        return "Knitting"
    return None


def _parse_search_intent(user_message: str) -> PatternSearchIntent:
    """Use gpt-4o-mini + instructor to extract free_only/min_rating/semantic_query filters.
    Craft is determined by keyword matching, not the LLM, to prevent context inference."""
    client = instructor.from_openai(OpenAI())
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=200,
        response_model=PatternSearchIntent,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract search filters from a yarn + pattern query. Rules:\n"
                    "- semantic_query: core search terms after removing yarn/weight info "
                    "(e.g. '300g Bulky 免费钩针帽子' → 'hat')\n"
                    "- craft: ONLY set if user explicitly mentions one of these keywords:\n"
                    "  Crochet triggers: '钩针', 'crochet', 'hook', '钩'\n"
                    "  Knitting triggers: '棒针', 'knitting', 'needle', '棒'\n"
                    "  If NONE of these words appear in the user message, craft MUST be null.\n"
                    "  DO NOT infer craft from context, yarn type, or project type.\n"
                    "  When in doubt, use null.\n"
                    "- free_only: '免费'/'free' → true; else false\n"
                    "- min_rating: '评分X以上'/'rating above X' → X; else 0.0\n"
                    "- yarn_weight: do NOT fill — leave null\n"
                    "- needle_size_min/max: do NOT fill\n"
                    "- include_fibers/exclude_fibers: do NOT fill\n"
                    "- categories: do NOT fill"
                ),
            },
            {"role": "user", "content": user_message},
        ],
    )
    # Override craft with reliable keyword check — prevents LLM from inferring via context
    return result.model_copy(update={"craft": _detect_craft(user_message)})


def _lookup_yarn_on_ravelry(yarn_name: str) -> Optional[dict]:
    """
    Search Ravelry yarn database for per-skein grams/yardage/weight.
    Returns dict(name, grams, yardage, yarn_weight) or None if not found / creds missing.
    """
    username = os.environ.get("RAVELRY_USERNAME", "")
    password = os.environ.get("RAVELRY_PASSWORD", "")
    if not username:
        return None

    auth = (username, password)
    try:
        search = requests.get(
            "https://api.ravelry.com/yarns/search.json",
            auth=auth,
            params={"query": yarn_name, "page_size": 5, "sort": "projects"},
            timeout=10,
        )
        search.raise_for_status()
        yarns = search.json().get("yarns", [])
        if not yarns:
            return None

        print(f"  [Ravelry] 搜索「{yarn_name}」前 3 结果：{[y.get('name') for y in yarns[:3]]}")

        detail = requests.get(
            f"https://api.ravelry.com/yarns/{yarns[0]['id']}.json",
            auth=auth,
            timeout=10,
        )
        detail.raise_for_status()
        yarn = detail.json().get("yarn", {})

        grams = yarn.get("grams")
        yardage = yarn.get("yardage")
        weight_name = (yarn.get("yarn_weight") or {}).get("name", "")

        return {
            "name": yarn.get("name", ""),
            "grams": float(grams) if grams else None,
            "yardage": float(yardage) if yardage else None,
            "yarn_weight": weight_name,
        }
    except Exception:
        return None


def _estimate_yarn_by_llm(
    yarn_name: str,
    skein_count: float,
    preferred_categories: list[str],
) -> YarnInfo:
    """LLM fallback estimation when Ravelry lookup fails; sets needs_confirmation=True."""
    client = instructor.from_openai(OpenAI())
    est = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=200,
        response_model=_YarnEstimate,
        messages=[
            {
                "role": "system",
                "content": (
                    "Estimate per-skein properties for this yarn based on its brand and name. "
                    "Common defaults if unsure: "
                    "DK 50g/115yd, Worsted 100g/200yd, Bulky 100g/120yd, "
                    "Fingering 100g/400yd, Lace 25g/200yd, Aran 100g/185yd."
                ),
            },
            {"role": "user", "content": f"Estimate properties for: {yarn_name}"},
        ],
    )

    total_grams = round(est.grams_per_skein * skein_count, 1)
    total_yardage = round(est.yardage_per_skein * skein_count, 1)
    msg = (
        f"在 Ravelry 未找到「{yarn_name}」。"
        f"AI 估算：每团约 {est.grams_per_skein:.0f}g / {est.yardage_per_skein:.0f} 码"
        f"（{est.yarn_weight}，置信度：{est.confidence}），"
        f"{skein_count:.0f} 团共约 {total_grams:.0f}g / {total_yardage:.0f} 码。"
        f"请核对包装标签后确认数据是否正确。"
    )
    return YarnInfo(
        yarn_grams=total_grams,
        yarn_weight=est.yarn_weight,
        yarn_yardage=total_yardage,
        preferred_categories=preferred_categories,
        needs_confirmation=True,
        confirmation_message=msg,
    )


# ── Tools ──────────────────────────────────────────────────────────────────────

def parse_yarn_input(user_message: str) -> YarnInfo:
    """
    Parse yarn input in two formats:
      Format 1 — direct: "200g DK 线"
        → LLM extracts grams + weight, code computes yardage via _YARDS_PER_GRAM.
      Format 2 — by name: "3 团 Lana Grossa Colors for You"
        → Ravelry yarn search API for per-skein grams/yardage/weight.
        → If not found: LLM estimates + sets needs_confirmation=True.
    """
    parsed = _classify_input(user_message)

    if parsed.input_type == "direct":
        weight = parsed.yarn_weight or "DK"
        grams = parsed.yarn_grams or 0.0
        yardage = round(grams * _YARDS_PER_GRAM.get(weight, 2.0), 1)
        return YarnInfo(
            yarn_grams=grams,
            yarn_weight=weight,
            yarn_yardage=yardage,
            preferred_categories=parsed.preferred_categories,
        )

    # by_name: try Ravelry first
    yarn_name = parsed.yarn_name or ""
    skein_count = parsed.skein_count or 1.0

    yarn_data = _lookup_yarn_on_ravelry(yarn_name)
    if yarn_data:
        per_g = yarn_data["grams"] or 0.0
        per_y = yarn_data["yardage"] or 0.0
        weight = yarn_data["yarn_weight"] or "DK"
        # Estimate yardage from grams if Ravelry has grams but not yardage
        if per_y == 0.0 and per_g > 0.0:
            per_y = per_g * _YARDS_PER_GRAM.get(weight, 2.0)
        return YarnInfo(
            yarn_grams=round(per_g * skein_count, 1),
            yarn_weight=weight,
            yarn_yardage=round(per_y * skein_count, 1),
            preferred_categories=parsed.preferred_categories,
        )

    # Ravelry not found → LLM estimate with confirmation warning
    return _estimate_yarn_by_llm(yarn_name, skein_count, parsed.preferred_categories)




def check_yardage(pattern: dict, available_yardage: float) -> dict:
    """Return a yardage-sufficiency verdict for one pattern.

    sufficient : available >= required_min
    tight      : available >= required_min × 0.9  (within 10%)
    insufficient: available < required_min × 0.9
    unknown    : pattern has no yardage data
    """
    raw_min = pattern.get("yardage")
    if raw_min is None:
        return {
            "pattern_name": pattern.get("name", ""),
            "permalink": pattern.get("permalink", ""),
            "required_min": None,
            "required_max": None,
            "available": available_yardage,
            "status": "unknown",
            "shortage": None,
        }

    required_min = float(raw_min)
    required_max = float(pattern.get("yardage_max") or required_min)

    if required_min == 0:
        status, shortage = "sufficient", None
    elif available_yardage >= required_min:
        status, shortage = "sufficient", None
    elif available_yardage >= required_min * 0.9:
        status, shortage = "tight", round(required_min - available_yardage, 1)
    else:
        status, shortage = "insufficient", round(required_min - available_yardage, 1)

    return {
        "pattern_name": pattern.get("name", ""),
        "permalink": pattern.get("permalink", ""),
        "required_min": required_min,
        "required_max": required_max,
        "available": available_yardage,
        "status": status,
        "shortage": shortage,
    }


def suggest_smaller_patterns(
    yarn_weight: str,
    max_yardage: float,
    intent: Optional[PatternSearchIntent] = None,
) -> list[dict]:
    """Find patterns within max_yardage; preserves craft/free_only from intent."""
    search_intent = (intent or PatternSearchIntent(semantic_query="")).model_copy(update={
        "yarn_weight": yarn_weight,
        "semantic_query": "small project",
    })
    results = hybrid_search(
        query="small project",
        patterns=_patterns,
        embeddings=_embeddings,
        openai_client=_openai_client,
        top_k=5,
        intent=search_intent,
    )
    return [
        p for p in results
        if not p.get("yardage") or p.get("yardage", 0) <= max_yardage
    ]


# ── Nodes ──────────────────────────────────────────────────────────────────────

def parse_input_node(state: State) -> dict:
    last_human = next(
        m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)
    )
    parsed = _classify_input(last_human)
    print(f"  [parse] input_type={parsed.input_type!r}  yarn_name={parsed.yarn_name!r}  grams={parsed.yarn_grams}  weight={parsed.yarn_weight!r}")
    info = parse_yarn_input(last_human)
    prefs = info.preferred_categories

    search_intent = _parse_search_intent(last_human)
    search_intent = search_intent.model_copy(update={
        "yarn_weight": info.yarn_weight,
        "semantic_query": " ".join(prefs) if prefs else info.yarn_weight or "knitting project",
    })
    print(f"  [intent] craft={search_intent.craft!r}  free_only={search_intent.free_only}  min_rating={search_intent.min_rating}  semantic_query={search_intent.semantic_query!r}")

    summary = (
        f"解析：{info.yarn_grams}g {info.yarn_weight} 线 ≈ {info.yarn_yardage:.0f} 码"
        + (f"，偏好：{', '.join(prefs)}" if prefs else "")
    )
    if info.needs_confirmation:
        summary += f"\n⚠️  {info.confirmation_message}"
    return {
        "messages": [AIMessage(content=summary)],
        "yarn_grams": info.yarn_grams,
        "yarn_weight": info.yarn_weight,
        "yarn_yardage": info.yarn_yardage,
        "preferred_categories": prefs,
        "search_query": " ".join(prefs) if prefs else info.yarn_weight,
        "search_intent": search_intent.model_dump(),
    }


def search_node(state: State) -> dict:
    intent_dict = state.get("search_intent") or {}
    intent = PatternSearchIntent(**intent_dict) if intent_dict else None
    query = (intent.semantic_query if intent else None) or state.get("search_query") or "knitting project"
    raw = hybrid_search(
        query=query,
        patterns=_patterns,
        embeddings=_embeddings,
        openai_client=_openai_client,
        top_k=10,
        intent=intent,
    )
    results = [
        {
            "id": p.get("id"),
            "name": p.get("name", ""),
            "permalink": p.get("permalink", ""),
            "yardage": p.get("yardage"),
            "yardage_max": p.get("yardage_max"),
            "yarn_weight_description": p.get("yarn_weight_description", ""),
            "_rrf_score": p.get("_rrf_score", 0),
        }
        for p in raw
    ]
    return {
        "messages": [AIMessage(content=f"搜索到 {len(results)} 个候选图解")],
        "candidates": results,
    }


def check_yardage_node(state: State) -> dict:
    available = state.get("yarn_yardage") or 0
    sufficient: list[dict] = []
    insufficient: list[dict] = []

    for p in state.get("candidates", []):
        result = check_yardage(p, available)
        if result["status"] in ("sufficient", "tight"):
            sufficient.append({**p, **result})
        elif result["status"] == "insufficient":
            insufficient.append({**p, **result})
        # unknown (no yardage data) → silently skip

    return {
        "messages": [AIMessage(
            content=(
                f"线量检查（可用 {available:.0f} 码）："
                f"线量够 {len(sufficient)} 个，不够 {len(insufficient)} 个"
            )
        )],
        "candidates": sufficient,
        "insufficient": insufficient,
    }


def human_feedback_node(state: State) -> dict:
    """Pause and ask the user whether to relax constraints."""
    examples = [
        p.get("pattern_name") or p.get("name", "")
        for p in state.get("insufficient", [])[:3]
    ]
    choice = interrupt({
        "message": "所有候选图解线量都不足，是否放宽条件搜索更小的项目？(放宽/放弃)",
        "insufficient_examples": examples,
        "available_yardage": state.get("yarn_yardage"),
    })
    return {
        "messages": [AIMessage(content=f"用户选择：{choice}")],
        "human_feedback": str(choice),
    }


def suggest_smaller_node(state: State) -> dict:
    intent_dict = state.get("search_intent") or {}
    intent = PatternSearchIntent(**intent_dict) if intent_dict else None
    results = suggest_smaller_patterns(
        yarn_weight=state.get("yarn_weight") or "",
        max_yardage=state.get("yarn_yardage") or 0,
        intent=intent,
    )
    return {
        "messages": [AIMessage(content=f"找到 {len(results)} 个线量合适的小项目")],
        "candidates": results,
    }


def recommend_node(state: State) -> dict:
    candidates = state.get("candidates", [])
    if not candidates:
        return {"messages": [AIMessage(content="很遗憾，没有找到线量合适的图解。")]}

    pattern_lines = "\n".join(
        f"- {p.get('pattern_name') or p.get('name', '')} "
        f"（需 {p.get('yardage', '?')}–{p.get('yardage_max', '?')} 码，"
        f"https://www.ravelry.com/patterns/library/{p.get('permalink', '')}）"
        for p in candidates[:5]
    )
    yarn_desc = (
        f"{state.get('yarn_grams', '?')}g {state.get('yarn_weight', '')} 线"
        f"（约 {(state.get('yarn_yardage') or 0):.0f} 码）"
    )
    prefs = state.get("preferred_categories", [])
    pref_line = f"偏好品类：{', '.join(prefs)}" if prefs else "无特定品类偏好"

    response = _llm.invoke([
        SystemMessage(content=(
            "你是一个专业编织顾问。根据用户的线量和候选图解，"
            "用中文给出简洁的推荐，说明每个图解为什么适合这批线，并附上 Ravelry 链接。"
        )),
        HumanMessage(content=(
            f"用户的线：{yarn_desc}\n{pref_line}\n\n候选图解：\n{pattern_lines}\n\n"
            "请推荐最适合的 1-3 个图解并说明理由。"
        )),
    ])
    return {"messages": [response]}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_check(state: State) -> str:
    return "recommend_node" if state.get("candidates") else "human_feedback_node"


def route_after_feedback(state: State) -> str:
    return "suggest_smaller_node" if "放宽" in (state.get("human_feedback") or "") else END


# ── Graph ──────────────────────────────────────────────────────────────────────

builder = StateGraph(State)
builder.add_node("parse_input_node", parse_input_node)
builder.add_node("search_node", search_node)
builder.add_node("check_yardage_node", check_yardage_node)
builder.add_node("human_feedback_node", human_feedback_node)
builder.add_node("suggest_smaller_node", suggest_smaller_node)
builder.add_node("recommend_node", recommend_node)

builder.add_edge(START, "parse_input_node")
builder.add_edge("parse_input_node", "search_node")
builder.add_edge("search_node", "check_yardage_node")
builder.add_conditional_edges("check_yardage_node", route_after_check)
builder.add_conditional_edges("human_feedback_node", route_after_feedback)
builder.add_edge("suggest_smaller_node", "recommend_node")
builder.add_edge("recommend_node", END)

graph = builder.compile(checkpointer=MemorySaver())


# ── Runner helpers ─────────────────────────────────────────────────────────────

def _print_update(node: str, update: dict) -> None:
    msgs = update.get("messages", [])
    if msgs:
        last = msgs[-1]
        content = last.content if hasattr(last, "content") else str(last)
        print(f"  [{node}] {content}")


def run_scenario(
    user_message: str,
    thread_id: str,
    simulate_feedback: Optional[str] = None,
) -> None:
    """Run one scenario; simulate_feedback bypasses the interactive prompt."""
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n{'═' * 60}")
    print(f"  {user_message}")
    print(f"{'═' * 60}")

    initial_state: State = {
        "messages": [HumanMessage(content=user_message)],
        "yarn_grams": None,
        "yarn_weight": None,
        "yarn_yardage": None,
        "preferred_categories": [],
        "candidates": [],
        "insufficient": [],
        "search_query": None,
        "human_feedback": None,
        "search_intent": None,
    }

    interrupt_payload: dict = {}
    for step in graph.stream(initial_state, config, stream_mode="updates"):
        for node, update in step.items():
            if node == "__interrupt__":
                try:
                    interrupt_payload = update[0].value
                except (IndexError, AttributeError):
                    interrupt_payload = {}
            else:
                _print_update(node, update)

    snapshot = graph.get_state(config)
    if not snapshot.next:
        return  # completed without interruption

    # ── Human-in-the-loop ────────────────────────────────────────────────────
    print(f"\n⏸  {interrupt_payload.get('message', '线量不足')}")
    examples = interrupt_payload.get("insufficient_examples", [])
    if examples:
        print(f"   示例不足图解：{', '.join(examples)}")

    if simulate_feedback is not None:
        user_choice = simulate_feedback
        print(f"   [模拟输入] {user_choice}")
    else:
        user_choice = input("   请输入 (放宽/放弃): ").strip()

    for step in graph.stream(Command(resume=user_choice), config, stream_mode="updates"):
        for node, update in step.items():
            if node != "__interrupt__":
                _print_update(node, update)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # craft=Crochet, free_only=True
    run_scenario("我有 300g Bulky 线，找免费钩针图解", thread_id="1")
    # min_rating=4.5, categories → Hat
    run_scenario("我有 200g DK 线，评分4.5以上的帽子", thread_id="2")
    # craft=Knitting
    run_scenario("我有 3 团 Drops Alpaca DK，棒针图解", thread_id="3")
    # low yardage → triggers human-in-the-loop
    run_scenario("50g lace 线，想织毛衣", thread_id="4", simulate_feedback="放宽")


if __name__ == "__main__":
    main()

"""
guardrails.py — Input validation and fallback strategies.
All checks are keyword-based (no LLM) for speed and cost efficiency.
"""

KNITTING_KEYWORDS = {
    # English
    "knit", "knitting", "crochet", "yarn", "pattern", "needle", "hook",
    "stitch", "wool", "fiber", "fibre", "gauge", "swatch", "ravelry",
    "cardigan", "sweater", "pullover", "vest", "shawl", "sock", "hat",
    "beanie", "mitten", "glove", "scarf", "cowl", "blanket", "lace",
    "cable", "colorwork", "dk", "worsted", "bulky", "fingering", "aran",
    # Chinese
    "棒针", "钩针", "编织", "毛线", "图解", "针", "线", "毛衣", "背心",
    "帽子", "袜子", "围巾", "披肩", "手套", "毯子",
}

OFF_TOPIC_RESPONSE = (
    "🧶 I can only help with knitting and crochet pattern searches. "
    "Try something like:\n"
    "- \"free beginner sweater pattern\"\n"
    "- \"DK weight cardigan rating above 4.5\"\n"
    "- \"粗线钩针帽子\""
)

def is_knitting_related(query: str) -> bool:
    """Keyword-based check — fast, no LLM cost."""
    q = query.lower()
    return any(kw in q for kw in KNITTING_KEYWORDS)


def get_fallback_patterns(patterns: list[dict], top_n: int = 10) -> list[dict]:
    """
    Fallback tier 2: return top-rated patterns when search yields no results.
    Sorted by rating_average descending, minimum 10 ratings required.
    """
    candidates = [
        p for p in patterns
        if (p.get("rating_average") or 0) >= 4.5
        and (p.get("rating_count") or 0) >= 10
    ]
    return sorted(
        candidates,
        key=lambda p: (p.get("rating_average") or 0),
        reverse=True,
    )[:top_n]


def relax_intent(intent):
    """
    Fallback tier 1: relax constraints one at a time.
    Returns a new intent with the most restrictive filter removed.
    Priority: min_rating > yarn_weight > needle_size > craft
    """
    relaxed = intent.model_copy()
    if relaxed.min_rating > 0:
        relaxed.min_rating = 0.0
        return relaxed, "Relaxed: removed minimum rating filter"
    if relaxed.yarn_weight:
        relaxed.yarn_weight = None
        return relaxed, "Relaxed: removed yarn weight filter"
    if relaxed.needle_size_min or relaxed.needle_size_max:
        relaxed.needle_size_min = None
        relaxed.needle_size_max = None
        return relaxed, "Relaxed: removed needle size filter"
    if relaxed.craft:
        relaxed.craft = None
        return relaxed, "Relaxed: removed craft filter"
    return relaxed, None  # nothing left to relax

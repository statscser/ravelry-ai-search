def clean_notes(notes: str) -> str:
    # 只保留英文部分，去掉西班牙语
    # 大多数双语图解都是英文在前，用 ___ 或 ESPAÑOL 分隔
    if "ESPAÑOL" in notes.upper():
        notes = notes[:notes.upper().index("ESPAÑOL")]
    if "---" in notes:
        notes = notes[:notes.index("---")]
    return notes.strip()

def build_text_for_embedding(pattern: dict) -> str:
    """
    把 pattern 的关键信息拼成一段文字，用于生成 embedding。
    想清楚：用户搜索时会用什么词？这些词应该出现在这段文字里。
    """
    parts = []
    if pattern.get("name"):
        parts.append(f"Name{pattern['name']}")
    if pattern.get("craft"):
        parts.append(f"Craft: {pattern['craft']['name']}")
    if pattern.get("yarn_weight_description"):
        parts.append(f"Yarn weight: {pattern['yarn_weight_description']}")
    # if pattern.get("yarn_weight"):
    #     parts.append(f"yarn_weight: {'; '.join(f'{k}: {v}' for k, v in pattern['yarn_weight'].items() if v)}")
    if pattern.get("pattern_needle_sizes"):
        needle_names = [n["name"] for n in pattern["pattern_needle_sizes"]]
        parts.append(f"Needle sizes: {', '.join(needle_names)}")
    if pattern.get("pattern_categories"):
        parts.append(f"Categories: {', '.join(c['name'] for c in pattern['pattern_categories'])}")
    if pattern.get("notes"):
        en_notes = clean_notes(pattern["notes"])
        parts.append(f"Description: {en_notes}")
    if pattern.get("pattern_attributes"):
        parts.append(f"Attributes: {', '.join(a['permalink'] for a in pattern['pattern_attributes'])}")
    # TODO Week 5：从 packs 字段提取 fiber 信息加入 embedding
    # pattern['packs'][0]['yarn_weight']['name'] 有线重信息

    return "\n".join(parts)
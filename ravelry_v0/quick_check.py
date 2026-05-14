import json
from pathlib import Path

patterns = json.loads(Path("data/patterns.json").read_text(encoding="utf-8"))

# 看看数据里有多少 crochet blanket
blankets = [
    p for p in patterns
    if "blanket" in p.get("name", "").lower()
    or any("blanket" in c.get("name", "").lower() 
           for c in (p.get("pattern_categories") or []))
]

print(f"数据里共有 {len(blankets)} 个 blanket 图解")
for b in blankets[:10]:
    craft = (b.get("craft") or {}).get("name", "")
    print(f"  {b['name']} · {craft}")
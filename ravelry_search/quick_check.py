# import json
# from pathlib import Path

# patterns = json.loads(Path("data/patterns.json").read_text(encoding="utf-8"))

# ### 看看数据里有多少 crochet blanket
# blankets = [
#     p for p in patterns
#     if "blanket" in p.get("name", "").lower()
#     or any("blanket" in c.get("name", "").lower() 
#            for c in (p.get("pattern_categories") or []))
# ]

# print(f"数据里共有 {len(blankets)} 个 blanket 图解")
# for b in blankets[:10]:
#     craft = (b.get("craft") or {}).get("name", "")
#     print(f"  {b['name']} · {craft}")


# ### check_attributes.py
# import json
# from pathlib import Path
# from collections import Counter

# patterns = json.loads(Path("data/patterns.json").read_text(encoding="utf-8"))

# all_attrs = []
# for p in patterns:
#     for a in (p.get("pattern_attributes") or []):
#         all_attrs.append(a["permalink"])

# counter = Counter(all_attrs)
# print(f"总共有 {len(counter)} 种不同的 attribute\n")

# print("=== 全部 attributes（按频率排序）===")
# for attr, count in counter.most_common():
#     print(f"  {count:4d}  {attr}")


### Check packs字段
import json
from pathlib import Path
from collections import Counter

patterns = json.loads(Path("data/patterns.json").read_text(encoding="utf-8"))

# 统计 packs 字段的覆盖率
has_packs = sum(1 for p in patterns if p.get("packs"))
print(f"有 packs 字段：{has_packs}/{len(patterns)} ({has_packs/len(patterns)*100:.0f}%)")

# 看看 packs 里有什么 fiber 信息
# 实际上 fiber 信息在 yarn.yarn_company_name 或者需要另外调 yarn API
# 先看看 packs 里有什么字段
sample_with_packs = [p for p in patterns if p.get("packs")][:3]
for p in sample_with_packs:
    print(f"\n{p['name']}:")
    for pack in p["packs"][:2]:
        print(f"  yarn_name: {pack.get('yarn_name')}")
        print(f"  yarn_weight: {(pack.get('yarn_weight') or {}).get('name')}")
        # fiber 信息在哪里？
        print(f"  pack keys: {list(pack.keys())}")
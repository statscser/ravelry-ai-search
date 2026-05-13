import json
from pathlib import Path

data = json.loads(Path("data/patterns.json").read_text(encoding="utf-8"))

print(f"总条数：{len(data)}")

# 检查关键字段的覆盖率
fields = ["name", "craft", "yarn_weight", "pattern_needle_sizes", 
          "pattern_categories", "notes", "text_for_embedding"]

for field in fields:
    has_field = sum(1 for p in data if p.get(field))
    print(f"{field}: {has_field}/{len(data)} ({has_field/len(data)*100:.0f}%)")

# 看第一条的 text_for_embedding
print("\n第一条 text_for_embedding：")
print(data[0]["text_for_embedding"])
import json
from pathlib import Path
from data_processor import build_text_for_embedding

path = Path("data/patterns.json")
patterns = json.loads(path.read_text(encoding="utf-8"))

for p in patterns:
    p["text_for_embedding"] = build_text_for_embedding(p)

path.write_text(json.dumps(patterns, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Updated {len(patterns)} patterns")
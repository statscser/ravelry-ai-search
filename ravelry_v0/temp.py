# check_data_quality.py
import json
from pathlib import Path

patterns = json.loads(Path("data/patterns.json").read_text(encoding="utf-8"))
print(f"总条数：{len(patterns)}")

no_notes = [p for p in patterns if not p.get("notes")]
no_photo = [p for p in patterns if not p.get("photos")]
no_either = [p for p in patterns if not p.get("notes") and not p.get("photos")]

print(f"没有 notes：{len(no_notes)}")
print(f"没有封面图：{len(no_photo)}")
print(f"两者都没有：{len(no_either)}")
print(f"清洗后剩余：{len(patterns) - len(set([p['id'] for p in no_notes] + [p['id'] for p in no_photo]))}")
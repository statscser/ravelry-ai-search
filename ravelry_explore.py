import json
import os
import time
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv("RAVELRY_USERNAME")
PASSWORD = os.getenv("RAVELRY_PASSWORD")
BASE_URL = "https://api.ravelry.com"

def search_patterns(query="", page=1, page_size=100):
    response = requests.get(
        f"{BASE_URL}/patterns/search.json",
        auth=(USERNAME, PASSWORD),
        params={
            "query": query,
            "page": page,
            "page_size": page_size,
            "sort": "best",
        }
    )
    response.raise_for_status()
    return response.json()

def get_pattern_detail(pattern_id: int):
    response = requests.get(
        f"{BASE_URL}/patterns/{pattern_id}.json",
        auth=(USERNAME, PASSWORD),
    )
    response.raise_for_status()
    return response.json()["pattern"]

def main():
    print("拉取数据中...")
    data = search_patterns(query="sweater", page_size=100)
    
    # 存成 JSON
    Path("data").mkdir(exist_ok=True)
    Path("data/sample_100.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    # 快速分析数据质量
    patterns = data["patterns"]
    print(f"\n总共拿到：{len(patterns)} 条")
    
    # 抽一条看完整结构
    sample = patterns[0]
    print(f"\n第一条数据的所有字段：")
    for key, value in sample.items():
        print(f"  {key}: {value}")

    # 拉取第一条的完整详情，看看有没有我们关心的字段
    print("拉取第一条的完整详情...")
    first_id = patterns[0]["id"]
    detail = get_pattern_detail(first_id)

    fields = ["name", "yarn_weight", "needle_sizes", "notes", 
              "pattern_categories", "rating_average", "free"]
    print("\n关键字段：")
    for f in fields:
        print(f"  {f}: {detail.get(f)}")
    
    # 完整详情存下来研究
    Path("data/sample_detail.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
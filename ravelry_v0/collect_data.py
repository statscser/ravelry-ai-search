import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from data_processor import build_text_for_embedding

load_dotenv()

USERNAME = os.environ["RAVELRY_USERNAME"]
PASSWORD = os.environ["RAVELRY_PASSWORD"]
AUTH = (USERNAME, PASSWORD)

BASE_URL = "https://api.ravelry.com"
OUTPUT_PATH = Path(__file__).parent / "data" / "patterns.json"

PER_PAGE = 100
SLEEP_BETWEEN = 0.7
MAX_RETRIES = 3

# 按 category 采集，每个 category 目标数量
CATEGORIES = [
    ("sweater",       500),
    ("cardigan",      500),
    ("hat",           400),
    ("socks",         400),
    ("shawl",         300),
    ("blanket",       300),
    ("baby",          300),
    ("mittens gloves",200),
    ("cowl",          200),
    ("top tank",      300),
    ("scarf",         200),
    ("vest",          200),
]

def get_with_retry(url: str, params: dict = None) -> dict:
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, auth=AUTH, params=params)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 10))
            print(f"  429 rate limited — waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
        else:
            resp.raise_for_status()
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {url}")


def fetch_search_page(query: str, page: int) -> list[dict]:
    data = get_with_retry(
        f"{BASE_URL}/patterns/search.json",
        params={
            "query": query,
            "sort": "best",
            "page": page,
            "page_size": PER_PAGE,
        },
    )
    return data.get("patterns", [])


def fetch_pattern_details_batch(pattern_ids: list[int]) -> list[dict]:
    ids_param = " ".join(str(pid) for pid in pattern_ids)
    data = get_with_retry(
        f"{BASE_URL}/patterns.json",
        params={"ids": ids_param},
    )
    return list(data["patterns"].values())


def fetch_category(query: str, target_count: int, seen_ids: set) -> list[dict]:
    """按 category query 采集 target_count 条，跳过 seen_ids 里已有的。"""
    collected = []
    page = 1

    while len(collected) < target_count:
        results = fetch_search_page(query=query, page=page)
        time.sleep(SLEEP_BETWEEN)

        if not results:
            break  # 没有更多结果了

        # 过滤掉已采集的
        new_ids = [r["id"] for r in results if r["id"] not in seen_ids]
        if not new_ids:
            page += 1
            continue

        # 批量拉详情
        details = fetch_pattern_details_batch(new_ids)
        time.sleep(SLEEP_BETWEEN)

        for detail in details:
            if len(collected) >= target_count:
                break
            detail["text_for_embedding"] = build_text_for_embedding(detail)
            collected.append(detail)
            seen_ids.add(detail["id"])

        page += 1

    return collected


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_patterns = []
    seen_ids = set()

    for query, target_count in CATEGORIES:
        print(f"\n📂 采集 '{query}'（目标 {target_count} 条）...")
        results = fetch_category(query, target_count, seen_ids)
        all_patterns.extend(results)
        print(f"  ✅ 新增 {len(results)} 条，总计 {len(all_patterns)} 条（去重后）")

    OUTPUT_PATH.write_text(
        json.dumps(all_patterns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n🎉 Done. {len(all_patterns)} patterns saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
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

PAGES = 10
PER_PAGE = 100
SLEEP_BETWEEN = 0.7
MAX_RETRIES = 3


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


def fetch_search_page(page: int) -> list[dict]:
    data = get_with_retry(
        f"{BASE_URL}/patterns/search.json",
        params={"sort": "best", "page": page, "page_size": PER_PAGE},
    )
    return data["patterns"]


# def fetch_pattern_detail(pattern_id: int) -> dict:
#     data = get_with_retry(f"{BASE_URL}/patterns/{pattern_id}.json")
#     return data["pattern"]

def fetch_pattern_details_batch(pattern_ids: list[int]) -> list[dict]:
    ids_param = " ".join(str(pid) for pid in pattern_ids)
    data = get_with_retry(
        f"{BASE_URL}/patterns.json",
        params={"ids": ids_param},
    )
    return list(data["patterns"].values())

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_patterns = []

    for page in range(1, PAGES + 1):
        print(f"Fetching search page {page}/{PAGES}...")
        results = fetch_search_page(page)
        time.sleep(SLEEP_BETWEEN)

        # 一次拿 100 个 id，批量拉详情
        pattern_ids = [r["id"] for r in results]
        details = fetch_pattern_details_batch(pattern_ids)
        time.sleep(SLEEP_BETWEEN)

        for detail in details:
            detail["text_for_embedding"] = build_text_for_embedding(detail)
            all_patterns.append(detail)

        print(f"  Collected {len(all_patterns)} patterns so far...")

    OUTPUT_PATH.write_text(json.dumps(all_patterns, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. {len(all_patterns)} patterns saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

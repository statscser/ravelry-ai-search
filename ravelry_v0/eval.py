"""
eval.py — Offline evaluation against golden_set.json.
Usage: python eval.py [--version v0]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rag_chroma import load_collection, parse_query, search
from hybrid_search import hybrid_search
import numpy as np

load_dotenv()

GOLDEN_SET_PATH  = Path(__file__).parent / "data" / "golden_set.json"

TOP_K = 20
RECALL_GT_LIMIT = 30  # queries with more GT patterns skip Recall


# ── Metric helpers ─────────────────────────────────────────────────────────────

def mrr_at_k(result_ids: list[int], relevant: set[int], k: int = 20) -> float:
    for rank, pid in enumerate(result_ids[:k], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def precision_at_k(result_ids: list[int], relevant: set[int], k: int = 10) -> float:
    hits = sum(1 for pid in result_ids[:k] if pid in relevant)
    return hits / k


def recall_at_k(
    result_ids: list[int], relevant: set[int], k: int = 10
) -> float | None:
    if len(relevant) > RECALL_GT_LIMIT:
        return None
    if not relevant:
        return 0.0
    hits = sum(1 for pid in result_ids[:k] if pid in relevant)
    return hits / len(relevant)


# ── Main ───────────────────────────────────────────────────────────────────────

def evaluate(version: str = "v0") -> None:
    EVAL_RESULTS_PATH = Path(__file__).parent / "data" / f"eval_results_{version}.json"

    raw = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    golden = [item for item in raw if item.get("relevant_pattern_ids")]
    if not golden:
        print("No annotated queries found in golden_set.json — run the annotator first.")
        return
    print(f"Loaded {len(golden)} annotated queries.\n")

    embeddings = np.load(Path(__file__).parent / "data" / "embeddings.npy")

    openai_client = OpenAI()
    collection, patterns = load_collection()

    rows = []

    for item in golden:
        query      = item["query"]
        relevant   = set(item["relevant_pattern_ids"])
        gt_count   = len(relevant)

        intent  = parse_query(query, openai_client)

        # ### Chroma vector search
        # results = search(
        #     query=intent.semantic_query,
        #     collection=collection,
        #     openai_client=openai_client,
        #     patterns=patterns,
        #     top_k=TOP_K,
        #     intent=intent,
        # )

        ### Hybrid search
        results = hybrid_search(
            query=intent.semantic_query,
            patterns=patterns,
            embeddings=embeddings,
            openai_client=openai_client,
            top_k=TOP_K,
            intent=intent,
        )

        result_ids = [int(p["id"]) for p in results]

        mrr  = mrr_at_k(result_ids, relevant, k=TOP_K)
        p10  = precision_at_k(result_ids, relevant, k=10)
        r10  = recall_at_k(result_ids, relevant, k=10)

        rows.append({
            "query":            query,
            "category":         item.get("category", ""),
            "gt_count":         gt_count,
            "mrr_at_20":        round(mrr, 4),
            "precision_at_10":  round(p10, 4),
            "recall_at_10":     round(r10, 4) if r10 is not None else None,
        })

        r10_str = f"{r10:.4f}" if r10 is not None else "  N/A"
        print(f"  {query[:42]:<42}  GT={gt_count:>3}  MRR={mrr:.4f}  P@10={p10:.4f}  R@10={r10_str}")

    # 在 for loop 结束后，打印前加这行
    print(f"rows count: {len(rows)}")
    print(f"first row: {rows[0] if rows else 'EMPTY'}")

    # ── Print table ────────────────────────────────────────────────────────────
    W = 78
    print("\n" + "=" * W)
    print(f"{'Query':<44} {'GT':>3} {'MRR@20':>8} {'P@10':>7} {'R@10':>7}")
    print("-" * W)
    for r in rows:
        r10_str = f"{r['recall_at_10']:.4f}" if r["recall_at_10"] is not None else "  N/A "
        print(f"{r['query']:<44} {r['gt_count']:>3} {r['mrr_at_20']:>8.4f} {r['precision_at_10']:>7.4f} {r10_str:>7}")

    valid_mrr = [r["mrr_at_20"] for r in rows]
    valid_p10 = [r["precision_at_10"] for r in rows]
    valid_r10 = [r["recall_at_10"] for r in rows if r["recall_at_10"] is not None]

    avg_mrr = sum(valid_mrr) / len(valid_mrr) if valid_mrr else 0.0
    avg_p10 = sum(valid_p10) / len(valid_p10) if valid_p10 else 0.0
    avg_r10 = sum(valid_r10) / len(valid_r10) if valid_r10 else 0.0

    print("-" * W)
    r10_note = f"(n={len(valid_r10)}/{len(rows)})"
    print(f"{'AVERAGE':<44} {len(rows):>3} {avg_mrr:>8.4f} {avg_p10:>7.4f} {avg_r10:>7.4f}  {r10_note}")
    print("=" * W)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    output = {
        "version":   version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "n_queries":           len(rows),
            "avg_mrr_at_20":       round(avg_mrr, 4),
            "avg_precision_at_10": round(avg_p10, 4),
            "avg_recall_at_10":    round(avg_r10, 4),
            "n_recall_excluded":   sum(1 for r in rows if r["recall_at_10"] is None),
        },
        "queries": rows,
    }

    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved → {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v0", help="Label for this eval run")
    args = parser.parse_args()
    evaluate(args.version)

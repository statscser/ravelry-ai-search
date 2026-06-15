"""
debug_memory.py — Memory profiling for the app startup pipeline.

Calls the real load_collection() from rag_chroma.py.
A background thread samples RSS every 50 ms to capture transient peaks
that wouldn't show up in a simple before/after comparison.

Run from ravelry_search/:
    # With internal step-by-step logs from load_collection():
    $env:RAVELRY_DEBUG_MEMORY=1; python debug_memory.py   # PowerShell
    RAVELRY_DEBUG_MEMORY=1 python debug_memory.py         # bash

    # Without internal logs (just peak sampler):
    python debug_memory.py
"""

import os
import threading
import time

import psutil
from rank_bm25 import BM25Okapi

# ── helpers ───────────────────────────────────────────────────────────────────

_proc = psutil.Process(os.getpid())

def rss_mb() -> float:
    return _proc.memory_info().rss / 1024 / 1024


class PeakSampler:
    """Background thread that samples RSS every `interval_s` seconds."""
    def __init__(self, interval_s: float = 0.05):
        self._interval = interval_s
        self._stop = threading.Event()
        self._peak = rss_mb()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, rss_mb())
            time.sleep(self._interval)

    def start(self) -> "PeakSampler":
        self._peak = rss_mb()
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return self._peak

    @property
    def peak(self) -> float:
        return self._peak


def step(label: str, before: float, peak: float | None = None) -> float:
    after = rss_mb()
    delta = after - before
    peak_str = f"  峰值={peak:,.1f} MB  瞬时膨胀=+{peak - after:,.1f} MB" if peak is not None else ""
    print(f"[内存] {label}: {after:,.1f} MB  (Δ{delta:+,.1f} MB){peak_str}")
    return after


# ── pipeline ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("Ravelry AI Search — startup memory profiler (peak sampler)")
print("=" * 60)

baseline = rss_mb()
print(f"[内存] 基准 (Python + imports): {baseline:,.1f} MB\n")

readings: list[tuple[str, float, float, float]] = []  # label, before, after, peak

# ── Step 1: load_collection() ─────────────────────────────────────────────────
print("--- Step 1: load_collection() [rag_chroma.py 真实函数] ---")
print("    (50ms 采样线程已启动，捕获瞬时峰值)")
if os.environ.get("RAVELRY_DEBUG_MEMORY") == "1":
    print("    (RAVELRY_DEBUG_MEMORY=1: 内部分步日志已开启)\n")

from rag_chroma import load_collection

before = rss_mb()
sampler = PeakSampler(interval_s=0.05).start()
collection, patterns, embeddings = load_collection()
peak = sampler.stop()
after = step(
    f"load_collection() → {collection.count():,} patterns, embeddings {embeddings.shape}",
    before,
    peak,
)
print(f"\n  ┌─ load_collection() 峰值 RSS : {peak:,.1f} MB")
print(f"  ├─ load_collection() 最终 RSS : {after:,.1f} MB")
print(f"  └─ 瞬时膨胀 (峰值-最终)       : {peak - after:+,.1f} MB")
print(f"     (这部分在调用完成后已释放，但 OOM 发生在此期间)\n")
readings.append(("load_collection() 全流程", before, after, peak))

# ── Step 2: BM25 corpus ───────────────────────────────────────────────────────
print("--- Step 2: BM25Okapi corpus build (每次查询时构建) ---")

def _bm25_text(p: dict) -> str:
    name       = p.get("name") or ""
    craft      = (p.get("craft") or {}).get("name", "")
    yarn_wt    = (p.get("yarn_weight") or {}).get("name", "")
    categories = " ".join(c["name"] for c in (p.get("pattern_categories") or []))
    attributes = " ".join(a["permalink"] for a in (p.get("pattern_attributes") or []))
    return f"{name} {craft} {yarn_wt} {categories} {attributes}"

before = rss_mb()
sampler2 = PeakSampler(interval_s=0.05).start()
corpus = [_bm25_text(p).lower().split() for p in patterns]
bm25   = BM25Okapi(corpus)
peak2  = sampler2.stop()
after  = step(f"BM25Okapi ({len(corpus):,} docs)", before, peak2)
readings.append(("BM25Okapi index (per-query)", before, after, peak2))
print()

# ── Summary ───────────────────────────────────────────────────────────────────

total_used = rss_mb() - baseline

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Step':<40} {'Δ RSS':>8}  {'Peak':>8}  {'膨胀':>8}")
print("-" * 68)
for label, b, a, pk in readings:
    delta = a - b
    print(f"{label:<40} {delta:>+8.1f}  {pk:>8.1f}  {pk-a:>+8.1f}")
print("-" * 68)
print(f"{'TOTAL (vs baseline)':<40} {total_used:>+8.1f}")
print(f"\n最终 RSS : {rss_mb():,.1f} MB  |  基准: {baseline:,.1f} MB")
print(f"绝对峰值  : {max(pk for *_, pk in readings):,.1f} MB")

#!/usr/bin/env python3
"""Profile GPU VRAM usage at different embedding batch sizes.

Runs embedding requests at increasing batch sizes and measures VRAM delta.
Helps find the maximum safe batch size for a 23607 MB VRAM cap.

Usage:
    python3 scripts/profile_vram.py --server http://localhost:8100 --max-batch 256
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

import torch
from servers.embedding_server.client import get_dense_vectors, get_sparse_vectors, health_check


def generate_test_texts(count: int, avg_words: int = 150) -> List[str]:
    """Generate deterministic test texts of ~avg_words length."""
    template = "The {adj} {noun} {verb} over the {adj2} {noun2} looking for {noun3}."
    adjs = ["quick", "brown", "lazy", "red", "blue", "green", "fast", "dark", "bright", "old"]
    nouns = ["fox", "dog", "cat", "bird", "fish", "bear", "wolf", "deer", "hawk", "owl"]
    verbs = ["jumps", "runs", "sleeps", "flies", "swims", "stares", "hunts", "waits"]
    texts = []
    for i in range(count):
        words = []
        for _ in range(avg_words):
            words.append(f"token{i}_{i % 100}")
        texts.append(" ".join(words))
    return texts


def get_vram_snapshot() -> Tuple[int, int]:
    """Return (allocated_bytes, reserved_bytes) on CUDA device 0."""
    allocated = torch.cuda.memory_allocated(device="cuda")
    reserved = torch.cuda.memory_reserved(device="cuda")
    return allocated, reserved


def profile_batch_size(
    batch_size: int,
    texts: List[str],
    label: str = "dense",
    server_url: str = "http://localhost:8100",
) -> Tuple[float, int, int]:
    """Embed a batch of texts and return (elapsed_sec, vram_delta_alloc, vram_delta_reserved)."""
    # Snapshot before
    alloc_before, res_before = get_vram_snapshot()

    # Run embedding
    t0 = time.time()
    if label == "dense":
        get_dense_vectors(texts[:batch_size], batch_size=batch_size)
    else:
        get_sparse_vectors(texts[:batch_size], is_query=False)
    elapsed = time.time() - t0

    # Snapshot after
    alloc_after, res_after = get_vram_snapshot()

    delta_alloc = (alloc_after - alloc_before) / 1e9  # GB
    delta_res = (res_after - res_before) / 1e9  # GB

    return elapsed, delta_alloc, delta_res


def main():
    parser = argparse.ArgumentParser(description="Profile GPU VRAM usage at different batch sizes.")
    parser.add_argument("--server", default="http://localhost:8100", help="Embedding server URL")
    parser.add_argument("--max-batch", type=int, default=256, help="Max batch size to test")
    parser.add_argument("--steps", type=int, default=8, help="Number of batch sizes to test")
    parser.add_argument("--test-both", action="store_true", help="Test both dense and sparse")
    args = parser.parse_args()

    if not health_check():
        log.error("Embedding server not reachable at %s", args.server)
        sys.exit(1)

    # Check GPU
    if not torch.cuda.is_available():
        log.error("No CUDA available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    log.info("GPU: %s | Total VRAM: %.1f GB", gpu_name, total_vram)
    log.info("VRAM cap: 23.6 GB (EMBEDDING_VRAM_FRACTION=0.90)")

    # Generate test texts
    log.info("Generating 512 test texts...")
    all_texts = generate_test_texts(512)

    # Calculate batch sizes to test
    batches = []
    step = max(1, args.max_batch // args.steps)
    for bs in range(step, args.max_batch + 1, step):
        batches.append(bs)
    if batches[-1] != args.max_batch:
        batches.append(args.max_batch)

    log.info("Testing batch sizes: %s", batches)
    log.info("VRAM snapshot BEFORE profiling:\n")

    alloc_before, res_before = get_vram_snapshot()
    log.info("  Allocated: %.2f GB | Reserved: %.2f GB", alloc_before / 1e9, res_before / 1e9)
    log.info("")
    log.info("=" * 80)
    log.info("%-10s %-10s %-12s %-16s %-16s", "BatchSize", "Elapsed", "DeltaAlloc", "DeltaRes", "Status")
    log.info("-" * 80)

    results = []
    for bs in batches:
        for label in (["dense"] if not args.test_both else ["dense", "sparse"]):
            texts = all_texts[:bs]
            try:
                elapsed, delta_alloc, delta_res = profile_batch_size(bs, texts, label, args.server)
                status = "OK"
                results.append((bs, label, elapsed, delta_alloc, delta_res, status))
                log.info(
                    "%-10s %-10s %.2f GB/s  %-12.3f GB %-16.3f GB %s",
                    bs, f"{elapsed:.2f}s", bs / elapsed if elapsed > 0 else 0,
                    delta_alloc, delta_res, status,
                )
            except Exception as e:
                err_msg = str(e)[:60]
                results.append((bs, label, 0, 0, 0, f"FAIL: {err_msg}"))
                log.info(
                    "%-10s %-10s %-12s %-16s %-16s FAIL: %s",
                    bs, "-", "-", "-", "-", err_msg,
                )

    log.info("=" * 80)

    # VRAM snapshot AFTER profiling
    alloc_after, res_after = get_vram_snapshot()
    log.info("")
    log.info("VRAM snapshot AFTER profiling:")
    log.info("  Allocated: %.2f GB | Reserved: %.2f GB", alloc_after / 1e9, res_after / 1e9)
    log.info("  Net change: Alloc %.3f GB | Reserved %.3f GB",
             (alloc_after - alloc_before) / 1e9, (res_after - res_before) / 1e9)

    # Recommendation
    vram_cap_gb = 23.6
    log.info("")
    log.info("RECOMMENDATION:")
    # Find largest batch that didn't cause OOM and had reasonable delta
    safe = [(bs, label, da, dr, st) for bs, label, _, da, dr, st in results if st == "OK"]
    if safe:
        # Suggest batch size where delta is < 2GB (leaves headroom for model weights)
        recommended = None
        for bs, label, da, dr, st in reversed(safe):
            if da < 2.0:  # < 2GB delta for this batch
                recommended = (bs, label, da, dr)
                break
        if recommended:
            bs, label, da, dr = recommended
            log.info("  For %s: batch_size=%d (VRAM delta: +%.3f GB alloc, +%.3f GB reserved)",
                     label, bs, da, dr)
        else:
            log.info("  All tested batch sizes caused >2GB VRAM delta.")
            log.info("  Try reducing max batch size below %d.", safe[0][0])
    else:
        log.info("  All tests failed. Check server and GPU memory.")


if __name__ == "__main__":
    main()
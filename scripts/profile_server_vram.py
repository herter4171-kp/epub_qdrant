#!/usr/bin/env python3
"""Profile server-side VRAM usage by hitting /profile/dense and /profile/sparse endpoints.

These endpoints run in the server process on the GPU, so they measure real VRAM delta.

Usage (run on GPU box):
    python3 scripts/profile_server_vram.py --server http://localhost:8100
    python3 scripts/profile_server_vram.py --server http://localhost:8100 --max-count 512
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def health_check(server: str) -> bool:
    try:
        resp = requests.get(f"{server}/health", timeout=10)
        resp.raise_for_status()
        return resp.json().get("dense") is True and resp.json().get("sparse") is True
    except Exception as e:
        log.error("Health check failed: %s", e)
        return False


def profile_dense(server: str, count: int) -> dict:
    resp = requests.post(f"{server}/profile/dense", json={"count": count}, timeout=300)
    resp.raise_for_status()
    return resp.json()


def profile_sparse(server: str, count: int, is_query: bool = False) -> dict:
    resp = requests.post(f"{server}/profile/sparse", json={"count": count, "is_query": is_query}, timeout=300)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Profile server-side VRAM at various batch sizes.")
    parser.add_argument("--server", default="http://localhost:8100", help="Embedding server URL")
    parser.add_argument("--max-count", type=int, default=256, help="Max batch size to test")
    parser.add_argument("--steps", type=int, default=8, help="Number of batch sizes to test")
    parser.add_argument("--test-sparse", action="store_true", help="Also test sparse")
    args = parser.parse_args()

    if not health_check(args.server):
        log.error("Embedding server not healthy at %s", args.server)
        sys.exit(1)

    log.info("Server healthy. Testing batch sizes...")

    # Calculate batch sizes to test
    step = max(1, args.max_count // args.steps)
    sizes = list(range(step, args.max_count + 1, step))
    if sizes[-1] != args.max_count:
        sizes.append(args.max_count)

    log.info("Sizes to test: %s", sizes)
    log.info("")
    log.info("=" * 90)
    log.info("%-10s %-10s %-12s %-16s %-16s", "Count", "Elapsed", "PerText", "VRAMDelta", "Status")
    log.info("-" * 90)

    results = []
    for count in sizes:
        # Dense
        try:
            t0 = time.time()
            r = profile_dense(args.server, count)
            elapsed = time.time() - t0
            per_text = elapsed / count * 1000 if count > 0 else 0
            log.info(
                "%-10s %-10.2fs %-12.1fms/text %-16.3fGB %-16s",
                count, r.get("elapsed_sec", 0), per_text,
                r.get("vram_delta_gb", 0), "OK",
            )
            results.append(("dense", count, r.get("vram_delta_gb", 0), "OK"))
        except Exception as e:
            log.info("%-10s %-10s %-12s %-16s %-16s FAIL: %s",
                     count, "-", "-", "-", "DENSE", str(e)[:40])
            results.append(("dense", count, 0, f"FAIL: {str(e)[:40]}"))

        # Sparse
        if args.test_sparse:
            try:
                t0 = time.time()
                r = profile_sparse(args.server, count)
                elapsed = time.time() - t0
                per_text = elapsed / count * 1000 if count > 0 else 0
                log.info(
                    "%-10s %-10s %-12.1fms/text %-16.3fGB %-16s",
                    f"  sparse {count}", f"{elapsed:.2f}s", per_text,
                    r.get("vram_delta_gb", 0), "OK",
                )
                results.append(("sparse", count, r.get("vram_delta_gb", 0), "OK"))
            except Exception as e:
                log.info("%-10s %-10s %-12s %-16s %-16s FAIL: %s",
                         "  sparse", "-", "-", "-", "SPARSE", str(e)[:40])
                results.append(("sparse", count, 0, f"FAIL: {str(e)[:40]}"))

    log.info("=" * 90)

    # Recommendation
    vram_cap_gb = 23.6
    log.info("")
    log.info("RECOMMENDATION (VRAM cap: %.1f GB):", vram_cap_gb)

    # Find safe batch sizes where delta is reasonable (< 2GB per batch)
    safe_dense = [(c, d) for _, c, d, s in results if "dense" in _ and s == "OK" and d < 2.0]
    safe_sparse = [(c, d) for _, c, d, s in results if "sparse" in _ and s == "OK" and d < 2.0]

    if safe_dense:
        best = max(safe_dense, key=lambda x: x[0])
        log.info("  Dense: batch_size=%d (VRAM delta: +%.3f GB)", best[0], best[1])
    else:
        first = results[0] if results else None
        if first and "dense" in first[0]:
            log.info("  All dense batches >2GB delta. Try batch_size=%d.", first[1])
        else:
            log.info("  Dense tests failed entirely.")

    if safe_sparse:
        best = max(safe_sparse, key=lambda x: x[0])
        log.info("  Sparse: batch_size=%d (VRAM delta: +%.3f GB)", best[0], best[1])
    elif args.test_sparse:
        log.info("  Sparse tests failed or skipped.")


if __name__ == "__main__":
    main()
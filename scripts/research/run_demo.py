"""P5.3 — run the full demo from cache, timed.

Defaults to replay mode: REFOLD_OFFLINE is set for you, recorded replies are
replayed instead of calling a model, and any structure that was not precomputed
raises rather than silently invoking ESMFold.

Usage:
    python scripts/research/run_demo.py --assert-no-misses
    python scripts/research/run_demo.py --live            # allow inference (not for judging)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

TARGET_SECONDS = 300.0  # the demo window; P5.3's pass/fail bar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", default=None)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--assert-no-misses", action="store_true",
                    help="fail if anything resolved outside the cache (C4)")
    ap.add_argument("--live", action="store_true",
                    help="allow live inference and a live model call")
    ap.add_argument("--target-seconds", type=float, default=TARGET_SECONDS)
    args = ap.parse_args()

    # Set the environment BEFORE importing anything that reads it.
    if not args.live:
        os.environ["REFOLD_OFFLINE"] = "1"

    from src import demo
    from src.cache import fold_cache
    from src.constants import DEFAULT_PROTEIN

    protein = args.protein or DEFAULT_PROTEIN
    generations = args.generations or demo.DEFAULT_GENERATIONS

    print(
        f"[demo] protein={protein} generations={generations} "
        f"mode={'live' if args.live else 'replay (cached)'}"
    )

    started = time.time()
    state = demo.run(
        protein=protein,
        generations=generations,
        replay_patches=not args.live,
        assert_no_misses=args.assert_no_misses,
        verbose=True,
    )
    elapsed = time.time() - started

    if state.get("synthetic_structures"):
        print("\n" + "!" * 62)
        print("SYNTHETIC STRUCTURES WERE USED (REFOLD_FOLD_BACKEND=synthetic).")
        print("Every TM-score, contact recovery and pLDDT below is a harness")
        print("fixture, NOT a prediction, and means nothing scientifically.")
        print("Run scripts/clear_synthetic_cache.py, then precompute with the")
        print("real ESMFold checkpoint, before reporting any of these numbers.")
        print("!" * 62)

    print("\n" + "=" * 62)
    print(f"fold backend: {state.get('fold_backend')}")
    print(f"wall clock: {elapsed:.1f}s (target <= {args.target_seconds:.0f}s)")
    print(f"fold cache: {state['cache_stats']}")
    print(f"score cache: {state['score_cache_stats']}")
    print(f"final policy weights: {state['final_policy']['position_score']}")
    print(f"final proposal: {state['final_policy']['proposal']}")

    accepted = [g for g in state["generations"] if g["accepted"]]
    counterexamples = [g for g in state["generations"] if g["counterexample"]]
    print(f"generations: {len(state['generations'])}  "
          f"accepted patches: {len(accepted)}  "
          f"counterexamples: {len(counterexamples)}")

    if state.get("holdout"):
        h = state["holdout"]
        print(
            f"holdout {h['variant']}: hidden {h['hidden_score']:.4f} "
            f"(delta {h['hidden_delta']:+.4f}) edits {h['mutations']}"
        )
    print("=" * 62)

    if elapsed > args.target_seconds:
        print(
            f"\nP5.3 FAIL: {elapsed:.1f}s exceeds the {args.target_seconds:.0f}s "
            f"demo window."
        )
        return 1

    print("\nP5.3 PASS")
    if args.assert_no_misses:
        print("C4 PASS (asserted inside the driver)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

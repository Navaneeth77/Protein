"""The final reveal — the ONLY place `reveal=True` is ever passed.

Prints the withheld decomposition (TM-score, contact recovery, sequence recovery)
for the seeded variants and for the held-out variant. Nothing in src/agent/ calls
this, and tests/test_evaluator.py greps the agent tree to keep it that way.

Usage:
    python scripts/research/final_reveal.py
    python scripts/research/final_reveal.py --protein 1ubq
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.constants import DEFAULT_PROTEIN  # noqa: E402
from src.demo import DEMO_STATE  # noqa: E402
from src.evaluator import HiddenEvaluator  # noqa: E402

COLUMNS = ("variant", "stage", "hidden", "tm", "contact_rec", "seq_rec", "plddt", "edits")


def row(variant: str, stage: str, metrics: dict) -> str:
    return (
        f"{variant:<12} {stage:<9} "
        f"{metrics['hidden_score']:>7.4f} "
        f"{metrics['tm_score']:>7.4f} "
        f"{metrics['contact_recovery']:>11.4f} "
        f"{metrics['sequence_recovery']:>8.4f} "
        f"{metrics['mean_plddt']:>7.4f} "
        f"{metrics['edit_count']:>5d}"
    )


def header() -> str:
    return (
        f"{'variant':<12} {'stage':<9} {'hidden':>7} {'tm':>7} "
        f"{'contact_rec':>11} {'seq_rec':>8} {'plddt':>7} {'edits':>5}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", default=DEFAULT_PROTEIN)
    ap.add_argument("--state", default=None)
    args = ap.parse_args()

    state_path = Path(args.state) if args.state else DEMO_STATE
    if not state_path.exists():
        raise SystemExit(f"no demo state at {state_path}; run scripts/research/run_demo.py first")
    state = json.loads(state_path.read_text(encoding="utf-8"))

    evaluator = HiddenEvaluator(protein=args.protein)

    print("GROUND TRUTH REVEAL")
    print("=" * 72)
    print(header())
    print("-" * 72)

    records = list(state.get("variants", {}).items())
    if state.get("holdout"):
        records.append((state["holdout"]["variant"], state["holdout"]))

    for name, record in records:
        origin = record["origin"]
        repaired = record["repaired"]
        before = evaluator.evaluate(origin, reveal=True, origin=origin)
        after = evaluator.evaluate(repaired, reveal=True, origin=origin)
        print(row(name, "corrupted", before))
        print(row(name, "repaired", after))
        delta = after["hidden_score"] - before["hidden_score"]
        tm_delta = after["tm_score"] - before["tm_score"]
        print(
            f"{'':<12} {'delta':<9} {delta:>+7.4f} {tm_delta:>+7.4f}"
            f"{'':>11}{'':>8}{'':>7}{'':>5}"
        )
        print("-" * 72)

    print("\nSelection used corrupt_01..03. The holdout row was never part of any")
    print("accept/reject decision — see scripts/research/check_holdout.py for the proof.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

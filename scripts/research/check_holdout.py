"""P5.2 verify — prove the held-out variant never influenced selection.

Greps every training artefact written before the reveal for the holdout's
filename, its sequence, and its cache hash. Zero hits anywhere is the pass
condition; a single hit means the variant leaked into the outer loop and is no
longer a blind test.

Usage:
    python scripts/research/check_holdout.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agent import counterexamples  # noqa: E402
from src.agent.outer_loop import GENERATION_LOG  # noqa: E402
from src.cache import fold_cache  # noqa: E402
from src.constants import DEFAULT_PROTEIN  # noqa: E402
from src.demo import HOLDOUT_VARIANT, SELECTION_VARIANTS  # noqa: E402
from src.paths import corruption_dir, read_fasta  # noqa: E402

# Files that record what the outer loop actually selected on. The holdout must
# not appear in any of them. demo_state.json is excluded on purpose: it is
# written after the loop finishes and legitimately holds the reveal.
TRAINING_ARTEFACTS = (
    counterexamples.DEFAULT_LOG,
    GENERATION_LOG,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", default=DEFAULT_PROTEIN)
    args = ap.parse_args()

    fasta = corruption_dir(args.protein) / f"{HOLDOUT_VARIANT}.fasta"
    if not fasta.exists():
        raise SystemExit(
            f"no holdout at {fasta}; run scripts/research/make_holdout.py first"
        )
    sequence = read_fasta(fasta)[1]
    needles = {
        "filename": f"{HOLDOUT_VARIANT}",
        "sequence": sequence,
        "cache hash": fold_cache.sequence_hash(sequence),
    }

    print(f"holdout: {HOLDOUT_VARIANT}")
    print(f"  sequence: {sequence}")
    print(f"  hash:     {needles['cache hash']}")
    print(f"  selection set: {', '.join(SELECTION_VARIANTS)}\n")

    # Sanity: the holdout must genuinely differ from every selection variant.
    for name in SELECTION_VARIANTS:
        other = corruption_dir(args.protein) / f"{name}.fasta"
        if other.exists() and read_fasta(other)[1] == sequence:
            raise SystemExit(f"FAIL: holdout is identical to {name}")

    failures = []
    for artefact in TRAINING_ARTEFACTS:
        if not artefact.exists():
            print(f"[skip] {artefact} does not exist")
            continue
        text = artefact.read_text(encoding="utf-8")
        for label, needle in needles.items():
            hits = text.count(needle)
            status = "FAIL" if hits else "ok"
            print(f"[{status:>4}] {artefact.name}: {label} -> {hits} hit(s)")
            if hits:
                failures.append((artefact.name, label, hits))

    # The generation log is also checked structurally, not only textually.
    if GENERATION_LOG.exists():
        for line in GENERATION_LOG.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for run in record["incumbent_runs"] + record["candidate_runs"]:
                if run["variant"] == HOLDOUT_VARIANT:
                    failures.append(("generations.jsonl", "run variant", 1))

    print()
    if failures:
        print("P5.2 FAIL — the holdout leaked into selection:")
        for name, label, hits in failures:
            print(f"  {name}: {label} ({hits})")
        return 1

    print("P5.2 PASS — zero hits in every training artefact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

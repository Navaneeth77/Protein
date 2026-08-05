"""P5.2 — generate exactly one blind held-out corruption.

A thin, named wrapper over scripts/make_corruptions.py with a different seed and
the `holdout` prefix, so the held-out variant is impossible to confuse with the
three variants the outer loop selects on.

Usage:
    python scripts/research/make_holdout.py
    python scripts/research/make_holdout.py --protein 1ubq --seed 90210
Then:
    python scripts/research/check_holdout.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import DEFAULT_PROTEIN  # noqa: E402
from src.demo import HOLDOUT_VARIANT, SELECTION_VARIANTS  # noqa: E402
from src.paths import corruption_dir, read_fasta  # noqa: E402

HOLDOUT_SEED = 90210


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", default=DEFAULT_PROTEIN)
    ap.add_argument("--seed", type=int, default=HOLDOUT_SEED)
    args = ap.parse_args()

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_corruptions.py"),
            args.protein,
            "--prefix", "holdout",
            "--n-variants", "1",
            "--seed", str(args.seed),
        ],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        return result.returncode

    path = corruption_dir(args.protein) / f"{HOLDOUT_VARIANT}.fasta"
    holdout = read_fasta(path)[1]
    for name in SELECTION_VARIANTS:
        other = corruption_dir(args.protein) / f"{name}.fasta"
        if other.exists() and read_fasta(other)[1] == holdout:
            raise SystemExit(
                f"FAIL: holdout collided with {name}; pick a different --seed"
            )

    print(f"\nholdout is distinct from all of {', '.join(SELECTION_VARIANTS)}")
    print("Next: python scripts/research/check_holdout.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

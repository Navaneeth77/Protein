"""Remove every cached structure produced by the synthetic backend.

Run this before the first real ESMFold precompute so no harness fixture can
survive into a run whose numbers are meant to mean something.

Usage:
    python scripts/clear_synthetic_cache.py --dry-run
    python scripts/clear_synthetic_cache.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cache import synthetic_backend  # noqa: E402
from src.paths import CACHE, LOGS  # noqa: E402

# Artefacts derived from synthetic structures are equally meaningless.
DERIVED = (
    LOGS / "demo_state.json",
    LOGS / "generations.jsonl",
    LOGS / "counterexamples.jsonl",
    LOGS / "ablation.json",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-logs", action="store_true",
                    help="leave logs/ alone and only clear data/cache/")
    args = ap.parse_args()

    structures = sorted(CACHE.glob("*.pdb"))
    synthetic, real = [], []
    for path in structures:
        text = path.read_text(encoding="utf-8", errors="replace")
        (synthetic if synthetic_backend.is_synthetic(text) else real).append(path)

    print(f"cache: {len(structures)} structure(s) — "
          f"{len(synthetic)} synthetic, {len(real)} real")

    for path in synthetic:
        if args.dry_run:
            print(f"  would remove {path.name}")
        else:
            path.unlink()

    if not args.keep_logs:
        for path in DERIVED:
            if not path.exists():
                continue
            if args.dry_run:
                print(f"  would remove {path}")
            else:
                path.unlink()

    if args.dry_run:
        print("\ndry run; nothing changed")
    else:
        print(f"\nremoved {len(synthetic)} synthetic structure(s)"
              + ("" if args.keep_logs else " and their derived logs"))
        if real:
            print(f"kept {len(real)} real structure(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

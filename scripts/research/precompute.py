"""P5.1 — precompute everything the live demo path will touch.

Runs the SAME driver the demo runs (src/demo.py), with live inference allowed, so
data/cache/ ends up holding exactly the structures the demo asks for — no more,
no less. Enumerating the whole reachable candidate tree instead would be
combinatorial (12 candidates per round, cubed) and would spend hours folding
sequences the search never visits.

Also folds the reference structure and the held-out variant, and warms the ESM
substitution-probability cache the heatmap reads.

Usage:
    python scripts/research/precompute.py
    python scripts/research/precompute.py --protein 1ubq --generations 2
Then:
    python scripts/research/run_demo.py --assert-no-misses
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import demo  # noqa: E402
from src.agent import esm_score  # noqa: E402
from src.cache import fold_cache  # noqa: E402
from src.constants import DEFAULT_PROTEIN  # noqa: E402
from src.paths import corruption_dir, protein_dir, read_fasta  # noqa: E402


def warm_roots(protein: str) -> int:
    """Fold and score the reference plus every variant on disk.

    The reference is needed by the reveal step; the extra variants beyond the
    three used for selection are needed by the UI's variant picker.
    """
    count = 0
    reference = protein_dir(protein) / "native_seq.fasta"
    if reference.exists():
        sequence = read_fasta(reference)[1]
        print(f"[warm] reference ({len(sequence)} aa)")
        fold_cache.fold(sequence)
        esm_score.masked_marginal_matrix(sequence)
        count += 1

    for fasta in sorted(corruption_dir(protein).glob("*.fasta")):
        sequence = read_fasta(fasta)[1]
        print(f"[warm] {fasta.stem}")
        fold_cache.fold(sequence)
        esm_score.masked_marginal_matrix(sequence)
        count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", default=DEFAULT_PROTEIN)
    ap.add_argument("--generations", type=int, default=demo.DEFAULT_GENERATIONS)
    args = ap.parse_args()

    if fold_cache.offline():
        raise SystemExit(
            "REFOLD_OFFLINE is set, but precompute needs live inference. Unset it."
        )

    started = time.time()
    fold_cache.reset_stats()
    esm_score.reset_stats()

    roots = warm_roots(args.protein)
    if not roots:
        raise SystemExit(
            f"nothing to precompute for {args.protein}; run scripts/fetch_protein.py "
            f"and scripts/make_corruptions.py first"
        )

    print(f"\n[precompute] warmed {roots} root sequence(s); now walking the demo\n")
    demo.run(
        protein=args.protein,
        generations=args.generations,
        replay_patches=False,
        assert_no_misses=False,
        verbose=True,
    )

    cached = len(list(fold_cache.CACHE.glob("*.pdb")))
    print(
        f"\n[precompute] done in {time.time() - started:.1f}s — "
        f"{cached} structure(s) cached, "
        f"{fold_cache.STATS['model_calls']} fold model call(s), "
        f"{esm_score.STATS['forward_batches']} score batch(es)"
    )
    print(f"[precompute] cache dir: {fold_cache.CACHE}")
    print(f"[precompute] recorded replies: {demo.PATCH_LOG}")
    print("\nNext: python scripts/research/run_demo.py --assert-no-misses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

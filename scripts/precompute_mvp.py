"""Warm the caches the MVP demo needs, so the button press is instant.

The problem this solves: one ESMFold call is ~57 s and peaks near 8.4 GB, while
Gemma is ~7.6 GB on a 14 GB machine. Folding during the demo would be both slow
and memory-hostile, and we cannot know in advance which patch Gemma will pick.

So: enumerate the shortlist under a spread of plausible policies — cheap, because
shortlisting only needs the cached ESM scores — and fold the union. Whatever
patch Gemma returns at demo time, its candidates are then almost certainly
already on disk.

Run this with Ollama unloaded. Usage:
    python scripts/precompute_mvp.py
    python scripts/precompute_mvp.py --list-only     # show what it would fold
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import esm_score, grounder, policy as policy_mod  # noqa: E402
from src.agent.policy_interpreter import apply_policy  # noqa: E402
from src.cache import fold_cache  # noqa: E402
from src.constants import DEFAULT_PROTEIN  # noqa: E402
from src.mvp import DEFAULT_VARIANT  # noqa: E402
from src.paths import corruption_dir, protein_dir, read_fasta  # noqa: E402

BASE_PROPOSAL = {
    "positions": 3,
    "substitutions_per_position": 4,
    "preserve_residue_class": True,
    "max_total_edits": 3,
}


def candidate_policies() -> list[tuple[str, dict]]:
    """A spread over the DSL wide enough to cover any single patch Gemma emits."""
    weight_sets = {
        "seed": {"esm_surprisal": 0.60, "low_plddt": 0.20, "contact_violation": 0.20},
        "long_range_on": {
            "esm_surprisal": 0.45, "low_plddt": 0.15,
            "contact_violation": 0.15, "long_range_contact_violation": 0.25,
        },
        "long_range_heavy": {
            "esm_surprisal": 0.30, "low_plddt": 0.10,
            "contact_violation": 0.10, "long_range_contact_violation": 0.50,
        },
        "esm_only": {"esm_surprisal": 0.80, "low_plddt": 0.20},
        "plddt_heavy": {"esm_surprisal": 0.20, "low_plddt": 0.60, "contact_violation": 0.20},
        "contact_heavy": {"esm_surprisal": 0.20, "low_plddt": 0.20, "contact_violation": 0.60},
    }
    proposal_sets = {
        "base": {},
        "wide": {"positions": 5},
        "free_class": {"preserve_residue_class": False},
        "more_subs": {"substitutions_per_position": 6},
    }

    policies = []
    for weight_name, weights in weight_sets.items():
        for proposal_name, override in proposal_sets.items():
            proposal = dict(BASE_PROPOSAL)
            proposal.update(override)
            policy = {"position_score": dict(weights), "proposal": proposal}
            policies.append((f"{weight_name}/{proposal_name}", policy_mod.validate_policy(policy)))
    return policies


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", default=DEFAULT_PROTEIN)
    ap.add_argument("--variant", default=DEFAULT_VARIANT)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    if fold_cache.offline():
        raise SystemExit("REFOLD_OFFLINE is set; unset it to precompute")

    started = time.time()
    fold_cache.reset_stats()

    reference = read_fasta(protein_dir(args.protein) / "native_seq.fasta")[1]
    corrupted = read_fasta(corruption_dir(args.protein) / f"{args.variant}.fasta")[1]
    print(f"protein {args.protein}, variant {args.variant}, {len(corrupted)} residues")

    # ESM scores first: cheap, and the shortlisting below depends on them.
    for label, sequence in (("reference", reference), ("corrupted", corrupted)):
        esm_score.masked_marginal_matrix(sequence)
        print(f"[esm] {label} substitution matrix cached")

    wanted: dict[str, list[str]] = {}
    # The reference is needed for the "what it should have been" panel; the
    # corrupted origin is the starting point of every round.
    for sequence in (reference, corrupted):
        wanted.setdefault(sequence, []).append("root")

    print("\n[plan] shortlisting under each policy (no folding yet)")
    fold = fold_cache.fold(corrupted)
    state = grounder.ground(corrupted, fold)

    for name, policy in candidate_policies():
        shortlist = apply_policy(policy, state, origin=corrupted)
        for candidate in shortlist:
            wanted.setdefault(candidate.sequence, []).append(f"{name}:{candidate.label()}")

    todo = [s for s in wanted if not fold_cache.is_cached(s)]
    print(f"\n[plan] {len(wanted)} sequence(s) wanted, {len(todo)} not yet cached")
    for sequence in todo:
        print(f"  {fold_cache.sequence_hash(sequence)}  {', '.join(wanted[sequence][:3])}")

    if args.list_only:
        print(f"\nestimated fold time: {len(todo)} x ~57s = ~{len(todo) * 57 / 60:.0f} min")
        return 0

    if not todo:
        print("\nnothing to do; every sequence is already cached")
        return 0

    print(f"\n[fold] folding {len(todo)} sequence(s); ~57s each on CPU")
    for index, sequence in enumerate(todo, 1):
        step = time.time()
        fold_cache.fold(sequence)
        print(
            f"  [{index}/{len(todo)}] {fold_cache.sequence_hash(sequence)} "
            f"done in {time.time() - step:.0f}s "
            f"({', '.join(wanted[sequence][:2])})"
        )

    cached = len(list(fold_cache.CACHE.glob("*.pdb")))
    print(
        f"\n[done] {time.time() - started:.0f}s total, "
        f"{fold_cache.STATS['model_calls']} fold(s) computed, "
        f"{cached} structure(s) in the cache"
    )
    print("\nNext: streamlit run app/streamlit_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

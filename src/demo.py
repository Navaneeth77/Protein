"""The single demo driver, shared by precompute and replay.

scripts/research/precompute.py runs this with live inference allowed, which populates
data/cache/. scripts/research/run_demo.py runs the SAME code path with REFOLD_OFFLINE=1
and asserts zero cache misses — that only works if both runs walk an identical
sequence of candidates, so patches are recorded on the live run and replayed
verbatim on the cached one (see `patch_log`).

Selection uses corrupt_01..03 (P3.4). The held-out variant is scored only after
the outer loop has finished and is never part of any accept/reject decision (P5.2).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .agent import esm_score, inner_loop, outer_loop, outer_loop_client, policy as policy_mod
from .cache import fold_cache
from .constants import DEFAULT_PROTEIN
from .evaluator import HiddenEvaluator
from .paths import LOGS, corruption_dir, read_fasta

DEMO_STATE = LOGS / "demo_state.json"
PATCH_LOG = LOGS / "demo_patches.json"

SELECTION_VARIANTS = ("corrupt_01", "corrupt_02", "corrupt_03")
HOLDOUT_VARIANT = "holdout_01"
DEFAULT_GENERATIONS = 3


class CacheMiss(RuntimeError):
    """A replay run reached for a structure that was never precomputed."""


def _any_synthetic(variants: dict, holdout: dict | None) -> bool:
    """Did any structure in this run come from the synthetic backend?"""
    sequences = []
    for record in variants.values():
        sequences += [record["origin"], record["repaired"]]
    if holdout:
        sequences += [holdout["origin"], holdout["repaired"]]
    for sequence in sequences:
        path = fold_cache.cache_path(sequence)
        if path.exists() and "SYNTHETIC_GEOMETRY" in path.read_text(encoding="utf-8"):
            return True
    return False


def load_variants(protein: str, names) -> dict[str, str]:
    out = {}
    for name in names:
        path = corruption_dir(protein) / f"{name}.fasta"
        if path.exists():
            out[name] = read_fasta(path)[1]
    return out


def _recorded_transport(replies: list, index_holder: dict):
    """Replay recorded raw replies verbatim instead of calling a model.

    Raw text, not the parsed patch: a reply that was *rejected* on the live run
    must be rejected identically on replay, or the two runs diverge and the
    zero-cache-miss assertion becomes meaningless.
    """

    def transport(prompt: str) -> str:
        i = index_holder["i"]
        index_holder["i"] += 1
        if i >= len(replies):
            raise CacheMiss(
                f"no recorded reply for generation {i}; re-run scripts/research/precompute.py"
            )
        return replies[i]

    return transport


def run(
    protein: str = DEFAULT_PROTEIN,
    generations: int = DEFAULT_GENERATIONS,
    replay_patches: bool = False,
    assert_no_misses: bool = False,
    patch_log: Path | None = None,
    state_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run the full outer loop, then score the held-out variant. Returns the state."""
    patch_log = Path(patch_log) if patch_log else PATCH_LOG
    state_path = Path(state_path) if state_path else DEMO_STATE

    fold_cache.reset_stats()
    esm_score.reset_stats()
    started = time.time()

    if generations < 1:
        raise ValueError("need at least one generation")

    corruption_set = load_variants(protein, SELECTION_VARIANTS)
    if len(corruption_set) < 3:
        raise SystemExit(
            f"need {len(SELECTION_VARIANTS)} selection variants, found "
            f"{sorted(corruption_set)}; run scripts/make_corruptions.py"
        )

    evaluator = HiddenEvaluator(protein=protein)
    loop = outer_loop.OuterLoop(evaluator, verbose=verbose)

    transport = None
    recorded: list = []
    if replay_patches:
        if not patch_log.exists():
            raise SystemExit(f"no recorded patches at {patch_log}; run precompute first")
        recorded = json.loads(patch_log.read_text(encoding="utf-8"))
        transport = _recorded_transport(recorded, {"i": 0})

    replies: list[str] = []
    last_result = None

    for generation in range(generations):
        if verbose:
            print(f"\n=== generation {generation} ===")

        outcomes = list(last_result.candidate_runs) if last_result else []
        counterexample = last_result.counterexample if last_result else None

        outcome = outer_loop_client.propose_patch(
            loop.incumbent_policy,
            outcomes,
            counterexample,
            transport=transport,
        )
        replies.append(outcome.raw)

        if not outcome.accepted:
            if verbose:
                print(f"[gen {generation}] patch rejected: {outcome.error}")
            # A rejected patch still counts as a generation: the incumbent is
            # re-tested against itself so the log keeps one row per generation.
            candidate_policy = policy_mod.clone(loop.incumbent_policy)
            patch_record = {"rejected": True, "error": outcome.error}
        else:
            candidate_policy = outcome.policy
            patch_record = outcome.patch

        last_result = loop.run_generation(
            candidate_policy, corruption_set, patch=patch_record
        )

    if not replay_patches:
        patch_log.parent.mkdir(parents=True, exist_ok=True)
        patch_log.write_text(json.dumps(replies, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ holdout
    holdout = None
    holdout_map = load_variants(protein, [HOLDOUT_VARIANT])
    if holdout_map:
        sequence = holdout_map[HOLDOUT_VARIANT]
        evaluator.set_origin(sequence)
        result = inner_loop.repair(sequence, loop.incumbent_policy, evaluator)
        holdout = {
            "variant": HOLDOUT_VARIANT,
            "origin": result.origin,
            "repaired": result.sequence,
            "mutations": [
                f"{m['from']}{m['position'] + 1}{m['to']}" for m in result.mutations
            ],
            "hidden_score": result.hidden_score,
            "hidden_delta": result.hidden_delta,
            "public_score": result.public_score,
            "predicted_delta": result.predicted_delta,
            "mean_plddt": float(result.final_metrics["mean_plddt"]),
            "esm_score": float(result.final_metrics["esm_score"]),
            "edit_count": int(result.final_metrics["edit_count"]),
        }
        if verbose:
            print(
                f"\n[holdout] {HOLDOUT_VARIANT}: hidden {holdout['hidden_score']:.4f} "
                f"(delta {holdout['hidden_delta']:+.4f}), edits {holdout['mutations']}"
            )
    elif verbose:
        print(f"\n[holdout] {HOLDOUT_VARIANT} not generated; skipping")

    # -------------------------------------------------------------- final state
    final_runs = last_result.candidate_runs if last_result.accepted else last_result.incumbent_runs
    variants = {
        run_record["variant"]: {
            "origin": run_record["origin"],
            "repaired": run_record["repaired"],
            "mutations": run_record["mutations"],
            "hidden_score": run_record["hidden_score"],
            "public_score": run_record["public_score"],
            "mean_plddt": run_record["mean_plddt"],
            "esm_score": run_record["esm_score"],
            "edit_count": run_record["edit_count"],
        }
        for run_record in final_runs
    }

    elapsed = time.time() - started
    state = {
        "protein": protein,
        "elapsed_seconds": round(elapsed, 3),
        "seed_policy": policy_mod.load_seed_policy(),
        "final_policy": loop.incumbent_policy,
        "generations": [asdict(g) for g in loop.history],
        "variants": variants,
        "holdout": holdout,
        "cache_stats": dict(fold_cache.STATS),
        "score_cache_stats": dict(esm_score.STATS),
        "offline": fold_cache.offline(),
        "fold_backend": fold_cache.backend(),
        # True if ANY structure this run touched was a harness fixture rather
        # than a prediction. Every consumer of this file must surface it.
        "synthetic_structures": _any_synthetic(variants, holdout),
    }

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    if verbose:
        print(
            f"\n[demo] {len(loop.history)} generation(s) in {elapsed:.1f}s — "
            f"fold hits={fold_cache.STATS['cache_hits']}, "
            f"misses={fold_cache.STATS['cache_misses']}, "
            f"model calls={fold_cache.STATS['model_calls']}"
        )
        print(f"[demo] wrote {state_path}")

    if assert_no_misses:
        misses = fold_cache.STATS["cache_misses"]
        calls = fold_cache.STATS["model_calls"]
        if misses or calls:
            raise CacheMiss(
                f"replay was not fully cached: cache_misses={misses}, "
                f"model_calls={calls} (expected 0 and 0)"
            )
        if verbose:
            print("[demo] C4 PASS — cache_misses == 0 and model_calls == 0")

    return state

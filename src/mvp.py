"""The MVP demo: one protein, one corruption, one Gemma patch, two repair rounds.

This is the whole idea in one file:

    corrupted sequence
      -> ESM-2 scores every residue          (src/agent/esm_score.py)
      -> policy picks suspicious positions   (src/agent/policy_interpreter.py)
      -> candidate mutations enumerated
      -> ESMFold predicts each candidate     (src/cache/fold_cache.py, cached)
      -> hidden evaluator scores them        (src/evaluator.py)
      -> state is grounded                   (src/agent/grounder.py)
      -> Gemma reads the state and returns ONE policy patch
      -> the interpreter runs again under the patched policy
      -> before/after comparison

Deliberately one generation. src/demo.py has the multi-generation version; this
module is what the Streamlit app drives.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agent import esm_score, grounder, outer_loop_client, policy as policy_mod
from .agent.inner_loop import public_score
from .agent.policy_interpreter import apply_policy, enumerate_candidates, select_positions
from .cache import fold_cache
from .constants import DEFAULT_PROTEIN
from .evaluator import HiddenEvaluator
from .paths import LOGS, corruption_dir, protein_dir, read_fasta

RESULT_PATH = LOGS / "mvp_result.json"
DEFAULT_VARIANT = "corrupt_01"


@dataclass
class Round:
    """One repair attempt under one policy."""

    label: str
    policy: dict
    selected_positions: list = field(default_factory=list)
    enumerated: list = field(default_factory=list)   # everything the policy proposed
    candidates: list = field(default_factory=list)   # the subset actually folded
    chosen: dict | None = None
    origin_metrics: dict = field(default_factory=dict)

    @property
    def hidden_score(self) -> float:
        return float(self.chosen["hidden_score"]) if self.chosen else float("nan")

    @property
    def hidden_delta(self) -> float:
        if not self.chosen:
            return 0.0
        return self.hidden_score - float(self.origin_metrics["hidden_score"])

    @property
    def public_delta(self) -> float:
        if not self.chosen:
            return 0.0
        return float(self.chosen["public_score"]) - public_score(self.origin_metrics)


@dataclass
class MvpResult:
    protein: str
    reference_sequence: str
    corrupted_sequence: str
    corruption_sites: list
    origin_metrics: dict
    state_summary: dict
    baseline: dict
    patched: dict
    gemma: dict
    timeline: list
    elapsed_seconds: float
    fold_backend: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, allow_nan=False)


def _mutation_sites(reference: str, corrupted: str) -> list:
    return [
        {"position": i, "from": a, "to": b}
        for i, (a, b) in enumerate(zip(reference, corrupted))
        if a != b
    ]


def _changes_behaviour(before: dict, after: dict, state: dict, origin: str) -> bool:
    """Would `after` actually make the search do something different?

    Compares the selected sites and the FULL enumerated candidate set, not the
    folded top-3. Enumeration is the search: a policy that proposes six
    substitutions per site instead of four has genuinely changed what gets
    explored, even when the three candidates that survive pre-ranking happen to
    be the same. Judging only the folded subset reported such patches as no-ops.

    Cheap either way: both quantities come from cached ESM scores, no folding.
    """
    if select_positions(before, state) != select_positions(after, state):
        return True
    old = {c.sequence for c in enumerate_candidates(before, state, origin=origin)}
    new = {c.sequence for c in enumerate_candidates(after, state, origin=origin)}
    return old != new


def _run_round(
    label: str,
    policy: dict,
    state: dict,
    origin: str,
    evaluator: HiddenEvaluator,
    origin_metrics: dict,
    log,
) -> Round:
    """Enumerate, fold and score one shortlist under `policy`."""
    result = Round(label=label, policy=policy, origin_metrics=origin_metrics)
    result.selected_positions = select_positions(policy, state)
    log(f"{label}: policy targets positions {[p + 1 for p in result.selected_positions]}")

    # Everything the policy proposes, before the cheap pre-rank narrows it down.
    # Recorded because this is where a widened search becomes visible.
    proposed = enumerate_candidates(policy, state, origin=origin)
    result.enumerated = [
        {
            "label": candidate.label(),
            "position": candidate.position,
            "to_aa": candidate.to_aa,
            "substitution_prob": round(candidate.substitution_prob, 6),
        }
        for candidate in proposed
    ]
    log(f"{label}: enumerated {len(proposed)} candidate mutation(s)")

    shortlist = apply_policy(policy, state, origin=origin)
    if not shortlist:
        log(f"{label}: policy proposed no candidates")
        return result

    for candidate in shortlist:
        metrics = evaluator.evaluate(candidate.sequence, origin=origin)
        result.candidates.append(
            {
                "label": candidate.label(),
                "sequence": candidate.sequence,
                "position": candidate.position,
                "from_aa": candidate.from_aa,
                "to_aa": candidate.to_aa,
                "substitution_prob": round(candidate.substitution_prob, 6),
                "prerank_score": round(float(candidate.prerank_score), 4),
                "public_score": round(public_score(metrics), 6),
                "hidden_score": metrics["hidden_score"],
                "mean_plddt": metrics["mean_plddt"],
                "esm_score": metrics["esm_score"],
                "edit_count": metrics["edit_count"],
            }
        )

    result.candidates.sort(key=lambda c: -c["public_score"])
    result.chosen = result.candidates[0]
    log(
        f"{label}: picked {result.chosen['label']} "
        f"(public {result.chosen['public_score']:.4f}, "
        f"hidden {result.chosen['hidden_score']:.4f})"
    )
    return result


def run(
    protein: str = DEFAULT_PROTEIN,
    variant: str = DEFAULT_VARIANT,
    gemma_transport=None,
    progress=None,
) -> MvpResult:
    """Run the full MVP flow once and return everything needed to display it."""
    started = time.time()
    timeline: list = []

    def log(message: str) -> None:
        stamp = round(time.time() - started, 1)
        timeline.append({"t": stamp, "message": message})
        if progress:
            progress(message)
        print(f"[mvp +{stamp:5.1f}s] {message}")

    reference = read_fasta(protein_dir(protein) / "native_seq.fasta")[1]
    corrupted = read_fasta(corruption_dir(protein) / f"{variant}.fasta")[1]
    sites = _mutation_sites(reference, corrupted)
    log(f"loaded {protein}/{variant}: {len(corrupted)} residues, {len(sites)} corrupted sites")

    evaluator = HiddenEvaluator(protein=protein, origin_sequence=corrupted)

    log("folding the corrupted input")
    fold = fold_cache.fold(corrupted)
    log(
        f"corrupted fold: mean pLDDT {fold.mean_plddt:.3f}, "
        f"{len(fold.contacts)} contacts, {fold.clashes} clashes"
        + (" (from cache)" if fold.from_cache else " (computed live)")
    )

    origin_metrics = evaluator.evaluate(corrupted, origin=corrupted)
    log(f"corrupted hidden score: {origin_metrics['hidden_score']:.4f}")

    log("grounding state: per-residue surprisal, pLDDT, contact graph")
    state = grounder.ground(corrupted, fold)

    seed_policy = policy_mod.load_seed_policy()
    baseline = _run_round(
        "baseline", seed_policy, state, corrupted, evaluator, origin_metrics, log
    )

    # ---------------------------------------------------------------- Gemma
    counterexample = None
    if baseline.chosen and baseline.public_delta > 0 and baseline.hidden_delta <= 0:
        counterexample = {
            "iteration": 0,
            "predicted_delta": round(baseline.public_delta, 6),
            "hidden_delta": round(baseline.hidden_delta, 6),
            "variant": variant,
            "mutations": [baseline.chosen["label"]],
            "state_before": {
                "sequence_length": state["sequence_length"],
                "summary": state["summary"],
                "top_residues": sorted(
                    state["residues"], key=lambda r: -r["esm_surprisal"]
                )[:8],
            },
        }
        log("COUNTEREXAMPLE: public signals improved but the hidden verifier did not")
    else:
        log("no counterexample this round; sending outcomes only")

    outcomes = [
        {
            "variant": variant,
            "mutations": [c["label"]],
            "predicted_delta": round(
                c["public_score"] - public_score(origin_metrics), 6
            ),
            "hidden_delta": round(c["hidden_score"] - origin_metrics["hidden_score"], 6),
            "mean_plddt": c["mean_plddt"],
            "esm_score": c["esm_score"],
        }
        for c in baseline.candidates
    ]

    # The residue table Gemma reasons over: the sites the policy currently
    # targets, plus the next most suspicious ones so it has something to move to.
    by_surprisal = sorted(state["residues"], key=lambda r: -r["esm_surprisal"])
    focus_rows = {r["position"]: r for r in by_surprisal[:6]}
    for position in baseline.selected_positions:
        focus_rows[position] = state["residues"][position]
    focus = {
        "selected_positions": baseline.selected_positions,
        "residues": sorted(focus_rows.values(), key=lambda r: r["position"]),
    }

    patched_policy = policy_mod.clone(seed_policy)
    gemma: dict = {}
    attempts: list = []
    best: dict | None = None      # the best VALID attempt seen so far

    # Two attempts at most. A patch that validates but selects exactly the same
    # sites and substitutions has not changed the strategy, which is the whole
    # point of the demo, so Gemma gets told that and asked once more. A valid
    # patch from an earlier attempt is never thrown away just because a later
    # attempt came back malformed.
    for attempt in range(2):
        if attempt:
            log("first patch validated but changed nothing; asking Gemma once more")
            focus["previous_attempt_changed_nothing"] = True
        else:
            log("asking Gemma for one policy patch")

        outcome = outer_loop_client.propose_patch(
            seed_policy, outcomes, counterexample,
            transport=gemma_transport, focus=focus,
        )
        record = {
            "source": outcome.source,
            "raw": outcome.raw,
            "accepted": outcome.accepted,
            "error": outcome.error,
            "patch": outcome.patch,
            "rationale": (outcome.patch or {}).get("rationale"),
            "kind": outer_loop_client.describe_patch(outcome.patch),
            "attempts": attempt + 1,
        }
        attempts.append({"accepted": outcome.accepted, "error": outcome.error,
                         "patch": outcome.patch})

        if not outcome.accepted:
            log(f"Gemma's patch was REJECTED by the schema: {outcome.error}")
            if best is None:
                gemma = record
            continue

        changed = _changes_behaviour(seed_policy, outcome.policy, state, corrupted)
        log(
            f"Gemma returned a valid {record['kind']} patch; schema accepted it"
            + ("" if changed else " (but it does not change the search)")
        )
        if best is None or changed:
            best = {"record": record, "policy": outcome.policy, "changed": changed}
        if changed:
            break

    if best is not None:
        gemma = best["record"]
        gemma["accepted"] = True
        gemma["changed_behaviour"] = best["changed"]
        patched_policy = best["policy"]
    else:
        log("no valid patch after 2 attempts; re-running the incumbent policy")
        gemma["changed_behaviour"] = False

    gemma["attempts"] = len(attempts)
    gemma["all_attempts"] = attempts

    # -------------------------------------------------------------- round two
    patched = _run_round(
        "patched", patched_policy, state, corrupted, evaluator, origin_metrics, log
    )

    if baseline.chosen and patched.chosen:
        moved = patched.chosen["sequence"] != baseline.chosen["sequence"]
        delta = patched.hidden_score - baseline.hidden_score
        log(
            f"result: hidden {baseline.hidden_score:.4f} -> {patched.hidden_score:.4f} "
            f"({delta:+.4f}); chosen mutation "
            + ("changed" if moved else "unchanged")
        )

    result = MvpResult(
        protein=protein,
        reference_sequence=reference,
        corrupted_sequence=corrupted,
        corruption_sites=sites,
        origin_metrics=origin_metrics,
        state_summary=state["summary"],
        baseline=asdict(baseline),
        patched=asdict(patched),
        gemma=gemma,
        timeline=timeline,
        elapsed_seconds=round(time.time() - started, 2),
        fold_backend=fold_cache.backend(),
    )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(result.to_json(), encoding="utf-8")
    log(f"wrote {RESULT_PATH}")
    return result


def load_result(path: Path | None = None) -> dict:
    path = Path(path) if path else RESULT_PATH
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def heatmap_matrix(sequence: str):
    """Substitution-probability matrix for the heatmap, or None if uncached."""
    try:
        return esm_score.masked_marginal_matrix(sequence)
    except esm_score.ScoringUnavailable:
        return None

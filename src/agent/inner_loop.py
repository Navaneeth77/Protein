"""P2.1-P2.6 wired together — the inner repair loop.

One round = ground the incumbent, let the policy interpreter propose a
shortlist, fold the shortlist, pick a winner. The loop runs until the edit
budget is spent or no candidate beats the incumbent.

The selection rule here uses ONLY public signals (ESM plausibility + pLDDT).
The hidden score is recorded alongside but never steers the search — that gap is
exactly what produces the counterexamples the outer loop learns from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..cache import fold_cache
from . import grounder
from .policy_interpreter import Candidate, apply_policy, sequence_of

# Public proxy objective. Deliberately *not* the hidden objective: no TM-score,
# no contact recovery, nothing derived from a reference structure.
PUBLIC_WEIGHTS = {"esm_score": 0.5, "mean_plddt": 0.5}


def public_score(metrics: dict) -> float:
    return sum(w * float(metrics[k]) for k, w in PUBLIC_WEIGHTS.items())


@dataclass
class Step:
    round_index: int
    incumbent: str
    incumbent_public: float
    incumbent_hidden: float
    candidates: list = field(default_factory=list)   # [{label, sequence, metrics}]
    chosen: dict | None = None
    accepted: bool = False
    state_before: dict | None = None


@dataclass
class InnerResult:
    origin: str
    sequence: str
    steps: list = field(default_factory=list)
    final_metrics: dict = field(default_factory=dict)
    origin_metrics: dict = field(default_factory=dict)

    @property
    def hidden_score(self) -> float:
        return float(self.final_metrics["hidden_score"])

    @property
    def public_score(self) -> float:
        return public_score(self.final_metrics)

    @property
    def hidden_delta(self) -> float:
        return self.hidden_score - float(self.origin_metrics["hidden_score"])

    @property
    def predicted_delta(self) -> float:
        return self.public_score - public_score(self.origin_metrics)

    @property
    def mutations(self) -> list:
        return [
            {"position": i, "from": a, "to": b}
            for i, (a, b) in enumerate(zip(self.origin, self.sequence))
            if a != b
        ]

    def final_state(self) -> dict:
        return self.steps[-1].state_before if self.steps else {}


def repair(
    origin: str,
    policy: dict,
    evaluator,
    max_rounds: int | None = None,
    verbose: bool = False,
) -> InnerResult:
    """Run the inner loop on one corrupted sequence under one policy."""
    budget = int(policy["proposal"]["max_total_edits"])
    rounds = budget if max_rounds is None else int(max_rounds)

    origin_metrics = evaluator.evaluate(origin, origin=origin)
    incumbent = origin
    incumbent_metrics = origin_metrics
    result = InnerResult(origin=origin, sequence=origin, origin_metrics=origin_metrics)

    parent_fold = fold_cache.fold(origin)

    for round_index in range(rounds):
        fold = fold_cache.fold(incumbent)
        state = grounder.ground(
            incumbent,
            fold,
            mutations=[
                {"position": i, "from": a, "to": b}
                for i, (a, b) in enumerate(zip(origin, incumbent))
                if a != b
            ],
            parent_structure=parent_fold if incumbent != origin else None,
        )

        step = Step(
            round_index=round_index,
            incumbent=incumbent,
            incumbent_public=public_score(incumbent_metrics),
            incumbent_hidden=float(incumbent_metrics["hidden_score"]),
            state_before=state,
        )

        shortlist = apply_policy(policy, state, origin=origin)
        if not shortlist:
            result.steps.append(step)
            break

        scored = []
        for cand in shortlist:
            metrics = evaluator.evaluate(cand.sequence, origin=origin)
            scored.append(
                {
                    "label": cand.label(),
                    "sequence": cand.sequence,
                    "position": cand.position,
                    "from_aa": cand.from_aa,
                    "to_aa": cand.to_aa,
                    "edit_count": cand.edit_count,
                    "prerank_score": cand.prerank_score,
                    "public_score": public_score(metrics),
                    "metrics": metrics,
                }
            )
        step.candidates = scored

        best = max(scored, key=lambda c: (c["public_score"], -c["edit_count"], c["sequence"]))
        step.chosen = best

        if best["public_score"] > step.incumbent_public:
            step.accepted = True
            incumbent = best["sequence"]
            incumbent_metrics = best["metrics"]
            if verbose:
                print(
                    f"  round {round_index}: accept {best['label']} "
                    f"public {step.incumbent_public:.4f} -> {best['public_score']:.4f}"
                )
        elif verbose:
            print(f"  round {round_index}: no public improvement, stopping")

        result.steps.append(step)
        if not step.accepted:
            break

    result.sequence = incumbent
    result.final_metrics = incumbent_metrics
    return result


def candidate_from_dict(d: dict) -> Candidate:
    """Rehydrate a serialised candidate (used by the UI)."""
    return Candidate(
        sequence=d["sequence"],
        position=d["position"],
        from_aa=d["from_aa"],
        to_aa=d["to_aa"],
        parent_sequence=d.get("parent_sequence", ""),
    )


def state_sequence(state: dict) -> str:
    return sequence_of(state)

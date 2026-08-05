"""P3.4 — the keep-if-better outer loop (median rule).

A candidate policy replaces the incumbent only if its MEDIAN hidden score across
the seeded corruption set beats the incumbent's median. Median, not mean: one
lucky variant must not carry a policy that is worse everywhere else.

The outer loop is the only place `hidden_score` influences a decision, and it
influences exactly one bit of information — accept or reject.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..paths import LOGS
from . import counterexamples, inner_loop, policy as policy_mod

GENERATION_LOG = LOGS / "generations.jsonl"


def median(values) -> float:
    """statistics.median: mean of the two middles for even-length input."""
    return float(statistics.median(list(values)))


def decide(incumbent_scores, candidate_scores) -> dict:
    """Pure accept/reject decision. Hand-checkable, no I/O.

    Accept the candidate iff median(candidate) > median(incumbent). Strictly
    greater: a tie keeps the incumbent, so the policy never drifts on noise.
    """
    inc = median(incumbent_scores)
    cand = median(candidate_scores)
    return {
        "incumbent_median": inc,
        "candidate_median": cand,
        "accepted": bool(cand > inc),
        "delta": cand - inc,
    }


@dataclass
class GenerationResult:
    iteration: int
    accepted: bool
    incumbent_median: float
    candidate_median: float
    delta: float
    incumbent_public_median: float
    candidate_public_median: float
    incumbent_runs: list = field(default_factory=list)
    candidate_runs: list = field(default_factory=list)
    policy_before: dict = field(default_factory=dict)
    policy_after: dict = field(default_factory=dict)
    counterexample: dict | None = None
    patch: dict | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False)


def _summarise(result: inner_loop.InnerResult, variant_id: str) -> dict:
    return {
        "variant": variant_id,
        "origin": result.origin,
        "repaired": result.sequence,
        "mutations": [
            f"{m['from']}{m['position'] + 1}{m['to']}" for m in result.mutations
        ],
        "hidden_score": float(result.final_metrics["hidden_score"]),
        "hidden_delta": result.hidden_delta,
        "public_score": result.public_score,
        "predicted_delta": result.predicted_delta,
        "mean_plddt": float(result.final_metrics["mean_plddt"]),
        "esm_score": float(result.final_metrics["esm_score"]),
        "edit_count": int(result.final_metrics["edit_count"]),
        "rounds": len(result.steps),
    }


class OuterLoop:
    def __init__(
        self,
        evaluator,
        incumbent_policy: dict | None = None,
        generation_log: Path | None = None,
        counterexample_log: Path | None = None,
        verbose: bool = True,
    ) -> None:
        self.evaluator = evaluator
        self.incumbent_policy = policy_mod.validate_policy(
            incumbent_policy or policy_mod.load_seed_policy()
        )
        self.iteration = 0
        self.history: list[GenerationResult] = []
        self.generation_log = Path(generation_log) if generation_log else GENERATION_LOG
        self.counterexample_log = Path(counterexample_log) if counterexample_log else None
        self.verbose = verbose

    # ------------------------------------------------------------------ #

    def run_policy(self, policy: dict, corruption_set: dict) -> list[dict]:
        """Inner loop over every corrupted variant under one policy."""
        runs = []
        for variant_id, sequence in corruption_set.items():
            self.evaluator.set_origin(sequence)
            result = inner_loop.repair(sequence, policy, self.evaluator, verbose=False)
            runs.append(_summarise(result, variant_id))
            runs[-1]["_result"] = result
        return runs

    def run_generation(
        self, candidate_policy: dict, corruption_set: dict, patch: dict | None = None
    ) -> GenerationResult:
        """One generation: score both policies, apply the median rule, log."""
        candidate_policy = policy_mod.validate_policy(candidate_policy)

        incumbent_runs = self.run_policy(self.incumbent_policy, corruption_set)
        candidate_runs = self.run_policy(candidate_policy, corruption_set)

        verdict = decide(
            [r["hidden_score"] for r in incumbent_runs],
            [r["hidden_score"] for r in candidate_runs],
        )

        inc_public = median(r["public_score"] for r in incumbent_runs)
        cand_public = median(r["public_score"] for r in candidate_runs)

        counterexample = None
        worst = _worst_disagreement(candidate_runs)
        if worst is not None:
            counterexample = counterexamples.record(
                self.iteration,
                worst["predicted_delta"],
                worst["hidden_delta"],
                policy_mod.clone(self.incumbent_policy),
                worst["_result"].final_state(),
                path=self.counterexample_log,
                extra={
                    "variant": worst["variant"],
                    "candidate_policy": policy_mod.clone(candidate_policy),
                    "mutations": worst["mutations"],
                },
            )
            if self.verbose:
                print(
                    f"[gen {self.iteration}] COUNTEREXAMPLE on {worst['variant']}: "
                    f"public {worst['predicted_delta']:+.4f} but hidden "
                    f"{worst['hidden_delta']:+.4f}"
                )

        result = GenerationResult(
            iteration=self.iteration,
            accepted=verdict["accepted"],
            incumbent_median=verdict["incumbent_median"],
            candidate_median=verdict["candidate_median"],
            delta=verdict["delta"],
            incumbent_public_median=inc_public,
            candidate_public_median=cand_public,
            incumbent_runs=[_strip(r) for r in incumbent_runs],
            candidate_runs=[_strip(r) for r in candidate_runs],
            policy_before=policy_mod.clone(self.incumbent_policy),
            policy_after=policy_mod.clone(
                candidate_policy if verdict["accepted"] else self.incumbent_policy
            ),
            counterexample=counterexample,
            patch=patch,
        )

        if verdict["accepted"]:
            self.incumbent_policy = policy_mod.clone(candidate_policy)

        if self.verbose:
            verb = "ACCEPT" if verdict["accepted"] else "reject"
            print(
                f"[gen {self.iteration}] {verb}  incumbent median "
                f"{verdict['incumbent_median']:.4f} vs candidate "
                f"{verdict['candidate_median']:.4f}"
            )

        self.history.append(result)
        self._log(result)
        self.iteration += 1
        return result

    def _log(self, result: GenerationResult) -> None:
        self.generation_log.parent.mkdir(parents=True, exist_ok=True)
        with self.generation_log.open("a", encoding="utf-8") as fh:
            fh.write(result.to_json() + "\n")


def _strip(run: dict) -> dict:
    return {k: v for k, v in run.items() if not k.startswith("_")}


def _worst_disagreement(runs: list[dict]) -> dict | None:
    """The repair whose public gain the hidden verifier contradicts most.

    A counterexample is judged per repair (origin -> repaired under the candidate
    policy), not per policy median: that is what "predicted improvement but
    hidden_score did not improve" literally describes, and it means a
    counterexample can exist on the very first generation, before any patch has
    been proposed. Exactly one row is written per generation — the worst one.
    """
    disagreements = [
        r
        for r in runs
        if counterexamples.is_counterexample(r["predicted_delta"], r["hidden_delta"])
    ]
    if not disagreements:
        return None
    return max(disagreements, key=lambda r: r["predicted_delta"] - r["hidden_delta"])


def load_generations(path: Path | None = None) -> list[dict]:
    path = Path(path) if path else GENERATION_LOG
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

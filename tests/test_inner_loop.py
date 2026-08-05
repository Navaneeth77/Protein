"""The inner repair loop: public-only selection, edit budget, trajectory shape."""

from __future__ import annotations

import pytest

from src.agent import inner_loop, policy as policy_mod
from src.cache import fold_cache
from tests.conftest import make_pdb

ORIGIN = "MTYKLILNGKTLKGETTTEAPDAATAEK"


class RecordingEvaluator:
    """Rewards edits publicly; records every call so leakage can be asserted."""

    def __init__(self, hidden_slope: float = 0.05):
        self.hidden_slope = hidden_slope
        self.origin = None
        self.seen = []

    def set_origin(self, origin_sequence):
        self.origin = origin_sequence

    def evaluate(self, candidate, reveal=False, origin=None):
        assert reveal is False
        sequence = getattr(candidate, "sequence", candidate)
        baseline = origin or self.origin or sequence
        edits = sum(1 for a, b in zip(baseline, sequence) if a != b)
        self.seen.append((sequence, edits))
        return {
            "hidden_score": round(0.60 + self.hidden_slope * edits, 6),
            "esm_score": round(0.50 + 0.05 * edits, 6),
            "pseudo_log_likelihood": -50.0,
            "mean_plddt": round(0.50 + 0.05 * edits, 6),
            "edit_count": edits,
            "edit_fraction": edits / 3,
            "clashes": 0,
            "from_cache": True,
        }


@pytest.fixture
def stubbed_structures(monkeypatch):
    monkeypatch.setattr(
        fold_cache,
        "fold",
        lambda seq: fold_cache.structure_features(seq, make_pdb(seq), from_cache=True),
    )


def test_public_score_uses_only_public_signals():
    """The proxy objective must not reference anything reference-derived."""
    assert set(inner_loop.PUBLIC_WEIGHTS) == {"esm_score", "mean_plddt"}
    assert sum(inner_loop.PUBLIC_WEIGHTS.values()) == pytest.approx(1.0)
    metrics = {"esm_score": 0.4, "mean_plddt": 0.8}
    assert inner_loop.public_score(metrics) == pytest.approx(0.6)


def test_repair_respects_the_edit_budget(stubbed_structures, stub_esm):
    seed = policy_mod.load_seed_policy()
    evaluator = RecordingEvaluator()
    evaluator.set_origin(ORIGIN)
    result = inner_loop.repair(ORIGIN, seed, evaluator)

    assert result.final_metrics["edit_count"] <= seed["proposal"]["max_total_edits"]
    assert len(result.mutations) == result.final_metrics["edit_count"]
    assert all(e <= seed["proposal"]["max_total_edits"] for _, e in evaluator.seen)


def test_repair_stops_when_the_public_score_stops_improving(stubbed_structures, stub_esm):
    """A flat public signal must terminate the loop rather than spend the budget."""

    class FlatEvaluator(RecordingEvaluator):
        def evaluate(self, candidate, reveal=False, origin=None):
            metrics = super().evaluate(candidate, reveal=reveal, origin=origin)
            metrics["esm_score"] = 0.5
            metrics["mean_plddt"] = 0.5
            return metrics

    evaluator = FlatEvaluator()
    evaluator.set_origin(ORIGIN)
    result = inner_loop.repair(ORIGIN, policy_mod.load_seed_policy(), evaluator)

    assert result.sequence == ORIGIN, "nothing improved, so nothing should be accepted"
    assert len(result.steps) == 1
    assert result.steps[0].accepted is False


def test_trajectory_records_every_candidate_it_folded(stubbed_structures, stub_esm):
    evaluator = RecordingEvaluator()
    evaluator.set_origin(ORIGIN)
    result = inner_loop.repair(ORIGIN, policy_mod.load_seed_policy(), evaluator)

    for step in result.steps:
        assert 0 < len(step.candidates) <= 3
        assert step.state_before is not None
        assert step.state_before["sequence_length"] == len(ORIGIN)
        for candidate in step.candidates:
            assert set(candidate) >= {"label", "sequence", "metrics", "public_score"}


def test_deltas_are_measured_against_the_origin(stubbed_structures, stub_esm):
    evaluator = RecordingEvaluator()
    evaluator.set_origin(ORIGIN)
    result = inner_loop.repair(ORIGIN, policy_mod.load_seed_policy(), evaluator)

    assert result.origin == ORIGIN
    assert result.predicted_delta == pytest.approx(
        result.public_score - inner_loop.public_score(result.origin_metrics)
    )
    assert result.hidden_delta == pytest.approx(
        result.hidden_score - result.origin_metrics["hidden_score"]
    )


def test_repair_is_deterministic(stubbed_structures, stub_esm):
    seed = policy_mod.load_seed_policy()

    def run():
        evaluator = RecordingEvaluator()
        evaluator.set_origin(ORIGIN)
        return inner_loop.repair(ORIGIN, seed, evaluator).sequence

    assert run() == run()


def test_a_wider_policy_folds_no_more_than_three_candidates_per_round(
    stubbed_structures, stub_esm
):
    wide = policy_mod.clone(policy_mod.load_seed_policy())
    wide["proposal"]["positions"] = 6
    wide["proposal"]["substitutions_per_position"] = 8
    evaluator = RecordingEvaluator()
    evaluator.set_origin(ORIGIN)
    result = inner_loop.repair(ORIGIN, policy_mod.validate_policy(wide), evaluator)

    # P2.4's whole point: enumeration can be wide, folding stays cheap.
    for step in result.steps:
        assert len(step.candidates) <= 3


def test_mutations_are_reported_against_the_origin(stubbed_structures, stub_esm):
    evaluator = RecordingEvaluator()
    evaluator.set_origin(ORIGIN)
    result = inner_loop.repair(ORIGIN, policy_mod.load_seed_policy(), evaluator)
    for mutation in result.mutations:
        position = mutation["position"]
        assert ORIGIN[position] == mutation["from"]
        assert result.sequence[position] == mutation["to"]
        assert mutation["from"] != mutation["to"]

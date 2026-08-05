"""P3.3 + P3.4 — counterexample recording and the median keep-if-better rule."""

from __future__ import annotations

import json

import pytest

from src.agent import counterexamples, outer_loop, policy as policy_mod
from src.agent.outer_loop import decide
from src.cache import fold_cache
from tests.conftest import make_pdb


# --------------------------------------------------------------------------- #
# P3.4 — the median rule, hand-checked
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "incumbent,candidate,expected_inc_median,expected_cand_median,expected_accept",
    [
        # Candidate wins on two of three but loses the median-relevant middle.
        ([0.50, 0.60, 0.70], [0.90, 0.55, 0.95], 0.60, 0.90, True),
        # Candidate wins two variants outright but its median is still lower.
        ([0.40, 0.80, 0.90], [0.85, 0.45, 0.50], 0.80, 0.50, False),
        # Exact tie on the median keeps the incumbent (strictly-greater rule).
        ([0.10, 0.50, 0.90], [0.49, 0.50, 0.51], 0.50, 0.50, False),
        # One outlier must not carry a policy that is worse elsewhere.
        ([0.60, 0.61, 0.62], [0.10, 0.11, 0.99], 0.61, 0.11, False),
        # Uniform improvement.
        ([0.20, 0.30, 0.40], [0.21, 0.31, 0.41], 0.30, 0.31, True),
    ],
)
def test_median_rule_matches_hand_calculation(
    incumbent, candidate, expected_inc_median, expected_cand_median, expected_accept
):
    verdict = decide(incumbent, candidate)
    assert verdict["incumbent_median"] == pytest.approx(expected_inc_median)
    assert verdict["candidate_median"] == pytest.approx(expected_cand_median)
    assert verdict["accepted"] is expected_accept
    assert verdict["delta"] == pytest.approx(expected_cand_median - expected_inc_median)


def test_median_is_not_the_mean():
    """A mean-based rule would accept this; the median rule must not."""
    incumbent = [0.50, 0.50, 0.50]
    candidate = [0.10, 0.40, 1.00]     # mean 0.50, median 0.40
    assert sum(candidate) / 3 == pytest.approx(0.5)
    assert decide(incumbent, candidate)["accepted"] is False


# --------------------------------------------------------------------------- #
# end-to-end generation with a stubbed evaluator
# --------------------------------------------------------------------------- #

# Positions the stub structure marks as low-confidence. A policy weighted on
# low_plddt targets exactly these; a policy weighted on contact_violation targets
# a disjoint set (verified by test_the_two_policies_target_disjoint_sites).
CRITICAL = (5, 6, 7)
PROPOSAL = {
    "positions": 3,
    "substitutions_per_position": 4,
    "preserve_residue_class": True,
    "max_total_edits": 3,
}
CONTACT_POLICY = {"position_score": {"contact_violation": 1.0}, "proposal": dict(PROPOSAL)}
PLDDT_POLICY = {"position_score": {"low_plddt": 1.0}, "proposal": dict(PROPOSAL)}


class StubEvaluator:
    """Deterministic stand-in whose public and hidden verdicts can be made to
    agree or disagree on demand.

    Public signals always reward edits, and reward edits at CRITICAL sites a
    little extra. The hidden score rewards edits too, but weights CRITICAL sites
    by `critical_coefficient`: negative means those edits actually damage the
    fold, which is precisely a counterexample.
    """

    def __init__(self, critical_coefficient: float = -0.30):
        self.critical_coefficient = critical_coefficient
        self.origin = None
        self.calls = 0

    def set_origin(self, origin_sequence):
        self.origin = origin_sequence

    def evaluate(self, candidate, reveal=False, origin=None):
        assert reveal is False, "loop code must never ask for the reveal"
        sequence = getattr(candidate, "sequence", candidate)
        baseline = origin or self.origin or sequence
        changed = [i for i, (a, b) in enumerate(zip(baseline, sequence)) if a != b]
        edits = len(changed)
        critical = sum(1 for i in changed if i in CRITICAL)
        self.calls += 1
        return {
            "hidden_score": round(
                0.60 + 0.05 * edits + self.critical_coefficient * critical, 6
            ),
            "esm_score": round(0.50 + 0.05 * edits + 0.02 * critical, 6),
            "pseudo_log_likelihood": -50.0,
            "mean_plddt": round(0.50 + 0.05 * edits, 6),
            "edit_count": edits,
            "edit_fraction": edits / 3,
            "clashes": 0,
            "from_cache": True,
        }


@pytest.fixture
def stubbed_structures(monkeypatch):
    """Fold stub with a pLDDT dip at CRITICAL, so low_plddt is informative."""
    import numpy as np

    def _fold(seq):
        plddt = np.array(
            [40.0 if i in CRITICAL else 92.0 for i in range(len(seq))], dtype=float
        )
        return fold_cache.structure_features(
            seq, make_pdb(seq, plddt=plddt), from_cache=True
        )

    monkeypatch.setattr(fold_cache, "fold", _fold)
    return _fold


@pytest.fixture
def corruption_set():
    return {
        "v1": "MTYKLILNGKTLKGETTTEAPDAATAEK",
        "v2": "MTYWLILNGKTLKGETTTEAVDAATAEK",
        "v3": "MTYKLILNGKTLKGETTTEAVDAATGEK",
    }


def _loop(evaluator, tmp_path, incumbent=None):
    return outer_loop.OuterLoop(
        evaluator,
        incumbent_policy=policy_mod.clone(incumbent or CONTACT_POLICY),
        generation_log=tmp_path / "generations.jsonl",
        counterexample_log=tmp_path / "counterexamples.jsonl",
        verbose=False,
    )


def test_the_two_policies_target_disjoint_sites(
    stubbed_structures, stub_esm, corruption_set
):
    """The premise the counterexample test rests on, checked explicitly."""
    from src.agent import grounder
    from src.agent.policy_interpreter import select_positions

    seq = corruption_set["v1"]
    state = grounder.ground(seq, stubbed_structures(seq))
    contact_sites = set(select_positions(policy_mod.validate_policy(CONTACT_POLICY), state))
    plddt_sites = set(select_positions(policy_mod.validate_policy(PLDDT_POLICY), state))
    assert plddt_sites == set(CRITICAL)
    assert not (contact_sites & plddt_sites)


def test_one_generation_records_exactly_one_counterexample(
    tmp_path, stubbed_structures, stub_esm, corruption_set
):
    """P3.3 verify: exactly one well-formed JSONL line, all required keys."""
    loop = _loop(StubEvaluator(critical_coefficient=-0.30), tmp_path)
    result = loop.run_generation(policy_mod.clone(PLDDT_POLICY), corruption_set)

    assert result.candidate_public_median > result.incumbent_public_median
    assert result.candidate_median < result.incumbent_median
    assert result.accepted is False, "hidden score got worse; the rule must reject"
    assert result.counterexample is not None

    log = tmp_path / "counterexamples.jsonl"
    lines = [l for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1, f"expected 1 counterexample row, got {len(lines)}"

    row = json.loads(lines[0])
    for key in counterexamples.REQUIRED_KEYS:
        assert key in row, f"missing key {key}"
    assert row["iteration"] == 0
    assert row["predicted_delta"] > 0
    assert row["hidden_delta"] <= 0
    assert row["policy_before"]["position_score"] == {"contact_violation": 1.0}
    assert row["state_before"]["sequence_length"] == len(corruption_set["v1"])
    assert "top_residues" in row["state_before"]


def test_no_counterexample_when_hidden_agrees(
    tmp_path, stubbed_structures, stub_esm, corruption_set
):
    """Same policies, but now CRITICAL edits genuinely help: accept, no row."""
    loop = _loop(StubEvaluator(critical_coefficient=+0.10), tmp_path)
    result = loop.run_generation(policy_mod.clone(PLDDT_POLICY), corruption_set)

    assert result.candidate_public_median > result.incumbent_public_median
    assert result.candidate_median > result.incumbent_median
    assert result.accepted is True
    assert result.counterexample is None

    log = tmp_path / "counterexamples.jsonl"
    assert not log.exists() or not log.read_text(encoding="utf-8").strip()


def test_accepted_generation_replaces_the_incumbent_policy(
    tmp_path, stubbed_structures, stub_esm, corruption_set
):
    loop = _loop(StubEvaluator(critical_coefficient=+0.10), tmp_path)
    before = policy_mod.clone(loop.incumbent_policy)
    result = loop.run_generation(policy_mod.clone(PLDDT_POLICY), corruption_set)

    assert result.accepted is True
    assert loop.incumbent_policy["position_score"] == {"low_plddt": 1.0}
    assert before["position_score"] == {"contact_violation": 1.0}
    assert loop.iteration == 1


def test_rejected_generation_keeps_the_incumbent_policy(
    tmp_path, stubbed_structures, stub_esm, corruption_set
):
    loop = _loop(StubEvaluator(critical_coefficient=-0.30), tmp_path)
    before = policy_mod.clone(loop.incumbent_policy)
    result = loop.run_generation(policy_mod.clone(PLDDT_POLICY), corruption_set)
    assert result.accepted is False
    assert loop.incumbent_policy == before


def test_generation_log_has_one_line_per_generation(
    tmp_path, stubbed_structures, stub_esm, corruption_set
):
    """P4.4 guard: no dropped generation 0, no double-counted seed policy."""
    loop = _loop(StubEvaluator(), tmp_path)
    for positions in (4, 5, 6):
        candidate = policy_mod.clone(PLDDT_POLICY)
        candidate["proposal"]["positions"] = positions
        loop.run_generation(candidate, corruption_set)

    log = tmp_path / "generations.jsonl"
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 3
    assert [r["iteration"] for r in rows] == [0, 1, 2]
    assert len(loop.history) == 3


def test_loop_code_never_requests_the_reveal(
    tmp_path, stubbed_structures, stub_esm, corruption_set
):
    """StubEvaluator asserts reveal is False on every single call."""
    loop = _loop(StubEvaluator(), tmp_path)
    loop.run_generation(policy_mod.clone(PLDDT_POLICY), corruption_set)
    assert loop.evaluator.calls > 0


def test_generation_result_is_json_serialisable(
    tmp_path, stubbed_structures, stub_esm, corruption_set
):
    loop = _loop(StubEvaluator(), tmp_path)
    result = loop.run_generation(policy_mod.clone(PLDDT_POLICY), corruption_set)
    reparsed = json.loads(result.to_json())
    assert reparsed["iteration"] == 0
    assert len(reparsed["incumbent_runs"]) == 3
    assert len(reparsed["candidate_runs"]) == 3
    assert all("_result" not in r for r in reparsed["candidate_runs"])

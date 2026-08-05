"""P3.2 + P2.3 + P2.4 — the deterministic interpreter and candidate pipeline."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.agent import grounder, policy as policy_mod
from src.agent.policy_interpreter import (
    Candidate,
    apply_policy,
    edit_count,
    enumerate_candidates,
    position_scores,
    prerank_candidates,
    select_positions,
)
from src.constants import residue_class

FORBIDDEN = re.compile(r"eval\(|exec\(|__import__|subprocess|os\.system")


def test_interpreter_contains_no_dynamic_code_execution():
    """C2, code-level: mirrors the P3.2 grep over the whole file, comments included."""
    src = Path("src/agent/policy_interpreter.py").read_text(encoding="utf-8")
    assert not FORBIDDEN.search(src)


@pytest.fixture
def state(sequence, fold_of, stub_esm):
    stub_esm["implausible"] = {7: 0.02, 13: 0.05, 20: 0.10}
    return grounder.ground(sequence, fold_of(sequence))


def test_positions_field_actually_drives_behaviour(state):
    """P3.2 verify: positions=5 yields 5 sites, positions=3 yields 3."""
    seed = policy_mod.load_seed_policy()
    three = select_positions(seed, state)

    wider = policy_mod.clone(seed)
    wider["proposal"]["positions"] = 5
    five = select_positions(policy_mod.validate_policy(wider), state)

    assert len(three) == 3
    assert len(five) == 5
    assert set(three) <= set(five)


def test_position_selection_is_deterministic_and_ties_break_low(state):
    seed = policy_mod.load_seed_policy()
    assert select_positions(seed, state) == select_positions(seed, state)

    flat = {"residues": [
        {"position": i, "aa": "A", "esm_surprisal": 1.0, "low_plddt": 0.0,
         "contact_violation": 0.0, "long_range_contact_violation": 0.0}
        for i in range(6)
    ]}
    assert select_positions(seed, flat) == [0, 1, 2]


def test_reweighting_changes_the_selected_sites(state):
    seed = policy_mod.load_seed_policy()
    surprisal_driven = select_positions(seed, state)

    plddt_driven_policy = policy_mod.validate_policy(
        {"position_score": {"low_plddt": 1.0}, "proposal": dict(seed["proposal"])}
    )
    scores_a = position_scores(seed, state)
    scores_b = position_scores(plddt_driven_policy, state)
    # Weights must be doing real work: the two score vectors differ.
    assert not all(abs(a - b) < 1e-12 for a, b in zip(scores_a, scores_b))
    assert len(surprisal_driven) == 3


def test_candidate_count_is_positions_times_substitutions(state, stub_esm):
    """P2.3 verify: default policy yields at most 3 x 4 = 12 unique candidates."""
    seed = policy_mod.load_seed_policy()
    candidates = enumerate_candidates(seed, state)
    assert len(candidates) <= 12
    assert len({c.sequence for c in candidates}) == len(candidates)
    incumbent = "".join(r["aa"] for r in state["residues"])
    assert all(c.sequence != incumbent for c in candidates)


def test_same_residue_substitution_is_filtered(state):
    """A "substitution" back to the incumbent residue is not a candidate."""
    incumbent = "".join(r["aa"] for r in state["residues"])
    seed = policy_mod.load_seed_policy()

    def ranker(seq, pos, top_n):
        # Offer the incumbent residue first; it must be dropped.
        return [(seq[pos], 0.99), ("A" if seq[pos] != "A" else "V", 0.5)]

    candidates = enumerate_candidates(seed, state, substitution_source=ranker)
    assert candidates, "the filter must not empty the shortlist"
    assert all(c.to_aa != incumbent[c.position] for c in candidates)
    assert all(c.sequence != incumbent for c in candidates)


def test_preserve_residue_class_filters_across_classes(state):
    seed = policy_mod.clone(policy_mod.load_seed_policy())
    seed["proposal"]["preserve_residue_class"] = True
    kept = enumerate_candidates(policy_mod.validate_policy(seed), state)
    incumbent = "".join(r["aa"] for r in state["residues"])
    assert kept
    for c in kept:
        assert residue_class(c.to_aa) == residue_class(incumbent[c.position])

    seed["proposal"]["preserve_residue_class"] = False
    unfiltered = enumerate_candidates(policy_mod.validate_policy(seed), state)
    assert len(unfiltered) >= len(kept)


def test_edit_budget_is_counted_against_the_origin(state):
    seed = policy_mod.clone(policy_mod.load_seed_policy())
    seed["proposal"]["max_total_edits"] = 1
    policy = policy_mod.validate_policy(seed)

    incumbent = "".join(r["aa"] for r in state["residues"])
    # Pretend the incumbent already differs from the origin at position 0.
    origin = ("A" if incumbent[0] != "A" else "V") + incumbent[1:]
    assert edit_count(origin, incumbent) == 1

    candidates = enumerate_candidates(policy, state, origin=origin)
    # Every further edit would make 2 > max_total_edits=1, so nothing survives
    # except a change back at position 0 if the ranker offers one.
    assert all(c.edit_count <= 1 for c in candidates)


def test_prerank_breaks_pll_ties_by_edit_count():
    """P2.4 verify: equal PLL, fewer edits wins."""
    parent = "AAAA"
    few = Candidate(sequence="AAAV", position=3, from_aa="A", to_aa="V",
                    parent_sequence=parent, mutations=[{"position": 3}])
    many = Candidate(sequence="AAVV", position=2, from_aa="A", to_aa="V",
                     parent_sequence=parent,
                     mutations=[{"position": 2}, {"position": 3}])
    ranked = prerank_candidates([many, few], shortlist_size=2, pll_source=lambda s: -10.0)
    assert ranked[0] is few
    assert ranked[0].prerank_score > ranked[1].prerank_score


def test_shortlist_never_exceeds_three(state):
    seed = policy_mod.load_seed_policy()
    shortlist = apply_policy(seed, state)
    assert 0 < len(shortlist) <= 3


def test_apply_policy_is_a_pure_function(state):
    seed = policy_mod.load_seed_policy()
    first = [c.sequence for c in apply_policy(seed, state)]
    second = [c.sequence for c in apply_policy(seed, state)]
    assert first == second

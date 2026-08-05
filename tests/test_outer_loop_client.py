"""P3.5 — bounded patch validation.

No live endpoint is used anywhere in this file: every reply is canned and fed in
through the injectable transport.
"""

from __future__ import annotations

import json

import pytest

from src.agent import grounder, outer_loop_client as client, policy as policy_mod
from src.agent.policy_interpreter import select_positions


def canned(reply: str):
    return lambda prompt: reply


@pytest.fixture
def seed():
    return policy_mod.load_seed_policy()


@pytest.fixture
def outcomes():
    return [
        {
            "variant": "corrupt_01",
            "mutations": ["A34K"],
            "predicted_delta": 0.04,
            "hidden_delta": -0.02,
            "mean_plddt": 0.71,
            "esm_score": 0.55,
        }
    ]


# --------------------------------------------------------------------------- #
# rejection
# --------------------------------------------------------------------------- #

def test_rejects_out_of_schema_patch(tmp_path, seed, outcomes):
    """P3.5 verify: a patch naming a field absent from state.json is rejected."""
    reply = json.dumps(
        {
            "kind": "representation",
            "rationale": "weight the hydrophobic moment",
            "position_score": {"hydrophobic_moment": 0.5, "esm_surprisal": 0.5},
        }
    )
    log = tmp_path / "calls.jsonl"
    outcome = client.propose_patch(
        seed, outcomes, transport=canned(reply), log_path=log
    )

    assert outcome.accepted is False
    assert outcome.policy is None
    assert "hydrophobic_moment" in outcome.error

    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["accepted"] is False
    assert "hydrophobic_moment" in rows[0]["error"]

    # And the incumbent policy is untouched — never force-applied.
    assert "hydrophobic_moment" not in seed["position_score"]


@pytest.mark.parametrize(
    "reply,fragment",
    [
        ("I think you should try weighting contacts more.", "no JSON object"),
        ('{"kind": "representation",}', "not valid JSON"),
        (json.dumps({"kind": "representation", "position_score": {}}), "changes nothing"),
        (json.dumps({"kind": "mechanism", "proposal": {}}), "changes nothing"),
        (json.dumps({"kind": "mechanism", "rationale": "none"}), "changes nothing"),
        (json.dumps({"kind": "mechanism", "proposal": {"exec": "os.system('x')"}}),
         "not in the DSL"),
        (json.dumps({"kind": "representation",
                     "position_score": {"esm_surprisal": 0.7, "low_plddt": 0.7}}),
         "sum to 1.0"),
        (json.dumps({"kind": "mechanism", "proposal": {"max_total_edits": -3}}),
         "schema violation"),
        (json.dumps({"kind": "mechanism", "proposal": {"positions": 999}}),
         "schema violation"),
    ],
)
def test_malformed_replies_are_rejected_with_a_clear_reason(
    tmp_path, seed, outcomes, reply, fragment
):
    outcome = client.propose_patch(
        seed, outcomes, transport=canned(reply), log_path=tmp_path / "c.jsonl"
    )
    assert outcome.accepted is False
    assert fragment in outcome.error, outcome.error


def test_a_patch_carrying_both_halves_is_applied(tmp_path, seed, outcomes):
    """Local models mix the two halves; both are applied and both are validated."""
    reply = json.dumps(
        {
            "kind": "representation",
            "position_score": {"esm_surprisal": 1.0},
            "proposal": {"positions": 9},
        }
    )
    outcome = client.propose_patch(
        seed, outcomes, transport=canned(reply), log_path=tmp_path / "c.jsonl"
    )
    assert outcome.accepted is True
    assert outcome.policy["position_score"] == {"esm_surprisal": 1.0}
    assert outcome.policy["proposal"]["positions"] == 9


def test_kind_is_inferred_when_the_model_omits_it(tmp_path, seed, outcomes):
    reply = json.dumps({"rationale": "widen", "proposal": {"positions": 5}})
    outcome = client.propose_patch(
        seed, outcomes, transport=canned(reply), log_path=tmp_path / "c.jsonl"
    )
    assert outcome.accepted is True
    assert outcome.policy["proposal"]["positions"] == 5


def test_a_declared_kind_inconsistent_with_the_payload_is_rejected(
    tmp_path, seed, outcomes
):
    reply = json.dumps({"kind": "mechanism", "position_score": {"esm_surprisal": 1.0}})
    outcome = client.propose_patch(
        seed, outcomes, transport=canned(reply), log_path=tmp_path / "c.jsonl"
    )
    assert outcome.accepted is False
    assert "declares kind" in outcome.error


# --------------------------------------------------------------------------- #
# acceptance
# --------------------------------------------------------------------------- #

VALID_REPRESENTATION_PATCH = {
    "kind": "representation",
    "rationale": "Public signals rose while the hidden verifier did not; the edits "
                 "sat at sites whose local packing was intact but whose long-range "
                 "contacts were lost.",
    "position_score": {
        "esm_surprisal": 0.45,
        "low_plddt": 0.15,
        "contact_violation": 0.15,
        "long_range_contact_violation": 0.25,
    },
}


def test_accepts_a_valid_representation_patch(tmp_path, seed, outcomes):
    outcome = client.propose_patch(
        seed,
        outcomes,
        transport=canned(json.dumps(VALID_REPRESENTATION_PATCH)),
        log_path=tmp_path / "c.jsonl",
    )
    assert outcome.accepted is True
    assert outcome.policy["position_score"]["long_range_contact_violation"] == 0.25
    # The proposal half is carried over untouched.
    assert outcome.policy["proposal"] == seed["proposal"]
    # The incumbent is not mutated in place.
    assert "long_range_contact_violation" not in seed["position_score"]


def _state_where_the_two_terms_disagree() -> dict:
    """A hand-built state: total-contact deficit and long-range deficit conflict.

    Positions 0-2 are locally under-packed but long-range intact; positions 3-5
    are the reverse. A seed-weighted policy must pick the first group and a
    long-range-weighted policy the second, so "the patch changed behaviour" is a
    statement about the weights and not about a particular synthetic geometry.
    """
    residues = []
    for i in range(8):
        local_deficit = 1.0 if i < 3 else 0.0
        long_deficit = 1.0 if 3 <= i < 6 else 0.0
        residues.append(
            {
                "position": i,
                "aa": "A",
                "esm_surprisal": 0.0,
                "low_plddt": 0.0,
                "contact_violation": local_deficit,
                "long_range_contact_violation": long_deficit,
            }
        )
    return {"residues": residues}


def test_accepted_patch_changes_the_interpreter_output(tmp_path, seed, outcomes):
    """P3.5 verify: activating the field really moves the interpreter.

    Cross-checks P3.2's test: same state, two policies, different sites chosen.
    """
    state = _state_where_the_two_terms_disagree()

    outcome = client.propose_patch(
        seed,
        outcomes,
        transport=canned(json.dumps(VALID_REPRESENTATION_PATCH)),
        log_path=tmp_path / "c.jsonl",
    )
    assert outcome.accepted

    before = select_positions(seed, state)
    after = select_positions(outcome.policy, state)
    assert before == [0, 1, 2], before
    assert after == [3, 4, 5], after
    assert before != after, "the patch parsed but did not change behaviour"


def test_activated_feature_is_informative_on_a_real_fold(sequence, fold_of, stub_esm):
    """The field the patch turns on must carry signal, not be a constant."""
    state = grounder.ground(sequence, fold_of(sequence))
    long_range = [r["long_range_contact_violation"] for r in state["residues"]]
    assert len(set(long_range)) > 1


def test_accepts_a_valid_mechanism_patch(tmp_path, seed, outcomes):
    reply = json.dumps({"kind": "mechanism", "proposal": {"positions": 5}})
    outcome = client.propose_patch(
        seed, outcomes, transport=canned(reply), log_path=tmp_path / "c.jsonl"
    )
    assert outcome.accepted is True
    assert outcome.policy["proposal"]["positions"] == 5
    assert outcome.policy["proposal"]["max_total_edits"] == 3
    assert outcome.policy["position_score"] == seed["position_score"]


def test_markdown_fenced_reply_is_tolerated(tmp_path, seed, outcomes):
    reply = "```json\n" + json.dumps(VALID_REPRESENTATION_PATCH) + "\n```"
    outcome = client.propose_patch(
        seed, outcomes, transport=canned(reply), log_path=tmp_path / "c.jsonl"
    )
    assert outcome.accepted is True


def test_prose_wrapped_reply_is_tolerated(tmp_path, seed, outcomes):
    reply = (
        "Here is my patch:\n"
        + json.dumps({"kind": "mechanism", "proposal": {"positions": 4}})
        + "\nHope that helps."
    )
    outcome = client.propose_patch(
        seed, outcomes, transport=canned(reply), log_path=tmp_path / "c.jsonl"
    )
    assert outcome.accepted is True
    assert outcome.policy["proposal"]["positions"] == 4


# --------------------------------------------------------------------------- #
# prompt content and the scripted fallback
# --------------------------------------------------------------------------- #

def test_prompt_carries_the_schema_policy_outcomes_and_counterexample(seed, outcomes):
    counterexample = {
        "iteration": 0,
        "predicted_delta": 0.04,
        "hidden_delta": -0.02,
        "variant": "corrupt_01",
        "mutations": ["A34K"],
        "state_before": {"sequence_length": 56, "summary": {}, "top_residues": []},
    }
    prompt = client.build_prompt(seed, outcomes, counterexample)
    assert "ALLOWED_FEATURES" in prompt
    assert "long_range_contact_violation" in prompt
    assert "esm_surprisal: 0.6" in prompt
    assert "COUNTEREXAMPLE" in prompt
    assert "corrupt_01" in prompt
    assert "ALLOWED_PROPOSAL_FIELDS" in prompt
    for field in client.MECHANISM_FIELDS:
        assert field in prompt


def test_prompt_never_mentions_the_reference_structure(seed, outcomes):
    prompt = client.build_prompt(seed, outcomes, None)
    assert "native" not in prompt.lower()
    assert "tm_score" not in prompt.lower()
    assert "contact_recovery" not in prompt.lower()


def test_scripted_fallback_is_labelled_and_produces_a_valid_patch(
    tmp_path, seed, outcomes, monkeypatch
):
    monkeypatch.setenv("REFOLD_GEMMA_MODE", "mock")
    outcome = client.propose_patch(seed, outcomes, log_path=tmp_path / "c.jsonl")
    assert outcome.source == "mock", "a scripted patch must never look model-authored"
    assert outcome.accepted is True
    assert "long_range_contact_violation" in outcome.policy["position_score"]

    rows = [json.loads(l) for l in (tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows[0]["source"] == "mock"


def test_scripted_fallback_switches_to_a_mechanism_patch_once_activated(seed):
    activated = policy_mod.clone(seed)
    activated["position_score"] = dict(VALID_REPRESENTATION_PATCH["position_score"])
    raw = client._call_mock("", policy_mod.validate_policy(activated))
    patch = client.parse_patch(raw)
    assert patch["kind"] == "mechanism"
    assert client.apply_patch(activated, patch)["proposal"]["positions"] == 4

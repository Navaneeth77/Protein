"""P3.1 — policy DSL schema.

This file doubles as the evidence for constraint C2: a payload carrying anything
outside the DSL is rejected by schema validation, never executed.
"""

from __future__ import annotations

import pytest
import yaml

from src.agent import policy as policy_mod
from src.agent.grounder import SCORABLE_FEATURES


def test_seed_policy_validates_and_has_the_documented_defaults():
    policy = policy_mod.load_seed_policy()
    assert policy["position_score"] == {
        "esm_surprisal": 0.60,
        "low_plddt": 0.20,
        "contact_violation": 0.20,
    }
    assert policy["proposal"] == {
        "positions": 3,
        "substitutions_per_position": 4,
        "preserve_residue_class": True,
        "max_total_edits": 3,
    }
    assert abs(sum(policy["position_score"].values()) - 1.0) < 1e-9


def test_seed_policy_leaves_the_long_range_feature_unused():
    """The representation patch needs an unused-but-computed field to activate."""
    policy = policy_mod.load_seed_policy()
    assert "long_range_contact_violation" not in policy["position_score"]
    assert "long_range_contact_violation" in SCORABLE_FEATURES


def _seed() -> dict:
    return policy_mod.clone(policy_mod.load_seed_policy())


def test_rejects_weights_that_do_not_sum_to_one():
    broken = _seed()
    broken["position_score"]["esm_surprisal"] = 0.9
    with pytest.raises(policy_mod.PolicyValidationError, match="sum to 1.0"):
        policy_mod.validate_policy(broken)


def test_rejects_negative_max_total_edits():
    broken = _seed()
    broken["proposal"]["max_total_edits"] = -1
    with pytest.raises(policy_mod.PolicyValidationError):
        policy_mod.validate_policy(broken)


def test_rejects_an_injected_executable_key():
    """C2 evidence: a code-shaped payload is rejected, not run."""
    broken = _seed()
    broken["exec"] = "os.system('rm -rf /')"
    with pytest.raises(policy_mod.PolicyValidationError):
        policy_mod.validate_policy(broken)


def test_rejects_a_feature_name_absent_from_the_state_schema():
    broken = _seed()
    broken["position_score"] = {"hydrophobic_moment": 1.0}
    with pytest.raises(policy_mod.PolicyValidationError):
        policy_mod.validate_policy(broken)


@pytest.mark.parametrize(
    "payload",
    [
        {"position_score": {"esm_surprisal": 1.0}},                       # no proposal
        {"proposal": {"positions": 3}},                                   # no position_score
        "eval('1+1')",                                                    # not a mapping
        {"position_score": {}, "proposal": {"positions": 1,
                                            "substitutions_per_position": 1,
                                            "preserve_residue_class": True,
                                            "max_total_edits": 1}},       # empty weights
    ],
)
def test_rejects_structurally_invalid_payloads(payload):
    with pytest.raises(policy_mod.PolicyValidationError):
        policy_mod.validate_policy(payload)


def test_rejects_an_unknown_proposal_field():
    broken = _seed()
    broken["proposal"]["import os"] = True
    with pytest.raises(policy_mod.PolicyValidationError):
        policy_mod.validate_policy(broken)


def test_schema_propertynames_stay_in_sync_with_the_grounder():
    schema = yaml.safe_load(policy_mod.SCHEMA_PATH.read_text(encoding="utf-8"))
    allowed = schema["properties"]["position_score"]["propertyNames"]["enum"]
    assert sorted(allowed) == sorted(SCORABLE_FEATURES)


def test_activating_the_long_range_feature_is_a_valid_policy():
    patched = _seed()
    patched["position_score"] = {
        "esm_surprisal": 0.45,
        "low_plddt": 0.15,
        "contact_violation": 0.15,
        "long_range_contact_violation": 0.25,
    }
    assert policy_mod.validate_policy(patched) is patched

"""P3.1 — policy loading and validation.

A policy is plain data. Loading one runs a JSON-Schema check plus two rules the
schema cannot express (weights sum to 1.0; every scored feature is a field the
grounder actually computes). Invalid payloads are *rejected*, never coerced —
that is the evidence for constraint C2.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .grounder import SCORABLE_FEATURES

SCHEMA_PATH = Path(__file__).with_name("policy.schema.yaml")
SEED_PATH = Path(__file__).with_name("policy.yaml")

WEIGHT_SUM_TOLERANCE = 1e-6


class PolicyValidationError(ValueError):
    """A policy payload was rejected. Never recovered from by coercion."""


def load_schema() -> dict:
    return yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_policy(policy: dict) -> dict:
    """Return the policy unchanged, or raise PolicyValidationError."""
    import jsonschema

    if not isinstance(policy, dict):
        raise PolicyValidationError(f"policy must be a mapping, got {type(policy).__name__}")

    try:
        jsonschema.validate(instance=policy, schema=load_schema())
    except jsonschema.ValidationError as exc:
        path = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        raise PolicyValidationError(f"schema violation at {path}: {exc.message}") from exc

    weights = policy["position_score"]
    unknown = sorted(set(weights) - set(SCORABLE_FEATURES))
    if unknown:
        raise PolicyValidationError(
            f"position_score refers to feature(s) the grounder does not compute: "
            f"{unknown}; allowed: {sorted(SCORABLE_FEATURES)}"
        )

    total = sum(float(v) for v in weights.values())
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise PolicyValidationError(
            f"position_score weights must sum to 1.0, got {total:.6f}"
        )

    return policy


def load_policy(path: str | Path | None = None) -> dict:
    path = Path(path) if path else SEED_PATH
    policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    return validate_policy(policy)


def load_seed_policy() -> dict:
    return load_policy(SEED_PATH)


def dump_policy(policy: dict) -> str:
    """Canonical YAML rendering, used by the diff viewer (P4.5)."""
    ordered = {
        "position_score": {k: policy["position_score"][k] for k in sorted(policy["position_score"])},
        "proposal": {k: policy["proposal"][k] for k in sorted(policy["proposal"])},
    }
    return yaml.safe_dump(ordered, sort_keys=False, default_flow_style=False)


def clone(policy: dict) -> dict:
    return copy.deepcopy(policy)


def renormalise(weights: dict) -> dict:
    """Scale weights to sum to 1.0. Used only when applying a validated patch."""
    total = sum(float(v) for v in weights.values())
    if total <= 0:
        raise PolicyValidationError("cannot renormalise all-zero position_score weights")
    return {k: round(float(v) / total, 6) for k, v in weights.items()}

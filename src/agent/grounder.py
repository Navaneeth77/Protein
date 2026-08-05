"""P2.7 — grounding a sequence + its predicted structure into `state.json`.

The state is an explicit object/relation graph, not an embedding: every number
here has a name, and the policy DSL may only refer to names that appear in this
file. That is what makes "add a state feature" a safe operation for the outer
loop (constraint C2) — the outer loop can re-weight a field, never invent a new
computation.

`long_range_contact_degree` and `long_range_contact_violation` are computed from
the start even though the seed policy gives them zero weight. The outer loop's
representation patch activates an existing-but-unused field; if the field were
absent, that patch would have nothing real to attach to.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..constants import LONG_RANGE_SEPARATION, residue_class
from . import esm_score

SCHEMA_PATH = Path(__file__).with_name("state.schema.json")

# Per-residue fields a policy weight is allowed to reference. Anything outside
# this set is rejected by the policy schema validator.
SCORABLE_FEATURES = (
    "esm_surprisal",
    "low_plddt",
    "contact_violation",
    "long_range_contact_violation",
)


def _deficit(values: np.ndarray) -> np.ndarray:
    """Shortfall of each entry against the structure's own mean, scaled to 0-1.

    Used for `contact_violation`. Note what this is *not*: it is not a comparison
    against a reference structure's contact map — the agent has no access to one.
    It is an internal under-packing signal, i.e. "this residue makes fewer
    contacts than the rest of this fold does".
    """
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    if mean <= 0:
        return np.zeros_like(values)
    return np.clip((mean - values) / mean, 0.0, 1.0)


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def ground(
    sequence: str,
    predicted_structure,
    mutations: list | None = None,
    parent_structure=None,
) -> dict:
    """Build the grounded state dict.

    `predicted_structure` is a `FoldResult` from src/cache/fold_cache.py —
    always a *predicted* structure. `mutations` is the list of edits that
    produced this sequence, as (position, from_aa, to_aa) or dicts; when
    `parent_structure` is also given, contact gain/loss per edit is filled in.
    """
    surprisal = esm_score.residue_surprisal(sequence)
    plddt = np.asarray(predicted_structure.plddt, dtype=float)
    degree = np.asarray(predicted_structure.contact_degree, dtype=float)
    long_degree = np.asarray(predicted_structure.long_range_contact_degree, dtype=float)

    contact_violation = _deficit(degree)
    long_range_violation = _deficit(long_degree)

    residues = []
    for i, aa in enumerate(sequence):
        residues.append(
            {
                "position": int(i),
                "aa": aa,
                "residue_class": residue_class(aa),
                "esm_surprisal": round(float(surprisal[i]), 6),
                "plddt": round(float(plddt[i]), 6),
                "low_plddt": round(float(np.clip(1.0 - plddt[i], 0.0, 1.0)), 6),
                "contact_degree": int(degree[i]),
                "long_range_contact_degree": int(long_degree[i]),
                "contact_violation": round(float(contact_violation[i]), 6),
                "long_range_contact_violation": round(float(long_range_violation[i]), 6),
                "ss_region": predicted_structure.ss_labels[i],
            }
        )

    contacts = [
        {
            "i": int(i),
            "j": int(j),
            "sequence_separation": int(abs(j - i)),
            "long_range": bool(abs(j - i) > LONG_RANGE_SEPARATION),
        }
        for i, j in sorted(predicted_structure.contacts)
    ]

    state = {
        "sequence_length": int(len(sequence)),
        "summary": {
            "mean_plddt": round(float(plddt.mean()), 6),
            "mean_esm_surprisal": round(float(surprisal.mean()), 6),
            "pseudo_log_likelihood": round(esm_score.pseudo_log_likelihood(sequence), 6),
            "radius_of_gyration": round(float(predicted_structure.radius_of_gyration), 6),
            "clashes": int(predicted_structure.clashes),
            "n_contacts": int(len(contacts)),
            "plddt_raw_range": [
                round(float(predicted_structure.plddt_raw_range[0]), 4),
                round(float(predicted_structure.plddt_raw_range[1]), 4),
            ],
        },
        "residues": residues,
        "relations": {
            "contacts": contacts,
            "helices": [[int(a), int(b)] for a, b in predicted_structure.helices],
            "strands": [[int(a), int(b)] for a, b in predicted_structure.strands],
            "mutation_effects": _mutation_effects(
                sequence, mutations, predicted_structure, parent_structure
            ),
        },
    }
    return state


def _mutation_effects(sequence, mutations, predicted_structure, parent_structure):
    if not mutations:
        return []

    child_contacts = predicted_structure.contacts
    parent_contacts = parent_structure.contacts if parent_structure is not None else None

    effects = []
    for mut in mutations:
        if isinstance(mut, dict):
            pos = int(mut["position"])
            to_aa = mut.get("to", sequence[pos])
            from_aa = mut.get("from")
        else:
            pos, from_aa, to_aa = int(mut[0]), mut[1], mut[2]

        lost = gained = 0
        if parent_contacts is not None:
            before = {p for p in parent_contacts if pos in p}
            after = {p for p in child_contacts if pos in p}
            lost = len(before - after)
            gained = len(after - before)

        effect = {
            "position": pos,
            "to": to_aa,
            "breaks_contact": bool(lost > 0),
            "contacts_lost": int(lost),
            "contacts_gained": int(gained),
        }
        if from_aa:
            effect["from"] = from_aa
        effects.append(effect)
    return effects


def validate(state: dict) -> None:
    """Raise jsonschema.ValidationError if the state is malformed."""
    import jsonschema

    jsonschema.validate(instance=state, schema=load_schema())


def to_json(state: dict) -> str:
    """Serialise, failing loudly if any numpy scalar leaked into the state."""
    return json.dumps(state, indent=2, allow_nan=False)

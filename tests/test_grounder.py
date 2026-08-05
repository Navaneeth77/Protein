"""P2.7 — the grounded state graph."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.agent import grounder


def test_state_serialises_and_validates(sequence, fold_of, stub_esm):
    state = grounder.ground(sequence, fold_of(sequence))
    text = grounder.to_json(state)          # raises on numpy scalars / NaN
    reparsed = json.loads(text)
    grounder.validate(reparsed)             # jsonschema
    assert reparsed == state


def test_no_numpy_types_leak_into_the_state(sequence, fold_of, stub_esm):
    state = grounder.ground(sequence, fold_of(sequence))

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                assert isinstance(k, str), k
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        else:
            assert not isinstance(node, np.generic), f"numpy scalar leaked: {node!r}"
            assert isinstance(node, (str, int, float, bool)) or node is None

    walk(state)


def test_one_residue_entry_per_position(sequence, fold_of, stub_esm):
    state = grounder.ground(sequence, fold_of(sequence))
    assert len(state["residues"]) == len(sequence)
    assert state["sequence_length"] == len(sequence)
    assert [r["position"] for r in state["residues"]] == list(range(len(sequence)))
    assert "".join(r["aa"] for r in state["residues"]) == sequence


def test_long_range_contact_degree_is_present_and_non_constant(fold_of, stub_esm):
    """P2.7 verify: the field the representation patch will activate is real."""
    # A long helix has genuinely varying long-range contact counts (ends differ
    # from the middle), which is what makes the field informative.
    seq = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    state = grounder.ground(seq, fold_of(seq))
    values = [r["long_range_contact_degree"] for r in state["residues"]]
    assert all(isinstance(v, int) for v in values)
    assert any(v > 0 for v in values), "no long-range contacts at all"
    assert len(set(values)) > 1, "long_range_contact_degree is constant"


def test_every_scorable_feature_exists_on_every_residue(sequence, fold_of, stub_esm):
    state = grounder.ground(sequence, fold_of(sequence))
    for residue in state["residues"]:
        for feature in grounder.SCORABLE_FEATURES:
            assert feature in residue, feature
            assert isinstance(residue[feature], (int, float))


def test_contact_violation_is_an_internal_deficit_not_a_reference_comparison(
    sequence, fold_of, stub_esm
):
    state = grounder.ground(sequence, fold_of(sequence))
    values = np.array([r["contact_violation"] for r in state["residues"]])
    degrees = np.array([r["contact_degree"] for r in state["residues"]])
    assert values.min() >= 0.0 and values.max() <= 1.0
    # Residues at or above the structure's mean degree have zero violation.
    mean_degree = degrees.mean()
    for v, d in zip(values, degrees):
        if d >= mean_degree:
            assert v == 0.0


def test_mutation_effects_report_contact_loss(fold_of, stub_esm):
    parent_seq = "MTYKLILNGKTLKGETTTEAVDAATAEK"
    child_seq = parent_seq[:5] + "G" + parent_seq[6:]
    parent = fold_of(parent_seq)
    child = fold_of(child_seq)
    state = grounder.ground(
        child_seq,
        child,
        mutations=[{"position": 5, "from": parent_seq[5], "to": "G"}],
        parent_structure=parent,
    )
    effects = state["relations"]["mutation_effects"]
    assert len(effects) == 1
    assert effects[0]["position"] == 5
    assert effects[0]["to"] == "G"
    assert isinstance(effects[0]["breaks_contact"], bool)
    # Glycine loses its CB, so its contact set is computed from CA and shifts.
    assert effects[0]["contacts_lost"] >= 0


def test_ss_regions_are_valid_labels(sequence, fold_of, stub_esm):
    state = grounder.ground(sequence, fold_of(sequence))
    assert {r["ss_region"] for r in state["residues"]} <= {"helix", "strand", "coil"}
    # The synthetic structure is an ideal helix, so it should read as helical.
    assert any(r["ss_region"] == "helix" for r in state["residues"])
    assert state["relations"]["helices"], "no helix span detected on an ideal helix"


def test_schema_rejects_a_state_with_an_invented_residue_field(
    sequence, fold_of, stub_esm
):
    state = grounder.ground(sequence, fold_of(sequence))
    state["residues"][0]["hydrophobic_moment"] = 1.0
    with pytest.raises(Exception):
        grounder.validate(state)

"""The MVP flow: one guard test for the demo path.

Deliberately small. It stubs the fold backend and Gemma so it runs in a second,
and checks the things the demo actually depends on: the loop completes, Gemma's
patch is applied, and the interpreter's behaviour is reported honestly.
"""

from __future__ import annotations

import json

import pytest

from src import mvp
from src.cache import fold_cache
from tests.conftest import make_pdb

LONG_RANGE_PATCH = {
    "kind": "representation",
    "rationale": "Long-range contact loss is unscored; activate it.",
    "position_score": {
        "esm_surprisal": 0.45,
        "low_plddt": 0.15,
        "contact_violation": 0.15,
        "long_range_contact_violation": 0.25,
    },
}
WIDEN_PATCH = {
    "kind": "mechanism",
    "rationale": "Widen the site search.",
    "proposal": {"positions": 5},
}


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """Fold from a synthetic PDB, and keep results out of the real tree."""
    monkeypatch.setattr(fold_cache, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(mvp, "RESULT_PATH", tmp_path / "mvp_result.json")
    monkeypatch.setattr(
        fold_cache,
        "fold",
        lambda seq: fold_cache.structure_features(seq, make_pdb(seq), from_cache=True),
    )


def _needs_data() -> None:
    from src.paths import corruption_dir, protein_dir

    if not (protein_dir("1pgb") / "native_seq.fasta").exists():
        pytest.skip("run scripts/fetch_protein.py first")
    if not (corruption_dir("1pgb") / f"{mvp.DEFAULT_VARIANT}.fasta").exists():
        pytest.skip("run scripts/make_corruptions.py first")


@pytest.mark.models
def test_mvp_completes_and_applies_gemmas_patch(stubbed):
    _needs_data()
    result = mvp.run(gemma_transport=lambda prompt: json.dumps(WIDEN_PATCH))

    assert result.gemma["accepted"] is True
    assert result.gemma["kind"] == "mechanism"
    # The patch must actually reach the interpreter.
    assert result.patched["policy"]["proposal"]["positions"] == 5
    assert result.baseline["policy"]["proposal"]["positions"] == 3
    assert len(result.patched["selected_positions"]) == 5
    assert len(result.baseline["selected_positions"]) == 3
    assert result.baseline["candidates"], "baseline produced no candidates"
    assert result.patched["candidates"], "patched policy produced no candidates"
    assert result.timeline


@pytest.mark.models
def test_mvp_survives_a_rejected_patch(stubbed):
    """Gemma returning prose must not break the demo."""
    _needs_data()
    result = mvp.run(gemma_transport=lambda prompt: "I would weight contacts more.")

    assert result.gemma["accepted"] is False
    assert "no JSON object" in result.gemma["error"]
    # Falls back to the incumbent policy and still finishes.
    assert result.patched["policy"] == result.baseline["policy"]
    assert result.patched["candidates"]


@pytest.mark.models
def test_result_is_json_serialisable_for_the_ui(stubbed):
    _needs_data()
    result = mvp.run(gemma_transport=lambda prompt: json.dumps(LONG_RANGE_PATCH))
    payload = json.loads(result.to_json())
    for key in ("reference_sequence", "corrupted_sequence", "corruption_sites",
                "baseline", "patched", "gemma", "timeline", "origin_metrics"):
        assert key in payload
    assert len(payload["corrupted_sequence"]) == len(payload["reference_sequence"])
    assert payload["corruption_sites"], "no corrupted sites recorded"


def test_changes_behaviour_detects_a_real_no_op(sequence, fold_of, stub_esm):
    """The retry trigger: same sites and same candidates means nothing changed."""
    from src.agent import grounder, policy as policy_mod

    state = grounder.ground(sequence, fold_of(sequence))
    seed = policy_mod.load_seed_policy()

    assert not mvp._changes_behaviour(seed, policy_mod.clone(seed), state, sequence)

    widened = policy_mod.clone(seed)
    widened["proposal"]["positions"] = 5
    assert mvp._changes_behaviour(
        seed, policy_mod.validate_policy(widened), state, sequence
    )

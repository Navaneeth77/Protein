"""P2.6 — the hidden evaluator: redaction, weighting, and C3 leakage checks."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from src import evaluator as ev
from src.cache import fold_cache
from tests.conftest import make_pdb

REFERENCE_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEK"


@pytest.fixture
def fake_reference(tmp_path, monkeypatch):
    """A self-contained reference protein so no real data tree is touched."""
    directory = tmp_path / "proteins" / "toy"
    directory.mkdir(parents=True)
    (directory / "native.pdb").write_text(make_pdb(REFERENCE_SEQ), encoding="utf-8")
    (directory / "native_seq.fasta").write_text(
        f">toy\n{REFERENCE_SEQ}\n", encoding="utf-8"
    )

    monkeypatch.setattr(ev, "protein_dir", lambda name: directory)
    monkeypatch.setattr(ev, "evaluator_sidecar_dir", lambda name: tmp_path / "side")
    monkeypatch.setattr(ev, "corruption_dir", lambda name: tmp_path / "corr")
    (tmp_path / "corr").mkdir()
    ev._load_reference.cache_clear()
    ev._default_evaluator.cache_clear()
    return directory


@pytest.fixture
def stubbed_pipeline(monkeypatch, fake_reference):
    """Fold + ESM stubs: the evaluator's arithmetic is what is under test."""
    monkeypatch.setattr(
        fold_cache,
        "fold",
        lambda seq: fold_cache.structure_features(seq, make_pdb(seq), from_cache=True),
    )
    monkeypatch.setattr(
        ev.esm_score, "pseudo_log_likelihood", lambda seq: -1.5 * len(seq)
    )
    monkeypatch.setattr(
        ev, "esm_calibration", lambda name: {"mean": -1.5, "std": 0.5, "n_batch": 4}
    )
    return ev.HiddenEvaluator(protein="toy", origin_sequence=REFERENCE_SEQ)


# --------------------------------------------------------------------------- #
# (b) hand-computed toy case
# --------------------------------------------------------------------------- #

def test_combine_matches_the_hand_computed_weighting():
    """P2.6 verify (b): tm=1, everything else 0 -> exactly 0.55."""
    assert ev.combine(1.0, 0.0, 0.0, 0.0, 0.0) == pytest.approx(0.55)
    assert ev.combine(0.0, 1.0, 0.0, 0.0, 0.0) == pytest.approx(0.20)
    assert ev.combine(0.0, 0.0, 1.0, 0.0, 0.0) == pytest.approx(0.15)
    assert ev.combine(0.0, 0.0, 0.0, 1.0, 0.0) == pytest.approx(0.10)
    assert ev.combine(0.0, 0.0, 0.0, 0.0, 1.0) == pytest.approx(-0.05)
    assert ev.combine(1.0, 1.0, 1.0, 1.0, 0.0) == pytest.approx(1.0)


def test_positive_weights_sum_to_one():
    positive = {k: v for k, v in ev.WEIGHTS.items() if v > 0}
    assert sum(positive.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# (a) redaction
# --------------------------------------------------------------------------- #

def test_no_ground_truth_key_survives_redaction(stubbed_pipeline):
    """P2.6 verify (a): no key mentions tm, reference identity, or contacts."""
    public = stubbed_pipeline.evaluate(REFERENCE_SEQ, reveal=False)
    assert not [
        k
        for k in public
        if "tm" in k.lower() or "native" in k.lower() or "contact" in k.lower()
    ]
    assert set(public) == set(ev.public_keys())
    assert "hidden_score" in public


def test_reveal_exposes_the_decomposition(stubbed_pipeline):
    full = stubbed_pipeline.evaluate(REFERENCE_SEQ, reveal=True)
    for key in ev.HIDDEN_KEYS:
        assert key in full
    assert full["tm_score"] == pytest.approx(1.0, abs=1e-6)
    assert full["contact_recovery"] == pytest.approx(1.0)
    assert full["sequence_recovery"] == pytest.approx(1.0)


def test_hidden_score_is_identical_with_and_without_reveal(stubbed_pipeline):
    public = stubbed_pipeline.evaluate(REFERENCE_SEQ, reveal=False)
    full = stubbed_pipeline.evaluate(REFERENCE_SEQ, reveal=True)
    assert public["hidden_score"] == full["hidden_score"]


# --------------------------------------------------------------------------- #
# behaviour
# --------------------------------------------------------------------------- #

def test_a_corrupted_sequence_scores_below_the_reference(stubbed_pipeline):
    corrupted = REFERENCE_SEQ[:5] + "P" + REFERENCE_SEQ[6:]
    ref = stubbed_pipeline.evaluate(REFERENCE_SEQ, reveal=True)
    bad = stubbed_pipeline.evaluate(corrupted, reveal=True)
    assert bad["edit_count"] == 1
    assert bad["hidden_score"] < ref["hidden_score"]


def test_edit_fraction_uses_the_pinned_denominator(stubbed_pipeline):
    corrupted = "P" + REFERENCE_SEQ[1:]
    result = stubbed_pipeline.evaluate(corrupted, reveal=True)
    assert result["edit_count"] == 1
    assert result["edit_fraction"] == pytest.approx(1 / ev.EDIT_FRACTION_DENOMINATOR)


def test_length_mismatch_is_rejected(stubbed_pipeline):
    with pytest.raises(ValueError, match="length"):
        stubbed_pipeline.evaluate(REFERENCE_SEQ + "A")


def test_rescale_esm_is_bounded_and_monotone():
    calib = {"mean": -2.0, "std": 0.5}
    low = ev.rescale_esm(-4.0 * 50, 50, calib)
    mid = ev.rescale_esm(-2.0 * 50, 50, calib)
    high = ev.rescale_esm(-1.0 * 50, 50, calib)
    assert 0.0 < low < mid < high < 1.0
    assert mid == pytest.approx(0.5)


def test_rescale_esm_is_length_robust():
    """A longer sequence of equal per-residue quality must not score differently."""
    calib = {"mean": -2.0, "std": 0.5}
    short = ev.rescale_esm(-2.0 * 20, 20, calib)
    long = ev.rescale_esm(-2.0 * 200, 200, calib)
    assert short == pytest.approx(long)


def test_tm_normalisation_choice_is_pinned_in_the_source():
    """The tm_norm_chain1/2 decision must stay documented where it is made."""
    src = Path("src/evaluator.py").read_text(encoding="utf-8")
    assert "tm_norm_chain2" in src
    assert "REFERENCE" in src


# --------------------------------------------------------------------------- #
# (c) C3 leakage checks
# --------------------------------------------------------------------------- #

def _agent_files():
    return sorted(Path("src/agent").rglob("*.py"))


def test_reveal_true_is_never_called_from_agent_code():
    """P2.6 verify (c)."""
    for path in _agent_files():
        assert "reveal=True" not in path.read_text(encoding="utf-8"), path


def test_agent_code_never_mentions_the_reference_structure():
    """C3: grep for 'native' and 'SEQRES' under src/agent/ returns nothing."""
    for path in _agent_files():
        text = path.read_text(encoding="utf-8")
        assert "native" not in text.lower(), f"{path} mentions the reference structure"
        assert "SEQRES" not in text, path


def test_only_the_evaluator_names_the_reference_structure_path():
    """C3: exactly one file under src/ contains native_pdb_path."""
    hits = [
        p for p in Path("src").rglob("*.py")
        if "native_pdb_path" in p.read_text(encoding="utf-8")
    ]
    assert [p.as_posix() for p in hits] == ["src/evaluator.py"]


def test_agent_code_never_imports_the_evaluator():
    pattern = re.compile(r"(from\s+\.\.?\s*evaluator|import\s+.*\bevaluator\b)")
    for path in _agent_files():
        assert not pattern.search(path.read_text(encoding="utf-8")), path

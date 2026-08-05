"""P2.5 — fold cache behaviour and structural feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from src.cache import fold_cache
from tests.conftest import make_pdb


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fold_cache, "CACHE", tmp_path)
    fold_cache.reset_stats()
    monkeypatch.delenv(fold_cache.OFFLINE_ENV, raising=False)
    return tmp_path


def test_second_call_is_a_cache_hit(isolated_cache, monkeypatch, sequence):
    """P2.5 verify: cache_misses == 1 and cache_hits == 1 across two calls."""
    calls = {"n": 0}

    def fake_predict(seq):
        calls["n"] += 1
        fold_cache.STATS["model_calls"] += 1
        return make_pdb(seq)

    monkeypatch.setattr(fold_cache, "_predict_pdb", fake_predict)

    first = fold_cache.fold(sequence)
    second = fold_cache.fold(sequence)

    assert fold_cache.STATS["cache_misses"] == 1
    assert fold_cache.STATS["cache_hits"] == 1
    assert calls["n"] == 1, "the model was re-run on a cached sequence"
    assert first.from_cache is False
    assert second.from_cache is True
    assert first.pdb_text == second.pdb_text
    assert fold_cache.cache_path(sequence).exists()


def test_offline_mode_refuses_an_uncached_sequence(isolated_cache, monkeypatch, sequence):
    """C4: with live inference disabled, a miss is a loud error."""
    monkeypatch.setenv(fold_cache.OFFLINE_ENV, "1")

    def explode(seq):
        raise AssertionError("the model must not be reached in offline mode")

    monkeypatch.setattr(fold_cache, "_predict_pdb", explode)
    with pytest.raises(fold_cache.FoldUnavailable):
        fold_cache.fold(sequence)
    assert fold_cache.STATS["model_calls"] == 0


def test_offline_mode_still_serves_cached_sequences(isolated_cache, monkeypatch, sequence):
    monkeypatch.setattr(fold_cache, "_predict_pdb", lambda seq: make_pdb(seq))
    fold_cache.fold(sequence)
    monkeypatch.setenv(fold_cache.OFFLINE_ENV, "1")
    fold_cache.reset_stats()
    result = fold_cache.fold(sequence)
    assert result.from_cache is True
    assert fold_cache.STATS["cache_misses"] == 0
    assert fold_cache.STATS["model_calls"] == 0


def test_plddt_scale_is_detected_not_assumed(sequence):
    """The 0-100 vs 0-1 question is answered from the data, not hardcoded."""
    hundred = fold_cache.structure_features(
        sequence, make_pdb(sequence, plddt=np.full(len(sequence), 90.0))
    )
    assert hundred.plddt_raw_range == (90.0, 90.0)
    assert abs(hundred.mean_plddt - 0.90) < 1e-9

    unit = fold_cache.structure_features(
        sequence, make_pdb(sequence, plddt=np.full(len(sequence), 0.9))
    )
    assert abs(unit.mean_plddt - 0.90) < 1e-9

    assert 0.0 <= hundred.plddt.min() <= hundred.plddt.max() <= 1.0


def test_features_are_extracted_for_every_residue(sequence):
    fold = fold_cache.structure_features(sequence, make_pdb(sequence))
    n = len(sequence)
    assert len(fold.plddt) == n
    assert len(fold.contact_degree) == n
    assert len(fold.long_range_contact_degree) == n
    assert len(fold.ss_labels) == n
    assert fold.radius_of_gyration > 0
    assert fold.clashes >= 0
    assert fold.ca_coords.shape == (n, 3)
    assert fold.cb_coords.shape == (n, 3)


def test_length_mismatch_is_rejected(sequence):
    with pytest.raises(ValueError, match="residues"):
        fold_cache.structure_features(sequence + "A", make_pdb(sequence))


def test_hash_is_stable_and_sequence_specific(sequence):
    assert fold_cache.sequence_hash(sequence) == fold_cache.sequence_hash(sequence)
    other = "A" + sequence[1:]
    assert fold_cache.sequence_hash(sequence) != fold_cache.sequence_hash(other)

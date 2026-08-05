"""P2.1 + P2.2 — ESM-2 surprisal scoring, against the real checkpoint.

These are the model-tier tests: they load facebook/esm2_t12_35M_UR50D and are
skipped with a clear reason when torch/transformers are unavailable. The first
run downloads the checkpoint (~130 MB) and warms data/cache/esm_score/.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agent import esm_score
from src.paths import corruption_dir, protein_dir, read_fasta

pytestmark = pytest.mark.models

PROTEIN = "1pgb"


@pytest.fixture(scope="module")
def reference() -> str:
    fasta = protein_dir(PROTEIN) / "native_seq.fasta"
    if not fasta.exists():
        pytest.skip("run scripts/fetch_protein.py first")
    return read_fasta(fasta)[1]


@pytest.fixture(scope="module")
def corrupted() -> dict:
    files = sorted(corruption_dir(PROTEIN).glob("corrupt_*.fasta"))
    if not files:
        pytest.skip("run scripts/make_corruptions.py first")
    return {p.stem: read_fasta(p)[1] for p in files}


# --------------------------------------------------------------------------- #
# P2.1
# --------------------------------------------------------------------------- #

def test_native_scores_more_natural_than_corrupted(reference, corrupted):
    """P2.1 verify: mean surprisal is lower on the reference than on every variant."""
    reference_mean = esm_score.mean_surprisal(reference)
    print(f"\nreference mean surprisal: {reference_mean:.4f}")
    failures = []
    for name, sequence in corrupted.items():
        variant_mean = esm_score.mean_surprisal(sequence)
        print(f"  {name}: {variant_mean:.4f}")
        if not variant_mean > reference_mean:
            failures.append((name, variant_mean))
    assert not failures, f"variants not less natural than the reference: {failures}"


def test_pseudo_log_likelihood_ranks_the_reference_highest(reference, corrupted):
    reference_pll = esm_score.pseudo_log_likelihood(reference)
    for name, sequence in corrupted.items():
        assert esm_score.pseudo_log_likelihood(sequence) < reference_pll, name


def test_masked_marginal_matrix_shape_and_normalisation(reference):
    matrix = esm_score.masked_marginal_matrix(reference)
    assert matrix.shape == (len(reference), 20)
    assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6)
    assert matrix.min() >= 0.0


def test_surprisal_is_non_negative_and_per_position(reference):
    surprisal = esm_score.residue_surprisal(reference)
    assert surprisal.shape == (len(reference),)
    assert (surprisal >= 0).all()


def test_scores_are_cached_on_disk(reference):
    """Second call must not re-run the model — this is what keeps the demo fast."""
    esm_score.masked_marginal_matrix(reference)     # ensure cached
    esm_score.reset_stats()
    esm_score.masked_marginal_matrix(reference)
    assert esm_score.STATS["cache_hits"] == 1
    assert esm_score.STATS["forward_batches"] == 0


def test_offline_mode_refuses_an_uncached_sequence(monkeypatch, reference):
    novel = "W" * len(reference)
    monkeypatch.setenv(esm_score.OFFLINE_ENV, "1")
    esm_score._log_probs_of_sequence.cache_clear()
    if esm_score._cache_path(novel).exists():
        pytest.skip("this sequence happens to be cached already")
    with pytest.raises(esm_score.ScoringUnavailable):
        esm_score.masked_marginal_matrix(novel)


# --------------------------------------------------------------------------- #
# P2.2
# --------------------------------------------------------------------------- #

def test_rank_suspicious_positions_finds_a_planted_implausible_residue(reference):
    """P2.2 verify: a deliberately implausible residue at a known index ranks top-3.

    Proline in the middle of the beta sheet of the B1 domain is the standard
    fold-breaking substitution, and it is the residue ESM-2 is most confident
    does not belong there.
    """
    index = 25
    planted = reference[:index] + "P" + reference[index + 1 :]
    assert planted != reference

    top3 = esm_score.rank_suspicious_positions(planted, top_k=3)
    print(f"\nplanted P at {index}; top-3 suspicious: {top3}")
    assert index in top3


def test_ranking_is_deterministic_and_bounded(reference):
    first = esm_score.rank_suspicious_positions(reference, top_k=5)
    second = esm_score.rank_suspicious_positions(reference, top_k=5)
    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5
    assert all(0 <= i < len(reference) for i in first)


def test_ranking_orders_by_descending_surprisal(reference):
    surprisal = esm_score.residue_surprisal(reference)
    ranked = esm_score.rank_suspicious_positions(reference, top_k=len(reference))
    values = [surprisal[i] for i in ranked]
    assert values == sorted(values, reverse=True)


# --------------------------------------------------------------------------- #
# P2.3 substitution proposals
# --------------------------------------------------------------------------- #

def test_substitution_ranking_excludes_the_incumbent_residue(reference):
    for position in (0, 10, 25, len(reference) - 1):
        ranked = esm_score.substitution_ranking(reference, position, 4)
        assert len(ranked) == 4
        assert all(aa != reference[position] for aa, _ in ranked)
        probabilities = [p for _, p in ranked]
        assert probabilities == sorted(probabilities, reverse=True)


def test_substitution_ranking_is_deterministic(reference):
    a = esm_score.substitution_ranking(reference, 12, 4)
    b = esm_score.substitution_ranking(reference, 12, 4)
    assert a == b

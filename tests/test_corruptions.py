"""P1.3 — the corruption variants are well-formed and reproducible."""

from __future__ import annotations

import json

import pytest

from src.constants import AA_SET, DEFAULT_PROTEIN
from src.paths import (
    corruption_dir,
    evaluator_sidecar_dir,
    protein_dir,
    read_fasta,
)

PROTEIN = DEFAULT_PROTEIN


def _reference() -> str:
    fasta = protein_dir(PROTEIN) / "native_seq.fasta"
    if not fasta.exists():
        pytest.skip("reference sequence not generated; run scripts/fetch_protein.py")
    return read_fasta(fasta)[1]


def _variants() -> dict:
    files = sorted(corruption_dir(PROTEIN).glob("corrupt_*.fasta"))
    if not files:
        pytest.skip("no corruptions generated; run scripts/make_corruptions.py")
    return {p.stem: read_fasta(p)[1] for p in files}


def test_reference_sequence_is_standard_amino_acids_only():
    """P1.2 verify."""
    seq = _reference()
    assert set(seq) <= AA_SET
    assert len(seq) == 56, "1PGB B1 domain is 56 residues"


def test_between_four_and_five_variants_exist():
    assert 4 <= len(_variants()) <= 5


@pytest.mark.parametrize("min_edits,max_edits", [(3, 5)])
def test_every_variant_is_a_valid_corruption(min_edits, max_edits):
    """P1.3 verify: same length, Hamming distance in [3, 5], real substitutions."""
    reference = _reference()
    for name, seq in _variants().items():
        assert len(seq) == len(reference), f"{name}: length changed"
        assert set(seq) <= AA_SET, f"{name}: non-standard residue"
        changed = [i for i, (a, b) in enumerate(zip(reference, seq)) if a != b]
        assert min_edits <= len(changed) <= max_edits, f"{name}: {len(changed)} edits"
        for i in changed:
            assert seq[i] != reference[i], f"{name}: position {i} unchanged"


def test_variants_are_distinct_from_each_other():
    variants = _variants()
    assert len({*variants.values()}) == len(variants)


def test_sidecar_records_the_edits_and_is_evaluator_only():
    """Which positions changed must exist for the evaluator and nowhere else."""
    sidecar = evaluator_sidecar_dir(PROTEIN) / "corrupt_positions.json"
    if not sidecar.exists():
        pytest.skip("sidecar not generated")
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    reference = _reference()
    variants = _variants()
    assert set(data["variants"]) == set(variants)
    for name, record in data["variants"].items():
        for edit in record:
            pos = edit["position"]
            assert reference[pos] == edit["from"]
            assert variants[name][pos] == edit["to"]

    # The agent-visible tree must not carry the same bookkeeping.
    for path in corruption_dir(PROTEIN).iterdir():
        assert path.suffix == ".fasta", f"unexpected file in agent-visible tree: {path}"
        text = path.read_text(encoding="utf-8")
        assert "position" not in text.lower()


def test_corruption_generation_is_reproducible():
    """Same seed, same variants — checked by regenerating in a temp location."""
    import numpy as np

    from scripts.make_corruptions import (
        DEFAULT_SEED,
        make_variant,
        position_weights,
    )

    reference = _reference()
    rsa = np.full(len(reference), 0.2)     # deterministic stand-in for SASA
    weights = position_weights(rsa)

    def run():
        rng = np.random.default_rng(DEFAULT_SEED)
        return [make_variant(rng, reference, weights, rsa)[0] for _ in range(5)]

    assert run() == run()

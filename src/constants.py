"""Shared constants: amino-acid alphabet, residue classes, geometry thresholds.

Nothing here touches reference structures, so it is safe to import from both the
agent path and the evaluator.
"""

from __future__ import annotations

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = frozenset(AA_ALPHABET)

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

# Three-way partition of the 20 standard residues, used by the DSL's
# `preserve_residue_class` switch. Partition is total and disjoint.
HYDROPHOBIC = frozenset("AVLIMFWP")
POLAR = frozenset("STNQCYG")
CHARGED = frozenset("DEKRH")

RESIDUE_CLASS = {}
for _aa in HYDROPHOBIC:
    RESIDUE_CLASS[_aa] = "hydrophobic"
for _aa in POLAR:
    RESIDUE_CLASS[_aa] = "polar"
for _aa in CHARGED:
    RESIDUE_CLASS[_aa] = "charged"

assert set(RESIDUE_CLASS) == AA_SET, "residue-class partition must cover all 20 AAs"

# Geometry thresholds (see src/geometry.py).
CONTACT_CUTOFF_ANGSTROM = 8.0
LONG_RANGE_SEPARATION = 4      # |i - j| > 4 counts as long-range
CLASH_CUTOFF_ANGSTROM = 2.0
CLASH_MIN_SEQ_SEPARATION = 2   # skip bonded / near-bonded pairs

# Model ids. esmfold_v1 is the structure module; the small esm2 is the scorer.
ESMFOLD_MODEL = "facebook/esmfold_v1"
ESM2_SCORE_MODEL = "facebook/esm2_t12_35M_UR50D"

DEFAULT_PROTEIN = "1pgb"


def residue_class(aa: str) -> str:
    """Class label for a one-letter amino-acid code."""
    try:
        return RESIDUE_CLASS[aa]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"not a standard amino acid: {aa!r}") from exc

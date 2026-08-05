"""Shared fixtures.

Two tiers of test run here:

* fast tests, which stub the ESM calls so the logic under test (policy schema,
  interpreter, median rule, patch validation, redaction) runs with no model;
* model tests, marked `@pytest.mark.models`, which need torch + transformers and
  are skipped with a clear reason when those are missing.

Structures for the fast tier come from an ideal-helix generator rather than a
recorded prediction, so the fast tier has no data dependency at all.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import AA_ALPHABET  # noqa: E402


def models_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def pytest_configure(config):
    config.addinivalue_line("markers", "models: needs torch + transformers")
    config.addinivalue_line("markers", "esmfold: needs the ESMFold checkpoint")


def pytest_collection_modifyitems(config, items):
    if models_available():
        return
    skip = pytest.mark.skip(reason="torch/transformers not installed")
    for item in items:
        if "models" in item.keywords or "esmfold" in item.keywords:
            item.add_marker(skip)


# --------------------------------------------------------------------------- #
# synthetic structures
# --------------------------------------------------------------------------- #

def helix_coords(n: int, radius: float = 2.3, rise: float = 1.5, turn_deg: float = 100.0):
    """Ideal alpha-helix CA coordinates and outward CB offsets."""
    ca, cb = [], []
    for i in range(n):
        theta = math.radians(turn_deg) * i
        c = np.array([radius * math.cos(theta), radius * math.sin(theta), rise * i])
        radial = np.array([math.cos(theta), math.sin(theta), 0.0])
        ca.append(c)
        cb.append(c + 1.5 * radial)
    return np.array(ca), np.array(cb)


def hairpin_coords(n: int, gap: float = 11.0):
    """Two antiparallel helices packed side by side.

    A single helix has no long-range contacts at all (i to i+5 is already past
    8 A), which would make every contact-based feature identically zero. The
    hairpin gives the fast tier real tertiary contacts to work with.
    """
    m = n // 2
    ca1, cb1 = helix_coords(m)
    ca2, cb2 = helix_coords(n - m)
    flip = np.array([1.0, 1.0, -1.0])
    offset = np.array([gap, 0.0, float(ca1[-1, 2])])
    return (
        np.vstack([ca1, ca2 * flip + offset]),
        np.vstack([cb1, cb2 * flip + offset]),
    )


ONE_TO_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def make_pdb(sequence: str, plddt=None, coords=None) -> str:
    """A minimal but well-formed PDB with pLDDT in the B-factor column (0-100)."""
    n = len(sequence)
    if coords is None:
        ca, cb = hairpin_coords(n)
    else:
        ca, cb = coords
    if plddt is None:
        plddt = np.full(n, 85.0)
    plddt = np.asarray(plddt, dtype=float)

    lines, serial = [], 1
    for i, aa in enumerate(sequence):
        resname = ONE_TO_THREE[aa]
        atoms = [("N", ca[i] + np.array([-1.2, 0.0, -0.4])),
                 ("CA", ca[i]),
                 ("C", ca[i] + np.array([1.2, 0.0, 0.4])),
                 ("O", ca[i] + np.array([1.8, 0.8, 0.6]))]
        if aa != "G":
            atoms.append(("CB", cb[i]))
        for name, xyz in atoms:
            # Fixed-column PDB ATOM record: serial 7-11, name 13-16, altLoc 17,
            # resName 18-20, chainID 22, resSeq 23-26, xyz 31-54, B-factor 61-66.
            lines.append(
                "ATOM  "
                f"{serial:5d}"
                " "
                f"{name:<4s}"
                " "
                f"{resname:>3s}"
                " A"
                f"{i + 1:4d}"
                "    "
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                "  1.00"
                f"{plddt[i]:6.2f}"
            )
            serial += 1
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


@pytest.fixture
def sequence() -> str:
    # 28 residues, deterministic, all standard letters.
    return "MTYKLILNGKTLKGETTTEAVDAATAEK"


@pytest.fixture
def fold_of():
    """Build a FoldResult for any sequence, no model involved."""
    from src.cache import fold_cache

    def _build(seq: str, plddt=None):
        return fold_cache.structure_features(seq, make_pdb(seq, plddt=plddt))

    return _build


# --------------------------------------------------------------------------- #
# deterministic stub for the ESM scorer
# --------------------------------------------------------------------------- #

def stub_matrix(seq: str, implausible: dict | None = None) -> np.ndarray:
    """(L, 20) probabilities: the incumbent residue is favoured, except where told.

    `implausible` maps position -> the probability mass given to the residue
    actually present, so a test can plant a "surprising" site at a known index.
    """
    implausible = implausible or {}
    n = len(seq)
    probs = np.zeros((n, 20))
    for i, aa in enumerate(seq):
        idx = AA_ALPHABET.index(aa)
        mass = implausible.get(i, 0.8)
        probs[i, :] = (1.0 - mass) / 19.0
        probs[i, idx] = mass
        # Break ties among alternatives so substitution ranking is deterministic
        # and depends on position: nearest alphabet neighbours score highest.
        for k in range(20):
            if k == idx:
                continue
            probs[i, k] *= 1.0 + 0.01 * ((idx - k) % 20)
        probs[i] /= probs[i].sum()
    return probs


@pytest.fixture
def stub_esm(monkeypatch):
    """Patch src.agent.esm_score so nothing touches a model."""
    from src.agent import esm_score

    state = {"implausible": {}}

    def _matrix(seq: str) -> np.ndarray:
        return stub_matrix(seq, state["implausible"])

    def _logprobs(seq: str):
        probs = _matrix(seq)
        idx = np.array([AA_ALPHABET.index(a) for a in seq])
        return tuple(np.log(probs[np.arange(len(seq)), idx]).tolist())

    monkeypatch.setattr(esm_score, "masked_marginal_matrix", _matrix)
    monkeypatch.setattr(esm_score, "_log_probs_of_sequence", _logprobs)
    return state

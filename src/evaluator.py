"""P2.6 — the immutable hidden evaluator.

THIS IS THE ONLY MODULE ALLOWED TO READ data/proteins/ (constraint C3).
Nothing under src/agent/ may import the reference structure, the reference
sequence, or any quantity derived from them except the scalar `hidden_score`.

The score decomposition (TM-score, contact recovery, sequence recovery) is
withheld unless `reveal=True`, which is called only from
scripts/research/final_reveal.py — never from inner- or outer-loop code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .agent import esm_score
from .cache import fold_cache
from .constants import DEFAULT_PROTEIN
from .geometry import contact_recovery, contact_set, tm_score_fixed_alignment
from .paths import corruption_dir, evaluator_sidecar_dir, protein_dir, read_fasta
from .pdb_io import first_chain

# Weights are fixed by the memo and must not be tuned by the outer loop.
WEIGHTS = {
    "tm_score": 0.55,
    "esm_score": 0.20,
    "plddt": 0.15,
    "contact_recovery": 0.10,
    "edit_fraction": -0.05,
}

# Denominator for edit_fraction. Pinned to the seed policy's max_total_edits so
# the score's scale does not move when a policy patch changes the edit budget:
# a candidate at the seed budget scores edit_fraction = 1.0, and a policy that
# raises its own budget is penalised past 1.0 rather than silently rescaled.
EDIT_FRACTION_DENOMINATOR = 3

# Keys never returned when reveal=False. Enforced by _redact().
HIDDEN_KEYS = (
    "tm_score",
    "contact_recovery",
    "sequence_recovery",
    "tm_backend",
    "reference_length",
)


@dataclass
class Reference:
    """The withheld ground truth. Constructed only inside this module."""

    name: str
    sequence: str
    ca_coords: np.ndarray
    contacts: set


@lru_cache(maxsize=4)
def _load_reference(name: str) -> Reference:
    directory = protein_dir(name)
    # The single string "native_pdb_path" exists in this file and nowhere else
    # under src/ — that is the grep-able form of constraint C3.
    native_pdb_path = directory / "native.pdb"
    if not native_pdb_path.exists():
        raise FileNotFoundError(
            f"missing reference structure {native_pdb_path}; run scripts/fetch_protein.py"
        )
    chain = first_chain(native_pdb_path)
    _, seq = read_fasta(directory / "native_seq.fasta")
    if seq != chain.sequence:
        raise ValueError(
            "reference FASTA and structure disagree; regenerate with scripts/fetch_protein.py"
        )
    return Reference(
        name=name,
        sequence=seq,
        ca_coords=chain.ca_coords(),
        contacts=contact_set(chain.cb_coords()),
    )


# --------------------------------------------------------------------------- #
# component scores
# --------------------------------------------------------------------------- #

def _tm_score(candidate_ca: np.ndarray, candidate_seq: str, ref: Reference):
    """TM-score of a candidate against the reference, normalised by reference length.

    Argument order matters and is easy to get wrong silently. With
    tm_align(candidate, reference, candidate_seq, reference_seq):
        .tm_norm_chain1 -> normalised by CHAIN 1 length == the CANDIDATE
        .tm_norm_chain2 -> normalised by CHAIN 2 length == the REFERENCE  <-- use this
    Both lengths are equal here (substitution-only variants), so the two numbers
    coincide today; tm_norm_chain2 is pinned anyway so an indel-tolerant future
    change cannot flip the hidden score unnoticed.
    """
    try:
        import tmtools
    except ImportError:
        return (
            tm_score_fixed_alignment(candidate_ca, ref.ca_coords),
            "geometry.tm_score_fixed_alignment",
        )

    result = tmtools.tm_align(
        candidate_ca.astype(np.float64),
        ref.ca_coords.astype(np.float64),
        candidate_seq,
        ref.sequence,
    )
    return float(result.tm_norm_chain2), "tmtools.tm_align:tm_norm_chain2"


def _calibration_path(name: str) -> Path:
    return evaluator_sidecar_dir(name) / "esm_calibration.json"


def esm_calibration(name: str) -> dict:
    """Mean/std of per-residue log-likelihood over a reference batch.

    Raw pseudo-log-likelihoods are unbounded negative numbers. Dropping one into
    a weighted sum next to a 0-1 TM-score would let it dominate or vanish
    depending only on sequence length, so it is squashed to 0-1 with a logistic
    on the z-score of the *per-residue* mean log-likelihood (length-robust).

    The batch is the reference sequence plus every seeded corruption variant, so
    "0.5" means "as plausible as the average sequence in the benchmark".
    """
    path = _calibration_path(name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    ref = _load_reference(name)
    batch = [ref.sequence]
    for fasta in sorted(corruption_dir(name).glob("*.fasta")):
        batch.append(read_fasta(fasta)[1])

    per_residue = [
        esm_score.pseudo_log_likelihood(seq) / len(seq) for seq in batch
    ]
    calib = {
        "mean": float(np.mean(per_residue)),
        "std": float(max(np.std(per_residue), 1e-3)),
        "n_batch": len(batch),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    return calib


def rescale_esm(pll: float, length: int, calib: dict) -> float:
    """Logistic-normalised ESM plausibility in (0, 1)."""
    z = (pll / max(length, 1) - calib["mean"]) / calib["std"]
    return float(1.0 / (1.0 + np.exp(-z)))


def combine(
    tm_score: float,
    esm_score_norm: float,
    plddt: float,
    contact_recovery_value: float,
    edit_fraction: float,
) -> float:
    """The hidden objective. Pure arithmetic, no I/O — hand-checkable.

    hidden = 0.55*tm + 0.20*esm + 0.15*plddt + 0.10*contact_recovery
             - 0.05*edit_fraction
    """
    return (
        WEIGHTS["tm_score"] * tm_score
        + WEIGHTS["esm_score"] * esm_score_norm
        + WEIGHTS["plddt"] * plddt
        + WEIGHTS["contact_recovery"] * contact_recovery_value
        + WEIGHTS["edit_fraction"] * edit_fraction
    )


def _redact(full: dict) -> dict:
    """Strip every ground-truth-derived key. See P2.6 verify (a)."""
    return {k: v for k, v in full.items() if k not in HIDDEN_KEYS}


# --------------------------------------------------------------------------- #
# evaluator
# --------------------------------------------------------------------------- #

class HiddenEvaluator:
    def __init__(
        self,
        protein: str = DEFAULT_PROTEIN,
        origin_sequence: str | None = None,
        edit_denominator: int = EDIT_FRACTION_DENOMINATOR,
    ) -> None:
        self.protein = protein
        self.origin_sequence = origin_sequence
        self.edit_denominator = int(edit_denominator)
        self.calls = 0

    def set_origin(self, origin_sequence: str) -> None:
        """The corrupted starting sequence edits are counted against."""
        self.origin_sequence = origin_sequence

    def evaluate(self, candidate, reveal: bool = False, origin: str | None = None) -> dict:
        """Score one candidate.

        `candidate` is a sequence string, or anything with a `.sequence`.
        With reveal=False the returned dict contains only the scalar objective
        plus the fields the agent is allowed to see: ESM score, pLDDT, edit count.
        """
        sequence = getattr(candidate, "sequence", candidate)
        if not isinstance(sequence, str):
            raise TypeError(f"candidate must be a sequence string, got {type(candidate)}")

        ref = _load_reference(self.protein)
        if len(sequence) != len(ref.sequence):
            raise ValueError(
                f"candidate length {len(sequence)} != reference length {len(ref.sequence)}"
            )

        fold = fold_cache.fold(sequence)
        tm, backend = _tm_score(fold.ca_coords, sequence, ref)
        recovery = contact_recovery(fold.contacts, ref.contacts)

        pll = esm_score.pseudo_log_likelihood(sequence)
        esm_norm = rescale_esm(pll, len(sequence), esm_calibration(self.protein))

        baseline = origin if origin is not None else self.origin_sequence
        # No origin configured means "measure the candidate as given": zero edits.
        # Loop code always supplies one; see outer_loop.run_generation.
        edits = 0 if baseline is None else sum(1 for a, b in zip(baseline, sequence) if a != b)
        edit_fraction = edits / self.edit_denominator

        hidden = combine(tm, esm_norm, fold.mean_plddt, recovery, edit_fraction)
        self.calls += 1

        full = {
            "hidden_score": round(float(hidden), 6),
            "esm_score": round(float(esm_norm), 6),
            "pseudo_log_likelihood": round(float(pll), 6),
            "mean_plddt": round(float(fold.mean_plddt), 6),
            "edit_count": int(edits),
            "edit_fraction": round(float(edit_fraction), 6),
            "clashes": int(fold.clashes),
            "from_cache": bool(fold.from_cache),
            # ground truth below this line
            "tm_score": round(float(tm), 6),
            "contact_recovery": round(float(recovery), 6),
            "sequence_recovery": round(
                sum(1 for a, b in zip(sequence, ref.sequence) if a == b) / len(ref.sequence), 6
            ),
            "tm_backend": backend,
            "reference_length": len(ref.sequence),
        }
        return full if reveal else _redact(full)


@lru_cache(maxsize=4)
def _default_evaluator(protein: str) -> HiddenEvaluator:
    return HiddenEvaluator(protein)


def evaluate(
    candidate, reveal: bool = False, protein: str = DEFAULT_PROTEIN, origin: str | None = None
) -> dict:
    """Module-level convenience wrapper around HiddenEvaluator.evaluate."""
    return _default_evaluator(protein).evaluate(candidate, reveal=reveal, origin=origin)


def public_keys() -> tuple:
    """Exactly what the agent may see."""
    return (
        "hidden_score",
        "esm_score",
        "pseudo_log_likelihood",
        "mean_plddt",
        "edit_count",
        "edit_fraction",
        "clashes",
        "from_cache",
    )

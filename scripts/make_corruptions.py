"""P1.3 — generate reproducible synthetic corruption variants.

"Bad protein" is operationalised as 3-5 point substitutions against the
reference sequence, chosen with a seeded RNG and biased toward buried positions
so the damage actually shows up in the predicted fold.

Agent-visible output:      data/corruptions/<name>/<prefix>_0N.fasta
Evaluator-only bookkeeping: data/evaluator_only/<name>/<prefix>_positions.json

Which positions were changed is *never* written anywhere the agent can read.

Usage:
    python scripts/make_corruptions.py 1pgb
    python scripts/make_corruptions.py 1pgb --prefix holdout --n-variants 1 --seed 90210
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import AA_ALPHABET, RESIDUE_CLASS  # noqa: E402
from src.paths import (  # noqa: E402
    corruption_dir,
    ensure_dirs,
    evaluator_sidecar_dir,
    protein_dir,
    read_fasta,
    write_fasta,
)

# Theoretical maximum per-residue solvent accessibility, Tien et al. 2013.
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "E": 223.0, "Q": 225.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}
BURIED_RSA = 0.25
DEFAULT_SEED = 1729


def relative_solvent_accessibility(pdb_path: Path, sequence: str) -> np.ndarray:
    """Per-residue RSA in [0, ~1] via Shrake-Rupley. Falls back to 0.5 uniform."""
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.SASA import ShrakeRupley
    except ImportError:  # pragma: no cover
        print("[rsa] Biopython SASA unavailable; using uniform weights")
        return np.full(len(sequence), 0.5)

    structure = PDBParser(QUIET=True).get_structure("ref", str(pdb_path))
    ShrakeRupley().compute(structure[0], level="R")
    chain = list(structure[0])[0]
    rsa = []
    for res, aa in zip(
        [r for r in chain if r.id[0] == " " and hasattr(r, "sasa")], sequence
    ):
        rsa.append(res.sasa / MAX_ASA[aa])
    if len(rsa) != len(sequence):
        print(f"[rsa] length mismatch ({len(rsa)} vs {len(sequence)}); uniform weights")
        return np.full(len(sequence), 0.5)
    return np.clip(np.array(rsa), 0.0, 2.0)


def position_weights(rsa: np.ndarray) -> np.ndarray:
    """Sampling weights that favour buried positions (low RSA)."""
    w = np.where(rsa < BURIED_RSA, 4.0, 1.0)
    w[0] = w[-1] = 0.1          # termini rarely carry the fold
    return w / w.sum()


def pick_substitution(rng: np.random.Generator, current: str, buried: bool) -> str:
    """A different residue; class-changing when the site is buried."""
    options = [a for a in AA_ALPHABET if a != current]
    if buried:
        cls = RESIDUE_CLASS[current]
        disruptive = [a for a in options if RESIDUE_CLASS[a] != cls]
        if disruptive:
            options = disruptive
    return str(rng.choice(options))


def make_variant(
    rng: np.random.Generator, sequence: str, weights: np.ndarray, rsa: np.ndarray
) -> tuple[str, list[dict]]:
    n_edits = int(rng.integers(3, 6))  # 3, 4 or 5
    positions = rng.choice(
        len(sequence), size=n_edits, replace=False, p=weights
    )
    chars = list(sequence)
    record = []
    for pos in sorted(int(p) for p in positions):
        original = sequence[pos]
        new = pick_substitution(rng, original, bool(rsa[pos] < BURIED_RSA))
        chars[pos] = new
        record.append(
            {
                "position": pos,
                "from": original,
                "to": new,
                "rsa": round(float(rsa[pos]), 4),
                "buried": bool(rsa[pos] < BURIED_RSA),
            }
        )
    return "".join(chars), record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="1pgb")
    ap.add_argument("--n-variants", type=int, default=5)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--prefix", default="corrupt")
    args = ap.parse_args()

    ensure_dirs()
    ref_dir = protein_dir(args.name)
    ref_pdb = ref_dir / "native.pdb"
    _, sequence = read_fasta(ref_dir / "native_seq.fasta")

    rsa = relative_solvent_accessibility(ref_pdb, sequence)
    weights = position_weights(rsa)
    print(f"[rsa] buried positions (<{BURIED_RSA}): {int((rsa < BURIED_RSA).sum())}/{len(sequence)}")

    rng = np.random.default_rng(args.seed)
    out_dir = corruption_dir(args.name)
    sidecar = {"seed": args.seed, "prefix": args.prefix, "variants": {}}

    for v in range(1, args.n_variants + 1):
        variant_id = f"{args.prefix}_{v:02d}"
        seq, record = make_variant(rng, sequence, weights, rsa)
        assert len(seq) == len(sequence)
        write_fasta(out_dir / f"{variant_id}.fasta", variant_id, seq)
        sidecar["variants"][variant_id] = record
        edits = ", ".join(f"{r['from']}{r['position'] + 1}{r['to']}" for r in record)
        print(f"[write] {variant_id}.fasta  edits={len(record)}  ({edits})")

    side_dir = evaluator_sidecar_dir(args.name)
    side_dir.mkdir(parents=True, exist_ok=True)
    side_path = side_dir / f"{args.prefix}_positions.json"
    side_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"[write] evaluator-only sidecar: {side_path}")
    print("\nP1.3 PASS (run tests/test_corruptions.py for the formal check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""P1.1 + P1.2 — download a reference protein, validate it, extract its sequence.

Writes, under the evaluator-only tree:
    data/proteins/<name>/native.pdb
    data/proteins/<name>/native_seq.fasta

Validation is programmatic, not trusted from the choice of PDB id: exactly one
chain, no missing backbone atoms, resolution better than the cutoff. A failure
here means pick a different entry rather than patch around gaps.

Usage:
    python scripts/fetch_protein.py 1pgb
    python scripts/fetch_protein.py 1ubq --name 1ubq
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import AA_SET, THREE_TO_ONE  # noqa: E402
from src.paths import ensure_dirs, protein_dir, write_fasta  # noqa: E402

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
BACKBONE = ("N", "CA", "C", "O")
MAX_RESOLUTION = 2.5


def download(pdb_id: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    print(f"[fetch] {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        dest.write_bytes(resp.read())
    print(f"[fetch] wrote {dest} ({dest.stat().st_size} bytes)")
    return dest


def validate(pdb_path: Path) -> dict:
    """Biopython-based structural validation. Raises SystemExit on failure."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("ref", str(pdb_path))
    model = structure[0]

    chains = list(model)
    n_chains = len(chains)
    print(f"[check] chains in first model: {n_chains}")
    if n_chains != 1:
        raise SystemExit(
            f"FAIL: expected exactly one chain, found {n_chains} "
            f"({[c.id for c in chains]}). Pick a different entry."
        )

    chain = chains[0]
    residues = [r for r in chain if r.get_resname() in THREE_TO_ONE and r.id[0] == " "]
    hetero = [r for r in chain if r.id[0] != " "]
    print(f"[check] standard residues: {len(residues)}  (hetero/water skipped: {len(hetero)})")

    missing = []
    for res in residues:
        absent = [a for a in BACKBONE if a not in res]
        if absent:
            missing.append((res.id[1], res.get_resname(), absent))
    if missing:
        raise SystemExit(f"FAIL: missing backbone atoms: {missing[:10]}")
    print("[check] backbone complete for every residue: OK")

    resolution = structure.header.get("resolution")
    print(f"[check] resolution from header: {resolution} A")
    if resolution is None or resolution >= MAX_RESOLUTION:
        raise SystemExit(
            f"FAIL: resolution {resolution} not better than {MAX_RESOLUTION} A."
        )

    return {
        "chain_id": chain.id,
        "n_residues": len(residues),
        "resolution": resolution,
        "residues": residues,
    }


def extract_sequence(residues) -> str:
    """P1.2 — sequence from coordinates (not SEQRES), evaluator-side only."""
    seq = "".join(THREE_TO_ONE[r.get_resname()] for r in residues)
    bad = sorted(set(seq) - AA_SET)
    if bad:
        raise SystemExit(f"FAIL: non-standard letters in sequence: {bad}")
    return seq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_id")
    ap.add_argument("--name", default=None, help="directory name (default: pdb id lowercased)")
    args = ap.parse_args()

    ensure_dirs()
    name = (args.name or args.pdb_id).lower()
    out_dir = protein_dir(name)
    pdb_path = out_dir / "native.pdb"

    if not pdb_path.exists():
        download(args.pdb_id, pdb_path)
    else:
        print(f"[fetch] reusing existing {pdb_path}")

    info = validate(pdb_path)
    seq = extract_sequence(info["residues"])

    if len(seq) != info["n_residues"]:
        raise SystemExit(
            f"FAIL: sequence length {len(seq)} != residue count {info['n_residues']}"
        )
    print(f"[check] sequence length matches residue count: {len(seq)}")

    fasta = out_dir / "native_seq.fasta"
    write_fasta(fasta, f"{name} chain {info['chain_id']} len={len(seq)}", seq)
    print(f"[write] {fasta}")
    print(f"[seq]   {seq}")
    print("\nP1.1 + P1.2 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

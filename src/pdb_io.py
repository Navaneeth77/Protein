"""Minimal, dependency-light PDB reading.

Kept separate from Biopython on purpose: this module is imported by the agent
path (via the fold cache) where the only thing being read is a *predicted*
structure. Biopython is still used for the one-off validation scripts in
Phase 1 where richer header parsing is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import THREE_TO_ONE


@dataclass
class Residue:
    index: int                      # 0-based position along the chain
    resseq: int                     # residue number as written in the file
    resname: str                    # three-letter code
    aa: str                         # one-letter code
    atoms: dict = field(default_factory=dict)   # atom name -> (3,) float array
    bfactors: dict = field(default_factory=dict)  # atom name -> float

    @property
    def ca(self) -> np.ndarray | None:
        return self.atoms.get("CA")

    @property
    def cb_or_ca(self) -> np.ndarray | None:
        """CB, falling back to CA (glycine has no CB)."""
        cb = self.atoms.get("CB")
        return cb if cb is not None else self.atoms.get("CA")

    @property
    def ca_bfactor(self) -> float:
        return float(self.bfactors.get("CA", 0.0))


@dataclass
class Chain:
    chain_id: str
    residues: list[Residue]

    @property
    def sequence(self) -> str:
        return "".join(r.aa for r in self.residues)

    def __len__(self) -> int:
        return len(self.residues)

    def ca_coords(self) -> np.ndarray:
        return np.array([r.ca for r in self.residues], dtype=float)

    def cb_coords(self) -> np.ndarray:
        return np.array([r.cb_or_ca for r in self.residues], dtype=float)

    def ca_bfactors(self) -> np.ndarray:
        return np.array([r.ca_bfactor for r in self.residues], dtype=float)

    def heavy_atoms(self) -> tuple[np.ndarray, np.ndarray]:
        """(coords (N,3), residue_index (N,)) for all non-hydrogen atoms."""
        coords, owners = [], []
        for r in self.residues:
            for name, xyz in r.atoms.items():
                if name.startswith("H") or name in ("D",):
                    continue
                coords.append(xyz)
                owners.append(r.index)
        return np.array(coords, dtype=float), np.array(owners, dtype=int)


def parse_pdb(path_or_text: str, *, is_text: bool = False) -> dict[str, Chain]:
    """Parse ATOM records of the first model into {chain_id: Chain}.

    Only the 20 standard residues are kept; altloc other than ' '/'A' is skipped
    so each residue has a single set of coordinates.
    """
    if is_text:
        lines = path_or_text.splitlines()
    else:
        with open(path_or_text, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()

    chains: dict[str, list[Residue]] = {}
    seen: dict[tuple[str, str], Residue] = {}

    for line in lines:
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        resname = line[17:20].strip()
        if resname not in THREE_TO_ONE:
            continue
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue
        atom_name = line[12:16].strip()
        chain_id = line[21].strip() or "A"
        resseq = line[22:26].strip()
        icode = line[26].strip()
        key = (chain_id, f"{resseq}{icode}")

        res = seen.get(key)
        if res is None:
            bucket = chains.setdefault(chain_id, [])
            res = Residue(
                index=len(bucket),
                resseq=int(resseq),
                resname=resname,
                aa=THREE_TO_ONE[resname],
            )
            bucket.append(res)
            seen[key] = res

        xyz = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float
        )
        res.atoms[atom_name] = xyz
        try:
            res.bfactors[atom_name] = float(line[60:66])
        except ValueError:
            res.bfactors[atom_name] = 0.0

    return {cid: Chain(cid, residues) for cid, residues in chains.items()}


def first_chain(path_or_text: str, *, is_text: bool = False) -> Chain:
    """The single chain of a monomeric structure (lowest chain id if several)."""
    chains = parse_pdb(path_or_text, is_text=is_text)
    if not chains:
        raise ValueError("no standard-residue ATOM records found")
    return chains[sorted(chains)[0]]

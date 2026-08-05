"""Canonical filesystem locations.

The split between `proteins/` (evaluator-only) and `corruptions/` (agent-visible)
is the whole point of the layout; see constraint C3 in refold_tasks.md.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
PROTEINS = DATA / "proteins"        # evaluator-only tree
CORRUPTIONS = DATA / "corruptions"  # agent-visible tree
CACHE = DATA / "cache"
EVALUATOR_ONLY = DATA / "evaluator_only"  # sidecars: which positions were changed
LOGS = ROOT / "logs"
DEMO = ROOT / "demo"
DOCS = ROOT / "docs"
AGENT = ROOT / "src" / "agent"


def protein_dir(name: str) -> Path:
    return PROTEINS / name


def corruption_dir(name: str) -> Path:
    return CORRUPTIONS / name


def evaluator_sidecar_dir(name: str) -> Path:
    return EVALUATOR_ONLY / name


def ensure_dirs() -> None:
    for p in (DATA, PROTEINS, CORRUPTIONS, CACHE, EVALUATOR_ONLY, LOGS, DEMO, DOCS):
        p.mkdir(parents=True, exist_ok=True)


def read_fasta(path: Path) -> tuple[str, str]:
    """(header, sequence) from a single-record FASTA."""
    header, seq = "", []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:]
        else:
            seq.append(line)
    return header, "".join(seq)


def write_fasta(path: Path, header: str, sequence: str, width: int = 60) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))
    path.write_text(f">{header}\n{body}\n", encoding="utf-8")

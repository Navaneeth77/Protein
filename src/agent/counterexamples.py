"""P3.3 — the counterexample recorder.

A counterexample is the moment the public-facing prediction and hidden reality
disagree: the agent's visible signals (ESM plausibility, pLDDT) improved, but the
hidden structural verifier did not. These rows are the only training signal the
outer loop gets, so they are written verbatim and never summarised away.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..paths import LOGS

DEFAULT_LOG = LOGS / "counterexamples.jsonl"

REQUIRED_KEYS = (
    "iteration",
    "predicted_delta",
    "hidden_delta",
    "policy_before",
    "state_before",
)


def is_counterexample(predicted_delta: float, hidden_delta: float) -> bool:
    """Predicted improvement that the hidden verifier did not confirm."""
    return predicted_delta > 0.0 and hidden_delta <= 0.0


def record(
    iteration: int,
    predicted_delta: float,
    hidden_delta: float,
    policy_before: dict,
    state_before: dict,
    *,
    path: Path | None = None,
    extra: dict | None = None,
) -> dict:
    """Append one JSONL row and return it."""
    path = Path(path) if path else DEFAULT_LOG
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "iteration": int(iteration),
        "predicted_delta": round(float(predicted_delta), 6),
        "hidden_delta": round(float(hidden_delta), 6),
        "policy_before": policy_before,
        "state_before": _compact_state(state_before),
    }
    if extra:
        row.update(extra)

    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, allow_nan=False) + "\n")
    return row


def _compact_state(state: dict) -> dict:
    """Keep the state row readable: summary, relations, and the top residues.

    Full per-residue arrays for every generation would make the log unusable by
    hand, and the outer loop only ever reasons about the highest-scoring sites.
    """
    if not isinstance(state, dict) or "residues" not in state:
        return state

    residues = sorted(
        state["residues"], key=lambda r: -float(r.get("esm_surprisal", 0.0))
    )[:8]
    return {
        "sequence_length": state.get("sequence_length"),
        "summary": state.get("summary", {}),
        "top_residues": residues,
        "relations": {
            "n_contacts": len(state.get("relations", {}).get("contacts", [])),
            "n_long_range_contacts": sum(
                1
                for c in state.get("relations", {}).get("contacts", [])
                if c.get("long_range")
            ),
            "helices": state.get("relations", {}).get("helices", []),
            "strands": state.get("relations", {}).get("strands", []),
            "mutation_effects": state.get("relations", {}).get("mutation_effects", []),
        },
    }


def load(path: Path | None = None) -> list[dict]:
    path = Path(path) if path else DEFAULT_LOG
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def latest(path: Path | None = None) -> dict | None:
    rows = load(path)
    return rows[-1] if rows else None

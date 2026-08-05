"""P3.2 + P2.3 + P2.4 — the deterministic policy interpreter.

`apply_policy` is a pure function of (policy, state): same inputs, same
candidates, every time. It contains no dynamic code execution of any kind: no
interpreter escape, no shell-out, no import hook, no attribute lookup driven by
policy content. A policy is read as data and its fields parameterise fixed logic
written here. tests/test_policy_interpreter.py runs the C2 grep over this file.

Pipeline implemented:
  select_positions  (P3.2) weighted, min-max normalised per-residue score
  enumerate_candidates (P2.3) top-n substitutions per position, class filter
  prerank_candidates (P2.4) PLL minus a per-edit penalty, keep the best few
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import residue_class
from . import esm_score

# P2.4: penalty per accumulated edit, in pseudo-log-likelihood units. Large
# enough that a tie in PLL is broken by edit count, small enough that a genuinely
# better sequence can still afford an extra edit.
EDIT_PENALTY_LAMBDA = 0.5
SHORTLIST_SIZE = 3


@dataclass
class Candidate:
    sequence: str
    position: int                     # 0-based site changed relative to parent
    from_aa: str
    to_aa: str
    parent_sequence: str
    position_score: float = 0.0
    substitution_prob: float = 0.0
    mutations: list = field(default_factory=list)   # cumulative, vs origin
    prerank_score: float | None = None

    @property
    def edit_count(self) -> int:
        return len(self.mutations)

    def label(self) -> str:
        return f"{self.from_aa}{self.position + 1}{self.to_aa}"


def sequence_of(state: dict) -> str:
    return "".join(r["aa"] for r in state["residues"])


def _normalise(values: np.ndarray) -> np.ndarray:
    """Min-max to 0-1 across positions.

    Required because the scored features live on different scales:
    `esm_surprisal` is an unbounded positive log quantity while `low_plddt` and
    the contact-violation fields are already 0-1. Without this, the surprisal
    weight would dominate regardless of what the policy says, and re-weighting
    would look like it did nothing.
    """
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def position_scores(policy: dict, state: dict) -> np.ndarray:
    """Weighted, normalised per-residue suspicion score."""
    residues = state["residues"]
    n = len(residues)
    score = np.zeros(n, dtype=float)
    for feature, weight in policy["position_score"].items():
        raw = np.array([float(r[feature]) for r in residues], dtype=float)
        score += float(weight) * _normalise(raw)
    return score


def select_positions(policy: dict, state: dict) -> list[int]:
    """Top-`positions` sites by score; ties break toward the lower index."""
    score = position_scores(policy, state)
    k = int(policy["proposal"]["positions"])
    order = np.lexsort((np.arange(len(score)), -score))
    return [int(i) for i in order[:k]]


def enumerate_candidates(
    policy: dict,
    state: dict,
    origin: str | None = None,
    history: list | None = None,
    substitution_source=None,
) -> list[Candidate]:
    """P2.3 — concrete single-substitution candidates from the selected sites.

    `origin` is the starting (corrupted) sequence; edits are counted against it
    so `max_total_edits` bounds the whole trajectory, not one step. Substitutions
    identical to the incumbent residue are dropped, as are candidates that would
    exceed the edit budget.
    """
    incumbent = sequence_of(state)
    origin = origin or incumbent
    history = list(history or [])
    ranker = substitution_source or esm_score.substitution_ranking

    proposal = policy["proposal"]
    per_position = int(proposal["substitutions_per_position"])
    preserve_class = bool(proposal["preserve_residue_class"])
    max_edits = int(proposal["max_total_edits"])

    scores = position_scores(policy, state)
    seen: set[str] = {incumbent}
    candidates: list[Candidate] = []

    for pos in select_positions(policy, state):
        current = incumbent[pos]
        current_class = residue_class(current)
        # Ask for extra so the class filter cannot starve the shortlist.
        pool = ranker(incumbent, pos, 19)
        kept = 0
        for aa, prob in pool:
            if kept >= per_position:
                break
            if aa == current:
                continue
            if preserve_class and residue_class(aa) != current_class:
                continue

            seq = incumbent[:pos] + aa + incumbent[pos + 1 :]
            if seq in seen:
                continue
            mutations = _mutations_vs_origin(origin, seq)
            if len(mutations) > max_edits:
                continue

            seen.add(seq)
            kept += 1
            candidates.append(
                Candidate(
                    sequence=seq,
                    position=pos,
                    from_aa=current,
                    to_aa=aa,
                    parent_sequence=incumbent,
                    position_score=float(scores[pos]),
                    substitution_prob=float(prob),
                    mutations=mutations,
                )
            )

    return candidates


def _mutations_vs_origin(origin: str, sequence: str) -> list[dict]:
    return [
        {"position": i, "from": o, "to": c}
        for i, (o, c) in enumerate(zip(origin, sequence))
        if o != c
    ]


def edit_count(sequence: str, other: str) -> int:
    return sum(1 for a, b in zip(sequence, other) if a != b)


def prerank_candidates(
    candidates: list[Candidate],
    shortlist_size: int = SHORTLIST_SIZE,
    lam: float = EDIT_PENALTY_LAMBDA,
    pll_source=None,
) -> list[Candidate]:
    """P2.4 — cheap ranking before the expensive fold call.

    score = PLL(candidate) - lam * edit_count(candidate, origin)

    Ties in PLL therefore resolve in favour of fewer edits. A further tie breaks
    on sequence so the shortlist is fully deterministic.
    """
    pll = pll_source or esm_score.pseudo_log_likelihood
    for cand in candidates:
        cand.prerank_score = float(pll(cand.sequence)) - lam * cand.edit_count
    ranked = sorted(
        candidates,
        key=lambda c: (-c.prerank_score, c.edit_count, c.sequence),
    )
    return ranked[:shortlist_size]


def apply_policy(
    policy: dict,
    state: dict,
    origin: str | None = None,
    history: list | None = None,
    shortlist_size: int = SHORTLIST_SIZE,
    substitution_source=None,
    pll_source=None,
) -> list[Candidate]:
    """Full deterministic step: state -> shortlisted candidates."""
    candidates = enumerate_candidates(
        policy, state, origin=origin, history=history,
        substitution_source=substitution_source,
    )
    if not candidates:
        return []
    return prerank_candidates(
        candidates, shortlist_size=shortlist_size, pll_source=pll_source
    )

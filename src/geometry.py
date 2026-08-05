"""Structure geometry: contact maps, compactness, clashes, superposition, TM-score.

Pure functions over coordinate arrays. This module never reads a file, so it can
be shared by the agent path (which only ever sees predicted structures) and the
evaluator (which is the only caller allowed to pass reference coordinates in).
"""

from __future__ import annotations

import numpy as np

from .constants import (
    CLASH_CUTOFF_ANGSTROM,
    CLASH_MIN_SEQ_SEPARATION,
    CONTACT_CUTOFF_ANGSTROM,
)

MIN_CONTACT_SEPARATION = 2  # |i-j| < 2 is trivially in contact; excluded


# --------------------------------------------------------------------------- #
# contacts
# --------------------------------------------------------------------------- #

def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def contact_set(
    cb_coords: np.ndarray,
    cutoff: float = CONTACT_CUTOFF_ANGSTROM,
    min_separation: int = MIN_CONTACT_SEPARATION,
) -> set[tuple[int, int]]:
    """{(i, j) : i < j, |i-j| >= min_separation, d(Cb_i, Cb_j) <= cutoff}."""
    d = pairwise_distances(cb_coords)
    n = len(cb_coords)
    i, j = np.triu_indices(n, k=min_separation)
    mask = d[i, j] <= cutoff
    return set(zip(i[mask].tolist(), j[mask].tolist()))


def contact_degrees(
    contacts: set[tuple[int, int]], n: int, long_range_separation: int
) -> tuple[np.ndarray, np.ndarray]:
    """(total_degree, long_range_degree) per residue."""
    total = np.zeros(n, dtype=int)
    long_range = np.zeros(n, dtype=int)
    for i, j in contacts:
        total[i] += 1
        total[j] += 1
        if abs(j - i) > long_range_separation:
            long_range[i] += 1
            long_range[j] += 1
    return total, long_range


def contact_recovery(
    candidate: set[tuple[int, int]], reference: set[tuple[int, int]]
) -> float:
    """|candidate ∩ reference| / |reference|. 0.0 when the reference has none."""
    if not reference:
        return 0.0
    return len(candidate & reference) / len(reference)


# --------------------------------------------------------------------------- #
# compactness / packing
# --------------------------------------------------------------------------- #

def radius_of_gyration(coords: np.ndarray) -> float:
    centroid = coords.mean(axis=0)
    d = coords - centroid
    return float(np.sqrt((d * d).sum(axis=1).mean()))


def clash_count(
    coords: np.ndarray,
    residue_index: np.ndarray,
    cutoff: float = CLASH_CUTOFF_ANGSTROM,
    min_seq_separation: int = CLASH_MIN_SEQ_SEPARATION,
) -> int:
    """Non-bonded heavy-atom pairs closer than `cutoff`.

    Pairs from residues within `min_seq_separation` of each other are skipped so
    that real covalent bonds and 1-3 neighbours are not counted as clashes.
    """
    d = pairwise_distances(coords)
    n = len(coords)
    i, j = np.triu_indices(n, k=1)
    sep_ok = np.abs(residue_index[i] - residue_index[j]) >= min_seq_separation
    return int(((d[i, j] < cutoff) & sep_ok).sum())


# --------------------------------------------------------------------------- #
# secondary structure (geometric heuristic, not DSSP)
# --------------------------------------------------------------------------- #

def secondary_structure(
    ca_coords: np.ndarray, contacts: set[tuple[int, int]]
) -> list[str]:
    """Per-residue 'helix' / 'strand' / 'coil' from CA geometry alone.

    A deliberately simple CA-distance heuristic, not a DSSP reimplementation:
    the state graph only needs a coarse regional label, and adding a hydrogen-
    bond solver here would add a dependency for no gain in the policy search.
    """
    n = len(ca_coords)
    labels = ["coil"] * n
    if n < 5:
        return labels

    d = pairwise_distances(ca_coords)
    long_range = np.zeros(n, dtype=bool)
    for i, j in contacts:
        if abs(j - i) > 4:
            long_range[i] = True
            long_range[j] = True

    for i in range(n - 4):
        d13 = d[i, i + 3]
        d14 = d[i, i + 4]
        if 4.5 <= d13 <= 6.5 and 5.0 <= d14 <= 7.5:
            for k in range(i, i + 5):
                labels[k] = "helix"

    for i in range(n - 4):
        if labels[i] != "coil":
            continue
        if d[i, i + 3] >= 9.0 and d[i, i + 4] >= 11.5 and long_range[i]:
            labels[i] = "strand"

    return labels


def contiguous_regions(labels: list[str], target: str, min_length: int = 4):
    """Inclusive [start, end] spans where `labels` equals `target`."""
    spans, start = [], None
    for i, lab in enumerate(labels):
        if lab == target and start is None:
            start = i
        elif lab != target and start is not None:
            if i - start >= min_length:
                spans.append([start, i - 1])
            start = None
    if start is not None and len(labels) - start >= min_length:
        spans.append([start, len(labels) - 1])
    return spans


# --------------------------------------------------------------------------- #
# superposition and TM-score
# --------------------------------------------------------------------------- #

def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimising |R @ mobile + t - target|."""
    mc = mobile.mean(axis=0)
    tc = target.mean(axis=0)
    p = mobile - mc
    q = target - tc
    u, _, vt = np.linalg.svd(p.T @ q)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    diag = np.diag([1.0, 1.0, sign])
    r = vt.T @ diag @ u.T
    t = tc - r @ mc
    return r, t


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    r, t = kabsch(a, b)
    moved = (r @ a.T).T + t
    return float(np.sqrt(((moved - b) ** 2).sum(axis=1).mean()))


def tm_d0(length: int) -> float:
    """TM-score length-normalisation constant d0 for a reference of `length`."""
    if length <= 15:
        return 0.5
    return max(0.5, 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8)


def tm_score_fixed_alignment(
    candidate_ca: np.ndarray, reference_ca: np.ndarray
) -> float:
    """TM-score with the identity residue correspondence.

    Valid here because every candidate is a substitution-only variant of the
    reference sequence, so residue i of the candidate *is* residue i of the
    reference — no sequence alignment step is needed. Normalisation is by the
    reference length (see `tm_norm_chain2` note in the evaluator).

    Implements the Zhang-Skolnick fragment-seeded iterative superposition
    search: seed on fragments of decreasing length, then repeatedly re-fit on
    the residues that are currently within a distance cutoff.
    """
    if candidate_ca.shape != reference_ca.shape:
        raise ValueError(
            f"identity correspondence requires equal shapes, got "
            f"{candidate_ca.shape} and {reference_ca.shape}"
        )
    n = len(reference_ca)
    if n < 3:
        return 0.0

    d0 = tm_d0(n)
    d0_search = float(np.clip(d0, 4.5, 8.0))
    best = 0.0

    frag_lengths = []
    length = n
    while length >= 4:
        frag_lengths.append(length)
        length //= 2
    if frag_lengths[-1] != 4:
        frag_lengths.append(4)

    # Accuracy envelope, measured against tmtools by scripts/validate_tm_score.py:
    # agreement is within 4e-4 for TM >= 0.74 and exact at TM = 1.0, but this
    # search underestimates by up to ~0.06 for badly dissimilar structures
    # (TM < 0.5). The cause is TM-align's multi-cutoff refinement schedule, not
    # seed density -- using every start offset instead of a stride costs 4x and
    # buys 4e-4. Every candidate scored here is a single-point mutant of a
    # 56-residue protein sitting at TM ~0.85-0.98, i.e. well inside the accurate
    # regime, so the stride stays.
    for frag_len in frag_lengths:
        for start in range(0, n - frag_len + 1, max(1, frag_len // 2) if frag_len < n else 1):
            sel = np.arange(start, start + frag_len)
            for _ in range(20):
                r, t = kabsch(candidate_ca[sel], reference_ca[sel])
                moved = (r @ candidate_ca.T).T + t
                d = np.sqrt(((moved - reference_ca) ** 2).sum(axis=1))
                score = float((1.0 / (1.0 + (d / d0) ** 2)).sum() / n)
                if score > best:
                    best = score
                cut = d0_search
                new_sel = np.where(d < cut)[0]
                while len(new_sel) < 4 and cut < 20.0:
                    cut += 0.5
                    new_sel = np.where(d < cut)[0]
                if len(new_sel) < 4:
                    break
                if len(new_sel) == len(sel) and np.array_equal(new_sel, sel):
                    break
                sel = new_sel

    return best

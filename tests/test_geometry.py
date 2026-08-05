"""Geometry primitives: contacts, compactness, superposition, TM-score."""

from __future__ import annotations

import numpy as np

from src.geometry import (
    clash_count,
    contact_degrees,
    contact_recovery,
    contact_set,
    kabsch,
    radius_of_gyration,
    rmsd,
    tm_d0,
    tm_score_fixed_alignment,
)
from tests.conftest import helix_coords


def test_contact_set_excludes_trivial_neighbours():
    ca, cb = helix_coords(20)
    contacts = contact_set(cb)
    assert all(abs(j - i) >= 2 for i, j in contacts)
    assert all(i < j for i, j in contacts)


def test_contact_degrees_counts_both_endpoints():
    contacts = {(0, 5), (0, 9), (3, 7)}
    total, long_range = contact_degrees(contacts, n=10, long_range_separation=4)
    assert total[0] == 2
    assert total[5] == 1
    # separation 5 and 9 are long range at threshold 4; separation 4 is not
    assert long_range[0] == 2
    assert long_range[3] == 0


def test_contact_recovery_bounds():
    ref = {(0, 5), (1, 6), (2, 7)}
    assert contact_recovery(ref, ref) == 1.0
    assert contact_recovery(set(), ref) == 0.0
    assert contact_recovery({(0, 5)}, ref) == 1 / 3
    assert contact_recovery({(0, 5)}, set()) == 0.0


def test_radius_of_gyration_of_known_shape():
    # Four points at distance 1 from the origin along +/-x, +/-y.
    coords = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]])
    assert abs(radius_of_gyration(coords) - 1.0) < 1e-12


def test_clash_count_skips_bonded_pairs():
    coords = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.2, 0, 0]])
    owners = np.array([0, 1, 5])
    # pairs (0,1) and (1,2) are both < 2.0 A apart, but only (1,2) is
    # sequence-separated enough to count.
    assert clash_count(coords, owners) == 2  # (0,2) at 1.2 and (1,2) at 0.2
    owners_adjacent = np.array([0, 1, 2])
    assert clash_count(coords, owners_adjacent) == 1  # only (0,2), separation 2


def test_kabsch_recovers_a_known_rigid_motion():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(12, 3))
    angle = 0.7
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    shift = np.array([3.0, -1.0, 2.0])
    b = (rot @ a.T).T + shift
    r, t = kabsch(a, b)
    assert np.allclose(r, rot, atol=1e-8)
    assert np.allclose(t, shift, atol=1e-8)
    assert rmsd(a, b) < 1e-8


def test_tm_score_identical_structures_is_one():
    ca, _ = helix_coords(56)
    assert abs(tm_score_fixed_alignment(ca, ca) - 1.0) < 1e-9


def test_tm_score_penalises_a_scrambled_structure():
    ca, _ = helix_coords(56)
    rng = np.random.default_rng(1)
    scrambled = ca + rng.normal(scale=8.0, size=ca.shape)
    score = tm_score_fixed_alignment(scrambled, ca)
    assert 0.0 < score < 0.5


def test_tm_score_is_invariant_to_rigid_motion():
    ca, _ = helix_coords(40)
    angle = 1.1
    rot = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )
    moved = (rot @ ca.T).T + np.array([10.0, 5.0, -3.0])
    assert abs(tm_score_fixed_alignment(moved, ca) - 1.0) < 1e-6


def test_tm_d0_matches_the_published_formula():
    assert abs(tm_d0(56) - (1.24 * (56 - 15) ** (1 / 3) - 1.8)) < 1e-12
    assert tm_d0(10) == 0.5

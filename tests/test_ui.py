"""P4 — UI helpers: gated reveal, honest heatmap caption, score history, diff."""

from __future__ import annotations

import json

import pytest

from app import streamlit_app as ui
from src.agent import policy as policy_mod
from tests.conftest import make_pdb

REFERENCE_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEK"
CORRUPTED_SEQ = "MTYKLILNGKTLKGETTTEAPDAATAEK"


@pytest.fixture
def fake_reference_pdb(tmp_path, monkeypatch):
    """A reference structure with an unmistakable marker in its atom lines."""
    directory = tmp_path / "toy"
    directory.mkdir()
    text = make_pdb(REFERENCE_SEQ).replace("ATOM  ", "ATOM  ")
    (directory / "native.pdb").write_text(
        "REMARK REFERENCE-STRUCTURE-MARKER\n" + text, encoding="utf-8"
    )
    monkeypatch.setattr(ui, "protein_dir", lambda name: directory)
    return directory


# --------------------------------------------------------------------------- #
# P4.2 — gated reveal
# --------------------------------------------------------------------------- #

def test_reference_structure_is_absent_from_the_dom_when_not_revealed(
    fake_reference_pdb,
):
    """P4.2 verify: no reference atom lines in the rendered HTML."""
    corrupted = make_pdb(CORRUPTED_SEQ)
    repaired = make_pdb(REFERENCE_SEQ)

    models = ui.build_viewer_models(corrupted, repaired, reveal=False, protein="toy")
    assert [m["label"] for m in models] == ["corrupted input", "repaired candidate"]
    assert all("REFERENCE-STRUCTURE-MARKER" not in m["pdb"] for m in models)

    html = ui.viewer_html(models)
    assert "REFERENCE-STRUCTURE-MARKER" not in html
    assert ui.CACHED_BADGE in {m["badge"] for m in models}


def test_reference_structure_appears_only_after_the_reveal(fake_reference_pdb):
    models = ui.build_viewer_models(
        make_pdb(CORRUPTED_SEQ), make_pdb(REFERENCE_SEQ), reveal=True, protein="toy"
    )
    assert [m["label"] for m in models][-1] == "ground truth (revealed)"
    html = ui.viewer_html(models)
    assert "REFERENCE-STRUCTURE-MARKER" in html


def test_reveal_defaults_to_off_in_the_signature():
    import inspect

    signature = inspect.signature(ui.build_viewer_models)
    assert signature.parameters["reveal"].default is False


def test_viewer_handles_missing_cached_structures(fake_reference_pdb):
    assert ui.build_viewer_models(None, None, reveal=False, protein="toy") == []


# --------------------------------------------------------------------------- #
# P4.3 — honest labelling (C1)
# --------------------------------------------------------------------------- #

def test_heatmap_caption_says_approximation():
    """P4.3 verify: the checkable form of C1's naming rule."""
    assert "approximation" in ui.HEATMAP_CAPTION


def test_heatmap_caption_makes_no_implementation_claim():
    lowered = ui.HEATMAP_CAPTION.lower()
    assert "pepcompass" not in lowered
    assert "jacobian" not in lowered
    assert "implementation of one" in lowered or "not an implementation" in lowered


def test_app_title_claim_stays_within_what_was_built():
    source = ui.__file__
    text = open(source, encoding="utf-8").read().lower()
    for forbidden in ("therapeutic", "restores function", "discovers"):
        # Any occurrence must be a negation inside the caption, never a claim.
        for line in text.splitlines():
            if forbidden in line:
                assert "never" in line or "nothing here" in line or "not " in line, line


def test_heatmap_dataframe_shape(sequence):
    import numpy as np

    matrix = np.full((len(sequence), 20), 0.05)
    frame = ui.heatmap_dataframe(sequence, matrix)
    assert frame.shape == (20, len(sequence))
    assert list(frame.columns)[0] == f"{sequence[0]}1"


# --------------------------------------------------------------------------- #
# P4.4 — score history
# --------------------------------------------------------------------------- #

def _generations(n: int) -> list[dict]:
    return [
        {
            "iteration": i,
            "accepted": i == 1,
            "candidate_public_median": 0.50 + 0.01 * i,
            "candidate_median": 0.60 + 0.02 * i,
            "incumbent_median": 0.60,
            "policy_before": policy_mod.load_seed_policy(),
            "policy_after": policy_mod.load_seed_policy(),
            "counterexample": None,
            "patch": None,
        }
        for i in range(n)
    ]


def test_history_length_equals_completed_generations():
    """P4.4 verify: no dropped generation 0, no double-counted seed policy."""
    for n in (1, 3, 5):
        history = ui.score_history(_generations(n))
        assert len(history["iteration"]) == n
        assert history["iteration"] == list(range(n))
        assert len(history["public_median"]) == n
        assert len(history["hidden_median"]) == n


def test_hidden_series_is_truncated_at_the_reveal_point():
    history = ui.score_history(_generations(5), reveal_up_to=2)
    assert history["public_median"] == [pytest.approx(0.50 + 0.01 * i) for i in range(5)]
    assert history["hidden_median"][:3] == [
        pytest.approx(0.60 + 0.02 * i) for i in range(3)
    ]
    assert history["hidden_median"][3:] == [None, None]


def test_full_reveal_shows_every_hidden_point():
    history = ui.score_history(_generations(4), reveal_up_to=3)
    assert all(v is not None for v in history["hidden_median"])


# --------------------------------------------------------------------------- #
# P4.5 — policy diff
# --------------------------------------------------------------------------- #

def test_diff_shows_a_plus_line_for_a_newly_weighted_feature():
    """P4.5 verify: the activated key appears as an added line."""
    before = policy_mod.load_seed_policy()
    after = policy_mod.clone(before)
    after["position_score"] = {
        "esm_surprisal": 0.45,
        "low_plddt": 0.15,
        "contact_violation": 0.15,
        "long_range_contact_violation": 0.25,
    }
    diff = ui.policy_diff(before, after)
    added = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    assert any("long_range_contact_violation" in l for l in added), diff


def test_diff_is_empty_for_an_unchanged_policy():
    seed = policy_mod.load_seed_policy()
    assert ui.policy_diff(seed, policy_mod.clone(seed)) == ""


def test_rendered_policy_view_is_labelled_display_only():
    seed = policy_mod.load_seed_policy()
    rendered = ui.rendered_policy_view(seed)
    assert "DISPLAY ONLY" in rendered
    assert "never executed" in rendered
    assert "esm_surprisal" in rendered


def test_accepted_patches_filters_on_the_accept_flag():
    generations = _generations(4)
    accepted = ui.accepted_patches(generations)
    assert [g["iteration"] for g in accepted] == [1]


def test_demo_state_round_trips(tmp_path):
    payload = {"protein": "1pgb", "generations": _generations(2), "variants": {}}
    path = tmp_path / "demo_state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert ui.load_demo_state(path)["protein"] == "1pgb"
    assert ui.load_demo_state(tmp_path / "missing.json") == {}

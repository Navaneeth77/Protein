"""ReFold — the MVP demo UI.

One page, one button. Pressing "Run ReFold" executes the whole loop once:

    corrupted protein -> ESM scores residues -> candidate mutations
    -> ESMFold predicts each -> evaluator scores them -> state is grounded
    -> Gemma returns ONE policy patch -> interpreter re-runs -> before/after

Layout: sequences on the left, structure/heatmap/candidates in the middle,
Gemma's reasoning and the policy diff on the right, button at the bottom.

Structures come from data/cache/ (populated by scripts/precompute_mvp.py) so the
button press does not spend a minute per fold. Gemma is called live.
"""

from __future__ import annotations

import difflib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import policy as policy_mod  # noqa: E402
from src.cache import fold_cache  # noqa: E402
from src.constants import AA_ALPHABET, DEFAULT_PROTEIN  # noqa: E402
from src.paths import LOGS, protein_dir  # noqa: E402

DEMO_STATE = LOGS / "demo_state.json"

# P4.3 — the caption is the honest-labelling contract, asserted in tests/test_ui.py.
# EXCLUSION (C1): this heatmap is a per-position ESM masked-marginal probability
# map. It is NOT an implementation of PepCompass, and it computes no tangent
# space and no decoder Jacobian. The wording below must keep saying so.
HEATMAP_CAPTION = (
    "Per-position ESM-2 masked-marginal substitution probabilities. This is an "
    "approximation of a geometry-inspired local mutation map, not an "
    "implementation of one: no tangent space and no decoder derivative is "
    "computed anywhere in this system."
)

CACHED_BADGE = "cached"
LIVE_BADGE = "live"
SYNTHETIC_BADGE = "SYNTHETIC - NOT A PREDICTION"

SYNTHETIC_WARNING = (
    "These structures came from the synthetic harness backend, not from ESMFold. "
    "Every TM-score, contact recovery and pLDDT shown here is a fixture used to "
    "test the pipeline and means nothing scientifically. Clear the cache and "
    "precompute with the real checkpoint before reading anything into these numbers."
)

REVEAL_KEY = "reveal_ground_truth"


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #

def load_demo_state(path: Path | None = None) -> dict:
    path = Path(path) if path else DEMO_STATE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cached_pdb(sequence: str) -> str | None:
    path = fold_cache.cache_path(sequence)
    return path.read_text(encoding="utf-8") if path.exists() else None


def reference_pdb(protein: str = DEFAULT_PROTEIN) -> str | None:
    """The withheld structure. Only ever called behind the reveal gate."""
    path = protein_dir(protein) / "native.pdb"
    return path.read_text(encoding="utf-8") if path.exists() else None


# --------------------------------------------------------------------------- #
# structure viewer with gated reveal
# --------------------------------------------------------------------------- #

def build_viewer_models(
    corrupted_pdb: str | None,
    repaired_pdb: str | None,
    reveal: bool = False,
    protein: str = DEFAULT_PROTEIN,
) -> list[dict]:
    """The models the viewer will draw.

    With `reveal` false the reference structure is not merely hidden in the UI —
    it is never loaded, so it cannot reach the DOM by accident.
    """

    def badge_for(pdb: str) -> str:
        return SYNTHETIC_BADGE if "SYNTHETIC_GEOMETRY" in pdb else CACHED_BADGE

    models = []
    if corrupted_pdb:
        models.append(
            {"label": "corrupted input", "colour": "salmon",
             "pdb": corrupted_pdb, "badge": badge_for(corrupted_pdb)}
        )
    if repaired_pdb:
        models.append(
            {"label": "repaired candidate", "colour": "skyblue",
             "pdb": repaired_pdb, "badge": badge_for(repaired_pdb)}
        )
    if reveal:
        ref = reference_pdb(protein)
        if ref:
            models.append(
                {"label": "ground truth (revealed)", "colour": "palegreen",
                 "pdb": ref, "badge": "ground truth"}
            )
    return models


def viewer_html(models: list[dict], width: int = 700, height: int = 460) -> str:
    """Render the models to standalone HTML via py3Dmol."""
    import py3Dmol

    view = py3Dmol.view(width=width, height=height)
    for index, model in enumerate(models):
        view.addModel(model["pdb"], "pdb")
        view.setStyle({"model": index}, {"cartoon": {"color": model["colour"]}})
    view.zoomTo()
    return view._make_html()


def show_structures(models: list[dict], height: int = 380, width: int = 460) -> None:
    """Draw the models in Streamlit, falling back to raw HTML if stmol is unhappy."""
    import streamlit as st

    try:
        import py3Dmol
        from stmol import showmol

        view = py3Dmol.view(width=width, height=height)
        for index, model in enumerate(models):
            view.addModel(model["pdb"], "pdb")
            view.setStyle({"model": index}, {"cartoon": {"color": model["colour"]}})
        view.zoomTo()
        showmol(view, height=height, width=width)
    except Exception:
        st.components.v1.html(viewer_html(models, width=width, height=height), height=height + 20)


# --------------------------------------------------------------------------- #
# heatmap
# --------------------------------------------------------------------------- #

def heatmap_matrix(sequence: str) -> np.ndarray | None:
    """(L, 20) masked-marginal probabilities, or None if not cached offline."""
    from src.agent import esm_score

    try:
        return esm_score.masked_marginal_matrix(sequence)
    except esm_score.ScoringUnavailable:
        return None


def heatmap_dataframe(sequence: str, matrix: np.ndarray):
    import pandas as pd

    return pd.DataFrame(
        matrix.T,
        index=list(AA_ALPHABET),
        columns=[f"{aa}{i + 1}" for i, aa in enumerate(sequence)],
    )


# --------------------------------------------------------------------------- #
# score history (used by the multi-generation driver, kept for src/demo.py)
# --------------------------------------------------------------------------- #

def score_history(generations: list[dict], reveal_up_to: int | None = None) -> dict:
    """Public series always; hidden series truncated at the reveal point."""
    iterations = [int(g["iteration"]) for g in generations]
    public = [float(g["candidate_public_median"]) for g in generations]
    hidden = [float(g["candidate_median"]) for g in generations]
    incumbent = [float(g["incumbent_median"]) for g in generations]

    if reveal_up_to is not None:
        cutoff = int(reveal_up_to)
        hidden = [h if it <= cutoff else None for it, h in zip(iterations, hidden)]
        incumbent = [h if it <= cutoff else None for it, h in zip(iterations, incumbent)]

    return {
        "iteration": iterations,
        "public_median": public,
        "hidden_median": hidden,
        "incumbent_hidden_median": incumbent,
    }


def score_history_dataframe(history: dict):
    import pandas as pd

    return pd.DataFrame(history).set_index("iteration")


# --------------------------------------------------------------------------- #
# policy diff
# --------------------------------------------------------------------------- #

def policy_diff(before: dict, after: dict, label: str = "policy.yaml") -> str:
    """Unified diff of the canonical YAML rendering of two policies."""
    return "".join(
        difflib.unified_diff(
            policy_mod.dump_policy(before).splitlines(keepends=True),
            policy_mod.dump_policy(after).splitlines(keepends=True),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
        )
    )


def accepted_patches(generations: list[dict]) -> list[dict]:
    return [g for g in generations if g.get("accepted")]


def rendered_policy_view(policy: dict) -> str:
    """A `policy.py`-shaped rendering for presentation flavour only.

    DISPLAY ONLY. Nothing in this string is ever executed: the deterministic
    interpreter in src/agent/policy_interpreter.py is the only thing that runs,
    and it reads the YAML data, not this text.
    """
    weights = ",\n".join(
        f"        {k!r}: {v}" for k, v in sorted(policy["position_score"].items())
    )
    proposal = ",\n".join(
        f"        {k!r}: {v!r}" for k, v in sorted(policy["proposal"].items())
    )
    return (
        "# DISPLAY ONLY - never executed\n"
        "POSITION_SCORE = {\n" + weights + ",\n}\n\n"
        "PROPOSAL = {\n" + proposal + ",\n}\n"
    )


# --------------------------------------------------------------------------- #
# sequence rendering
# --------------------------------------------------------------------------- #

def sequence_html(
    sequence: str,
    corrupted_positions: set | None = None,
    repaired_positions: set | None = None,
) -> str:
    """Monospace sequence with corrupted sites red and repaired sites green."""
    corrupted_positions = corrupted_positions or set()
    repaired_positions = repaired_positions or set()
    chunks = []
    for i, aa in enumerate(sequence):
        if i in repaired_positions:
            style = "background:#1b7f3b;color:#fff;font-weight:700;border-radius:2px;"
        elif i in corrupted_positions:
            style = "background:#b3261e;color:#fff;font-weight:700;border-radius:2px;"
        else:
            style = "color:#444;"
        chunks.append(f"<span style='{style}padding:1px 2px;'>{aa}</span>")
    return (
        "<div style='font-family:ui-monospace,Consolas,monospace;font-size:15px;"
        "line-height:2.0;word-break:break-all;'>" + "".join(chunks) + "</div>"
    )


# --------------------------------------------------------------------------- #
# the MVP page
# --------------------------------------------------------------------------- #

RESULT_KEY = "mvp_result"


def main() -> None:
    import pandas as pd
    import streamlit as st

    from src import mvp

    st.set_page_config(page_title="ReFold", layout="wide")
    st.title("ReFold")
    st.caption(
        "A falsifiable, self-correcting protein repair harness in which "
        "representations and executable repair rules co-evolve under a hidden "
        "structural verifier."
    )
    # Kept on one line each so the claim and its negation cannot drift apart;
    # tests/test_ui.py and the C1 check in tests/test_constraints.py both read
    # this file line by line.
    st.caption(
        "Scope: this does not design a therapeutic, does not repair function in any organism, and makes no claim to have found new protein physics."
    )

    st.sidebar.header("setup")
    st.sidebar.write(f"fold backend: `{fold_cache.backend()}`")
    st.sidebar.write(
        f"inference: `{'replay only' if fold_cache.offline() else 'live allowed'}`"
        f" (REFOLD_OFFLINE={os.environ.get('REFOLD_OFFLINE', '0')})"
    )
    st.sidebar.write(f"cached structures: {len(list(fold_cache.CACHE.glob('*.pdb')))}")
    st.sidebar.write(f"Gemma transport: `{os.environ.get('REFOLD_GEMMA_MODE', 'mock')}`")
    st.sidebar.write(f"Gemma model: `{os.environ.get('REFOLD_GEMMA_MODEL', 'gemma4:12b')}`")
    reveal = st.sidebar.checkbox("Reveal ground-truth structure", value=False, key=REVEAL_KEY)

    result = st.session_state.get(RESULT_KEY) or mvp.load_result()

    if not result:
        st.info(
            "Press **Run ReFold** at the bottom of the page. The loop runs once: "
            "ESM scores the corrupted protein, the policy proposes mutations, "
            "ESMFold predicts them, the hidden evaluator scores them, Gemma reads "
            "the grounded state and rewrites one part of the policy, and the "
            "interpreter runs again under the new policy."
        )

    left, middle, right = st.columns([1.05, 1.25, 1.15], gap="large")

    # ------------------------------------------------------------------- LEFT
    with left:
        st.subheader("1. The protein")
        if not result:
            st.write("—")
        else:
            corrupted_positions = {s["position"] for s in result["corruption_sites"]}
            chosen = (result["patched"].get("chosen") or result["baseline"].get("chosen"))
            repaired_positions = {chosen["position"]} if chosen else set()

            st.markdown("**Original (ground truth, shown for the demo)**")
            st.markdown(
                sequence_html(result["reference_sequence"]), unsafe_allow_html=True
            )

            st.markdown("**Corrupted input** — red = corrupted site")
            st.markdown(
                sequence_html(result["corrupted_sequence"], corrupted_positions),
                unsafe_allow_html=True,
            )

            st.markdown("**After repair** — green = the mutation ReFold applied")
            repaired_sequence = chosen["sequence"] if chosen else result["corrupted_sequence"]
            st.markdown(
                sequence_html(
                    repaired_sequence,
                    corrupted_positions - repaired_positions,
                    repaired_positions,
                ),
                unsafe_allow_html=True,
            )

            st.markdown("**Corrupted sites**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "site": f"{s['from']}{s['position'] + 1}{s['to']}",
                            "position": s["position"] + 1,
                            "was": s["from"],
                            "now": s["to"],
                        }
                        for s in result["corruption_sites"]
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "The harness is never shown which sites these are — only the "
                "corrupted sequence and its own predicted structure."
            )

    # ----------------------------------------------------------------- MIDDLE
    with middle:
        st.subheader("2. What the harness sees")
        if not result:
            st.write("—")
        else:
            chosen = (result["patched"].get("chosen") or result["baseline"].get("chosen"))
            models = build_viewer_models(
                cached_pdb(result["corrupted_sequence"]),
                cached_pdb(chosen["sequence"]) if chosen else None,
                reveal=reveal,
                protein=result["protein"],
            )
            if models:
                st.caption(
                    " | ".join(f"{m['label']} ({m['badge']})" for m in models)
                )
                show_structures(models)
            else:
                st.info("No cached structure. Run scripts/precompute_mvp.py.")

            if result.get("fold_backend") == "synthetic":
                st.error(SYNTHETIC_WARNING, icon="⚠️")

            st.markdown("**Mutation sensitivity heatmap**")
            matrix = heatmap_matrix(result["corrupted_sequence"])
            if matrix is None:
                st.info("Substitution probabilities not cached.")
            else:
                st.dataframe(
                    heatmap_dataframe(result["corrupted_sequence"], matrix)
                    .style.background_gradient(axis=None, cmap="magma")
                    .format("{:.2f}"),
                    height=300,
                )
            st.caption(HEATMAP_CAPTION)

            st.markdown("**Candidate mutations**")
            for name, key in (("baseline policy", "baseline"), ("Gemma's policy", "patched")):
                round_data = result[key]
                candidates = round_data.get("candidates") or []
                enumerated = round_data.get("enumerated") or []
                if not candidates and not enumerated:
                    continue
                sites = [p + 1 for p in round_data.get("selected_positions", [])]
                st.markdown(
                    f"*{name}* — sites {sites}, "
                    f"{len(enumerated)} mutation(s) proposed, "
                    f"{len(candidates)} folded and scored"
                )
                if enumerated:
                    st.caption(
                        "proposed: " + ", ".join(c["label"] for c in enumerated)
                    )
                if candidates:
                    st.dataframe(
                        pd.DataFrame(candidates)[
                            ["label", "public_score", "hidden_score", "mean_plddt", "esm_score"]
                        ].round(4),
                        hide_index=True,
                        use_container_width=True,
                    )

    # ------------------------------------------------------------------ RIGHT
    with right:
        st.subheader("3. Gemma rewrites the policy")
        if not result:
            st.write("—")
        else:
            gemma = result["gemma"]
            badge = "accepted" if gemma["accepted"] else "REJECTED"
            st.markdown(
                f"**Transport:** `{gemma['source']}` &nbsp;&nbsp; "
                f"**Patch kind:** `{gemma.get('kind')}` &nbsp;&nbsp; "
                f"**Schema:** `{badge}`"
            )
            if gemma.get("rationale"):
                st.markdown("**Gemma's reasoning**")
                st.info(gemma["rationale"])
            if gemma.get("error"):
                st.warning(f"Schema rejected the patch: {gemma['error']}")
            with st.expander("raw model output"):
                st.code(gemma.get("raw") or "(none)", language="json")

            st.markdown("**Policy diff**")
            diff = policy_diff(result["baseline"]["policy"], result["patched"]["policy"])
            st.code(diff or "(policy unchanged)", language="diff")

            st.markdown("**Score comparison**")
            base_chosen = result["baseline"].get("chosen")
            patch_chosen = result["patched"].get("chosen")
            origin_hidden = result["origin_metrics"]["hidden_score"]
            cols = st.columns(3)
            cols[0].metric("corrupted", f"{origin_hidden:.4f}")
            if base_chosen:
                cols[1].metric(
                    "baseline repair",
                    f"{base_chosen['hidden_score']:.4f}",
                    f"{base_chosen['hidden_score'] - origin_hidden:+.4f}",
                )
            if patch_chosen:
                delta_vs_base = (
                    patch_chosen["hidden_score"] - base_chosen["hidden_score"]
                    if base_chosen else 0.0
                )
                cols[2].metric(
                    "after Gemma's patch",
                    f"{patch_chosen['hidden_score']:.4f}",
                    f"{delta_vs_base:+.4f} vs baseline",
                )

            if base_chosen and patch_chosen:
                base_sites = [p + 1 for p in result["baseline"]["selected_positions"]]
                patch_sites = [p + 1 for p in result["patched"]["selected_positions"]]
                n_base = len(result["baseline"].get("enumerated") or [])
                n_patch = len(result["patched"].get("enumerated") or [])
                changed = gemma.get("changed_behaviour")
                st.markdown(
                    f"- Baseline policy: sites **{base_sites}**, "
                    f"**{n_base}** mutations proposed, chose **{base_chosen['label']}**\n"
                    f"- Gemma's policy: sites **{patch_sites}**, "
                    f"**{n_patch}** mutations proposed, chose **{patch_chosen['label']}**\n"
                    f"- Search behaviour "
                    f"{'**changed**' if changed else 'did NOT change'}"
                )

            st.markdown("**Timeline**")
            st.dataframe(
                pd.DataFrame(result["timeline"]),
                hide_index=True,
                use_container_width=True,
                height=260,
            )
            st.caption(f"total {result['elapsed_seconds']:.1f}s")

    # ----------------------------------------------------------------- BUTTON
    st.divider()
    button_col, note_col = st.columns([1, 3])
    with button_col:
        run = st.button("Run ReFold", type="primary", use_container_width=True)
    with note_col:
        st.caption(
            "Runs the full loop once. Structures resolve from cache; Gemma is "
            "called live and takes about a minute on CPU."
        )

    if run:
        status = st.status("Running ReFold…", expanded=True)

        def progress(message: str) -> None:
            status.write(message)

        try:
            outcome = mvp.run(progress=progress)
            st.session_state[RESULT_KEY] = json.loads(outcome.to_json())
            status.update(label="Done", state="complete")
            st.rerun()
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.exception(exc)


if __name__ == "__main__":
    main()

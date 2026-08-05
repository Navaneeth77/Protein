"""Generate the static web demo's data bundle from committed artefacts.

The Vercel site is a *replay*: it cannot run ESMFold or Gemma (see
`web/README.md`), so everything it shows has to be derivable from what is in
this repository. This script is the only thing that produces `web/data/`, and it
draws a hard line between two kinds of number:

  computed  — recomputed here, now, by the project's own code. This covers far
              more than it looks like it should. TM-score, contact recovery,
              sequence recovery, pLDDT and clash counts come from
              src/geometry.py over the committed ESMFold predictions and the
              real 1PGB crystal structure, and need no model at all. The site
              lists, the grounded state and the proposed mutation counts come
              from src/agent/policy_interpreter.py driven by ESM-2, which is a
              130 MB scorer rather than the 8.4 GB folding model.

  recorded  — transcribed from the verified run of 2026-07-30 in DEMO.md. Only
              two things are left in here: the composite hidden score (its ESM
              term is fine, but the score is produced by the withheld evaluator
              against the reference structure, and logs/ is gitignored so the
              run's own JSON was never committed) and Gemma's verbatim
              rationale, which is a model output that cannot be re-derived.

Recomputing the interpreter reproduces DEMO.md exactly — sites [50, 21, 55],
12 proposals, then [50, 21, 55, 48, 11] and 31 — which is the cross-check that
the two categories agree.

Usage:
    python scripts/build_web_data.py
    python scripts/build_web_data.py --allow-missing-esm   # degrade knowingly
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.agent import grounder, policy as policy_mod  # noqa: E402
from src.agent.policy_interpreter import enumerate_candidates, select_positions  # noqa: E402
from src.cache import fold_cache  # noqa: E402
from src.constants import AA_ALPHABET, DEFAULT_PROTEIN  # noqa: E402
from src.geometry import (  # noqa: E402
    contact_recovery,
    contact_set,
    kabsch,
    tm_score_fixed_alignment,
)
from src.paths import corruption_dir, evaluator_sidecar_dir, protein_dir, read_fasta  # noqa: E402
from src.pdb_io import first_chain  # noqa: E402

WEB = ROOT / "web"
OUT_DATA = WEB / "data"
OUT_STRUCTURES = OUT_DATA / "structures"
BUNDLE = OUT_DATA / "demo.json"

VARIANT = "corrupt_01"

# Only what genuinely cannot be re-derived. Everything else is computed below.
RECORDED = {
    "date": "2026-07-30",
    "source": "DEMO.md",
    "model_policy_reviser": "gemma4:12b",
    "total_seconds": 105,
    "hidden_score": {"corrupted": 0.5699, "baseline": 0.8026, "patched": 0.8080},
    "picked": {"baseline": "P50A", "patched": "P50M"},
    "gemma_rationale": (
        "Increasing positions and substitutions per position expands the search "
        "space, allowing for more diverse mutations to be explored."
    ),
    "counterexample_fired": False,
}

# The patch Gemma applied. `positions` is pinned by the recorded site list.
# `substitutions_per_position` is recovered by search below rather than assumed.
PATCHED_POSITIONS = 5
RECORDED_PROPOSAL_COUNTS = {"baseline": 12, "patched": 31}
RECORDED_SITES = {"baseline": [50, 21, 55], "patched": [50, 21, 55, 48, 11]}


def superpose_pdb(pdb_text: str, mobile_ca: np.ndarray, target_ca: np.ndarray) -> str:
    """Rewrite every ATOM/HETATM coordinate into the reference's frame.

    ESMFold emits each prediction in its own arbitrary frame, so drawing a
    prediction and the crystal structure in one scene puts them side by side
    rather than on top of each other — an overlay that looks like disagreement
    but is only a choice of origin. The rotation comes from the project's own
    Kabsch implementation over CA atoms.

    Display-only, and it changes no reported number: TM-score, RMSD and the
    contact set are invariant under rigid motion, and the metrics in this bundle
    are computed from the untransformed coordinates regardless.
    """
    rot, trans = kabsch(mobile_ca, target_ca)
    out = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
            out.append(line)
            continue
        xyz = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float
        )
        moved = rot @ xyz + trans
        out.append(f"{line[:30]}{moved[0]:8.3f}{moved[1]:8.3f}{moved[2]:8.3f}{line[54:]}")
    return "\n".join(out) + "\n"


def _fold_from_cache(sequence: str):
    """FoldResult for a sequence whose structure is committed in data/cache/."""
    path = fold_cache.cache_path(sequence)
    if not path.exists():
        raise FileNotFoundError(
            f"no committed structure for {fold_cache.sequence_hash(sequence)}; "
            f"this demo requires it to be in data/cache/"
        )
    text = path.read_text(encoding="utf-8")
    return fold_cache.structure_features(sequence, text, from_cache=True), text


def _metrics(fold, ref_ca, ref_contacts, ref_seq, origin_seq):
    """The hidden evaluator's decomposition minus its ESM term.

    The composite hidden_score is not reproduced here — it belongs to the
    withheld evaluator. See src/evaluator.py:combine.
    """
    return {
        "tm_score": round(float(tm_score_fixed_alignment(fold.ca_coords, ref_ca)), 4),
        "contact_recovery": round(float(contact_recovery(fold.contacts, ref_contacts)), 4),
        "sequence_recovery": round(
            sum(1 for a, b in zip(fold.sequence, ref_seq) if a == b) / len(ref_seq), 4
        ),
        "mean_plddt": round(float(fold.mean_plddt), 4),
        "clashes": int(fold.clashes),
        "radius_of_gyration": round(float(fold.radius_of_gyration), 2),
        "edit_count": sum(1 for a, b in zip(origin_seq, fold.sequence) if a != b),
        "plddt_per_residue": [round(float(v), 4) for v in fold.plddt],
        "helices": fold.helices,
        "strands": fold.strands,
        "synthetic": bool(fold.synthetic),
    }


def esm_available(sequences) -> bool:
    """True if every sequence can be scored without a live model download."""
    from src.agent import esm_score

    return all(esm_score._cache_path(s).exists() for s in sequences)


def recover_substitutions_per_position(policy, state, origin, target_count):
    """Find the smallest substitutions_per_position reproducing the recorded count.

    Gemma's rationale says it raised this parameter but the value itself lived
    only in logs/, which is gitignored. Rather than guess it, re-run the real
    interpreter across the DSL's legal range and report which values reproduce
    the recorded proposal count. The residue-class filter saturates, so more
    than one value can match; the smallest is reported as the patch and the full
    set is carried so the page can say the recovery is not unique.
    """
    matches = []
    for spp in range(1, 20):
        trial = policy_mod.clone(policy)
        trial["proposal"]["substitutions_per_position"] = spp
        n = len(enumerate_candidates(trial, state, origin=origin))
        if n == target_count:
            matches.append(spp)
    return (matches[0] if matches else None), matches


def build(allow_missing_esm: bool = False) -> dict:
    protein = DEFAULT_PROTEIN

    _, native_seq = read_fasta(protein_dir(protein) / "native_seq.fasta")
    _, corrupt_seq = read_fasta(corruption_dir(protein) / f"{VARIANT}.fasta")

    ref_chain = first_chain(protein_dir(protein) / "native.pdb")
    ref_ca = ref_chain.ca_coords()
    ref_contacts = contact_set(ref_chain.cb_coords())
    if ref_chain.sequence != native_seq:
        raise ValueError("reference FASTA and native.pdb disagree")

    def mutate(seq: str, label: str) -> str:
        from_aa, pos_1based, to_aa = label[0], int(label[1:-1]), label[-1]
        i = pos_1based - 1
        if seq[i] != from_aa:
            raise ValueError(f"{label}: expected {from_aa} at {pos_1based}, found {seq[i]}")
        return seq[:i] + to_aa + seq[i + 1:]

    baseline_seq = mutate(corrupt_seq, RECORDED["picked"]["baseline"])
    patched_seq = mutate(corrupt_seq, RECORDED["picked"]["patched"])

    OUT_STRUCTURES.mkdir(parents=True, exist_ok=True)

    folds = {}
    stages = {}
    for key, seq, label in (
        ("corrupted", corrupt_seq, "corrupted input"),
        ("baseline", baseline_seq, f"baseline repair ({RECORDED['picked']['baseline']})"),
        ("patched", patched_seq, f"patched repair ({RECORDED['picked']['patched']})"),
    ):
        fold, pdb_text = _fold_from_cache(seq)
        if fold.synthetic:
            raise RuntimeError(
                f"{key} resolved to a synthetic fixture, not an ESMFold prediction; "
                f"refusing to publish it as a result"
            )
        folds[key] = fold
        (OUT_STRUCTURES / f"{key}.pdb").write_text(
            superpose_pdb(pdb_text, fold.ca_coords, ref_ca), encoding="utf-8"
        )
        stages[key] = {
            "label": label,
            "sequence": seq,
            "structure": f"data/structures/{key}.pdb",
            "cache_hash": fold_cache.sequence_hash(seq),
            "computed": _metrics(fold, ref_ca, ref_contacts, native_seq, corrupt_seq),
            "recorded_hidden_score": RECORDED["hidden_score"][key],
        }

    (OUT_STRUCTURES / "native.pdb").write_text(
        (protein_dir(protein) / "native.pdb").read_text(encoding="utf-8"), encoding="utf-8"
    )

    corrupt_sidecar = json.loads(
        (evaluator_sidecar_dir(protein) / "corrupt_positions.json").read_text(encoding="utf-8")
    )
    seed_policy = policy_mod.load_seed_policy()

    # --------------------------------------------------------------- ESM tier
    wanted = [corrupt_seq, baseline_seq, patched_seq]
    have_esm = esm_available(wanted)
    if not have_esm and not allow_missing_esm:
        raise SystemExit(
            "ESM-2 scores are not cached for every demo sequence, so the grounded\n"
            "state, the substitution heatmap and the interactive policy editor\n"
            "cannot be regenerated. Warm the cache first (needs torch +\n"
            "transformers; the scorer is ~130 MB, not the 8.4 GB folder):\n\n"
            "    python -c \"import sys; sys.path.insert(0,'.');"
            "from src.agent import esm_score; from src.paths import corruption_dir, read_fasta;"
            "esm_score.masked_marginal_matrix(read_fasta(corruption_dir('1pgb')/'corrupt_01.fasta')[1])\"\n\n"
            "Or pass --allow-missing-esm to publish a bundle without them. That\n"
            "silently removes interactive features, so it is not the default."
        )

    esm_block = None
    interpreter = None
    if have_esm:
        from src.agent import esm_score

        state = grounder.ground(corrupt_seq, folds["corrupted"])

        seed_sites = select_positions(seed_policy, state)
        seed_cands = enumerate_candidates(seed_policy, state, origin=corrupt_seq)

        patched_policy = policy_mod.clone(seed_policy)
        patched_policy["proposal"]["positions"] = PATCHED_POSITIONS
        spp, spp_matches = recover_substitutions_per_position(
            patched_policy, state, corrupt_seq, RECORDED_PROPOSAL_COUNTS["patched"]
        )
        if spp is None:
            raise RuntimeError(
                "no substitutions_per_position in the DSL's range reproduces the "
                f"recorded {RECORDED_PROPOSAL_COUNTS['patched']} proposals"
            )
        patched_policy["proposal"]["substitutions_per_position"] = spp
        patched_sites = select_positions(patched_policy, state)
        patched_cands = enumerate_candidates(patched_policy, state, origin=corrupt_seq)

        # Cross-check the recomputation against what DEMO.md wrote down. A
        # mismatch means the two provenance tiers disagree and the page would be
        # telling two different stories, so fail rather than publish.
        checks = {
            "baseline_sites": ([p + 1 for p in seed_sites], RECORDED_SITES["baseline"]),
            "patched_sites": ([p + 1 for p in patched_sites], RECORDED_SITES["patched"]),
            "baseline_count": (len(seed_cands), RECORDED_PROPOSAL_COUNTS["baseline"]),
            "patched_count": (len(patched_cands), RECORDED_PROPOSAL_COUNTS["patched"]),
        }
        for name, (got, want) in checks.items():
            if got != want:
                raise RuntimeError(
                    f"recomputed {name} = {got!r} but DEMO.md records {want!r}"
                )
        for which, cands in (("baseline", seed_cands), ("patched", patched_cands)):
            labels = {c.label() for c in cands}
            if RECORDED["picked"][which] not in labels:
                raise RuntimeError(
                    f"{RECORDED['picked'][which]} is not among the {which} proposals"
                )

        def cand_json(cands):
            return [
                {
                    "label": c.label(),
                    "position": c.position,
                    "from": c.from_aa,
                    "to": c.to_aa,
                    "substitution_prob": round(float(c.substitution_prob), 6),
                    "position_score": round(float(c.position_score), 6),
                    "edit_count": c.edit_count,
                }
                for c in cands
            ]

        interpreter = {
            "baseline": {
                "sites_1based": [p + 1 for p in seed_sites],
                "proposals": len(seed_cands),
                "candidates": cand_json(seed_cands),
            },
            "patched": {
                "sites_1based": [p + 1 for p in patched_sites],
                "proposals": len(patched_cands),
                "candidates": cand_json(patched_cands),
                "substitutions_per_position_recovered": spp,
                "substitutions_per_position_all_matches": spp_matches,
            },
            "state": {
                "residues": [
                    {
                        "position": r["position"],
                        "aa": r["aa"],
                        "esm_surprisal": r["esm_surprisal"],
                        "low_plddt": r["low_plddt"],
                        "contact_violation": r["contact_violation"],
                        "long_range_contact_violation": r["long_range_contact_violation"],
                    }
                    for r in state["residues"]
                ],
            },
        }

        esm_block = {
            "alphabet": AA_ALPHABET,
            "model": "facebook/esm2_t12_35M_UR50D",
            "browser_model": "Xenova/esm2_t12_35M_UR50D",
            "matrices": {},
            "surprisal": {},
        }
        for key, seq in (
            ("corrupted", corrupt_seq),
            ("baseline", baseline_seq),
            ("patched", patched_seq),
        ):
            mat = esm_score.masked_marginal_matrix(seq)
            esm_block["matrices"][key] = [[round(float(v), 5) for v in row] for row in mat]
            esm_block["surprisal"][key] = [
                round(float(v), 5) for v in esm_score.residue_surprisal(seq)
            ]

        patched_policy_out = patched_policy
    else:
        patched_policy_out = policy_mod.clone(seed_policy)
        patched_policy_out["proposal"]["positions"] = PATCHED_POSITIONS
        patched_policy_out["proposal"]["substitutions_per_position"] = None

    return {
        "protein": protein.upper(),
        "variant": VARIANT,
        "generated_from": "committed artefacts only; see scripts/build_web_data.py",
        "native": {
            "sequence": native_seq,
            "structure": "data/structures/native.pdb",
            "length": len(native_seq),
            "contacts": len(ref_contacts),
        },
        "corruptions": corrupt_sidecar["variants"][VARIANT],
        "stages": stages,
        "policy": {
            "seed": seed_policy,
            "patched": patched_policy_out,
            "schema": {
                "scorable_features": list(grounder.SCORABLE_FEATURES),
                "positions": [1, 10],
                "substitutions_per_position": [1, 19],
                "max_total_edits": [1, 10],
            },
        },
        "interpreter": interpreter,
        "esm": esm_block,
        "recorded": RECORDED,
        "evaluator_weights": {
            "tm_score": 0.55,
            "esm_score": 0.20,
            "plddt": 0.15,
            "contact_recovery": 0.10,
            "edit_fraction": -0.05,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--allow-missing-esm",
        action="store_true",
        help="publish without ESM-derived data instead of failing (drops interactivity)",
    )
    args = ap.parse_args()

    bundle = build(allow_missing_esm=args.allow_missing_esm)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    BUNDLE.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {BUNDLE.relative_to(ROOT)}  ({BUNDLE.stat().st_size / 1024:.0f} KB)")
    for key, stage in bundle["stages"].items():
        c = stage["computed"]
        print(
            f"  {key:10s} TM={c['tm_score']:.4f} contacts={c['contact_recovery']:.4f} "
            f"pLDDT={c['mean_plddt']:.4f} clashes={c['clashes']}"
        )
    it = bundle["interpreter"]
    if it:
        print(
            f"  interpreter reproduces DEMO.md: "
            f"baseline {it['baseline']['sites_1based']} n={it['baseline']['proposals']}, "
            f"patched {it['patched']['sites_1based']} n={it['patched']['proposals']} "
            f"(substitutions_per_position={it['patched']['substitutions_per_position_recovered']}, "
            f"also matched by {it['patched']['substitutions_per_position_all_matches']})"
        )
    else:
        print("  interpreter/esm: OMITTED (--allow-missing-esm)")
    print(f"wrote {OUT_STRUCTURES.relative_to(ROOT)}/*.pdb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

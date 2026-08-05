"""Generate the static web demo's data bundle from committed artefacts.

The Vercel site is a *replay*: it cannot run ESMFold or Gemma (see
`web/README.md`), so everything it shows has to be derivable from what is in
this repository. This script is the only thing that produces `web/data/`, and
it draws a hard line between two kinds of number:

  computed  — recomputed here, now, by the project's own code (src/geometry.py,
              src/pdb_io.py, src/cache/fold_cache.py) from the committed ESMFold
              predictions in data/cache/ and the real 1PGB crystal structure in
              data/proteins/. TM-score, contact recovery, sequence recovery,
              pLDDT and clash counts all fall in here. None of them need torch.

  recorded  — transcribed from the verified run of 2026-07-30 written down in
              DEMO.md. The hidden score and Gemma's reasoning are here because
              the hidden score's ESM term needs the model, and logs/ is
              gitignored so the run's own JSON was never committed.

Every field in the emitted JSON carries its provenance, and the page renders the
two differently. Nothing is invented: a quantity that is neither computable nor
recorded is emitted as null and the page says so.

Usage:
    python scripts/build_web_data.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.agent import policy as policy_mod  # noqa: E402
from src.cache import fold_cache  # noqa: E402
from src.constants import DEFAULT_PROTEIN  # noqa: E402
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

VARIANT = "corrupt_01"

# The 2026-07-30 run, transcribed from DEMO.md. Anything here is labelled
# "recorded" in the UI and is never presented as if it were computed now.
RECORDED = {
    "date": "2026-07-30",
    "source": "DEMO.md",
    "model_policy_reviser": "gemma4:12b",
    "total_seconds": 105,
    "hidden_score": {"corrupted": 0.5699, "baseline": 0.8026, "patched": 0.8080},
    "proposals": {"baseline": 12, "patched": 31},
    "sites_1based": {"baseline": [50, 21, 55], "patched": [50, 21, 55, 48, 11]},
    "picked": {"baseline": "P50A", "patched": "P50M"},
    "gemma_rationale": (
        "Increasing positions and substitutions per position expands the search "
        "space, allowing for more diverse mutations to be explored."
    ),
    "counterexample_fired": False,
}


def superpose_pdb(pdb_text: str, mobile_ca: np.ndarray, target_ca: np.ndarray) -> str:
    """Rewrite every ATOM/HETATM coordinate into the reference's frame.

    ESMFold emits each prediction in its own arbitrary frame, so drawing a
    prediction and the crystal structure in one scene puts them side by side
    rather than on top of each other — an overlay that looks like disagreement
    but is only a choice of origin. The rotation comes from the project's own
    Kabsch implementation over CA atoms.

    This is display-only and changes no reported number: TM-score, RMSD and the
    contact set are all invariant under rigid motion, and the metrics in this
    bundle are computed from the untransformed coordinates regardless.
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
        out.append(
            f"{line[:30]}{moved[0]:8.3f}{moved[1]:8.3f}{moved[2]:8.3f}{line[54:]}"
        )
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
    """Ground-truth-derived metrics, recomputed now from committed structures.

    This is the evaluator's decomposition minus its ESM term. The ESM term needs
    facebook/esm2_t12_35M_UR50D, and data/cache/esm_score/ is gitignored, so the
    composite hidden_score cannot be reproduced here — it is carried as a
    recorded value instead. See src/evaluator.py:combine.
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


def build() -> dict:
    protein = DEFAULT_PROTEIN

    _, native_seq = read_fasta(protein_dir(protein) / "native_seq.fasta")
    _, corrupt_seq = read_fasta(corruption_dir(protein) / f"{VARIANT}.fasta")

    ref_chain = first_chain(protein_dir(protein) / "native.pdb")
    ref_ca = ref_chain.ca_coords()
    ref_contacts = contact_set(ref_chain.cb_coords())

    if ref_chain.sequence != native_seq:
        raise ValueError("reference FASTA and native.pdb disagree")

    # The two repairs the recorded run picked, rebuilt from the corrupted
    # sequence rather than hard-coded, so a wrong label cannot slip through.
    def mutate(seq: str, label: str) -> str:
        from_aa, pos_1based, to_aa = label[0], int(label[1:-1]), label[-1]
        i = pos_1based - 1
        if seq[i] != from_aa:
            raise ValueError(f"{label}: expected {from_aa} at position {pos_1based}, found {seq[i]}")
        return seq[:i] + to_aa + seq[i + 1:]

    baseline_seq = mutate(corrupt_seq, RECORDED["picked"]["baseline"])
    patched_seq = mutate(corrupt_seq, RECORDED["picked"]["patched"])

    OUT_STRUCTURES.mkdir(parents=True, exist_ok=True)

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
        filename = f"{key}.pdb"
        # Metrics below come from `fold`, i.e. the untransformed coordinates.
        # Only the published file is moved into the reference frame.
        (OUT_STRUCTURES / filename).write_text(
            superpose_pdb(pdb_text, fold.ca_coords, ref_ca), encoding="utf-8"
        )
        stages[key] = {
            "label": label,
            "sequence": seq,
            "structure": f"data/structures/{filename}",
            "cache_hash": fold_cache.sequence_hash(seq),
            "computed": _metrics(fold, ref_ca, ref_contacts, native_seq, corrupt_seq),
            "recorded_hidden_score": RECORDED["hidden_score"][key],
        }

    # The withheld reference. Committed, but gated behind an explicit toggle in
    # the UI exactly as the Streamlit app gates it.
    native_pdb = (protein_dir(protein) / "native.pdb").read_text(encoding="utf-8")
    (OUT_STRUCTURES / "native.pdb").write_text(native_pdb, encoding="utf-8")

    corrupt_sidecar = json.loads(
        (evaluator_sidecar_dir(protein) / "corrupt_positions.json").read_text(encoding="utf-8")
    )

    seed_policy = policy_mod.load_seed_policy()
    # The patch the recorded run applied. `positions` is directly evidenced by
    # the site lists (3 -> 5). substitutions_per_position is known to have risen
    # from Gemma's own rationale, but its exact value was only in logs/, which is
    # gitignored — so it is emitted as null and the page says "increased".
    patched_policy = policy_mod.clone(seed_policy)
    patched_policy["proposal"]["positions"] = len(RECORDED["sites_1based"]["patched"])
    patched_policy["proposal"]["substitutions_per_position"] = None

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
            "seed_yaml": policy_mod.dump_policy(seed_policy),
            "patched": patched_policy,
            "seed": seed_policy,
        },
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
    bundle = build()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    out = OUT_DATA / "demo.json"
    out.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out.relative_to(ROOT)}")
    for key, stage in bundle["stages"].items():
        c = stage["computed"]
        print(
            f"  {key:10s} TM={c['tm_score']:.4f} "
            f"contacts={c['contact_recovery']:.4f} "
            f"pLDDT={c['mean_plddt']:.4f} clashes={c['clashes']}"
        )
    print(f"wrote {OUT_STRUCTURES.relative_to(ROOT)}/*.pdb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

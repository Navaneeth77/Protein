"""P6.2 — ablation: ESM-only vs the contact-aware evolved policy.

Runs the inner loop over the same variants under two policies:
  ESM-only       — contact terms zeroed out, weight moved to esm_surprisal/low_plddt
  contact-aware  — the evolved policy from logs/demo_state.json (or the seed)

Both medians are printed regardless of which wins. If ESM-only wins, that is the
result; it does not get massaged, and the pitch simply does not claim
contact-awareness helps.

Usage:
    python scripts/research/ablation.py
    python scripts/research/ablation.py --protein 1pgb --variants holdout_01
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agent import inner_loop, policy as policy_mod  # noqa: E402
from src.constants import DEFAULT_PROTEIN  # noqa: E402
from src.demo import DEMO_STATE, HOLDOUT_VARIANT, load_variants  # noqa: E402
from src.evaluator import HiddenEvaluator  # noqa: E402
from src.paths import LOGS  # noqa: E402

REPORT = LOGS / "ablation.json"

# The ESM-only arm: every contact term at zero. Weight has to go somewhere for the
# sum-to-1.0 rule, and moving it to the two non-contact terms in their original
# 3:1 ratio keeps the comparison about contact-awareness and nothing else.
ESM_ONLY_WEIGHTS = {"esm_surprisal": 0.75, "low_plddt": 0.25}


def contact_aware_policy() -> tuple[dict, str]:
    if DEMO_STATE.exists():
        state = json.loads(DEMO_STATE.read_text(encoding="utf-8"))
        return policy_mod.validate_policy(state["final_policy"]), "evolved"
    return policy_mod.load_seed_policy(), "seed"


def esm_only_policy(reference: dict) -> dict:
    ablated = policy_mod.clone(reference)
    ablated["position_score"] = dict(ESM_ONLY_WEIGHTS)
    return policy_mod.validate_policy(ablated)


def run_arm(label: str, policy: dict, variants: dict, protein: str) -> dict:
    evaluator = HiddenEvaluator(protein=protein)
    scores, rows = [], []
    for name, sequence in variants.items():
        evaluator.set_origin(sequence)
        result = inner_loop.repair(sequence, policy, evaluator)
        scores.append(result.hidden_score)
        rows.append(
            {
                "variant": name,
                "hidden_score": result.hidden_score,
                "hidden_delta": result.hidden_delta,
                "edits": [
                    f"{m['from']}{m['position'] + 1}{m['to']}" for m in result.mutations
                ],
            }
        )
        print(
            f"  {label:<14} {name:<12} hidden {result.hidden_score:.4f} "
            f"(delta {result.hidden_delta:+.4f})"
        )
    return {
        "label": label,
        "policy": policy,
        "median_hidden_score": float(statistics.median(scores)),
        "runs": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", default=DEFAULT_PROTEIN)
    ap.add_argument("--variants", nargs="*", default=None,
                    help="variant names (default: the holdout plus corrupt_04/05)")
    args = ap.parse_args()

    names = args.variants or [HOLDOUT_VARIANT, "corrupt_04", "corrupt_05"]
    variants = load_variants(args.protein, names)
    if not variants:
        raise SystemExit(f"none of {names} exist; run scripts/make_corruptions.py")

    aware, provenance = contact_aware_policy()
    ablated = esm_only_policy(aware)

    print(f"variants: {', '.join(variants)}")
    print(f"contact-aware policy source: {provenance}")
    print(f"  contact-aware weights: {aware['position_score']}")
    print(f"  ESM-only weights:      {ablated['position_score']}\n")

    esm_arm = run_arm("esm-only", ablated, variants, args.protein)
    print()
    aware_arm = run_arm("contact-aware", aware, variants, args.protein)

    print("\n" + "=" * 58)
    print(f"ESM-only      median hidden score: {esm_arm['median_hidden_score']:.4f}")
    print(f"contact-aware median hidden score: {aware_arm['median_hidden_score']:.4f}")
    delta = aware_arm["median_hidden_score"] - esm_arm["median_hidden_score"]
    print(f"difference (contact-aware - ESM-only): {delta:+.4f}")
    print("=" * 58)

    supports_claim = delta > 0
    if supports_claim:
        print("\nContact-awareness helps on this set. The ablation may go in the pitch.")
    else:
        print(
            "\nContact-awareness does NOT help on this set. That is the finding; "
            "the pitch must not claim otherwise, and the ablation should be "
            "reported as a negative result or left out."
        )

    report = {
        "protein": args.protein,
        "variants": sorted(variants),
        "contact_aware_policy_source": provenance,
        "esm_only": esm_arm,
        "contact_aware": aware_arm,
        "difference": delta,
        "supports_contact_awareness_claim": supports_claim,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

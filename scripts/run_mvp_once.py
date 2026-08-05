"""Run the MVP flow once from the command line and print a compact summary.

Same code path the Streamlit button uses. Handy for verifying the demo works
before presenting, and for warming the Gemma model.

Usage:
    python scripts/run_mvp_once.py
    python scripts/run_mvp_once.py --mock      # skip Gemma, use a canned patch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import mvp  # noqa: E402

CANNED = json.dumps(
    {
        "kind": "mechanism",
        "rationale": "Widen the site search so more candidate positions are examined.",
        "proposal": {"positions": 5},
    }
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use a canned patch, no model call")
    ap.add_argument("--variant", default=mvp.DEFAULT_VARIANT)
    args = ap.parse_args()

    transport = (lambda prompt: CANNED) if args.mock else None
    result = mvp.run(variant=args.variant, gemma_transport=transport)
    data = json.loads(result.to_json())

    base = data["baseline"]
    patch = data["patched"]
    gemma = data["gemma"]

    print("\n" + "=" * 66)
    print(f"corrupted hidden score : {data['origin_metrics']['hidden_score']:.4f}")
    print(f"corrupted sites        : "
          f"{[f'{s['from']}{s['position'] + 1}{s['to']}' for s in data['corruption_sites']]}")
    print("-" * 66)
    print(f"gemma transport        : {gemma['source']}")
    print(f"gemma patch kind       : {gemma.get('kind')}")
    print(f"gemma accepted         : {gemma['accepted']}  (attempts {gemma.get('attempts')})")
    if gemma.get("error"):
        print(f"gemma error            : {gemma['error']}")
    if gemma.get("rationale"):
        print(f"gemma rationale        : {gemma['rationale'][:200]}")
    print("-" * 66)
    print(f"baseline sites         : {[p + 1 for p in base['selected_positions']]}")
    if base.get("chosen"):
        print(f"baseline pick          : {base['chosen']['label']}  "
              f"hidden {base['chosen']['hidden_score']:.4f}")
    print(f"patched sites          : {[p + 1 for p in patch['selected_positions']]}")
    if patch.get("chosen"):
        print(f"patched pick           : {patch['chosen']['label']}  "
              f"hidden {patch['chosen']['hidden_score']:.4f}")

    if base.get("chosen") and patch.get("chosen"):
        changed = base["chosen"]["sequence"] != patch["chosen"]["sequence"]
        sites_changed = base["selected_positions"] != patch["selected_positions"]
        print("-" * 66)
        print(f"selected sites changed : {sites_changed}")
        print(f"chosen mutation changed: {changed}")
        print(f"hidden delta vs baseline: "
              f"{patch['chosen']['hidden_score'] - base['chosen']['hidden_score']:+.4f}")
    print("=" * 66)
    print(f"total {data['elapsed_seconds']:.1f}s; wrote {mvp.RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

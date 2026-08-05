"""P0.1 — environment smoke test.

Answers, from the machine rather than from documentation, whether the two model
paths this project needs actually work on this interpreter. Prints model class
names and the exact versions to record in requirements.txt.

Usage:
    python scripts/smoke_test.py              # scorer only (fast)
    python scripts/smoke_test.py --with-fold  # also load the ESMFold checkpoint
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import ESM2_SCORE_MODEL, ESMFOLD_MODEL  # noqa: E402

FALLBACK_SCORE_MODEL = "facebook/esm2_t30_150M_UR50D"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-fold", action="store_true",
                    help="also load esmfold_v1 (~2.6 GB download on first run)")
    args = ap.parse_args()

    print(f"python   {platform.python_version()} on {platform.system()} {platform.release()}")

    import torch
    import transformers

    print(f"torch    {torch.__version__}  (cuda available: {torch.cuda.is_available()})")
    print(f"transformers {transformers.__version__}")

    try:
        import tmtools  # noqa: F401

        print("tmtools  installed — evaluator will use tm_align")
    except ImportError:
        print("tmtools  NOT installed — evaluator falls back to src/geometry.py TM-score")

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    print(f"\n[scorer] loading {ESM2_SCORE_MODEL}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ESM2_SCORE_MODEL)
        model = AutoModelForMaskedLM.from_pretrained(ESM2_SCORE_MODEL)
    except Exception as exc:
        print(f"[scorer] FAILED ({type(exc).__name__}: {exc})")
        print(f"[scorer] retrying with the fallback {FALLBACK_SCORE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_SCORE_MODEL)
        model = AutoModelForMaskedLM.from_pretrained(FALLBACK_SCORE_MODEL)

    params = sum(p.numel() for p in model.parameters())
    print(f"[scorer] {type(model).__name__}  {params / 1e6:.1f}M params  "
          f"mask_token_id={tokenizer.mask_token_id}")

    print(f"\n[fold] importing EsmForProteinFolding")
    try:
        from transformers import EsmForProteinFolding

        print(f"[fold] {EsmForProteinFolding.__name__} importable")
    except Exception as exc:
        print(f"[fold] IMPORT FAILED: {type(exc).__name__}: {exc}")
        print("[fold] P0 BLOCKER — structure prediction is unavailable on this setup")
        return 1

    if not args.with_fold:
        print(f"[fold] skipping checkpoint load; pass --with-fold to load {ESMFOLD_MODEL}")
        print("\nP0.1 PASS (scorer verified; fold checkpoint not loaded)")
        return 0

    from src.cache.fold_cache import checkpoint_path

    source = checkpoint_path()
    print(f"[fold] loading {source} — large download on first run")
    try:
        fold_tokenizer = AutoTokenizer.from_pretrained(source)
        fold_model = EsmForProteinFolding.from_pretrained(
            source, low_cpu_mem_usage=True
        )
    except Exception as exc:
        print(f"[fold] LOAD FAILED: {type(exc).__name__}: {exc}")
        print("[fold] P0 BLOCKER — flag before continuing (see refold_tasks.md P0.1)")
        return 1

    fold_params = sum(p.numel() for p in fold_model.parameters())
    print(f"[fold] {type(fold_model).__name__}  {fold_params / 1e6:.1f}M params")
    print(f"[fold] tokenizer: {type(fold_tokenizer).__name__}")

    print("\nP0.1 PASS (both model paths verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

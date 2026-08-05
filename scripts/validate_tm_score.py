"""Cross-validate src/geometry.py's TM-score against the reference tmtools.

Why this script exists: tmtools ships no cp313 wheel and its sdist needs MSVC, so
the main environment (Python 3.13) uses the TM-score implementation in
src/geometry.py. That implementation is only trustworthy if it agrees with the
real thing, so this script runs both and reports the deviation.

Run it with an interpreter that HAS tmtools installed (here: Python 3.12):

    C:\\Python312\\python.exe scripts/validate_tm_score.py

It imports nothing from src/ except geometry and pdb_io, both of which are pure
numpy, so it works on any interpreter with numpy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.geometry import tm_score_fixed_alignment  # noqa: E402
from src.pdb_io import first_chain  # noqa: E402

# Agreement required in the regime the evaluator actually operates in. Every
# candidate it scores is a single-point mutant of a 56-residue protein, which sits
# at TM ~0.85-0.98. Below RELEVANT_TM our fragment search is known to underestimate
# by up to ~0.06 because it does not reproduce TM-align's multi-cutoff refinement
# schedule; that is reported but does not fail the check, because a structure that
# far from the reference is already scored "bad" either way.
TOLERANCE = 0.02
RELEVANT_TM = 0.5


def perturb(coords: np.ndarray, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return coords + rng.normal(scale=scale, size=coords.shape)


def rotate_and_shift(coords: np.ndarray, angle: float) -> np.ndarray:
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return (rotation @ coords.T).T + np.array([12.0, -7.0, 3.0])


def main() -> int:
    try:
        import tmtools
    except ImportError:
        print("tmtools is not installed for this interpreter.")
        print("Run this with an interpreter that has it, e.g. C:\\Python312\\python.exe")
        return 2

    reference_pdb = ROOT / "data" / "proteins" / "1pgb" / "native.pdb"
    if not reference_pdb.exists():
        print(f"missing {reference_pdb}; run scripts/fetch_protein.py first")
        return 1

    chain = first_chain(reference_pdb)
    reference = chain.ca_coords().astype(np.float64)
    sequence = chain.sequence
    print(f"reference: 1pgb, {len(sequence)} residues\n")

    cases = [("identical", reference.copy()), ("rigid motion", rotate_and_shift(reference, 0.9))]
    for scale in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        cases.append((f"noise sigma={scale}", perturb(reference, scale, seed=int(scale * 10))))

    print(f"{'case':<18} {'tmtools':>9} {'geometry.py':>12} {'abs diff':>9}  status")
    print("-" * 64)

    worst_relevant = 0.0
    worst_overall = 0.0
    failures = 0
    for label, candidate in cases:
        result = tmtools.tm_align(candidate, reference, sequence, sequence)
        # Argument order (candidate, reference) means chain 2 is the reference,
        # so tm_norm_chain2 is the reference-length normalisation the evaluator
        # pins. Same choice as src/evaluator.py::_tm_score.
        expected = float(result.tm_norm_chain2)
        actual = tm_score_fixed_alignment(candidate, reference)
        diff = abs(actual - expected)
        worst_overall = max(worst_overall, diff)

        if expected >= RELEVANT_TM:
            worst_relevant = max(worst_relevant, diff)
            status = "ok" if diff <= TOLERANCE else "FAIL"
            failures += 0 if diff <= TOLERANCE else 1
        else:
            status = "below-range"
        print(f"{label:<18} {expected:9.5f} {actual:12.5f} {diff:9.5f}  {status}")

    print("-" * 64)
    print(f"worst deviation at TM >= {RELEVANT_TM}: {worst_relevant:.5f} "
          f"(tolerance {TOLERANCE})")
    print(f"worst deviation overall:      {worst_overall:.5f} "
          f"(below-range cases are informational)")

    # Also confirm the normalisation convention is what the evaluator assumes.
    result = tmtools.tm_align(reference, reference, sequence, sequence)
    print(
        f"\nequal-length sanity: tm_norm_chain1={result.tm_norm_chain1:.5f} "
        f"tm_norm_chain2={result.tm_norm_chain2:.5f}"
    )

    if failures:
        print(f"\nFAIL: {failures} case(s) outside tolerance in the relevant range")
        return 1
    print(
        f"\nPASS: src/geometry.py agrees with tmtools within {TOLERANCE} for every "
        f"case at TM >= {RELEVANT_TM}, which is the range the evaluator scores in."
    )
    print(
        "Known bias below that range: this search underestimates TM by up to ~0.06 "
        "because it does not reproduce TM-align's multi-cutoff refinement."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

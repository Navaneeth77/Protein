# Contributing

## Setup

```powershell
pip install -r requirements.txt
ollama pull gemma4:12b
python scripts/fetch_esmfold.py        # ~8.4 GB, resumable
python scripts/smoke_test.py --with-fold
```

## Running the checks

```powershell
python -m pytest tests/ -q                    # everything (167 tests)
python -m pytest tests/ -q -m "not models"    # skip anything needing torch
python -m pytest tests/test_constraints.py -q # just the safety boundary
```

Tests come in two tiers. The fast tier stubs ESM and folds from a synthetic helix,
so it needs no models and no data. The `@pytest.mark.models` tier loads real
checkpoints and skips with a clear reason when they are absent.

## The rules that matter

Four constraints are enforced by `tests/test_constraints.py`, and breaking one fails
the suite. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before touching the
agent or the evaluator, but in short:

1. **Nothing under `src/agent/` may see ground truth.** No file there may contain
   the string `native` or `SEQRES`, import the evaluator, or reference
   `data/proteins/`. If you need a new signal, compute it from the candidate
   sequence or its *predicted* structure.
2. **`src/evaluator.py` is the only reader of `data/proteins/`.** It is also the
   only file allowed to contain `native_pdb_path`.
3. **No dynamic execution on the policy path.** No `eval(`, `exec(`,
   `__import__`, `subprocess` or `os.system` under `src/` or `app/`. A policy is
   data; the interpreter reads named fields from it and nothing else.
4. **`reveal=True` is passed from one script only**
   (`scripts/research/final_reveal.py`).

## Adding a state feature

The policy DSL can only weight fields the grounder actually computes. To add one:

1. Compute it in `src/agent/grounder.py::ground` from the sequence and predicted
   structure.
2. Add it to `SCORABLE_FEATURES` in the same file.
3. Add it to `src/agent/state.schema.json` and to the `propertyNames` enum in
   `src/agent/policy.schema.yaml` — `test_schema_propertynames_stay_in_sync_with_the_grounder`
   asserts those two lists match.
4. Normalise it to 0–1 if it is unbounded, or rely on the interpreter's per-feature
   min-max (see ARCHITECTURE.md).

## Layout

| Directory | Contents |
|---|---|
| `src/agent/` | anything that must not see ground truth |
| `src/` (top level) | the evaluator, geometry, PDB I/O, drivers |
| `src/cache/` | ESMFold cache and the labelled synthetic backend |
| `scripts/` | setup and the MVP demo path |
| `scripts/research/` | the fuller roadmap: multi-generation, holdout, ablation, reveal |
| `tests/` | fast tier plus `@pytest.mark.models` tier |
| `docs/` | architecture and roadmap |

Scripts in `scripts/research/` sit one directory deeper, so they add
`parent.parent.parent` to `sys.path`. Keep that in mind if you move files.

## Style

Match the surrounding code. A few habits this codebase keeps:

- **Comments explain why, not what.** Non-obvious choices carry the reason and, where
  relevant, the failure that motivated them.
- **Errors are loud.** An unexpected cache miss raises; an invalid patch is rejected
  with the reason, never silently coerced into something valid.
- **Empirical over assumed.** pLDDT scale is detected from the data, not hardcoded.
  If you find yourself assuming a model's output convention, check it instead.
- **Determinism.** Ties break on a stable key so the same inputs give the same
  candidates. Seeded RNG for anything random.

## Regenerating data

```powershell
python scripts/fetch_protein.py 1pgb        # download + validate the reference
python scripts/make_corruptions.py 1pgb     # seeded corrupted variants
.\run_mvp.ps1 -Precompute                   # warm the fold cache (~13 min)
python scripts/clear_synthetic_cache.py     # drop fixture structures
```

`data/cache/*.pdb` is committed on purpose so the demo needs no folding. If you add
structures, keep the cache small and never commit anything from the synthetic
backend — it writes to `data/cache/synthetic/`, which is gitignored.

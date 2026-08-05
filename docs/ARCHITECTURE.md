# Architecture and design decisions

Companion to the [README](../README.md). This is the "why it is built this way"
document — the constraints that shape the code, and the environment findings that
cost real time to discover.

## The four constraints

These are not conventions. Each is enforced by a test in
[`tests/test_constraints.py`](../tests/test_constraints.py), re-run on every commit.

### C1 — Scope lock

The system must not claim more than it does. No JEPA adaptation, no PepCompass
implementation, no general protein-fitness optimisation, and no claim to design
therapeutics, repair biological function, or discover protein physics.

*Enforced by:* a scan of `src/`, `app/` and `docs/` for the terms that would
signal scope creep. Every hit must sit in a comment or caption that explicitly
negates the claim. The mutation heatmap caption, for instance, is asserted to
contain the word "approximation".

### C2 — Gemma never executes code

The outer loop's only output channel is the policy DSL: four scoring weights and
three search parameters, defined in
[`src/agent/policy.schema.yaml`](../src/agent/policy.schema.yaml). A patch is
validated before the interpreter sees it, and an invalid patch is **rejected, not
coerced**.

*Enforced by:* a grep for `eval(`, `exec(`, `__import__`, `subprocess` and
`os.system` across the policy path, plus parametrised tests feeding code-shaped
payloads (`{"exec": "os.system(...)"}`) through validation and asserting they raise.

Two rules cannot be expressed in JSON Schema and live in
`src/agent/policy.py::validate_policy`:

1. `position_score` weights must sum to 1.0 (tolerance 1e-6).
2. Every `position_score` key must name a field the grounder actually computes.

Rule 2 is what makes "add a state feature" safe: Gemma can re-weight an existing
computed quantity, never invent a new computation.

### C3 — The reference structure never reaches the agent

`src/evaluator.py` is the only module permitted to read `data/proteins/`.

*Enforced by:* no file under `src/agent/` may contain the string `native` or
`SEQRES`; exactly one file under `src/` may contain `native_pdb_path`; no agent
file may import the evaluator; `reveal=True` may be passed from exactly one script
(`scripts/research/final_reveal.py`). The UI has an independent check — with the
reveal toggle off, the reference structure is never *loaded*, so it cannot reach
the DOM by accident.

The redaction is positive as well as negative: with `reveal=False` the returned
dict must contain no key whose name includes `tm`, `native` or `contact`.

### C4 — No live inference on the judged path

Every ESMFold call on the demo path resolves from `data/cache/`.

*Enforced by:* `REFOLD_OFFLINE=1` turns an unexpected cache miss into a loud
`FoldUnavailable` rather than a silent 40-second model call, and the flag is read at
call time so it cannot be defeated by import order.

## Why the inner loop selects on public signals only

The repair loop ranks candidates by `0.5·esm_score + 0.5·mean_plddt` — deliberately
*not* the hidden objective. If the agent optimised the hidden score directly there
would be no gap between prediction and reality, and therefore no counterexample to
learn from. The gap is the product.

`src/agent/inner_loop.py::PUBLIC_WEIGHTS` contains no term derived from a reference
structure, and that is asserted in `tests/test_inner_loop.py`.

## Why the median, not the mean

`src/agent/outer_loop.py::decide` accepts a candidate policy only if its **median**
hidden score across the corruption set strictly exceeds the incumbent's. Mean would
let one lucky variant carry a policy that is worse everywhere else;
`test_median_is_not_the_mean` pins exactly that case.

Strictly greater, so a tie keeps the incumbent and the policy never drifts on noise.

## Scoring details worth knowing

**ESM rescaling.** Raw pseudo-log-likelihoods are unbounded negative numbers. Adding
one to a weighted sum next to a 0–1 TM-score would let it dominate or vanish based on
sequence length alone. The evaluator squashes it with a logistic on the z-score of
the *per-residue* mean log-likelihood, calibrated against the reference sequence plus
every corrupted variant, so 0.5 means "as plausible as the average sequence in this
benchmark".

**Feature normalisation in the interpreter.** `esm_surprisal` is an unbounded positive
log quantity; `low_plddt` and the contact-violation fields are already 0–1. Without
per-feature min-max normalisation across positions the surprisal weight would dominate
regardless of what the policy says, and re-weighting would appear to do nothing.

**`contact_violation` is not a comparison against ground truth.** The agent has no
reference contact map. It is an internal under-packing signal: how far a residue's
contact count falls below the mean of the fold it is currently in.

**TM-score normalisation.** With `tm_align(candidate, reference, ...)`,
`tm_norm_chain2` is the reference-length normalisation. Both lengths are equal here
because every candidate is substitution-only, so the two numbers coincide today —
`tm_norm_chain2` is pinned anyway so a future indel-tolerant change cannot flip the
hidden score unnoticed.

## The synthetic backend

`src/cache/synthetic_backend.py` produces deterministic, plausible-looking
coordinates from a sequence. **It is not structure prediction** — no learned
parameters, no physics. It exists so the loop, the UI and the cache assertions can be
exercised on a machine where the 8.4 GB checkpoint has not finished downloading.

Four guardrails, added after a fixture leaked into the real cache during development
and skewed a whole precompute run:

1. Every structure it writes carries a `SYNTHETIC_GEOMETRY` REMARK line.
2. It writes to `data/cache/synthetic/`, a **separate directory**, so a run
   configured for ESMFold can never be served a fixture.
3. `FoldResult.synthetic` is True for anything carrying that remark, and the UI
   shows a red banner plus a `SYNTHETIC - NOT A PREDICTION` badge.
4. `scripts/clear_synthetic_cache.py` removes fixtures and their derived logs.

Enable with `REFOLD_FOLD_BACKEND=synthetic`. Default is `esmfold`.

## Environment findings

Discovered the hard way on the development machine (Windows 11, 13.9 GB RAM, 4 GB
GPU, Python 3.13.13). Recorded so they do not have to be rediscovered.

**ESMFold's pLDDT arrives on a 0–1 scale here, not 0–100.** The observed range was
0.71–0.95. `fold_cache.structure_features` detects the scale from the data rather
than assuming it.

**The fp32 checkpoint is 8.4 GB and does not fit in available RAM.** A plain
`from_pretrained` was OOM-killed mid-load. The fix: load in bfloat16, then upcast the
folding trunk and heads back to fp32. Safe because `EsmForProteinFolding.forward`
casts the language-model output with `esm_s.to(self.esm_s_combine.dtype)`, so
promoting `esm_s_combine` promotes the activations. bfloat16 rather than float16
because torch 2.x has far better CPU kernel coverage for it. Override with
`REFOLD_FOLD_DTYPE=float32`.

**Ollama needed three separate fixes to be usable.**

| Symptom | Cause | Fix |
|---|---|---|
| Model stuck in `Stopping...`, HTTP response never arrived | `keep_alive: 0` raced the unload against the reply | `keep_alive: 60s` |
| HTTP 500, `CUDA error: out of memory` | a 12B model does not fit a 4 GB card, and Ollama's automatic split tried anyway | `num_gpu: 14` fixed layers, with automatic CPU retry on OOM |
| `no JSON object found in reply` | the model opened with prose and `num_predict` truncated it before any JSON | constrained decoding against `PATCH_FORMAT_SCHEMA` |

Pure CPU inference worked but took over ten minutes per call. 14 GPU layers brings it
to ~70 s.

**`max_total_edits` was removed from Gemma's action space.** Constrained decoding
enforces shape but not numeric bounds, and the model repeatedly answered
`max_total_edits: 5000000000000000`, burning an attempt on a schema rejection. The
field has no effect on a single-round repair, so removing it costs nothing and
eliminates the failure mode without coercing anything the model said.

**`tmtools` has no cp313 wheel** and its sdist needs MSVC. TM-score is computed by
`src/geometry.py` (Zhang–Skolnick fragment-seeded iterative superposition), exercised
by `tests/test_geometry.py`: identity → 1.0, rigid-motion invariant, scrambled → below
0.5. `src/evaluator.py` picks up `tmtools` automatically if it is ever installed and
reports which backend produced the number.

`scripts/validate_tm_score.py` measures the two against each other on real 1PGB
coordinates, run under an interpreter that has `tmtools`
(`C:\Python312\python.exe scripts/validate_tm_score.py`):

| case | tmtools | geometry.py | diff |
|---|---|---|---|
| identical | 1.00000 | 1.00000 | 0.00000 |
| rigid motion | 1.00000 | 1.00000 | 0.00000 |
| noise σ=0.5 | 0.91563 | 0.91563 | 0.00000 |
| noise σ=1.0 | 0.74074 | 0.74088 | 0.00015 |
| noise σ=2.0 | 0.43820 | 0.40709 | 0.03111 |
| noise σ=3.0 | 0.34391 | 0.27919 | 0.06472 |
| noise σ=8.0 | 0.16278 | 0.10651 | 0.05627 |

So: essentially exact above TM 0.74, and a systematic **underestimate** of up to 0.06
below TM 0.5. The cause is TM-align's multi-cutoff refinement schedule, not seed
density — using every fragment start offset instead of a stride costs 4× and improves
agreement by 4e-4. Every candidate this evaluator scores is a single-point mutant of a
56-residue protein at TM ≈ 0.85–0.98, comfortably inside the accurate band, so the
stride stays and the bias is documented rather than chased.

## Relaxations made for the MVP

Honest accounting of where the implementation is looser than the roadmap.

- **"Change one thing per patch" was relaxed.** Local models routinely omit `kind`
  or send both halves of a patch. `outer_loop_client.infer_kind` decides from the
  keys actually present, and a patch carrying both halves is applied as a combined
  patch. Only a presentation convention was relaxed — every field still has to
  survive the P3.1 schema, which is what C2 rests on. A *declared* kind that
  contradicts the payload is still rejected.
- **Behaviour change is judged on the enumerated candidate set,** not the folded
  top-3. A policy that proposes six substitutions per site instead of four has
  genuinely changed what the search explores, even when the three survivors of
  pre-ranking happen to be the same. Judging only the folded subset reported such
  patches as no-ops.
- **Gemma gets two attempts.** If its first valid patch would leave the sites and
  candidate set identical, it is told so and asked once more. A valid patch from the
  first attempt is never discarded because the second came back malformed.

## Mapping to the roadmap

| Concept | Implementation |
|---|---|
| Raw observation | sequence + predicted PDB (`src/cache/fold_cache.py`) |
| State grounding | `state.json` residue/contact graph (`src/agent/grounder.py`) |
| Action | amino-acid substitution (`policy_interpreter.enumerate_candidates`) |
| Predicted transition | PLL/pLDDT pre-rank (`policy_interpreter.prerank_candidates`) |
| Mechanism program | policy DSL + interpreter (`src/agent/policy.schema.yaml`, `src/agent/policy_interpreter.py`) |
| New observation | reconstructed candidate structure (`src/cache/fold_cache.py::fold`) |
| Counterexample | predicted-improved-but-hidden-worsened (`src/agent/counterexamples.py`) |
| Representation revision | reweight/activate a `state.json` feature (`src/agent/outer_loop_client.py`) |
| Mechanism revision | change a proposal field (`src/agent/outer_loop_client.py`) |

Full plan in [ROADMAP.md](ROADMAP.md).

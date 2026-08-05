# ReFold — MVP demo runbook

## The one-sentence claim

A falsifiable, self-correcting protein repair harness in which representations
and executable repair rules co-evolve under a hidden structural verifier.

It does **not** design a therapeutic, does not repair function in any organism,
and makes no claim to have found new protein physics.

## The loop, once

```
corrupted 1PGB (5 point substitutions)
  -> ESM-2 scores every residue                (facebook/esm2_t12_35M_UR50D)
  -> policy picks the 3 most suspicious sites  (policy.yaml, weighted score)
  -> 12 candidate mutations enumerated
  -> cheap pre-rank keeps the best 3           (PLL - lambda * edits)
  -> ESMFold predicts those 3 structures       (facebook/esmfold_v1, cached)
  -> hidden evaluator scores them              (TM-score etc., withheld)
  -> state is grounded into an explicit graph  (state.json)
  -> Gemma reads the state + the counterexample
  -> Gemma returns ONE policy patch in a fixed DSL
  -> schema validates it; the interpreter re-runs under the new policy
  -> before/after comparison
```

Gemma's only output channel is the policy DSL. It never emits code, and the
patch is schema-validated before the interpreter will touch it.

## Launch

```powershell
.\run_mvp.ps1 -Check      # pre-flight: ollama, checkpoint, cache coverage
.\run_mvp.ps1             # start the app
```

Then open <http://localhost:8501> and press **Run ReFold** at the bottom.

`data/cache/` ships with the 38 structures this demo needs, so no folding is
required. Only if you change the protein or corruption:

```powershell
.\run_mvp.ps1 -Precompute   # ~13 min, ~40 s per structure
```

Setup from a fresh clone is in the [README](README.md#installation).

## Verified run (2026-07-30, everything real)

```
corrupted 1PGB (G9A T18A A34K K50P T55D)      hidden 0.5699
  baseline policy   sites [50, 21, 55]        12 mutations proposed
                    picked P50A               hidden 0.8026   (+0.233)
  gemma4:12b        valid mechanism patch, accepted first attempt
                    "Increasing positions and substitutions per position
                     expands the search space, allowing for more diverse
                     mutations to be explored."
  patched policy    sites [50, 21, 55, 48, 11]  31 mutations proposed
                    picked P50M               hidden 0.8080   (+0.005 vs baseline)
  total 105 s
```

Position 50 is the K50P corruption — a proline dropped into a beta strand. ESM-2
flagged it as the most implausible residue without ever seeing the reference
structure, and reverting it is what produced the +0.233 jump.

## Demo order

1. **Left column — the problem.** Original 1PGB sequence, the corrupted input
   with the 5 damaged sites in red, and the table of what was changed. Say: the
   harness never sees this table. It only ever sees the corrupted sequence and
   its own predicted structure.
2. **Press Run ReFold.** The status panel narrates each step live.
3. **Middle column — what the harness sees.** The predicted structure of the
   corrupted protein, the per-residue substitution heatmap, and the ranked
   candidate mutations under each policy.
4. **Right column — the actual point.** Gemma's reasoning in its own words, the
   unified diff of the policy before and after, and the score comparison.
5. **Land the claim.** ESM-2 found the fold-breaking proline at position 50 without
   ever seeing the reference structure, and reverting it moved the hidden score
   0.5699 → 0.8026. Gemma then read the grounded state, rewrote the policy's search
   parameters, and the agent examined five sites instead of three — finding a better
   substitution at the same position. Observe, reason, modify strategy, try again.

   **Be accurate about one thing.** On this variant the public and hidden signals
   *agree*, so no counterexample fires. The disagreement machinery is real and
   test-covered (`src/agent/counterexamples.py`), but do not narrate a
   counterexample that the timeline does not show — the timeline will say
   "no counterexample this round".

## What is real vs. what is simplified

Real:

- ESM-2 35M masked-marginal scoring, running locally.
- ESMFold (`esmfold_v1`, 8.4 GB checkpoint) predicting every candidate structure.
- The hidden evaluator: TM-score against the withheld crystal structure, contact
  recovery, pLDDT, edit penalty. Its decomposition is never returned to the agent.
- gemma4:12b via Ollama, called live, producing the patch you see.
- Schema validation that rejects out-of-DSL patches rather than coercing them.

Simplified for the MVP:

- One protein (1PGB), one corruption (`corrupt_01`), one generation, one patch.
- One repair round per policy rather than the full edit budget.
- Structures come from `data/cache/`. Precompute walks a spread of plausible
  policies so whatever patch Gemma picks is already folded; if it picks something
  unforeseen the app folds live instead, which costs ~40 s per candidate.
- Gemma gets two attempts: if its first valid patch would leave the selected
  sites and candidate set identical, it is told so and asked once more.

## Known failure points

| Risk | Symptom | Mitigation |
|---|---|---|
| Ollama not running | Gemma panel shows a connection error | `ollama serve`, then `ollama list` |
| Zombie `llama-server` holding VRAM | HTTP 500 "CUDA error: out of memory" | kill `llama-server` only — **not** `*llama*`, that pattern matches `ollama` itself and takes the server down |
| Gemma call is slow | 70–110 s per patch | expected: 12B, partial GPU offload. Two attempts is the worst case (~2 min) |
| Uncached candidate | ~40 s pause per fold | pre-warmed by `-Precompute`; the timeline says "computed live" when it happens. Gemma releases its memory after 60 s so the two rarely overlap |
| Gemma returns prose, not JSON | patch shown as REJECTED | constrained decoding against a JSON schema makes this unlikely now; if it happens the run completes on the incumbent policy, and the rejection is itself a legitimate "the DSL is a real boundary" talking point |
| Gemma's patch changes nothing | "Search behaviour did NOT change" | one automatic retry that tells it so; a valid patch from attempt 1 is never discarded if attempt 2 fails |
| Gemma emits an out-of-range number | patch REJECTED | `max_total_edits` was removed from the action space for exactly this reason; the remaining fields are bounded in the decode schema |
| Streamlit port in use | won't start | `streamlit run app/streamlit_app.py --server.port 8502` |

## Likely questions

**"Is the LLM writing code?"** No. Its only output channel is a policy DSL of four
scoring weights and three search parameters, validated against a JSON Schema before
the interpreter runs. Show the right-hand panel: the raw model output is a small JSON
object, and the diff below it is the resulting change to `src/agent/policy.yaml`.

**"How do you know it isn't cheating?"** The agent never sees the reference
structure. That is enforced by tests, not convention — `tests/test_constraints.py`
greps the agent tree for any reference to ground truth and fails the suite on a hit.
The verifier returns one scalar; its decomposition is unlockable from exactly one
script.

**"Why is the second improvement so small?"** Because the first one was large. The
baseline repair already recovered most of what a single substitution can on a
56-residue protein. The demo's claim is that the strategy changed and stayed
non-worse, not that round two is where the value is.

**"Did you pick the corruption to make this work?"** `corrupt_01` is the first of
five seeded variants generated by `scripts/make_corruptions.py` with a fixed seed.
The others are in `data/corruptions/1pgb/` and can be run by changing one argument.

## Environment notes

Full detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#environment-findings).
The three that bite most often:

- ESMFold's pLDDT arrives on a **0–1** scale here, not 0–100 (detected, not assumed).
- The checkpoint is loaded in bfloat16 with the folding trunk upcast to fp32,
  because the fp32 model does not fit in available RAM.
- Ollama needs `keep_alive: 60s` and a fixed `num_gpu`, not its automatic split.

# web/ — the deployed demo

This directory plus `api/fold.mjs` is the whole of what gets deployed to Vercel:
a static site with no framework, no build step and no npm install, and a single
dependency-free serverless function that proxies structure prediction.
`vercel.json` points `outputDirectory` here and `.vercelignore` keeps the Python
project out of the upload entirely.

It is a replay of the recorded run, plus two things you can actually drive:

- **Rewrite the policy yourself** (`index.html` §4). The DSL controls run the
  project's real interpreter, ported to `policy.js` from
  `src/agent/policy_interpreter.py` — same min-max normalisation, same
  tie-breaking toward lower indices, same residue-class filter and edit budget,
  same rejection-not-coercion validation. It needs no model and no download,
  because the interpreter is pure arithmetic over the grounded state. Setting
  the seed policy reproduces round 1 exactly (3 sites, 12 proposals); pressing
  "Apply Gemma's patch" reproduces round 2 (5 sites, 31 proposals).
- **Repair your own protein** (§5). Paste a sequence — protein, or DNA which is
  translated first — and it is folded, scored residue by residue by ESM-2 running
  in the browser, repaired, and folded again, with the corrected sequence
  returned as FASTA. Structure prediction goes through `api/fold.mjs`.

## Precision is a correctness decision

The in-browser model is loaded as **fp16**, not int8. Measured against the
PyTorch masked-marginal matrix for corrupted 1PGB:

| dtype | size | max abs Δp | top-5 suspicious positions |
|---|---|---|---|
| q8 | 35 MB | 0.081 | 50, 21, 34, 9, **35** — wrong at rank 5 |
| fp16 | 68 MB | 0.00038 | 50, 21, 34, 9, **55** — matches PyTorch |

int8 error is large enough to reorder the position ranking, and the patched
policy selects five sites, so q8 would hand back a different search than the
reference pipeline. fp16 costs 33 MB more and scores at the same speed. q8 stays
as a fallback for devices that cannot load fp16, and the page says so when that
happens. Pre-rank scores cross-checked against Python agree to ~0.006 on a sum
of 56 log-probabilities, with identical ordering.

## Repairing a sequence you supply (§5)

The full loop on arbitrary input: paste a sequence, it is folded, scored residue
by residue, repaired, and folded again.

1. **Translate** if the input is DNA/RNA — detected by alphabet and reported,
   never applied silently. Frame 1, stopping at the first stop codon.
2. **Fold as given** through `/api/fold`.
3. **Score** every residue with ESM-2 in the browser (masked marginals, one
   forward pass per residue).
4. **Select and substitute** using the ported interpreter: the highest-surprisal
   positions, each replaced by the residue ESM-2 most expects there. The
   residue-class filter is off here — reverting damage often has to cross
   classes, and the informative answer is simply what the model expects.
5. **Re-score and re-fold** the repaired sequence.
6. **Compare**: TM-score and RMSD between the two predicted structures (computed
   in `structure.js`), pLDDT from each fold, and pseudo-log-likelihood from ESM-2.

Output is the modification table and a copyable FASTA.

### The honesty constraint

A sequence supplied by a visitor **has no reference structure**, so there is no
hidden verifier and nothing is scored against truth — that is exactly what §3's
withheld evaluator provides for 1PGB and cannot provide here. Only computable
quantities are reported: sequence likelihood, the model's own confidence, and how
far the predicted fold moved. A repair that improves all three is a good
hypothesis, not a verified fix, and the page says so.

Structure prediction is **not** the 8.4 GB checkpoint running on Vercel. It is
`api/fold.mjs`, a dependency-free proxy to Meta's hosted ESMFold v1 — the browser
cannot call it directly because ESM Atlas's CORS preflight returns 403. That
service is free and intermittently unavailable (observed answering the same
56-residue chain in 2.2 s and then timing out minutes later), so the proxy retries
three times before giving up, and a fold failure degrades the page rather than
breaking it: the sequence analysis and the repaired FASTA still appear, without
the structures.

## Local development

`python -m http.server` cannot serve `/api/fold`, so §5 will not work under it.
Use the stand-in, which mirrors the Vercel function's contract exactly:

```bash
python scripts/dev_server.py        # http://localhost:8765
```

## Why the real app is not deployed here

The Streamlit app in `app/streamlit_app.py` cannot run on Vercel, and no amount
of configuration changes that:

| Requirement | Vercel |
|---|---|
| Streamlit's long-lived server + WebSocket session | Functions are stateless and request-scoped |
| `torch` + `transformers` + `scipy` + `pandas` | Well past the 250 MB unzipped function limit — `torch` alone exceeds it |
| ESMFold checkpoint, 8.4 GB | Not in the repository (`data/models/` is gitignored) and far past any bundle limit |
| `gemma4:12b` served by local Ollama | No Ollama process exists in a serverless runtime |

The error Vercel reports first — `No python entrypoint found` — is only zero-config
detection tripping over `requirements.txt`. It is the first wall, not the only one.

So this page replays a recorded run instead, and says so on its face.

## Where the numbers come from

`data/demo.json` is generated by `scripts/build_web_data.py`, which distinguishes
two kinds of value and labels every one of them in the UI:

- **computed** — recomputed at build time by the project's own code. TM-score,
  contact recovery, sequence recovery, pLDDT and clash counts come from
  `src/geometry.py` over the committed ESMFold predictions in `data/cache/`
  scored against the real 1PGB crystal structure, and need no model at all. The
  site lists, the grounded state, the substitution matrices and the proposal
  counts come from `src/agent/policy_interpreter.py` driven by ESM-2 — the
  130 MB scorer, not the 8.4 GB folder.
- **recorded** — transcribed from the verified run of 2026-07-30 in `DEMO.md`.
  Only two things remain here: the composite hidden score, which is produced by
  the withheld evaluator against the reference structure and whose run JSON was
  never committed (`logs/` is gitignored), and Gemma's verbatim rationale, which
  is a model output that cannot be re-derived.

The build **asserts the two tiers agree**: recomputing the interpreter must
reproduce DEMO.md's `[50, 21, 55]` / 12 and `[50, 21, 55, 48, 11]` / 31 exactly,
and both recorded picks must appear among the proposals. A mismatch fails the
build rather than publishing a page that tells two different stories.

`substitutions_per_position` was never written down, so it is **recovered**
rather than guessed: the build re-runs the interpreter across the DSL's legal
range and reports the smallest value reproducing the recorded 31 proposals,
which is 7. The residue-class filter saturates, so 7–19 all give the same count;
the page says so.

The reference structure is gated behind a toggle. With the toggle off it is never
fetched, mirroring `build_viewer_models()` in `app/streamlit_app.py`.

## Regenerating

```bash
python scripts/build_web_data.py
```

Rewrites `web/data/demo.json` and `web/data/structures/*.pdb`. It refuses to run
if any structure resolves to the synthetic harness backend rather than a real
ESMFold prediction, and it refuses to run if the ESM-2 scores are not cached —
regenerating without them would silently strip the heatmap and both interactive
sections. Warm that cache once (needs torch + transformers; the scorer is
~130 MB, not the 8.4 GB folder), or pass `--allow-missing-esm` to publish a
reduced bundle deliberately.

## Files

```
index.html          markup
styles.css          dark-first theme, light-mode counterpart, no webfonts
app.js              rendering, viewers, policy editor, the repair pipeline
policy.js           port of the DSL + interpreter; no model, pure arithmetic
esm.js              in-browser ESM-2 masked-marginal scoring
structure.js        PDB parsing, Kabsch, TM-score, codon translation
_verify.html        asserts structure.js agrees with src/geometry.py
data/demo.json      generated bundle (~98 KB)
data/structures/    corrupted / baseline / patched predictions + the reference

../api/fold.mjs     the only server-side code: a proxy to hosted ESMFold
```

External dependencies, both from CDNs and both degrading with a visible message
rather than a blank panel: 3Dmol.js for the structure viewer, and
transformers.js for §5 (loaded lazily — nothing is fetched until you press
**Run ESM-2**).

# ReFold — Build Task List for Claude Code

. Each task has a **Goal**, concrete **Do** steps, and a pass/fail **Verify** check — treat "Verify" as the actual definition of done, not the "Do" steps. The four constraints below are cross-cutting: re-check them at the end of every phase, not just once at the start.

**Assumptions:** Python 3.10+ environment with pip access to PyPI and the Hugging Face Hub; some Gemma endpoint (API or local) reachable from code; a GPU speeds things up but nothing here strictly requires one — CPU fallbacks are noted where they matter.


## Non-negotiable constraints (recheck every phase)

- [ ] **C1 — Scope lock.** Never implement JEPA-DNA adaptation/pretraining, full PepCompass (SORBES, MUTANG, LE-BO, Riemannian tangent spaces, decoder Jacobians), or general protein-fitness optimization (stability/expression/binding/aggregation). Never claim the system designs therapeutics, restores function, or discovers protein physics.
  **Verify:** `grep -rniE "jepa|pepcompass|jacobian|therapeutic|restores function|discovers" src/ app/ docs/` — every hit must sit inside a comment explaining an exclusion, never in shipped logic or pitch text.
- [ ] **C2 — Gemma never executes code.** The outer loop's only output channel is the fixed policy DSL, validated by schema, run only by a deterministic interpreter.
  **Verify:** feed the interpreter a policy payload containing `eval(`, `import os`, or an unknown key — it must be rejected by schema validation, not executed. This should be an actual test, not a manual check (see P3.1/P3.2).
- [ ] **C3 — Native structure never reaches the agent.** Corrupted sequence + the agent's own predicted structure are the only things ESM/Gemma-facing code ever sees. Native PDB and SEQRES load only inside the evaluator.
  **Verify:** `grep -rln "native" src/agent/` and `grep -rln "SEQRES" src/agent/` both return nothing; `grep -rl "native_pdb_path" src/` returns exactly one file (the evaluator).
- [ ] **C4 — No live-inference dependency at judging time.** Every ESMFold call on the live demo path must resolve from cache.
  **Verify:** stub/disable the ESMFold call object and re-run the full demo script; it completes with zero exceptions and the UI shows a "cached" badge on every structure except the one live ESM-scoring step you choose to keep live.

---

## Phase 0 — Setup

- [ ] **P0.1 — Environment**
  **Goal:** a working install before the clock starts.
  **Do:** `pip install torch transformers biopython tmtools streamlit py3Dmol stmol pyyaml jsonschema numpy pandas`. Smoke-test the two model paths you'll actually use:
  ```python
  from transformers import AutoTokenizer, EsmForProteinFolding, AutoModelForMaskedLM
  fold_tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
  fold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
  score_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D")
  ```
  Note: the original ESMFold docs recommend Python ≤3.9; the smoke test above is how you find out if that still matters on your setup, rather than assuming it does.
  **Verify:** the smoke-test script runs to completion and prints model class names with no import errors; record exact versions in `requirements.txt`. If `esmfold_v1` fails to load, fall back to `facebook/esm2_t30_150M_UR50D` for scoring only and flag ESMFold as a P0 blocker before continuing.

- [ ] **P0.2 — Repo scaffold**
  **Goal:** leakage-safe directory layout exists before any code is written.
  **Do:** create
  ```
  refold/
    data/proteins/<name>/        # native.pdb, native_seq.fasta — evaluator-only
    data/corruptions/<name>/     # corrupt_0{1..5}.fasta — agent-visible
    data/cache/                  # cached ESMFold outputs
    src/agent/                   # everything ESM/Gemma-facing
    src/evaluator.py             # the only module allowed to import data/proteins/
    app/streamlit_app.py
    logs/
    tests/
  ```
  **Verify:** `find refold -maxdepth 3` matches the layout; `data/proteins/` and `src/agent/` exist as separate trees before any corruption or scoring code is written (order matters — this is what makes C3 checkable later rather than retrofitted).

---

## Phase 1 — Protein and corruptions

- [ ] **P1.1 — Select the reference protein**
  **Goal:** one small, benign, single-chain, well-resolved protein.
  **Do:** default to the B1 domain of protein G (PDB `1PGB`, 56 residues) — small enough for fast ESMFold, large enough to have real tertiary contacts, and one of the most-benchmarked single domains in ML-for-protein work. Ubiquitin (`1UBQ`, 76 residues) is the reserve pick for Phase 6. Download the PDB, save to `data/proteins/<name>/native.pdb`.
  **Verify:** programmatically confirm — don't just trust the choice — via `Bio.PDB`: exactly one chain (`len(structure[0]) == 1`), no missing backbone atoms, crystallographic resolution < 2.5 Å from the header. If any check fails, pick a different entry rather than patching around gaps.

- [ ] **P1.2 — Extract the native sequence**
  **Goal:** a clean sequence string, stored evaluator-side only.
  **Do:** `Bio.PDB.Polypeptide` (or `PPBuilder`) to pull the sequence from `native.pdb`; write to `data/proteins/<name>/native_seq.fasta`. Do **not** copy this file or its contents into `data/corruptions/` or anything under `src/agent/`.
  **Verify:** sequence contains only the 20 standard amino-acid letters; length matches the residue count from P1.1's chain check exactly.

- [ ] **P1.3 — Generate 4–5 synthetic corruption variants**
  **Goal:** the operational definition of "bad protein" from the memo: 3–5 point substitutions, fixed seed, reproducible.
  **Do:** `scripts/make_corruptions.py`, seeded RNG, picks 3–5 distinct positions per variant and substitutes to a different residue (bias toward buried/hydrophobic-core positions if you want a guaranteed fold hit — check burial via relative solvent accessibility on the native structure, evaluator-side only). Write `data/corruptions/<name>/corrupt_0{1..5}.fasta`. Do not persist which positions were changed anywhere agent-visible — keep that bookkeeping in an evaluator-side sidecar file.
  **Verify:** `tests/test_corruptions.py` asserts, for every variant: same length as native; Hamming distance to native in `[3, 5]`; substituted residue ≠ original at every changed position.

- [ ] **P1.4 — Confirm corruptions actually degrade the fold**
  **Goal:** don't build the rest of the system on corruptions that ESMFold shrugs off.
  **Do:** run ESMFold once on native and once per corrupted variant; compute TM-score of each corrupted prediction against the native structure (see P2.6 for the reusable scoring code — fine to write a throwaway version here first).
  **Verify:** every corrupted variant's TM-score to native is measurably below 1.0 and below whatever threshold you pick as "degraded" (e.g. < 0.8) — regenerate with a different seed/position bias for any variant that doesn't clear this bar.

---

## Phase 2  — Inner loop, evaluator, grounder

- [ ] **P2.1 — ESM-2 surprisal scoring**
  **Goal:** per-residue "how unnatural is this position" score.
  **Do:** `src/agent/esm_score.py`, load `facebook/esm2_t12_35M_UR50D` via `AutoModelForMaskedLM`; implement `residue_surprisal(sequence) -> np.ndarray` (masked-marginal or WT-marginal negative log-likelihood per position) and `pseudo_log_likelihood(sequence) -> float`.
  **Verify:** `tests/test_esm_score.py::test_native_scores_more_natural_than_corrupted` — mean surprisal on the native sequence is lower than on each P1.3 corrupted variant.

- [ ] **P2.2 — Rank suspicious positions**
  **Goal:** `rank_suspicious_positions(sequence, top_k=3) -> list[int]`.
  **Do:** sort positions by surprisal descending, return top-k.
  **Verify:** unit test with a hand-built sequence containing one deliberately implausible residue at a known index — that index must appear in the top-3 output.

- [ ] **P2.3 — Candidate substitution enumeration**
  **Goal:** turn suspicious positions into concrete candidate sequences.
  **Do:** for each of the top-3 positions, take the top-4 substitution amino acids by ESM masked-marginal probability (matches the DSL defaults in P3.1); if `preserve_residue_class` is set, filter to same hydrophobic/polar/charged class as the current residue; drop any "substitution" identical to the current residue.
  **Verify:** default policy on a real corrupted sequence yields exactly ≤ 3×4 = 12 unique candidates, none equal to the incumbent; a test asserts the same-residue case is filtered.

- [ ] **P2.4 — Cheap pre-ranking**
  **Goal:** pick 2–3 candidates worth the expensive ESMFold call.
  **Do:** score all P2.3 candidates by `pseudo_log_likelihood(candidate) − λ · edit_count(candidate, incumbent)`; keep the top 2–3.
  **Verify:** test that two candidates with equal PLL but different edit counts rank in edit-count order (fewer edits wins); shortlist size is always ≤ 3.

- [ ] **P2.5 — ESMFold shortlist + cache + feature extraction**
  **Goal:** structural features for the shortlist only, fully cached.
  **Do:** `src/cache/fold_cache.py`: `hash(sequence)` → check `data/cache/<hash>.pdb` before invoking `EsmForProteinFolding`; on a miss, run the model, call `model.output_to_pdb(outputs)[0]`, write to cache. Extract: mean + per-residue pLDDT (it's in the B-factor column of the output PDB — don't recompute it), a contact map (Cβ–Cβ, or Cα for glycine, within ~8 Å), radius of gyration, and a clash count (any non-bonded heavy-atom pair below ~2.0 Å).
  **Verify:** call the same sequence twice — second call is a logged cache hit, not a re-run of the model; a test asserts `cache_misses == 1` and `cache_hits == 1` across two calls; extracted pLDDT values fall in the model's actual output range (check empirically rather than assuming 0–100 vs 0–1). Note in a code comment that high pLDDT is model confidence, not correctness — the hidden evaluator (P2.6), not this module, decides if a fold is actually right.

- [ ] **P2.6 — Immutable hidden evaluator**
  **Goal:** the one module allowed to touch the native structure, per C3.
  **Do:** `src/evaluator.py` loads `data/proteins/<name>/native.pdb` (nowhere else does). Compute:
  - `tm_score` via `tmtools.tm_align(candidate_coords, native_coords, candidate_seq, native_seq)` — **pin and comment which of `.tm_norm_chain1` / `.tm_norm_chain2` corresponds to native-length normalization**, since flipping the argument order silently changes the hidden score and this is an easy mistake to make unnoticed.
  - `contact_recovery = |contacts(candidate) ∩ contacts(native)| / |contacts(native)|`.
  - `edit_fraction = edit_count / max_total_edits` (state the denominator explicitly in a comment — the memo doesn't pin this, and it changes the score's scale).
  - `esm_score`: **rescale the raw PLL before using it** — raw pseudo-log-likelihoods are unbounded negative numbers, not a 0–1 quantity, so plugging them straight into a weighted sum with TM-score will let ESM score silently dominate or vanish. Min-max or logistic-normalize against a reference batch first.
  - Combine: `hidden_score = 0.55*tm_score + 0.20*esm_score + 0.15*plddt + 0.10*contact_recovery - 0.05*edit_fraction`.
  Expose `evaluate(candidate, reveal=False) -> dict`; when `reveal=False`, the returned dict must not contain `tm_score`, `contact_recovery`, or native-sequence-recovery keys — only `hidden_score` plus whatever public-facing fields (ESM score, pLDDT, edit count) the agent is allowed to see per the memo's visibility table.
  **Verify:** (a) `assert not any(k for k in evaluate(x, reveal=False) if "tm" in k.lower() or "native" in k.lower() or "contact" in k.lower())`; (b) a hand-computed toy case (tm_score=1, esm_score=0, plddt=0, contact_recovery=0, edit_fraction=0) returns `hidden_score == 0.55`; (c) `reveal=True` is called only from a dedicated final-reveal script, never from inner/outer loop code — `grep -rn "reveal=True" src/agent/` returns nothing.

- [ ] **P2.7 — Grounder → `state.json`**
  **Goal:** the explicit object/relation schema the memo insists on — not a raw ESM embedding.
  **Do:** `src/agent/grounder.py::ground(sequence, predicted_structure) -> dict`:
  ```json
  {
    "residues": [{"position": 0, "aa": "M", "esm_surprisal": 0.0, "plddt": 0.0,
                  "contact_degree": 0, "long_range_contact_degree": 0, "ss_region": "coil"}],
    "relations": {
      "contacts": [{"i": 0, "j": 12, "sequence_separation": 12}],
      "helices": [[4, 10]],
      "mutation_effects": [{"position": 5, "to": "L", "breaks_contact": false}]
    }
  }
  ```
  Precompute **both** total and long-range (`sequence_separation` above some threshold, e.g. > 4) contact degree from the start, even though the seed policy in P3.1 only weights total `contact_violation`. This matters beyond tidiness: the memo's key live-demo moment is Gemma adding a long-range-contact feature mid-run — if the grounder never computed that number, the "patch" has nothing real to attach to and the moment has to be staged rather than earned.
  **Verify:** `json.dumps(state)` succeeds with no numpy scalar types leaking through; validates against a `state.schema.json` you write via `jsonschema.validate`; `len(state["residues"]) == len(sequence)`; `long_range_contact_degree` is present and non-constant across residues on a real fold (not all zero).

---

## Phase 3— Policy DSL and outer loop

- [ ] **P3.1 — Policy DSL schema + seed policy**
  **Goal:** the only vocabulary Gemma is allowed to speak.
  **Do:** `src/agent/policy.schema.yaml` formalizing:
  ```yaml
  position_score:          # keys beyond the three below are allowed, but ONLY if the
    esm_surprisal: 0.60    # key names a field the grounder (P2.7) already computes —
    low_plddt: 0.20        # this is what makes "add a state feature" safe without
    contact_violation: 0.20  # letting Gemma invent new computations (ties to C2)
  proposal:
    positions: 3
    substitutions_per_position: 4
    preserve_residue_class: true
    max_total_edits: 3
  ```
  Seed `src/agent/policy.yaml` with exactly these default values. Weights in `position_score` must sum to 1.0.
  **Verify:** seed file validates; three deliberately broken payloads (weights not summing to 1.0; `max_total_edits` negative; an extra key like `exec: "os.system(...)"` or a feature name absent from `state.json`'s residue schema) are all rejected with a clear error — this test doubles as your C2 evidence.

- [ ] **P3.2 — Deterministic interpreter**
  **Goal:** `apply_policy(policy, state) -> list[Candidate]`, pure function, no dynamic code execution.
  **Do:** `src/agent/policy_interpreter.py` implements exactly the position-selection/substitution logic parameterized by the schema fields from P3.1.
  **Verify:** `grep -nE "eval\(|exec\(|__import__|subprocess|os\.system" src/agent/policy_interpreter.py` returns nothing; running the interpreter with `positions: 5` vs the seed `positions: 3` on identical state produces 5 vs 3 candidate positions — proves the DSL fields actually drive behavior rather than just parsing.

- [ ] **P3.3 — Counterexample recorder**
  **Goal:** log the exact moments where public-facing prediction and hidden reality disagree.
  **Do:** after each outer-loop generation, if predicted improvement (ESM score / pLDDT went up) but `hidden_score` did not, append to `logs/counterexamples.jsonl`: iteration, predicted vs. hidden delta, policy-before, state-before.
  **Verify:** mock the evaluator to always return a worsening hidden score; run one generation; confirm exactly one well-formed JSONL line was appended with all required keys (read back with `jsonlines`, check keys).

- [ ] **P3.4 — Keep-if-better loop (median rule)**
  **Goal:** a candidate policy replaces the incumbent only if it wins on the memo's explicit rule.
  **Do:** `src/agent/outer_loop.py::run_generation(candidate_policy, corruption_set)` runs the full inner loop (P2.1–P2.6) on all 3 seeded corruption variants under both the incumbent and candidate policy; keeps the candidate only if its **median** hidden score across the 3 exceeds the incumbent's median.
  **Verify:** hand-construct 3 fake (incumbent_score, candidate_score) triples where you've computed the medians by hand; the function's accept/reject decision must match your hand calculation exactly — a hardcoded-expected-output test, not a smoke test.

- [ ] **P3.5 — Gemma client + bounded patch validation**
  **Goal:** Gemma proposes exactly one patch per counterexample; anything out-of-schema is rejected, not coerced.
  **Do:** `src/agent/outer_loop_client.py` sends Gemma: current `state.json` schema, current policy, candidate outcomes, predicted-vs-observed deltas, and the relevant counterexample from P3.3. Parse the response into either a `representation` patch (activate/reweight an existing-but-unused `state.json` field — e.g. `long_range_contact_degree` from P2.7) or a `mechanism` patch (change `positions`, `substitutions_per_position`, `preserve_residue_class`, or `max_total_edits`). Validate against the P3.1 schema before ever calling P3.2's interpreter with it.
  **Verify:** `tests/test_outer_loop_client.py::test_rejects_out_of_schema_patch` uses a canned malformed Gemma response (mocked — no live API call needed for this test) referencing a field not in `state.json`; confirm it's rejected and logged, never force-applied. Separately, with a canned valid patch fixture that turns on `long_range_contact_degree`, confirm P3.2's interpreter output actually changes afterward (cross-check against P3.2's test).

---

## Phase 4  — Streamlit + py3Dmol UI

- [ ] **P4.1 — App skeleton**
  **Goal:** a running Streamlit app with the four sections the memo asks for.
  **Do:** `app/streamlit_app.py` with sections: structure viewer, mutation heatmap, score history, policy diff.
  **Verify:** `streamlit run app/streamlit_app.py --server.headless true &` starts with no exception; the local port responds (e.g. `curl -s -o /dev/null -w "%{http_code}" http://localhost:8501` returns `200`).

- [ ] **P4.2 — py3Dmol viewer with gated reveal**
  **Goal:** before/after/hidden-target structures, native one gated behind an explicit action.
  **Do:** render corrupted-candidate and repaired/incumbent structures via `py3Dmol`/`stmol` always; render the native structure only after a "Reveal ground truth" button is clicked, disabled/hidden by default.
  **Verify:** with reveal untoggled, inspect the rendered HTML/DOM string and confirm no native-structure atom lines are present — this is the UI-level companion check to C3's code-level check in P2.6, and both need to independently pass.

- [ ] **P4.3 — Mutation sensitivity heatmap**
  **Goal:** the memo's "geometry-inspired local mutation search," honestly labeled.
  **Do:** position × amino-acid heatmap of ESM masked-token probabilities — an approximation of PepCompass's tangent mutation map, not an implementation of it.
  **Verify:** the caption/legend string rendered in the app literally contains "approximation" and does not claim to be PepCompass itself — a substring assertion on the caption constant (`assert "approximation" in HEATMAP_CAPTION`) — this is the checkable form of C1's naming rule.

- [ ] **P4.4 — Score history chart**
  **Goal:** public vs. hidden score per outer-loop generation.
  **Do:** public score shown live every generation; hidden score shown up to the last reveal point (or fully, at final reveal).
  **Verify:** after running 3+ real generations, the chart's underlying data array length equals the number of completed generations exactly — check for the common off-by-one bugs (dropped generation 0, double-counted seed policy).

- [ ] **P4.5 — Policy diff viewer**
  **Goal:** show exactly what changed when a patch was accepted.
  **Do:** unified diff (`difflib.unified_diff`) of the policy YAML before/after each accepted patch; optionally render a generated `policy.py`-style view for presentation flavor — it is display-only and is never executed (P3.2's interpreter is the only thing that runs).
  **Verify:** after one accepted representation patch, the diff view shows a `+` line for the newly-weighted feature key, and re-running P3.2's test after the patch confirms the interpreter's actual candidate output changed the same way — diff view and executed behavior must never drift apart; treat any mismatch as a bug, not a display nuance.

---

## Phase 5  — Precompute, holdout, rehearsal, fallback

- [ ] **P5.1 — Precompute everything on the live path**
  **Goal:** zero live ESMFold calls during judging (C4).
  **Do:** batch-run ESMFold over native, all seeded corruptions, and every candidate sequence touched during a full rehearsal; fully populate `data/cache/`.
  **Verify:** monkeypatch the ESMFold call to raise if invoked, run the full demo script in "replay mode," and confirm zero exceptions — i.e. `cache_misses == 0` for the entire run, asserted and logged, not eyeballed.

- [ ] **P5.2 — One blind held-out corruption**
  **Goal:** a variant the evolved policy has never touched during selection.
  **Do:** generate exactly one additional corrupted variant not among the 3 used in P3.4's keep-if-better loop; run the final evolved policy on it (precomputed per P5.1/C4); hold its hidden score for the live reveal.
  **Verify:** grep `logs/counterexamples.jsonl` and any P3.4 training logs for this variant's filename/hash — zero hits anywhere before the reveal moment.

- [ ] **P5.3 — Full rehearsal**
  **Goal:** the entire presentation, timed, end to end, on cached data.
  **Do:** run the full script/narration start to finish using only P5.1's cache.
  **Verify:** completes within your target demo window (pick one, e.g. ≤ 5 minutes) with zero unhandled exceptions; capture logs for post-mortem if anything breaks.

- [ ] **P5.4 — Fallback recording**
  **Goal:** insurance against live-demo failure.
  **Do:** screen-record the P5.3 rehearsal to `demo/fallback_recording.mp4`.
  **Verify:** file exists, has non-zero duration (`ffprobe`), and visibly contains the "counterexample → patch → success" moment — the memo's specified strongest live moment — not just the final score screen.

---

## Phase 6  — Stretch goals

- [ ] **P6.1 — Second held-out protein**
  **Goal:** evidence the pipeline generalizes.
  **Do:** repeat P1.1–P1.4 for ubiquitin (`1UBQ`) or another small single-chain protein.
  **Verify:** same P1.1–P1.4 checks pass on the new protein; confirm zero changes were needed under `src/` — only new files under `data/` — which is itself the evidence for generalization, not an assumption.

- [ ] **P6.2 — Ablation: ESM-only vs. contact-aware policy**
  **Goal:** a number that supports (or honestly fails to support) the "contact-awareness helps" claim.
  **Do:** run the inner loop with `contact_violation` weight zeroed out (ESM-only) vs. the evolved contact-aware policy, same held-out protein/corruptions; compare median hidden scores.
  **Verify:** report both medians side by side regardless of outcome. Only include the ablation in the pitch if contact-aware's median genuinely exceeds ESM-only's — if it doesn't, that's a real finding to report, not a result to massage (the same evidentiary standard C1 asks for elsewhere).

---

## Phase 7 — Final claims-alignment check

- [ ] **P7.1 — The spoken pitch matches what was built**
  **Goal:** close the loop on C1: the only claim made out loud is the one the memo licenses.
  **Do:** write out the actual demo narration/pitch script.
  **Verify:** line-by-line, confirm the script never claims therapeutic design, function restoration, or discovery of protein physics, and does claim — using this or materially equivalent wording — "a falsifiable, self-correcting protein repair harness in which representations and executable repair rules co-evolve under a hidden structural verifier." If the script says more than the system does, cut the script, not the verification.

---

## Appendix — Design-pattern → implementation cross-reference

| Concept | Where it's implemented |
|---|---|
| Raw observation | Sequence + predicted PDB (P2.5) |
| State grounding | `state.json` residue/contact graph (P2.7) |
| Action | Amino-acid substitution (P2.3) |
| Predicted transition | Expected PLL/pLDDT/contact change (P2.4) |
| Mechanism program | Policy DSL + interpreter (P3.1, P3.2) |
| New observation | Reconstructed candidate structure (P2.5) |
| Counterexample | Predicted-improved-but-hidden-worsened (P3.3) |
| Representation revision | Reweight/activate a `state.json` feature (P3.5) |
| Mechanism revision | Change a proposal field (P3.5) |

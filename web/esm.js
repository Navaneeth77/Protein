/* In-browser ESM-2 masked-marginal scoring.
 *
 * A port of src/agent/esm_score.py's estimator onto transformers.js. It runs
 * Xenova/esm2_t12_35M_UR50D — the ONNX export of the very model the Python path
 * uses (facebook/esm2_t12_35M_UR50D) — so a sequence pasted into the page is
 * scored by the same estimator that found the fold-breaking proline at position
 * 50, not by an approximation of it.
 *
 * This is the scorer, not the folder. ESMFold is a different, 8.4 GB model and
 * is not involved here: a pasted sequence gets sequence-plausibility signals and
 * the policy search that runs on them, and gets no predicted structure, no
 * TM-score and no hidden score.
 *
 * Cost: masked-marginal scoring is one forward pass per residue (Meier et al.),
 * deliberately not the cheaper wild-type-marginal shortcut, matching the Python
 * implementation. That is why callers get progress callbacks and why sequence
 * length is capped.
 */

'use strict';

import { AA_ALPHABET } from './policy.js';

const CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6';
const MODEL_ID = 'Xenova/esm2_t12_35M_UR50D';

// ESM-2 vocabulary constants, asserted against the real tokenizer on load.
const MASK_ID = 32;
const CLS_ID = 0;
const EOS_ID = 2;
const OFFSET = 1; // ESM prepends <cls>, so residue i is token i + 1.

const BATCH_SIZE = 16; // mirrors esm_score.BATCH_SIZE

let _lib = null;
let _tok = null;
let _model = null;
let _aaTokenIds = null;
let _precision = null;

const _matrixCache = new Map();

export const stats = { cache_hits: 0, cache_misses: 0, forward_batches: 0 };

/** Load transformers.js, the tokenizer and the model. Idempotent. */
export async function load({ onStatus } = {}) {
  if (_model) return;
  const say = (m) => onStatus && onStatus(m);

  say('loading transformers.js…');
  _lib = await import(/* @vite-ignore */ CDN);
  _lib.env.allowLocalModels = false;

  say('loading ESM-2 tokenizer…');
  _tok = await _lib.AutoTokenizer.from_pretrained(MODEL_ID);

  const vocab = _tok.model.tokens_to_ids;
  _aaTokenIds = [...AA_ALPHABET].map((aa) => {
    const id = vocab.get(aa);
    if (id === undefined) throw new Error(`ESM vocabulary has no token for ${aa}`);
    return id;
  });
  for (const [name, want, got] of [
    ['<mask>', MASK_ID, vocab.get('<mask>')],
    ['<cls>', CLS_ID, vocab.get('<cls>')],
    ['<eos>', EOS_ID, vocab.get('<eos>')],
  ]) {
    if (got !== want) throw new Error(`unexpected ${name} id: expected ${want}, got ${got}`);
  }

  // Precision is a correctness decision here, not a size one. Measured against
  // the PyTorch masked-marginal matrix for corrupted 1PGB (56 residues):
  //
  //   dtype  size   max |Δp|   top-5 suspicious positions
  //   q8     35 MB  0.081      50,21,34,9,35   <- rank 5 is WRONG
  //   fp16   68 MB  0.00038    50,21,34,9,55   <- matches PyTorch
  //
  // int8 error is large enough to reorder the position ranking, and the patched
  // policy selects five sites, so q8 would hand back a different search than the
  // reference pipeline. fp16 costs 33 MB more and scores at the same speed.
  // q8 remains a fallback for devices where fp16 will not load, and the caller
  // is told when that happens so the page can say the numbers are approximate.
  say('downloading ESM-2 weights (~68 MB, cached by the browser afterwards)…');
  const opts = (dtype) => ({
    dtype,
    progress_callback: (p) => {
      if (p && p.status === 'progress' && p.progress != null) {
        say(`downloading ESM-2 weights… ${Math.round(p.progress)}%`);
      }
    },
  });
  try {
    _model = await _lib.AutoModelForMaskedLM.from_pretrained(MODEL_ID, opts('fp16'));
    _precision = 'fp16';
  } catch (err) {
    say('fp16 unavailable on this device, falling back to int8…');
    _model = await _lib.AutoModelForMaskedLM.from_pretrained(MODEL_ID, opts('q8'));
    _precision = 'q8';
  }
  say('ready');
}

/** 'fp16' (matches the PyTorch pipeline) or 'q8' (approximate fallback). */
export function precision() {
  return _precision;
}

export function isLoaded() {
  return Boolean(_model);
}

function encode(sequence) {
  const ids = new BigInt64Array(sequence.length + 2);
  ids[0] = BigInt(CLS_ID);
  for (let i = 0; i < sequence.length; i++) {
    const id = _tok.model.tokens_to_ids.get(sequence[i]);
    if (id === undefined) throw new Error(`unscorable residue ${sequence[i]} at ${i + 1}`);
    ids[i + 1] = BigInt(id);
  }
  ids[sequence.length + 1] = BigInt(EOS_ID);
  return ids;
}

function softmaxOver(logitsRow, indices) {
  let max = -Infinity;
  for (const k of indices) if (logitsRow[k] > max) max = logitsRow[k];
  let sum = 0;
  const out = new Array(indices.length);
  for (let i = 0; i < indices.length; i++) {
    const e = Math.exp(logitsRow[indices[i]] - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

/**
 * (L, 20) masked-marginal probabilities, columns ordered by AA_ALPHABET.
 * Mirrors esm_score.masked_marginal_matrix.
 */
export async function maskedMarginalMatrix(sequence, { onProgress, signal } = {}) {
  const hit = _matrixCache.get(sequence);
  if (hit) { stats.cache_hits += 1; return hit; }
  stats.cache_misses += 1;

  if (!_model) throw new Error('call load() first');

  const L = sequence.length;
  const T = L + 2;
  const base = encode(sequence);
  const probs = Array.from({ length: L }, () => new Array(20).fill(0));

  for (let start = 0; start < L; start += BATCH_SIZE) {
    if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError');

    const positions = [];
    for (let p = start; p < Math.min(start + BATCH_SIZE, L); p++) positions.push(p);
    const B = positions.length;

    const inputIds = new BigInt64Array(B * T);
    const attention = new BigInt64Array(B * T).fill(1n);
    for (let r = 0; r < B; r++) {
      inputIds.set(base, r * T);
      inputIds[r * T + positions[r] + OFFSET] = BigInt(MASK_ID);
    }

    const out = await _model({
      input_ids: new _lib.Tensor('int64', inputIds, [B, T]),
      attention_mask: new _lib.Tensor('int64', attention, [B, T]),
    });
    stats.forward_batches += 1;

    const logits = out.logits;            // [B, T, V]
    const V = logits.dims[2];
    const data = logits.data;
    for (let r = 0; r < B; r++) {
      const pos = positions[r];
      const rowStart = (r * T + pos + OFFSET) * V;
      const row = data.subarray ? data.subarray(rowStart, rowStart + V)
                                : data.slice(rowStart, rowStart + V);
      probs[pos] = softmaxOver(row, _aaTokenIds);
    }

    if (onProgress) onProgress(Math.min(start + BATCH_SIZE, L), L);
    // Yield so the progress bar can paint between batches.
    await new Promise((r) => setTimeout(r, 0));
  }

  _matrixCache.set(sequence, probs);
  return probs;
}

/** Per-position -log p(observed residue). esm_score.residue_surprisal */
export function residueSurprisal(sequence, matrix) {
  return [...sequence].map((aa, i) => {
    const p = matrix[i][AA_ALPHABET.indexOf(aa)];
    return -Math.log(Math.max(p, 1e-12));
  });
}

/** Sum of per-position log-likelihoods. esm_score.pseudo_log_likelihood */
export function pseudoLogLikelihood(sequence, matrix) {
  let total = 0;
  for (let i = 0; i < sequence.length; i++) {
    const p = matrix[i][AA_ALPHABET.indexOf(sequence[i])];
    total += Math.log(Math.max(p, 1e-12));
  }
  return total;
}

/**
 * Top-n substitutions at `position`, incumbent excluded, sorted by descending
 * probability then ascending letter. esm_score.substitution_ranking
 */
export function substitutionRanking(matrix, sequence, position, topN) {
  const current = sequence[position];
  const ranked = [...AA_ALPHABET]
    .map((aa, k) => [aa, matrix[position][k]])
    .filter(([aa]) => aa !== current)
    .sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : 1));
  return ranked.slice(0, topN);
}

/** Deficit against the mean, clipped to 0-1. grounder.py::_deficit */
export function deficit(values) {
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  if (mean <= 0) return values.map(() => 0);
  return values.map((v) => Math.min(Math.max((mean - v) / mean, 0), 1));
}

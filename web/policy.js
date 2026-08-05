/* A faithful browser port of the ReFold policy DSL and its interpreter.
 *
 * Ported from, and kept deliberately line-comparable with:
 *   src/agent/policy.py               validate_policy
 *   src/agent/policy_interpreter.py   _normalise, position_scores,
 *                                     select_positions, enumerate_candidates,
 *                                     prerank_candidates
 *   src/agent/policy.schema.yaml      the DSL's ranges
 *   src/constants.py                  the residue-class partition
 *
 * The interpreter is pure arithmetic over a grounded state — no model, no
 * dynamic execution — which is exactly why it can run here at all. Anything
 * requiring ESM-2 (substitution rankings, pseudo-log-likelihood) is injected
 * from esm.js rather than reimplemented.
 *
 * Tie-breaking matters and is easy to get subtly wrong, so it mirrors NumPy:
 * `select_positions` uses np.lexsort((arange, -score)), i.e. descending score
 * with ties resolved toward the LOWER index.
 */

'use strict';

export const AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY';
export const AA_SET = new Set(AA_ALPHABET);

const HYDROPHOBIC = new Set('AVLIMFWP');
const POLAR = new Set('STNQCYG');
const CHARGED = new Set('DEKRH');

export function residueClass(aa) {
  if (HYDROPHOBIC.has(aa)) return 'hydrophobic';
  if (POLAR.has(aa)) return 'polar';
  if (CHARGED.has(aa)) return 'charged';
  throw new Error(`not a standard amino acid: ${aa}`);
}

// src/agent/grounder.py::SCORABLE_FEATURES — the only names a policy may score.
export const SCORABLE_FEATURES = [
  'esm_surprisal',
  'low_plddt',
  'contact_violation',
  'long_range_contact_violation',
];

export const WEIGHT_SUM_TOLERANCE = 1e-6;
export const EDIT_PENALTY_LAMBDA = 0.5;
export const SHORTLIST_SIZE = 3;

/* ------------------------------------------------------------------ validation */

export class PolicyValidationError extends Error {}

/** Mirrors policy.py::validate_policy plus the schema's numeric ranges.
 *  Rejects; never coerces. Returns the policy unchanged on success. */
export function validatePolicy(policy) {
  if (!policy || typeof policy !== 'object') {
    throw new PolicyValidationError('policy must be a mapping');
  }
  for (const key of ['position_score', 'proposal']) {
    if (!(key in policy)) throw new PolicyValidationError(`missing required key: ${key}`);
  }

  const weights = policy.position_score;
  const names = Object.keys(weights);
  if (names.length < 1) {
    throw new PolicyValidationError('position_score needs at least one weight');
  }
  const unknown = names.filter((n) => !SCORABLE_FEATURES.includes(n)).sort();
  if (unknown.length) {
    throw new PolicyValidationError(
      `position_score refers to feature(s) the grounder does not compute: ` +
        `[${unknown.join(', ')}]; allowed: ${[...SCORABLE_FEATURES].sort().join(', ')}`
    );
  }
  for (const [k, v] of Object.entries(weights)) {
    if (typeof v !== 'number' || !isFinite(v)) {
      throw new PolicyValidationError(`position_score.${k} must be a number`);
    }
    if (v < 0 || v > 1) {
      throw new PolicyValidationError(`position_score.${k} must be within [0, 1], got ${v}`);
    }
  }
  const total = names.reduce((s, k) => s + Number(weights[k]), 0);
  if (Math.abs(total - 1.0) > WEIGHT_SUM_TOLERANCE) {
    throw new PolicyValidationError(
      `position_score weights must sum to 1.0, got ${total.toFixed(6)}`
    );
  }

  const p = policy.proposal;
  const ranges = {
    positions: [1, 10],
    substitutions_per_position: [1, 19],
    max_total_edits: [1, 10],
  };
  for (const [k, [lo, hi]] of Object.entries(ranges)) {
    if (!(k in p)) throw new PolicyValidationError(`proposal.${k} is required`);
    if (!Number.isInteger(p[k])) {
      throw new PolicyValidationError(`proposal.${k} must be an integer, got ${p[k]}`);
    }
    if (p[k] < lo || p[k] > hi) {
      throw new PolicyValidationError(`proposal.${k} must be within [${lo}, ${hi}], got ${p[k]}`);
    }
  }
  if (typeof p.preserve_residue_class !== 'boolean') {
    throw new PolicyValidationError('proposal.preserve_residue_class must be a boolean');
  }
  const extra = Object.keys(p).filter(
    (k) => !['positions', 'substitutions_per_position', 'preserve_residue_class', 'max_total_edits'].includes(k)
  );
  if (extra.length) {
    throw new PolicyValidationError(`proposal has unknown key(s): ${extra.join(', ')}`);
  }
  return policy;
}

/* ------------------------------------------------------------------ scoring */

/** Min-max to 0-1. Features live on different scales — an unbounded surprisal
 *  next to a 0-1 pLDDT deficit — so without this the weights would not mean
 *  what the policy says they mean. policy_interpreter.py::_normalise */
export function normalise(values) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (hi - lo < 1e-12) return values.map(() => 0);
  return values.map((v) => (v - lo) / (hi - lo));
}

export function positionScores(policy, state) {
  const residues = state.residues;
  const score = new Array(residues.length).fill(0);
  for (const [feature, weight] of Object.entries(policy.position_score)) {
    const raw = residues.map((r) => Number(r[feature]));
    const norm = normalise(raw);
    for (let i = 0; i < score.length; i++) score[i] += Number(weight) * norm[i];
  }
  return score;
}

/** Top-k by score, ties toward the lower index. */
export function selectPositions(policy, state) {
  const score = positionScores(policy, state);
  const order = score.map((s, i) => [s, i]);
  order.sort((a, b) => (b[0] - a[0]) || (a[1] - b[1]));
  return order.slice(0, Number(policy.proposal.positions)).map(([, i]) => i);
}

/* ------------------------------------------------------------------ candidates */

function mutationsVsOrigin(origin, sequence) {
  const out = [];
  for (let i = 0; i < sequence.length; i++) {
    if (origin[i] !== sequence[i]) out.push({ position: i, from: origin[i], to: sequence[i] });
  }
  return out;
}

/**
 * policy_interpreter.py::enumerate_candidates.
 *
 * `substitutionRanking(seq, pos, topN)` must return [aa, prob] pairs sorted by
 * descending probability then ascending letter, with the incumbent excluded —
 * matching esm_score.substitution_ranking.
 */
export function enumerateCandidates(policy, state, origin, substitutionRanking) {
  const incumbent = state.residues.map((r) => r.aa).join('');
  origin = origin || incumbent;

  const perPosition = Number(policy.proposal.substitutions_per_position);
  const preserveClass = Boolean(policy.proposal.preserve_residue_class);
  const maxEdits = Number(policy.proposal.max_total_edits);

  const scores = positionScores(policy, state);
  const seen = new Set([incumbent]);
  const candidates = [];

  for (const pos of selectPositions(policy, state)) {
    const current = incumbent[pos];
    const currentClass = residueClass(current);
    // Ask for extra so the class filter cannot starve the shortlist.
    const pool = substitutionRanking(incumbent, pos, 19);
    let kept = 0;

    for (const [aa, prob] of pool) {
      if (kept >= perPosition) break;
      if (aa === current) continue;
      if (preserveClass && residueClass(aa) !== currentClass) continue;

      const seq = incumbent.slice(0, pos) + aa + incumbent.slice(pos + 1);
      if (seen.has(seq)) continue;
      const mutations = mutationsVsOrigin(origin, seq);
      if (mutations.length > maxEdits) continue;

      seen.add(seq);
      kept += 1;
      candidates.push({
        sequence: seq,
        position: pos,
        from_aa: current,
        to_aa: aa,
        parent_sequence: incumbent,
        position_score: scores[pos],
        substitution_prob: prob,
        mutations,
        edit_count: mutations.length,
        prerank_score: null,
        label: `${current}${pos + 1}${aa}`,
      });
    }
  }
  return candidates;
}

/**
 * policy_interpreter.py::prerank_candidates.
 *
 * score = PLL(candidate) - lambda * edit_count, ties toward fewer edits then
 * sequence order. `pll` is async here because in the browser each call is a
 * fresh set of masked forward passes.
 */
export async function prerankCandidates(candidates, pll, shortlistSize = SHORTLIST_SIZE, lam = EDIT_PENALTY_LAMBDA, onProgress) {
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    c.prerank_score = (await pll(c.sequence)) - lam * c.edit_count;
    if (onProgress) onProgress(i + 1, candidates.length);
  }
  const ranked = [...candidates].sort(
    (a, b) =>
      (b.prerank_score - a.prerank_score) ||
      (a.edit_count - b.edit_count) ||
      (a.sequence < b.sequence ? -1 : a.sequence > b.sequence ? 1 : 0)
  );
  return ranked.slice(0, shortlistSize);
}

/* ------------------------------------------------------------------ helpers */

/** Parse FASTA or a raw sequence into uppercase one-letter codes.
 *  Throws with a specific reason rather than silently dropping characters. */
export function parseSequence(text, { minLength = 10, maxLength = 400 } = {}) {
  const lines = String(text).split(/\r?\n/);
  const body = lines.filter((l) => !l.trim().startsWith('>')).join('');
  const seq = body.replace(/\s+/g, '').toUpperCase();

  if (!seq) throw new Error('no sequence found');

  const bad = [...new Set([...seq].filter((c) => !AA_SET.has(c)))];
  if (bad.length) {
    throw new Error(
      `not one of the 20 standard amino acids: ${bad.slice(0, 6).map((c) => `"${c}"`).join(', ')}` +
        (bad.includes('X') || bad.includes('B') || bad.includes('Z')
          ? ' — ambiguity codes are not scorable'
          : '')
    );
  }
  if (seq.length < minLength) throw new Error(`too short: ${seq.length} residues, minimum ${minLength}`);
  if (seq.length > maxLength) {
    throw new Error(
      `too long: ${seq.length} residues, maximum ${maxLength}. Masked-marginal ` +
        `scoring costs one forward pass per residue, so this is a time limit, not a model limit.`
    );
  }
  return seq;
}

/** Build the minimal grounded state the interpreter needs.
 *  `features` maps each SCORABLE_FEATURE to a per-residue array; anything
 *  absent (because it needs a predicted structure) is filled with zeros and
 *  reported back so the UI can say so rather than imply the number is real. */
export function buildState(sequence, features) {
  const missing = SCORABLE_FEATURES.filter((f) => !features[f]);
  const residues = [...sequence].map((aa, i) => {
    const r = { position: i, aa, residue_class: residueClass(aa) };
    for (const f of SCORABLE_FEATURES) r[f] = features[f] ? features[f][i] : 0;
    return r;
  });
  return { state: { sequence_length: sequence.length, residues }, missing };
}

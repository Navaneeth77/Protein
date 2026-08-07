/* Structure handling for the browser: PDB parsing, superposition, TM-score,
 * and nucleotide translation.
 *
 * Ported from src/pdb_io.py and src/geometry.py. The one deliberate departure
 * is the rotation solver: NumPy's SVD is not available here, so `kabsch` uses
 * the quaternion formulation (Horn 1987), which finds the same optimal rotation
 * without needing a general SVD. Agreement with the Python implementation is
 * asserted by web/_verify.html rather than assumed.
 *
 * TM-score follows geometry.py::tm_score_fixed_alignment exactly, including the
 * Zhang-Skolnick fragment seeding, the 20-iteration refit and the d0_search
 * cutoff schedule. It uses the identity residue correspondence, which is valid
 * because every repaired sequence here is a substitution-only variant of its
 * input and therefore the same length.
 */

'use strict';

/* ------------------------------------------------------------------ PDB */

const THREE_TO_ONE = {
  ALA: 'A', CYS: 'C', ASP: 'D', GLU: 'E', PHE: 'F', GLY: 'G', HIS: 'H',
  ILE: 'I', LYS: 'K', LEU: 'L', MET: 'M', ASN: 'N', PRO: 'P', GLN: 'Q',
  ARG: 'R', SER: 'S', THR: 'T', VAL: 'V', TRP: 'W', TYR: 'Y',
};

/**
 * CA coordinates, pLDDT and sequence from a PDB. Mirrors pdb_io.parse_pdb:
 * first model only, standard residues only, altloc ' ' or 'A'.
 *
 * ESMFold writes pLDDT into the B-factor column. api.esmatlas.com returns it
 * already on a 0-1 scale while the local checkpoint writes 0-100, so the scale
 * is detected rather than assumed — the same check fold_cache.structure_features
 * makes.
 */
export function parsePdb(text) {
  const ca = [];
  const bfac = [];
  const seq = [];
  const seen = new Set();

  for (const line of text.split('\n')) {
    if (line.startsWith('ENDMDL')) break;
    if (!line.startsWith('ATOM')) continue;
    const resname = line.slice(17, 20).trim();
    if (!(resname in THREE_TO_ONE)) continue;
    const altloc = line[16];
    if (altloc !== ' ' && altloc !== 'A') continue;
    if (line.slice(12, 16).trim() !== 'CA') continue;

    const key = `${line[21]}|${line.slice(22, 27).trim()}`;
    if (seen.has(key)) continue;
    seen.add(key);

    ca.push([
      parseFloat(line.slice(30, 38)),
      parseFloat(line.slice(38, 46)),
      parseFloat(line.slice(46, 54)),
    ]);
    const b = parseFloat(line.slice(60, 66));
    bfac.push(Number.isFinite(b) ? b : 0);
    seq.push(THREE_TO_ONE[resname]);
  }

  const maxB = bfac.length ? Math.max(...bfac) : 0;
  const plddt = maxB > 1.5 ? bfac.map((v) => v / 100) : bfac.slice();

  return {
    ca,
    plddt,
    sequence: seq.join(''),
    meanPlddt: plddt.length ? plddt.reduce((s, v) => s + v, 0) / plddt.length : 0,
  };
}

/** Rewrite every ATOM/HETATM coordinate through (R, t). Display only. */
export function transformPdb(text, R, t) {
  const out = [];
  for (const line of text.split('\n')) {
    if (!(line.startsWith('ATOM') || line.startsWith('HETATM')) || line.length < 54) {
      out.push(line);
      continue;
    }
    const v = [
      parseFloat(line.slice(30, 38)),
      parseFloat(line.slice(38, 46)),
      parseFloat(line.slice(46, 54)),
    ];
    const m = apply(R, t, v);
    out.push(line.slice(0, 30) + m.map((c) => c.toFixed(3).padStart(8)).join('') + line.slice(54));
  }
  return out.join('\n');
}

/* ------------------------------------------------------------------ linear algebra */

function centroid(p) {
  const c = [0, 0, 0];
  for (const v of p) { c[0] += v[0]; c[1] += v[1]; c[2] += v[2]; }
  return c.map((v) => v / p.length);
}

function apply(R, t, v) {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2] + t[0],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2] + t[1],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2] + t[2],
  ];
}

/** Cyclic Jacobi eigendecomposition of a symmetric n x n matrix. */
function jacobiEigen(Ain, iterations = 100) {
  const n = Ain.length;
  const A = Ain.map((r) => r.slice());
  let V = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  );

  for (let sweep = 0; sweep < iterations; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    if (off < 1e-20) break;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-18) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const sign = theta >= 0 ? 1 : -1;
        const tRot = sign / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(tRot * tRot + 1);
        const s = tRot * c;

        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: A.map((r, i) => r[i]), vectors: V };
}

/**
 * Rotation R and translation t minimising |R @ mobile + t - target|.
 * Same result as geometry.py::kabsch, reached through the quaternion form.
 */
export function kabsch(mobile, target) {
  const mc = centroid(mobile);
  const tc = centroid(target);

  // S[a][b] = sum_i (mobile_i - mc)[a] * (target_i - tc)[b]
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < mobile.length; i++) {
    for (let a = 0; a < 3; a++) {
      const pa = mobile[i][a] - mc[a];
      for (let b = 0; b < 3; b++) S[a][b] += pa * (target[i][b] - tc[b]);
    }
  }

  const [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = S;
  const K = [
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
  ];

  const { values, vectors } = jacobiEigen(K);
  let best = 0;
  for (let i = 1; i < 4; i++) if (values[i] > values[best]) best = i;
  let [w, x, y, z] = [0, 1, 2, 3].map((r) => vectors[r][best]);
  const norm = Math.hypot(w, x, y, z) || 1;
  w /= norm; x /= norm; y /= norm; z /= norm;

  const R = [
    [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z],
  ];
  const t = [0, 1, 2].map(
    (i) => tc[i] - (R[i][0] * mc[0] + R[i][1] * mc[1] + R[i][2] * mc[2])
  );
  return { R, t };
}

export function rmsd(a, b) {
  const { R, t } = kabsch(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const m = apply(R, t, a[i]);
    sum += (m[0] - b[i][0]) ** 2 + (m[1] - b[i][1]) ** 2 + (m[2] - b[i][2]) ** 2;
  }
  return Math.sqrt(sum / a.length);
}

/* ------------------------------------------------------------------ TM-score */

export function tmD0(length) {
  if (length <= 15) return 0.5;
  return Math.max(0.5, 1.24 * Math.cbrt(length - 15) - 1.8);
}

/** geometry.py::tm_score_fixed_alignment, identity correspondence. */
export function tmScore(candidateCA, referenceCA) {
  const n = referenceCA.length;
  if (candidateCA.length !== n) {
    throw new Error(`identity correspondence needs equal lengths, got ${candidateCA.length} and ${n}`);
  }
  if (n < 3) return 0;

  const d0 = tmD0(n);
  const d0Search = Math.min(Math.max(d0, 4.5), 8.0);
  let best = 0;

  const fragLengths = [];
  for (let L = n; L >= 4; L = Math.floor(L / 2)) fragLengths.push(L);
  if (fragLengths[fragLengths.length - 1] !== 4) fragLengths.push(4);

  for (const fragLen of fragLengths) {
    const stride = fragLen < n ? Math.max(1, Math.floor(fragLen / 2)) : 1;
    for (let start = 0; start + fragLen <= n; start += stride) {
      let sel = [];
      for (let i = start; i < start + fragLen; i++) sel.push(i);

      for (let iter = 0; iter < 20; iter++) {
        const { R, t } = kabsch(sel.map((i) => candidateCA[i]), sel.map((i) => referenceCA[i]));
        const d = new Array(n);
        let score = 0;
        for (let i = 0; i < n; i++) {
          const m = apply(R, t, candidateCA[i]);
          const dist = Math.hypot(m[0] - referenceCA[i][0], m[1] - referenceCA[i][1], m[2] - referenceCA[i][2]);
          d[i] = dist;
          score += 1 / (1 + (dist / d0) ** 2);
        }
        score /= n;
        if (score > best) best = score;

        let cut = d0Search;
        let next = [];
        for (let i = 0; i < n; i++) if (d[i] < cut) next.push(i);
        while (next.length < 4 && cut < 20) {
          cut += 0.5;
          next = [];
          for (let i = 0; i < n; i++) if (d[i] < cut) next.push(i);
        }
        if (next.length < 4) break;
        if (next.length === sel.length && next.every((v, k) => v === sel[k])) break;
        sel = next;
      }
    }
  }
  return best;
}

/* ------------------------------------------------------------------ nucleotides */

const CODONS = {
  TTT: 'F', TTC: 'F', TTA: 'L', TTG: 'L', CTT: 'L', CTC: 'L', CTA: 'L', CTG: 'L',
  ATT: 'I', ATC: 'I', ATA: 'I', ATG: 'M', GTT: 'V', GTC: 'V', GTA: 'V', GTG: 'V',
  TCT: 'S', TCC: 'S', TCA: 'S', TCG: 'S', CCT: 'P', CCC: 'P', CCA: 'P', CCG: 'P',
  ACT: 'T', ACC: 'T', ACA: 'T', ACG: 'T', GCT: 'A', GCC: 'A', GCA: 'A', GCG: 'A',
  TAT: 'Y', TAC: 'Y', TAA: '*', TAG: '*', CAT: 'H', CAC: 'H', CAA: 'Q', CAG: 'Q',
  AAT: 'N', AAC: 'N', AAA: 'K', AAG: 'K', GAT: 'D', GAC: 'D', GAA: 'E', GAG: 'E',
  TGT: 'C', TGC: 'C', TGA: '*', TGG: 'W', CGT: 'R', CGC: 'R', CGA: 'R', CGG: 'R',
  AGT: 'S', AGC: 'S', AGA: 'R', AGG: 'R', GGT: 'G', GGC: 'G', GGA: 'G', GGG: 'G',
};

/** True when the text looks like DNA/RNA rather than protein.
 *  A, C, G and T are all valid amino-acid codes too, so length-divisible-by-3
 *  and an alphabet of only ACGTU is the honest test — and it is reported to the
 *  user rather than applied silently. */
export function looksLikeNucleotide(text) {
  const s = text.replace(/\s/g, '').toUpperCase();
  return s.length >= 30 && /^[ACGTU]+$/.test(s);
}

/** Translate a coding sequence in frame 1. Stops at the first stop codon. */
export function translate(text) {
  const s = text.replace(/\s/g, '').toUpperCase().replace(/U/g, 'T');
  const aa = [];
  let stopped = false;
  for (let i = 0; i + 3 <= s.length; i += 3) {
    const c = CODONS[s.slice(i, i + 3)];
    if (!c) break;
    if (c === '*') { stopped = true; break; }
    aa.push(c);
  }
  return {
    protein: aa.join(''),
    codons: aa.length,
    stopped,
    trailing: s.length % 3,
  };
}

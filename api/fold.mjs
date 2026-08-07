/**
 * Structure prediction proxy.
 *
 * The browser cannot call ESM Atlas directly — its preflight returns 403 — so
 * this forwards the request server-side. That is the only reason this function
 * exists: it holds no state, has no dependencies, and does no science. Every
 * other part of the repair loop (ESM-2 scoring, the policy interpreter,
 * candidate enumeration) already runs client-side in web/esm.js and
 * web/policy.js.
 *
 * What it is NOT: the 8.4 GB ESMFold checkpoint running on Vercel. It is a
 * ~40-line HTTP forwarder to Meta's hosted ESMFold v1, which folds a 56-residue
 * chain in about two seconds. A serverless function can do that; it could never
 * host the model itself.
 *
 * POST { sequence: "MTYK..." } -> { pdb, sequence, length, ms }
 */

const UPSTREAM = 'https://api.esmatlas.com/foldSequence/v1/pdb/';
const AA = /^[ACDEFGHIKLMNPQRSTVWY]+$/;

// ESM Atlas rejects long chains, and every residue costs wall-clock time in a
// function that Vercel will cut off. 400 keeps a fold comfortably inside the
// limit on the Hobby plan.
const MIN_LEN = 10;
const MAX_LEN = 400;

// Successful folds of a 56-residue chain come back in 1-2s, so 12s per attempt
// is already ~8x headroom. Three attempts plus backoff is ~40s worst case, which
// has to fit inside the function's maxDuration (60s, set in vercel.json — the
// Hobby default is 10s and would kill the retry loop mid-flight).
const UPSTREAM_TIMEOUT_MS = 12_000;
const ATTEMPTS = 3;
const BACKOFF_MS = 1_200;

function fail(res, status, title, detail) {
  res.status(status).json({ error: title, detail });
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return fail(res, 405, 'method not allowed', 'POST a JSON body: {"sequence": "..."}');
  }

  let body = req.body;
  if (typeof body === 'string') {
    try { body = JSON.parse(body); } catch { return fail(res, 400, 'malformed JSON body'); }
  }
  const sequence = String(body?.sequence ?? '').trim().toUpperCase();

  if (!sequence) return fail(res, 400, 'missing sequence');
  if (!AA.test(sequence)) {
    const bad = [...new Set([...sequence].filter((c) => !AA.test(c)))].slice(0, 6);
    return fail(
      res,
      400,
      'sequence is not protein',
      `expected the 20 standard amino acids; found ${bad.map((c) => `"${c}"`).join(', ')}`
    );
  }
  if (sequence.length < MIN_LEN || sequence.length > MAX_LEN) {
    return fail(
      res,
      400,
      'sequence length out of range',
      `got ${sequence.length}; this endpoint accepts ${MIN_LEN}-${MAX_LEN} residues`
    );
  }

  const started = Date.now();
  let lastDetail = '';

  // ESM Atlas is a free public service and returns a transient 504
  // ("Endpoint request timed out") fairly often — observed answering the same
  // 56-residue sequence in 2.2s and then timing out on the next call minutes
  // later. Retrying recovers most of those. A 400 is the caller's fault and is
  // never retried.
  for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
    try {
      const upstream = await fetch(UPSTREAM, {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: sequence,
        signal: controller.signal,
      });
      const text = await upstream.text();

      if (upstream.status === 400) {
        return fail(res, 400, 'the folding service rejected this sequence', text.slice(0, 300));
      }
      if (!upstream.ok) {
        lastDetail = `ESM Atlas returned ${upstream.status}. ${text.slice(0, 200)}`;
      } else if (!text.includes('ATOM')) {
        lastDetail = `no coordinates in the response. ${text.slice(0, 200)}`;
      } else {
        // Predictions are deterministic for a sequence, so they cache well.
        res.setHeader('Cache-Control', 'public, s-maxage=86400, stale-while-revalidate=604800');
        return res.status(200).json({
          pdb: text,
          sequence,
          length: sequence.length,
          ms: Date.now() - started,
          attempts: attempt,
          source: 'ESMFold v1 via api.esmatlas.com',
        });
      }
    } catch (err) {
      lastDetail =
        err?.name === 'AbortError'
          ? `no response within ${UPSTREAM_TIMEOUT_MS / 1000}s`
          : String(err?.message ?? err);
    } finally {
      clearTimeout(timer);
    }

    if (attempt < ATTEMPTS) {
      await new Promise((r) => setTimeout(r, BACKOFF_MS * attempt));
    }
  }

  return fail(
    res,
    502,
    'the folding service did not answer',
    `${ATTEMPTS} attempts over ${Math.round((Date.now() - started) / 1000)}s. ` +
      `Last error: ${lastDetail}. This is api.esmatlas.com being unavailable, not a ` +
      `problem with your sequence — the analysis below still runs without it.`
  );
}

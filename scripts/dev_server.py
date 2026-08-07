"""Serve web/ locally with the /api/fold endpoint that Vercel provides in production.

`python -m http.server` is enough for the replay sections, but section 5 needs a
structure prediction, and the browser cannot call ESM Atlas directly — its
preflight returns 403. In production that proxy is api/fold.mjs, a Vercel
function. This is the local stand-in so the repair flow can be developed and
tested without deploying.

It deliberately mirrors api/fold.mjs's contract exactly — same route, same JSON
shape, same validation and the same error bodies — so a bug found here is a bug
in production too.

    python scripts/dev_server.py           # http://localhost:8765
    python scripts/dev_server.py --port N
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

UPSTREAM = "https://api.esmatlas.com/foldSequence/v1/pdb/"
AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
MIN_LEN, MAX_LEN = 10, 400
TIMEOUT_S = 12          # ~8x the observed 1-2s success latency
ATTEMPTS = 3
BACKOFF_S = 1.2


class Handler(SimpleHTTPRequestHandler):
    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.rstrip("/") != "/api/fold":
            return self._json(404, {"error": "not found"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "malformed JSON body"})

        sequence = str(payload.get("sequence", "")).strip().upper()
        if not sequence:
            return self._json(400, {"error": "missing sequence"})
        if not AA.match(sequence):
            bad = sorted({c for c in sequence if not AA.match(c)})[:6]
            return self._json(400, {
                "error": "sequence is not protein",
                "detail": "expected the 20 standard amino acids; found "
                          + ", ".join(f'"{c}"' for c in bad),
            })
        if not (MIN_LEN <= len(sequence) <= MAX_LEN):
            return self._json(400, {
                "error": "sequence length out of range",
                "detail": f"got {len(sequence)}; this endpoint accepts "
                          f"{MIN_LEN}-{MAX_LEN} residues",
            })

        started = time.time()
        last_detail = ""

        # ESM Atlas returns a transient 504 fairly often — it answered the same
        # 56-residue sequence in 2.2s and then timed out on the next call. Retry
        # those; a 400 is the caller's fault and is never retried. Mirrors the
        # retry policy in api/fold.mjs.
        for attempt in range(1, ATTEMPTS + 1):
            request = urllib.request.Request(
                UPSTREAM, data=sequence.encode(),
                headers={"Content-Type": "text/plain"}, method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                    text = response.read().decode("utf-8", "replace")
                if "ATOM" not in text:
                    last_detail = f"no coordinates in the response. {text[:200]}"
                else:
                    return self._json(200, {
                        "pdb": text,
                        "sequence": sequence,
                        "length": len(sequence),
                        "ms": int((time.time() - started) * 1000),
                        "attempts": attempt,
                        "source": "ESMFold v1 via api.esmatlas.com",
                    })
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:200]
                if exc.code == 400:
                    return self._json(400, {
                        "error": "the folding service rejected this sequence",
                        "detail": detail,
                    })
                last_detail = f"ESM Atlas returned {exc.code}. {detail}"
            except Exception as exc:  # timeout, DNS, connection reset
                last_detail = str(exc)

            if attempt < ATTEMPTS:
                time.sleep(BACKOFF_S * attempt)

        elapsed = int(time.time() - started)
        return self._json(502, {
            "error": "the folding service did not answer",
            "detail": (
                f"{ATTEMPTS} attempts over {elapsed}s. Last error: {last_detail}. "
                f"This is api.esmatlas.com being unavailable, not a problem with "
                f"your sequence — the analysis below still runs without it."
            ),
        })

    def end_headers(self) -> None:
        # Never cache during development: an edited app.js should take effect on
        # reload rather than after a hard refresh.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        print(f"  {self.address_string()} {fmt % args}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    handler = partial(Handler, directory=str(WEB))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"serving {WEB.relative_to(ROOT)}/ on http://localhost:{args.port}")
    print("  POST /api/fold -> api.esmatlas.com (mirrors api/fold.mjs)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

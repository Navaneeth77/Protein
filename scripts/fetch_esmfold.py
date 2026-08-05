"""Resumable downloader for the ESMFold checkpoint (~8.4 GB).

Why this exists: on a slow link `huggingface_hub.snapshot_download` stalls (its
xet transport made no progress at all here), and a single-shot download of this
size that dies at 90% has to start over. This fetches each file with HTTP range
resume and retries indefinitely, so it can be left running and interrupted
freely.

Writes a plain `from_pretrained`-compatible directory:
    data/models/esmfold_v1/{config.json,vocab.txt,...,pytorch_model.bin}

Point the fold cache at it with:
    REFOLD_ESMFOLD_PATH=data/models/esmfold_v1

Usage:
    python scripts/fetch_esmfold.py
    python scripts/fetch_esmfold.py --status
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import ROOT  # noqa: E402

BASE = "https://huggingface.co/facebook/esmfold_v1/resolve/main/"
TARGET = ROOT / "data" / "models" / "esmfold_v1"

SMALL_FILES = (
    "config.json",
    "vocab.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
LARGE_FILES = ("pytorch_model.bin",)

CHUNK = 1 << 20  # 1 MiB


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, ValueError, TimeoutError):
        return None


def fetch(name: str, retries: int, quiet: bool = False) -> bool:
    """Download `name` with range resume. Returns True once complete."""
    url = BASE + name
    destination = TARGET / name
    destination.parent.mkdir(parents=True, exist_ok=True)

    total = remote_size(url)
    attempt = 0
    while True:
        have = destination.stat().st_size if destination.exists() else 0
        if total is not None and have >= total:
            print(f"[done] {name}  {have / 1e6:.1f} MB")
            return True

        attempt += 1
        if retries and attempt > retries:
            print(f"[give up] {name} at {have / 1e6:.1f} MB after {retries} attempt(s)")
            return False

        request = urllib.request.Request(url)
        if have:
            request.add_header("Range", f"bytes={have}-")

        started = time.time()
        got = 0
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if have and response.status != 206:
                    # Server ignored the range header; restart from scratch.
                    print(f"[warn] {name}: no range support, restarting")
                    destination.unlink(missing_ok=True)
                    continue
                with destination.open("ab" if have else "wb") as handle:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        got += len(block)
                        if not quiet and total:
                            done = have + got
                            rate = got / max(time.time() - started, 1e-6) / 1024
                            pct = 100.0 * done / total
                            eta = (total - done) / max(got / max(time.time() - started, 1e-6), 1)
                            print(
                                f"\r[{name}] {done / 1e6:8.1f}/{total / 1e6:.1f} MB "
                                f"{pct:5.1f}%  {rate:6.1f} kB/s  eta {eta / 60:5.1f} min",
                                end="",
                                flush=True,
                            )
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            print(f"\n[retry {attempt}] {name}: {type(exc).__name__} {exc}")
            time.sleep(min(5 * attempt, 60))
            continue

        print()
        if total is None:
            print(f"[done?] {name}: server gave no size; got {got / 1e6:.1f} MB")
            return True


def status() -> int:
    print(f"target: {TARGET}")
    incomplete = 0
    for name in SMALL_FILES + LARGE_FILES:
        path = TARGET / name
        total = remote_size(BASE + name)
        have = path.stat().st_size if path.exists() else 0
        total_text = f"{total / 1e6:.1f}" if total else "?"
        flag = "ok" if total and have >= total else "incomplete"
        if flag != "ok":
            incomplete += 1
        print(f"  [{flag:>10}] {name:<26} {have / 1e6:8.1f} / {total_text} MB")
    print("\nready to load" if not incomplete else f"\n{incomplete} file(s) outstanding")
    return 0 if not incomplete else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="report progress and exit")
    ap.add_argument("--retries", type=int, default=0,
                    help="attempts per file; 0 means retry forever")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.status:
        return status()

    print(f"downloading facebook/esmfold_v1 into {TARGET}")
    print("this is ~8.4 GB; the run is resumable, so interrupting it is safe\n")

    for name in SMALL_FILES:
        if not fetch(name, args.retries, quiet=True):
            print(f"could not fetch {name}; aborting")
            return 1
    for name in LARGE_FILES:
        if not fetch(name, args.retries, quiet=args.quiet):
            return 1

    print(f"\ncomplete. Now run:\n  .\\run_mvp.ps1 -Precompute")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

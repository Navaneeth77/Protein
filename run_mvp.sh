#!/usr/bin/env bash
# Launch the ReFold MVP demo. macOS / Linux counterpart of run_mvp.ps1.
#
#   ./run_mvp.sh              # start the Streamlit app
#   ./run_mvp.sh --precompute # warm the fold cache first (~15 min, do this once)
#   ./run_mvp.sh --check      # pre-flight: ollama, checkpoint, cache coverage
#
# Structures resolve from data/cache/, Gemma is called live through Ollama.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

PRECOMPUTE=0
CHECK=0
for arg in "$@"; do
  case "$arg" in
    --precompute|-Precompute) PRECOMPUTE=1 ;;
    --check|-Check)           CHECK=1 ;;
    -h|--help)
      sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "unknown option: $arg (try --help)" >&2
      exit 2 ;;
  esac
done

# ESMFold: local checkpoint rather than the hub, so nothing re-downloads.
export REFOLD_ESMFOLD_PATH="$ROOT/data/models/esmfold_v1"

# Gemma runs locally through Ollama. See run_mvp.ps1 for why keep_alive is not 0.
export REFOLD_GEMMA_MODE="ollama"
export REFOLD_GEMMA_MODEL="${REFOLD_GEMMA_MODEL:-gemma4:12b}"
export REFOLD_GEMMA_KEEP_ALIVE="60s"
export REFOLD_GEMMA_TIMEOUT="900"

# REFOLD_GEMMA_NUM_GPU is deliberately not set here. It exists to cap how many
# layers go to a small discrete GPU; on Apple silicon Ollama manages unified
# memory itself and pinning a layer count only makes it slower. Export it
# yourself if you are on a Linux box with a small CUDA card.

# Let the app fold live if a candidate turns out not to be cached. Set
# REFOLD_OFFLINE=1 to make an unexpected cache miss a loud error instead of a
# ~57-second pause.
unset REFOLD_OFFLINE REFOLD_FOLD_BACKEND 2>/dev/null || true

echo "ESMFold checkpoint : $REFOLD_ESMFOLD_PATH"
echo "Gemma              : $REFOLD_GEMMA_MODEL via $REFOLD_GEMMA_MODE"

if [[ "$CHECK" -eq 1 ]]; then
  echo
  echo "--- ollama models ---"
  if command -v ollama >/dev/null 2>&1; then
    ollama list
  else
    echo "ollama not on PATH — install from https://ollama.com"
  fi
  echo
  echo "--- checkpoint ---"
  "$PYTHON" scripts/fetch_esmfold.py --status
  echo
  echo "--- cache plan ---"
  "$PYTHON" scripts/precompute_mvp.py --list-only
  exit 0
fi

if [[ "$PRECOMPUTE" -eq 1 ]]; then
  echo
  echo "Warming the fold cache. This is the slow part (~57s per structure)."
  "$PYTHON" scripts/precompute_mvp.py
fi

echo
echo "Starting Streamlit on http://localhost:8501"
echo
exec streamlit run app/streamlit_app.py

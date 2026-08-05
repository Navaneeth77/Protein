# Launch the ReFold MVP demo.
#
#   .\run_mvp.ps1              # start the Streamlit app
#   .\run_mvp.ps1 -Precompute  # warm the fold cache first (~15 min, do this once)
#
# Structures resolve from data/cache/, Gemma is called live through Ollama.

param(
    [switch]$Precompute,
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# ESMFold: local checkpoint rather than the hub, so nothing re-downloads.
$env:REFOLD_ESMFOLD_PATH = Join-Path $root "data\models\esmfold_v1"

# Gemma runs locally through Ollama.
#   keep_alive: NOT 0. With 0 the model got stuck in "Stopping..." while the HTTP
#     reply never arrived, i.e. the unload raced the response. 60s releases the
#     ~9 GB soon enough and never wedges.
#   num_gpu 14: the whole 12B model does not fit this 4 GB card (Ollama's own
#     split OOM'd), but pure CPU took over ten minutes per call. 14 layers on the
#     GPU brings one call to ~70s. The client falls back to CPU on a CUDA OOM.
$env:REFOLD_GEMMA_MODE       = "ollama"
$env:REFOLD_GEMMA_MODEL      = "gemma4:12b"
$env:REFOLD_GEMMA_KEEP_ALIVE = "60s"
$env:REFOLD_GEMMA_NUM_GPU    = "14"
$env:REFOLD_GEMMA_TIMEOUT    = "900"

# Let the app fold live if a candidate turns out not to be cached. Set this to 1
# to make an unexpected cache miss a loud error instead of a 57-second pause.
Remove-Item Env:REFOLD_OFFLINE     -ErrorAction SilentlyContinue
Remove-Item Env:REFOLD_FOLD_BACKEND -ErrorAction SilentlyContinue

Write-Host "ESMFold checkpoint : $env:REFOLD_ESMFOLD_PATH"
Write-Host "Gemma              : $env:REFOLD_GEMMA_MODEL via $env:REFOLD_GEMMA_MODE"

if ($Check) {
    Write-Host "`n--- ollama models ---"
    ollama list
    Write-Host "`n--- checkpoint ---"
    python scripts/fetch_esmfold.py --status
    Write-Host "`n--- cache plan ---"
    python scripts/precompute_mvp.py --list-only
    exit $LASTEXITCODE
}

if ($Precompute) {
    Write-Host "`nWarming the fold cache. This is the slow part (~57s per structure)."
    python scripts/precompute_mvp.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "`nStarting Streamlit on http://localhost:8501`n"
streamlit run app/streamlit_app.py

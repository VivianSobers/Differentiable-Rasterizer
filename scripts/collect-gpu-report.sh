#!/usr/bin/env bash
# Collect everything needed to decide what to optimize next on the GPU path.
#
# Run this on a GPU box and share the file it writes. It is read-only apart from
# this repo's target/ directory.
#
#   ./scripts/collect-gpu-report.sh              # writes docs/gpu-report.txt
#   ./scripts/collect-gpu-report.sh out.txt      # custom path
#   SKIP_GPU=1 ./scripts/collect-gpu-report.sh   # CPU sections only
#
# Every stage runs under a timeout. An earlier version did not, and a stalled
# GPU submission left it blocked for hours with no output. Nothing here can
# now run longer than STAGE_TIMEOUT; a stage that exceeds it is recorded as
# TIMED OUT and the report continues.

set -uo pipefail

OUT="${1:-docs/gpu-report.txt}"
STAGE_TIMEOUT="${STAGE_TIMEOUT:-600}"   # seconds per stage
SKIP_GPU="${SKIP_GPU:-0}"

cd "$(dirname "$0")/.." || exit 1
mkdir -p "$(dirname "$OUT")"

# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

section() {
    printf '\n===== %s =====\n' "$1" | tee -a "$OUT"
}

# Run a stage under a timeout, recording the outcome either way.
stage() {
    local label="$1"
    shift
    local start elapsed status
    start=$(date +%s)
    timeout --kill-after=30s "$STAGE_TIMEOUT" "$@" 2>&1 \
        | grep -vE "^(   Compiling|    Finished|     Running|running |test result: ok\. 0 passed|^test .* ignored)" \
        | tee -a "$OUT"
    status=${PIPESTATUS[0]}
    elapsed=$(( $(date +%s) - start ))

    if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
        echo "!! $label TIMED OUT after ${STAGE_TIMEOUT}s — see notes at the end" | tee -a "$OUT"
    elif [ "$status" -ne 0 ]; then
        echo "!! $label exited with status $status (${elapsed}s)" | tee -a "$OUT"
    else
        echo "(${label} ok, ${elapsed}s)" | tee -a "$OUT"
    fi
}

: > "$OUT"
{
    echo "diffrast GPU report"
    echo "date:   $(date -Is)"
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "host:   $(uname -srm)"
} | tee -a "$OUT"

section "CPU"
{
    echo "cores: $(nproc)"
    grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//'
} 2>&1 | tee -a "$OUT"

section "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version \
        --format=csv,noheader 2>&1 | tee -a "$OUT"
    # Other processes on the card explain both slow results and stalls, and are
    # invisible from inside our own benchmark.
    echo "-- processes currently using the GPU --" | tee -a "$OUT"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader 2>&1 | tee -a "$OUT"
else
    echo "nvidia-smi not found" | tee -a "$OUT"
fi

section "Toolchain"
{
    rustc --version 2>&1
    cargo --version 2>&1
    python3 --version 2>&1
} | tee -a "$OUT"

section "Build"
stage "build" cargo build --release --all

section "Tests (CPU only)"
stage "cpu tests" cargo test --release -p diffrast

if [ "$SKIP_GPU" = "1" ]; then
    section "Tests (GPU)"
    echo "skipped: SKIP_GPU=1" | tee -a "$OUT"
else
    section "Tests (GPU)"
    # Single-threaded: the tests share one device, and serial execution keeps a
    # stall attributable to one named test rather than to the whole binary.
    stage "gpu tests" cargo test --release -p diffrast-gpu -- --test-threads=1 --nocapture
fi

section "CPU benchmarks"
stage "cpu bench" cargo bench --bench raster

if [ "$SKIP_GPU" = "1" ]; then
    section "GPU benchmarks"
    echo "skipped: SKIP_GPU=1" | tee -a "$OUT"
else
    section "GPU benchmarks"
    stage "gpu bench" cargo run --release --bin gpu_bench
fi

{
    echo
    echo "===== notes ====="
    echo "stage timeout: ${STAGE_TIMEOUT}s (override with STAGE_TIMEOUT=...)"
    echo "If a GPU stage timed out, check the process list above — a card shared"
    echo "with another job will queue our submissions behind it. Re-run with"
    echo "SKIP_GPU=1 to collect the CPU numbers regardless."
} | tee -a "$OUT"

printf '\nwrote %s\n' "$OUT"

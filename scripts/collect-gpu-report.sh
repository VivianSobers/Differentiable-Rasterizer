#!/usr/bin/env bash
# Collect everything needed to decide what to optimize next on the GPU path.
#
# Run this on a 4090 box and share the file it writes. It is read-only: it
# builds, runs tests and benchmarks, and records hardware details. It changes
# nothing outside this repo's target/ directory.
#
#   ./scripts/collect-gpu-report.sh            # writes docs/gpu-report.txt
#   ./scripts/collect-gpu-report.sh out.txt

set -uo pipefail

OUT="${1:-docs/gpu-report.txt}"
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$(dirname "$OUT")"

# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

section() {
    printf '\n===== %s =====\n' "$1" | tee -a "$OUT"
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
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
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

section "Tests"
# Failures are recorded rather than fatal: a partial report is far more useful
# than none, and a failing test here is itself the finding.
cargo test --release 2>&1 | grep -E "test result|^error|FAILED" | tee -a "$OUT"

section "CPU benchmarks"
# --bench raster, not a bare `cargo bench`: the latter also runs every test
# target in bench mode and buries the numbers under a wall of "ignored" lines.
cargo bench --bench raster 2>&1 \
    | grep -vE "^(   Compiling|    Finished|     Running|running |test result|^test )" \
    | tee -a "$OUT"

section "GPU benchmarks"
cargo run --release --bin gpu_bench 2>&1 \
    | grep -vE "^(   Compiling|    Finished|     Running)" | tee -a "$OUT"

# The default adapter is not always the fastest one present; on a two-card box
# it may not even be the card you meant.
section "GPU benchmarks (forced Vulkan)"
WGPU_BACKEND=vulkan cargo run --release --bin gpu_bench 2>&1 \
    | grep -vE "^(   Compiling|    Finished|     Running)" | tee -a "$OUT"

printf '\nwrote %s\n' "$OUT"

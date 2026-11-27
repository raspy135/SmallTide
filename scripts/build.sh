#!/usr/bin/env bash
# Build SmallTide (smalltide-cli) on Linux. Linux counterpart of build.ps1.
#
# Usage: scripts/build.sh [--test] [--offline] [--debug] [--stop] [--serve] [cargo args...]
#   --test     run the workspace tests first
#   --offline  pass --offline to cargo
#   --debug    debug build instead of --release
#   --stop     stop running smalltide-cli processes before building
#   --serve    stop running servers, build, then start `smalltide-cli serve`
# hipcc is found via SMALLTIDE_HIPCC, ROCM_PATH/HIP_PATH, /opt/rocm*, or PATH (see build_linux.rs).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TEST=0 OFFLINE=0 DEBUG=0 STOP=0 SERVE=0
CARGO_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test) TEST=1 ;;
        --offline) OFFLINE=1 ;;
        --debug) DEBUG=1 ;;
        --stop) STOP=1 ;;
        --serve) SERVE=1 ;;
        -h | --help) sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) CARGO_ARGS+=("$1") ;;
    esac
    shift
done

find_cargo() {
    if command -v cargo >/dev/null 2>&1; then command -v cargo; return; fi
    if [[ -x "$HOME/.cargo/bin/cargo" ]]; then printf '%s\n' "$HOME/.cargo/bin/cargo"; return; fi
    echo "[build] error: cargo not found on PATH or at ~/.cargo/bin/cargo. Install Rust from https://rustup.rs" >&2
    exit 1
}

CARGO="$(find_cargo)"
PROFILE_ARGS=(--release)
PROFILE_DIR=release
if [[ "$DEBUG" -eq 1 ]]; then PROFILE_ARGS=(); PROFILE_DIR=debug; fi
OFFLINE_ARGS=()
[[ "$OFFLINE" -eq 1 ]] && OFFLINE_ARGS=(--offline)
EXE_PATH="$ROOT/target/$PROFILE_DIR/smalltide-cli"

cd "$ROOT"

if [[ "$TEST" -eq 1 ]]; then
    echo "[build] Running workspace tests..."
    "$CARGO" test "${OFFLINE_ARGS[@]}" "${PROFILE_ARGS[@]}" --workspace
fi

# Linux does not lock a running binary, but a running server still holds the GPU and port 8000.
if pids="$(pgrep -x smalltide-cli)"; then
    if [[ "$STOP" -eq 1 || "$SERVE" -eq 1 ]]; then
        echo "[build] Stopping running smalltide-cli process(es): $(echo $pids)"
        kill $pids 2>/dev/null || true
        for _ in 1 2 3 4 5 6 7 8 9 10; do pgrep -x smalltide-cli >/dev/null || break; sleep 0.3; done
    else
        echo "[build] note: smalltide-cli is running (PID $(echo $pids)); it keeps the old binary until restarted. Use --stop or --serve to stop it."
    fi
fi

start=$(date +%s.%N)
echo "[build] Building smalltide-cli ($PROFILE_DIR)..."
"$CARGO" build "${OFFLINE_ARGS[@]}" "${PROFILE_ARGS[@]}" -p smalltide-cli "${CARGO_ARGS[@]}"

size_mb=0
[[ -f "$EXE_PATH" ]] && size_mb="$(awk -v b="$(stat -c %s "$EXE_PATH")" 'BEGIN { printf "%.2f", b / 1048576 }')"
elapsed="$(awk -v a="$start" -v b="$(date +%s.%N)" 'BEGIN { printf "%.2f", b - a }')"
echo "[build] Ready: $EXE_PATH ($size_mb MB) in ${elapsed}s"

if [[ "$SERVE" -eq 1 ]]; then
    echo "[build] Starting smalltide-cli serve..."
    exec "$EXE_PATH" serve
fi

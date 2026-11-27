#!/usr/bin/env bash
# SmallTide launcher for Linux (works with a pre-built release or a source checkout).
# Linux counterpart of smalltide.ps1 / smalltide.cmd: same commands, same .env files.
#
#   setup       Copy .env.*.example -> local .env files, verify AMD HIP GPU, download Inkling-2-Mini, pack experts.stide
#   serve       Start the OpenAI-compatible HTTP server and /expert-map dashboard
#   run ...     Run one-shot generation (forwards all flags to `smalltide-cli run`)
#   probe       Verify the HIP runtime library, GPU device, embedded kernels, and PCIe DMA bandwidth
#   vram-probe  Measure VRAM allocation bandwidth chunk by chunk
#   bench-gpu   Verify the GPU kernels and run GPU microbenchmarks
#   bench ...   Run bench/bench.py against the running server (http://127.0.0.1:8000)
#   pack        Pack model-00001-of-00001.safetensors into 4KB-aligned experts.stide
#   build       Build smalltide-cli from source (requires Rust + ROCm hipcc)
#   help        Show this help text
#
# Examples:
#   ./smalltide.sh setup
#   ./smalltide.sh serve .env.headless
#   ./smalltide.sh run -p "Explain speculative decoding in 3 bullet points." -n 256
#   ./smalltide.sh bench --only prose code --runs 2

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { echo "[smalltide] error: $*" >&2; exit 1; }

# read_dotenv_value <env-file> <key> <default>: first non-empty `KEY=value` (optionally
# `export`ed, quoted, or followed by an inline `# comment`), like Read-DotEnvValue in smalltide.ps1.
read_dotenv_value() {
    local env_path="$1" key="$2" default="$3" v
    if [[ -f "$env_path" ]]; then
        v="$(awk -v key="$key" '
            { sub(/\r$/, ""); t = $0; gsub(/^[ \t]+|[ \t]+$/, "", t) }
            t == "" || substr(t, 1, 1) == "#" { next }
            {
                sub(/^export[ \t]+/, "", t)
                if (index(t, key) != 1) next
                rest = substr(t, length(key) + 1)
                if (rest !~ /^[ \t]*=/) next
                sub(/^[ \t]*=[ \t]*/, "", rest)
                q = substr(rest, 1, 1)
                if ((q == "\"" || q == "'\''") && length(rest) >= 2 && substr(rest, length(rest), 1) == q) {
                    rest = substr(rest, 2, length(rest) - 2)
                } else {
                    sub(/[ \t]+#.*$/, "", rest)
                }
                if (rest != "") { print rest; exit }
            }' "$env_path")"
        if [[ -n "$v" ]]; then
            printf '%s\n' "$v"
            return
        fi
    fi
    printf '%s\n' "$default"
}

# Absolute model directory: $SMALLTIDE_MODEL_DIR, else SMALLTIDE_MODEL_DIR from .env, else the default.
resolve_model_dir() {
    local rel="${SMALLTIDE_MODEL_DIR:-}"
    [[ -n "$rel" ]] || rel="$(read_dotenv_value "$ROOT/.env" SMALLTIDE_MODEL_DIR models/Inkling-2-Mini)"
    if [[ "$rel" == /* ]]; then printf '%s\n' "$rel"; else printf '%s\n' "$ROOT/$rel"; fi
}

resolve_exe() {
    # 1. Pre-built release layout: smalltide-cli right next to smalltide.sh
    if [[ -x "$ROOT/smalltide-cli" ]]; then printf '%s\n' "$ROOT/smalltide-cli"; return; fi
    # 2. Source checkout release build
    local cargo_release="$ROOT/target/release/smalltide-cli"
    if [[ -x "$cargo_release" ]]; then printf '%s\n' "$cargo_release"; return; fi
    # 3. Build from source if this is a checkout and cargo is available
    if [[ -f "$ROOT/Cargo.toml" ]] && command -v cargo >/dev/null 2>&1; then
        echo "[smalltide] No binary found; building smalltide-cli from source..." >&2
        (cd "$ROOT" && cargo build --release -p smalltide-cli) >&2 || die "cargo build failed"
        printf '%s\n' "$cargo_release"
        return
    fi
    die "smalltide-cli not found. Download the pre-built Linux release, or install Rust + ROCm (hipcc) and run './smalltide.sh build'."
}

find_python() {
    local c
    for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1; then command -v "$c"; return; fi
    done
    die "Python 3 not found on PATH (needed only for bench/bench.py)."
}

show_help() {
    cat <<'EOF'
SmallTide - Adaptive MoE inference for Inkling-2-Mini on AMD RDNA 4 (Linux)

Usage: ./smalltide.sh <command> [args...]

Commands:
  setup       Create .env, verify AMD HIP GPU, download Inkling-2-Mini (~21 GB), pack experts.stide
  serve       Start OpenAI-compatible server (http://127.0.0.1:8000) + /expert-map
  run ...     Generate from the CLI (e.g. ./smalltide.sh run -p "Hello" -n 256)
  probe       Verify HIP runtime library, GPU, embedded kernels (gfx1201 / gfx1200), and DMA bandwidth
  vram-probe  Measure VRAM allocation bandwidth chunk by chunk
  bench-gpu   Run GPU kernel verification and GPU microbenchmarks
  bench ...   Run bench/bench.py against the running server
  pack        Build 4KB-aligned experts.stide from model-00001-of-00001.safetensors
  build       Compile smalltide-cli from source (requires Rust + ROCm hipcc)
EOF
}

cmd_setup() {
    local src dst label
    while read -r src dst label; do
        if [[ ! -e "$ROOT/$dst" ]]; then
            if [[ -f "$ROOT/$src" ]]; then
                cp "$ROOT/$src" "$ROOT/$dst"
                echo "[setup] Created $label from $src"
            fi
        else
            echo "[setup] Keeping existing $label"
        fi
    done <<'EOF'
.env.headless.example .env.headless .env.headless
.env.monitor.example .env.monitor .env.monitor
.env.headless.example .env .env
EOF

    local exe
    exe="$(resolve_exe)"
    echo
    echo "[setup] 1/3 Probing AMD HIP GPU..."
    "$exe" probe || die "GPU probe failed (see the message above). SmallTide needs an AMD Radeon RDNA 4 card (RX 9070 or RX 9060 family) and a ROCm HIP runtime (6.x/7.x) with RDNA 4 support."

    local model_dir
    model_dir="$(resolve_model_dir)"
    echo
    echo "[setup] 2/3 Downloading Inkling-2-Mini weights into $model_dir ..."
    "$ROOT/scripts/download_inkling2.sh" --out-dir "$model_dir" || die "Download failed"

    local st_path="$model_dir/model-00001-of-00001.safetensors"
    local pack_path="$model_dir/experts.stide"
    echo
    if [[ -f "$pack_path" ]]; then
        local pack_gb
        pack_gb="$(awk -v b="$(stat -c %s "$pack_path")" 'BEGIN { printf "%.2f", b / 1073741824 }')"
        echo "[setup] 3/3 Packed expert store already exists: $pack_path ($pack_gb GB)"
    else
        echo "[setup] 3/3 Packing 3,072 routed experts into 4KB-aligned $pack_path (~15s)..."
        "$exe" pack "$st_path" "$pack_path" || die "Packing experts.stide failed"
    fi

    echo
    echo "[setup] Done! Start SmallTide with:"
    echo "  ./smalltide.sh serve .env.monitor    # monitor plugged into the Radeon card"
    echo "  ./smalltide.sh serve .env.headless   # monitor on motherboard / iGPU"
    echo "Then open http://127.0.0.1:8000/expert-map or connect any OpenAI client to http://127.0.0.1:8000/v1"
}

main() {
    local command="${1:-help}"
    [[ $# -gt 0 ]] && shift
    cd "$ROOT"

    case "${command,,}" in
        help | -h | --help)
            show_help
            ;;
        build)
            [[ -f "$ROOT/Cargo.toml" ]] || die "Cargo.toml not found here. This looks like a pre-built release package (no build needed!)."
            if [[ -x "$ROOT/scripts/build.sh" ]]; then
                exec "$ROOT/scripts/build.sh" "$@"
            fi
            exec cargo build --release -p smalltide-cli "$@"
            ;;
        setup)
            cmd_setup
            ;;
        pack)
            local exe
            exe="$(resolve_exe)"
            if [[ $# -ge 2 ]]; then
                exec "$exe" pack "$@"
            fi
            local model_dir
            model_dir="$(resolve_model_dir)"
            exec "$exe" pack "$model_dir/model-00001-of-00001.safetensors" "$model_dir/experts.stide"
            ;;
        bench)
            local py
            py="$(find_python)"
            [[ -f "$ROOT/bench/bench.py" ]] || die "bench/bench.py not found."
            exec "$py" "$ROOT/bench/bench.py" "$@"
            ;;
        *)
            # serve, run, probe, vram-probe, bench-gpu and any other smalltide-cli command
            local exe
            exe="$(resolve_exe)"
            exec "$exe" "$command" "$@"
            ;;
    esac
}

main "$@"

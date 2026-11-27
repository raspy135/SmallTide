#!/usr/bin/env bash
# Resumable Inkling-2-Mini (+ DFlash 2 drafter) download for Linux.
# Linux counterpart of download_inkling2.ps1: same sources, same layout, same env variables.
#
# Usage: scripts/download_inkling2.sh [--out-dir DIR] [--skip-dflash2]
#   HF_TOKEN               optional, for higher Hugging Face rate limits
#   SMALLTIDE_MODEL_DIR    default output directory (relative to the repository root)
#   SMALLTIDE_DFLASH2_DIR  drafter directory (default: <out-dir>/../Inkling-2-Mini-DFlash2)
#   SMALLTIDE_KEEP_COMPRESSION=1  keep btrfs compression on the model directories (see below)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "[Download] error: $*" >&2; exit 1; }
abs_path() { if [[ "$1" == /* ]]; then printf '%s\n' "$1"; else printf '%s\n' "$REPO_ROOT/$1"; fi; }

OUT_DIR=""
SKIP_DFLASH2=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir) [[ $# -ge 2 ]] || die "--out-dir needs a value"; OUT_DIR="$2"; shift 2 ;;
        --out-dir=*) OUT_DIR="${1#*=}"; shift ;;
        --skip-dflash2) SKIP_DFLASH2=1; shift ;;
        -h | --help) sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$(abs_path "${SMALLTIDE_MODEL_DIR:-models/Inkling-2-Mini}")"
fi

command -v curl >/dev/null 2>&1 || die "curl was not found on PATH (install it with your package manager)."

# SmallTide streams experts with O_DIRECT so they bypass the page cache. On btrfs, compressed
# files are always read through the page cache, which defeats that; turn compression off on the
# directory before any file is written into it (new files inherit the setting).
prepare_dir() {
    local dir="$1"
    mkdir -p "$dir"
    [[ "${SMALLTIDE_KEEP_COMPRESSION:-0}" == "1" ]] && return 0
    command -v findmnt >/dev/null 2>&1 || return 0
    [[ "$(findmnt -no FSTYPE -T "$dir" 2>/dev/null)" == "btrfs" ]] || return 0  # only btrfs needs this
    if command -v btrfs >/dev/null 2>&1 && btrfs property set "$dir" compression none 2>/dev/null; then
        echo "[Download] btrfs: compression disabled on $dir (new files are stored uncompressed)"
    else
        echo "[Download] warning: $dir is on btrfs and compression could not be disabled;" >&2
        echo "[Download]          run 'btrfs property set \"$dir\" compression none' before downloading." >&2
    fi
}

CURL_ARGS=(-fL -C - --retry 3 --retry-delay 2)

# fetch <url> <out>: resumable download. HF_TOKEN goes to curl on stdin (`-H @-`), never on the
# command line, where any local user could read it from /proc/<pid>/cmdline.
fetch() {
    if [[ -n "${HF_TOKEN:-}" ]]; then
        curl "${CURL_ARGS[@]}" -H @- "$1" -o "$2" <<<"Authorization: Bearer $HF_TOKEN"
    else
        curl "${CURL_ARGS[@]}" "$1" -o "$2"
    fi
}

prepare_dir "$OUT_DIR"

BASE="https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main"
FILES=(
    config.json
    generation_config.json
    hf_quant_config.json
    model.safetensors.index.json
    special_tokens_map.json
    tokenizer_config.json
    chat_template.jinja
    tokenizer.json
    mtp.safetensors
    model-00001-of-00001.safetensors
)

echo "[Download] Target directory: $OUT_DIR"
echo "[Download] Source: https://huggingface.co/thinkingmachines/Inkling-2-Mini (~21 GB total, resumable)"

for f in "${FILES[@]}"; do
    out="$OUT_DIR/$f"
    echo "[Download] $f -> $out"
    fetch "$BASE/$f" "$out" || die "curl failed for $f (exit $?)"
done

echo "[Download] All Inkling-2-Mini files completed!"

if [[ "$SKIP_DFLASH2" -eq 0 ]]; then
    if [[ -n "${SMALLTIDE_DFLASH2_DIR:-}" ]]; then
        DFLASH2_DIR="$(abs_path "$SMALLTIDE_DFLASH2_DIR")"
    else
        DFLASH2_DIR="$(dirname "$OUT_DIR")/Inkling-2-Mini-DFlash2"
    fi
    prepare_dir "$DFLASH2_DIR"
    DFLASH2_BASE="https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/resolve/main"
    echo
    echo "[Download] DFlash 2 drafter: https://huggingface.co/incoai/Inkling-2-Mini-DFlash2 (~944 MB, resumable) -> $DFLASH2_DIR"
    for f in config.json model.safetensors; do
        out="$DFLASH2_DIR/$f"
        echo "[Download] $f -> $out"
        if ! fetch "$DFLASH2_BASE/$f" "$out"; then
            echo "[Download] warning: could not download DFlash 2 ($f); SmallTide will fall back to built-in DSpark (mtp.safetensors)." >&2
            break
        fi
    done
fi

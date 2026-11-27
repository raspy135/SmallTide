#!/usr/bin/env bash
# Builds a self-contained Linux release archive for SmallTide (no Rust or ROCm SDK needed by end
# users; they still need the ROCm HIP runtime library, libamdhip64.so). Linux counterpart of
# package-release.ps1.
#
# Usage: scripts/package-release.sh [--skip-build] [--no-probe] [--arch gfx1201,gfx1200] [--target-dir DIR]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "[package] error: $*" >&2; exit 1; }

SKIP_BUILD=0
NO_PROBE=0
ARCH="gfx1201,gfx1200" # kernel builds to embed; the first one names the package
TARGET_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build) SKIP_BUILD=1; shift ;;
        --no-probe) NO_PROBE=1; shift ;;
        --arch) [[ $# -ge 2 ]] || die "--arch needs a value"; ARCH="$2"; shift 2 ;;
        --target-dir) [[ $# -ge 2 ]] || die "--target-dir needs a value"; TARGET_DIR="$2"; shift 2 ;;
        -h | --help) sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

cd "$REPO_ROOT"

VERSION="$(tr -d '[:space:]' < VERSION)"
[[ -n "$VERSION" ]] || die "VERSION file is empty."

CARGO_ARGS=(build --release -p smalltide-cli)
[[ -n "$TARGET_DIR" ]] && CARGO_ARGS+=(--target-dir "$TARGET_DIR")

if [[ "$SKIP_BUILD" -eq 0 ]]; then
    echo "[package] Building release binary for $ARCH ..."
    SMALLTIDE_HIP_ARCH="$ARCH" cargo "${CARGO_ARGS[@]}" || die "cargo build --release failed."
fi

if [[ -z "$TARGET_DIR" ]]; then
    BASE_TARGET="$REPO_ROOT/target"
elif [[ "$TARGET_DIR" == /* ]]; then
    BASE_TARGET="$TARGET_DIR"
else
    BASE_TARGET="$REPO_ROOT/$TARGET_DIR"
fi
EXE_PATH="$BASE_TARGET/release/smalltide-cli"
[[ -x "$EXE_PATH" ]] || die "Release binary not found at $EXE_PATH"

if [[ "$NO_PROBE" -eq 0 ]]; then
    echo "[package] Verifying embedded .hsaco kernels via probe..."
    probe_out="$("$EXE_PATH" probe 2>&1)" || die "smalltide-cli probe failed:"$'\n'"$probe_out"
    hsaco_line="$(grep -E 'Embedded HSACO[[:space:]]*:[[:space:]]*[0-9]+[[:space:]]*bytes' <<<"$probe_out" | head -n1 || true)"
    hsaco_bytes="$(grep -oE '[0-9]+' <<<"$hsaco_line" | head -n1 || true)"
    if [[ -z "$hsaco_bytes" || "$hsaco_bytes" -le 0 ]]; then
        die "Embedded HSACO is 0 bytes! Rebuild with ROCm hipcc installed before releasing."
    fi
    echo "[package] $hsaco_line"
fi

PKG_ARCH="$(cut -d, -f1 <<<"$ARCH" | tr -d '[:space:]')"
PKG_NAME="smalltide-v$VERSION-linux-x64-$PKG_ARCH"
DIST_DIR="$REPO_ROOT/dist"
STAGE_DIR="$DIST_DIR/$PKG_NAME"
TAR_PATH="$DIST_DIR/$PKG_NAME.tar.gz"
SHA_PATH="$TAR_PATH.sha256"

rm -rf "$STAGE_DIR"
rm -f "$TAR_PATH" "$SHA_PATH"
mkdir -p "$STAGE_DIR/scripts" "$STAGE_DIR/profiles" "$STAGE_DIR/bench"

# Core binary (self-contained, with the embedded .hsaco kernels of every architecture in $ARCH)
install -m 755 "$EXE_PATH" "$STAGE_DIR/smalltide-cli"

# Launcher, config templates, docs & license
for f in smalltide.sh .env.headless.example .env.monitor.example README.md LICENSE LIMITATIONS.md CREDITS.md VERSION; do
    [[ -f "$f" ]] || die "Missing required release file: $f"
    cp -p "$f" "$STAGE_DIR/$f"
done
chmod 755 "$STAGE_DIR/smalltide.sh"

# Build provenance, so a downloaded bundle says which toolchain produced its kernels
commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"
[[ -n "$commit" ]] || commit="${GITHUB_SHA:-unknown}"
hip_version="$(hipcc --version 2>/dev/null | sed -n 's/^HIP version: //p' | head -n1 || true)"
{
    echo "SmallTide v$VERSION"
    echo "Commit       : $commit"
    echo "Built        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "GPU kernels  : $ARCH"
    echo "HIP (hipcc)  : ${hip_version:-not found, kernels may be empty}"
    echo "Rust         : $(rustc --version 2>/dev/null || echo unknown)"
    echo
    echo "Runtime needs: an AMD RDNA 4 GPU and the ROCm HIP runtime (libamdhip64.so)."
    echo "Check with   : ./smalltide.sh probe"
} > "$STAGE_DIR/BUILD_INFO"

# Resumable model downloader (curl only)
install -m 755 scripts/download_inkling2.sh "$STAGE_DIR/scripts/download_inkling2.sh"

# Fresh-start aggregate routing priors (floats only, no user data)
cp profiles/inkling2-mini-priors.json "$STAGE_DIR/profiles/"

# Benchmark harness
cp bench/bench.py bench/edit_source.py.txt "$STAGE_DIR/bench/"

echo "[package] Creating $TAR_PATH ..."
tar -C "$DIST_DIR" -czf "$TAR_PATH" "$PKG_NAME"

hash="$(sha256sum "$TAR_PATH" | cut -d' ' -f1)"
echo "$hash  $PKG_NAME.tar.gz" > "$SHA_PATH"

size_mb="$(awk -v b="$(stat -c %s "$TAR_PATH")" 'BEGIN { printf "%.2f", b / 1048576 }')"
echo
echo "[package] Release bundle ready!"
echo "  Archive : $TAR_PATH ($size_mb MB)"
echo "  SHA256  : $hash"
echo "  Staging : $STAGE_DIR"

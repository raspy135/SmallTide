# syntax=docker/dockerfile:1
#
# Turnkey SmallTide container for Linux hosts with an AMD RDNA 4 GPU (RX 9070 / RX 9060 / R9700).
# Ships the ROCm HIP runtime, so the host only needs the amdgpu kernel driver.
#
# Build:  docker build -f docker/Dockerfile -t smalltide .
# Run:    see docker/README.md

ARG ROCM=10.2

# ROCm 10 apt repository (TheRock packages), shared by both stages.
FROM ubuntu:24.04 AS rocm-repo
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && install -d /etc/apt/keyrings \
 && curl -fsSL https://stable.repo.amd.com/rocm/gpg/packages.gpg -o /etc/apt/keyrings/amdrocm.asc \
 && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.asc] https://stable.repo.amd.com/rocm/core/packages/ubuntu2404 stable main" \
      > /etc/apt/sources.list.d/rocm.list \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------------------------
FROM rocm-repo AS build
ARG ROCM
ARG ARCH=gfx1201,gfx1200
# Commit recorded in BUILD_INFO (the build context has no .git).
ARG GITHUB_SHA=unknown

RUN apt-get update \
 && apt-get install -y --no-install-recommends "amdrocm-runtime-dev${ROCM}" build-essential pkg-config \
 && rm -rf /var/lib/apt/lists/*

ENV ROCM_PATH=/opt/rocm/core-${ROCM} \
    PATH=/opt/rocm/core-${ROCM}/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
 && hipcc --version | grep -E "HIP version|clang version"

WORKDIR /src
COPY . .

# Build, package, then check that every requested architecture produced a kernel code object
# (without hipcc the build still succeeds but embeds empty placeholders).
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/src/target \
    GITHUB_SHA="$GITHUB_SHA" scripts/package-release.sh --no-probe --arch "$ARCH" \
 && for a in $(echo "$ARCH" | tr ',' ' '); do \
        grep -aq "amdgcn-amd-amdhsa--$a" dist/*/smalltide-cli \
          || { echo "no kernel code object for $a (was hipcc found?)" >&2; exit 1; }; \
    done \
 && mkdir /out && cp -a dist/smalltide-v*-linux-x64-*/. /out/

# ---------------------------------------------------------------------------------------------
FROM rocm-repo
ARG ROCM

LABEL org.opencontainers.image.title="SmallTide" \
      org.opencontainers.image.description="Adaptive MoE inference for Inkling-2-Mini on AMD RDNA 4, with the ROCm ${ROCM} HIP runtime"

# HIP runtime only (no compiler, no math libraries); tini as init; python3 for bench.py, btrfs-progs so the
# downloader can turn off compression on btrfs volumes (experts are read with O_DIRECT).
RUN apt-get update \
 && apt-get install -y --no-install-recommends "amdrocm-runtime${ROCM}" python3 btrfs-progs tini \
 && rm -rf /var/lib/apt/lists/* \
 && echo "/opt/rocm/core-${ROCM}/lib" > /etc/ld.so.conf.d/rocm.conf \
 && ldconfig \
 && ldd "/opt/rocm/core-${ROCM}/lib/libamdhip64.so.7" | { ! grep "not found"; }

COPY --from=build /out /opt/smalltide
COPY docker/entrypoint.sh /usr/local/bin/smalltide-entrypoint

# /data holds .env, the model weights, experts.stide and the learned routing priors.
WORKDIR /data
VOLUME /data
EXPOSE 8000
# tini as PID 1 forwards stop signals (the server has no handler of its own, and PID 1 ignores them).
ENTRYPOINT ["tini", "--", "smalltide-entrypoint"]
CMD ["serve"]

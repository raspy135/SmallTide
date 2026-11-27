# SmallTide container

A turnkey image for Linux hosts with an AMD RDNA 4 GPU (RX 9070 / RX 9060 family, Radeon AI PRO
R9700). It ships the ROCm HIP runtime (10.2 by default), so the host only needs:

- the `amdgpu` kernel driver (any recent distribution kernel has it), and
- Docker or Podman.

No ROCm install on the host, no Rust, no build.

Tested on an RX 9070 XT (16 GB, Fedora, Podman) and a Radeon AI PRO R9700 (32 GB, Docker Compose).

## Run

Pick one of the three ways below. In every case, the first start creates `/data/.env`, checks the
GPU, downloads Inkling-2-Mini (about 21 GB) and packs `experts.stide`; later starts go straight to
the server. Then open `http://localhost:8000/expert-map`, or point any OpenAI client at
`http://localhost:8000/v1`:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "Explain speculative decoding in 3 bullet points."}],
  "max_tokens": 1000
}'
```

The model thinks before it answers, so keep `max_tokens` generous: a small limit can be spent on
thinking (`reasoning_content`) and leave `content` empty.

### docker run

```bash
docker run -d --name smalltide --restart unless-stopped \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined \
  -p 127.0.0.1:8000:8000 \
  -v smalltide-data:/data \
  ghcr.io/raspy135/smalltide:latest
docker logs -f smalltide
```

`-p 127.0.0.1:8000:8000` keeps the server on this machine; use `-p 8000:8000` to open it to your
network (read [Network access](#network-access) first). With rootless Podman, replace
`--group-add video` with `--group-add keep-groups` so the container keeps your access to the GPU
devices. On SELinux hosts such as Fedora, also add `--security-opt label=disable`. To change
settings, add `--env-file docker/.env` (see [Configuration](#configuration)).

### Docker Compose

[`compose.yaml`](compose.yaml) works with `docker compose` and `podman-compose`:

```bash
docker compose -f docker/compose.yaml up -d          # pull the published image
docker compose -f docker/compose.yaml up -d --build  # or build it from this checkout
docker compose -f docker/compose.yaml logs -f
docker compose -f docker/compose.yaml down           # stop (the data volume is kept)
```

Compose reads `docker/.env` (see [Configuration](#configuration)). For example, rootless Podman
with the monitor on the Radeon card:

```bash
cp docker/.env.example docker/.env
# in docker/.env: SMALLTIDE_GROUP=keep-groups and SMALLTIDE_CONFIG=monitor
podman-compose -f docker/compose.yaml up -d
```

### Podman Quadlet (systemd service)

[`smalltide.container`](smalltide.container) runs SmallTide as a systemd service that starts at
boot and restarts on failure:

```bash
mkdir -p ~/.config/containers/systemd
cp docker/smalltide.container ~/.config/containers/systemd/
systemctl --user daemon-reload
systemctl --user start smalltide
journalctl --user -u smalltide -f
loginctl enable-linger "$USER"   # keep it running when you are logged out
```

For a system-wide service, copy it to `/etc/containers/systemd/` and drop `--user`. Edit the file
to change the config (`Environment=SMALLTIDE_CONFIG=...`), to open the port to your network (the
commented `PublishPort=` line), to load engine settings from a file (the commented
`EnvironmentFile=` line, see [Configuration](#configuration)), or to mount a model you already
have. It is marked for `podman auto-update`: with `podman-auto-update.timer` enabled, new images are
pulled and the service restarted automatically.

## Configuration

You need no file to start: the image carries both engine templates and creates the engine `.env`
in the data volume on first start. To change anything, use one settings file on your side:

```bash
cp docker/.env.example docker/.env
```

[`.env.example`](.env.example) lists and explains everything you can set, in two parts:

- **Container settings**, which decide how the container is started:

  | Variable | Default | |
  |---|---|---|
  | `SMALLTIDE_CONFIG` | `headless` | Engine template for the first start: `headless` (monitor on the motherboard or another GPU) or `monitor` (monitor plugged into the Radeon card) |
  | `SMALLTIDE_BIND` | `127.0.0.1` | `0.0.0.0` opens the port to your network (see [Network access](#network-access)) |
  | `SMALLTIDE_PUBLISH_PORT` | `8000` | Port on this machine |
  | `SMALLTIDE_DATA` | `smalltide-data` volume | An absolute host directory for the engine `.env` and the model |
  | `SMALLTIDE_GROUP` | `video` | `keep-groups` for rootless Podman |
  | `SMALLTIDE_IMAGE` | `ghcr.io/raspy135/smalltide:latest` | Image to run |
  | `SMALLTIDE_AUTO_SETUP` | `1` | `0` stops instead of downloading the model on first start |
  | `HF_TOKEN` | | Optional, for higher Hugging Face rate limits |

  `SMALLTIDE_BIND`, `SMALLTIDE_PUBLISH_PORT`, `SMALLTIDE_DATA`, `SMALLTIDE_GROUP` and
  `SMALLTIDE_IMAGE` are read by Compose; with `docker run` or the quadlet, use the matching flags or
  lines instead.

- **Engine settings**: any key of the engine `.env`, such as the VRAM margin left for the desktop
  (`SMALLTIDE_RESIDENT_MARGIN_MB`), the context length (`SMALLTIDE_MAX_CTX`) or the RAM tier
  (`SMALLTIDE_RAM_GB`). The example lists the most useful ones; the full list is in the main
  README under "Tuning `.env`". A value set here wins over the same key in the engine `.env`, which
  keeps supplying the defaults for everything you leave out.

How the file reaches the container: Compose reads `docker/.env` by itself, `docker run` takes
`--env-file docker/.env`, and the quadlet takes `EnvironmentFile=`. Restart the container after a
change (`docker compose -f docker/compose.yaml up -d` recreates it).

`SMALLTIDE_CONFIG` only picks the template on first start; once `/data/.env` exists it is ignored.
The `monitor` template trades some context (131,072 tokens instead of 196,608) and rescue budget for
VRAM headroom for the desktop. To switch templates later, remove the old file and restart with the
new value:

```bash
docker exec smalltide rm /data/.env
docker rm -f smalltide   # then run it again with -e SMALLTIDE_CONFIG=monitor
```

With Compose, change `SMALLTIDE_CONFIG` in `docker/.env` and run `up -d` again after the `rm`; with
the quadlet, edit `Environment=SMALLTIDE_CONFIG=`, then `systemctl --user daemon-reload` and
`systemctl --user restart smalltide` after the `rm`.

To keep the model on a specific disk, bind-mount a directory instead of the named volume:
`-v /path/to/smalltide:/data`. Experts are read with O_DIRECT, so on btrfs the downloader turns off
compression for the model directory.

### More than one GPU

The engine picks the discrete AMD card with the most VRAM, whether or not that VRAM is free. If
another workload already fills that card, the start fails with `upload dense GPU: ... hipMalloc
failed with hipError_t=2` (out of memory). To pick the card yourself, set its HIP device index (0,
1, ...) in `docker/.env`:

```bash
SMALLTIDE_HIP_DEVICE=1
```

The startup log shows the chosen card's VRAM total, so you can check you got the right one;
`rocm-smi --showproductname` on the host usually lists the cards in the same order. The other way
is `HIP_VISIBLE_DEVICES=1`, which hides every other card from the container. Use one or the other,
not both: with `HIP_VISIBLE_DEVICES` set, the remaining card becomes index 0.

If the cards have the same VRAM size, the engine cannot match the chosen card to its amdgpu VRAM
counters, so the pool sizing falls back to HIP's free memory figure, which leaves out what other
programs use on that card. If you see out of memory errors after startup, raise
`SMALLTIDE_RESIDENT_MARGIN_MB` or cap `SMALLTIDE_NVFP4_SLOTS`.

### System RAM

The engine keeps two things in host RAM: the RAM tier for experts that do not fit on the card
(`SMALLTIDE_RAM_GB`, 10 GB in both templates) and the prompt prefix cache, which grows with long
conversations (about 4.2 GB when full with the headless template). Most of the cache is page-locked
for the GPU, so it barely shows in the process's memory in `ps` or `docker stats`. If the host runs
out, the kernel kills the server: the container exits with code 137 and restarts, while `docker
inspect` still reports `OOMKilled=false`, because the whole host ran out, not the container's
limit. The kernel log shows it:

```bash
sudo journalctl -k -b | grep -iE "out of memory|killed process"
```

To use less RAM, set lower values in `docker/.env`:

```bash
SMALLTIDE_RAM_GB=0               # only when every expert fits on the card (32 GB R9700)
SMALLTIDE_PREFIX_TOKENS=65536    # prefix cache: about 2 GB when full
SMALLTIDE_PREFIX_SNAPSHOTS=6
```

The startup log tells you whether the card holds every expert: `[priors] resident set covers 100.0%
of the expected routing mass per token`. Below 100%, keep a few GB of RAM tier, or experts that
miss the card are read from disk.

## Network access

All three ways publish the port on `127.0.0.1` by default, so only this machine can reach the
server. The API has **no authentication**: anyone who can reach it can run the model, change the
engine settings (`POST /v1/tide/config`) and feed the routing priors. Before opening it to your
network:

- only do it on a network you trust, or put it behind a reverse proxy that checks credentials
  (Caddy, nginx, Traefik) and keep the container itself on `127.0.0.1`;
- with Docker, do not count on the host firewall: Docker's port rules bypass firewalld and ufw.

## Other commands

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined -v smalltide-data:/data ghcr.io/raspy135/smalltide probe

docker exec smalltide smalltide-entrypoint run -p "Hello" -n 64
docker exec smalltide smalltide-entrypoint bench --only prose --runs 1
docker exec smalltide cat /opt/smalltide/BUILD_INFO
```

`setup` reruns the first-start steps (it resumes a partial download), `shell` opens bash, and any
other command goes to `smalltide-cli` (`probe`, `run`, `vram-probe`, `bench-gpu`, ...).

## Build

```bash
docker build -f docker/Dockerfile -t smalltide .
# other ROCm release or kernel targets:
docker build -f docker/Dockerfile --build-arg ROCM=10.1 --build-arg ARCH=gfx1201 -t smalltide .
```

The GitHub Actions workflow **Container** (manual) builds the image and pushes it to
`ghcr.io/<owner>/smalltide` (the repository owner, `raspy135`; a fork publishes under its own owner) with the tags `latest`, `<version>` and `<version>-rocm<rocm>`.

# SmallTide

**Adaptive native-weight multimodal MoE inference for [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (30B total / 3.8B active, text + vision) on a single 16 GB AMD Radeon RX 9070 XT on Windows (Linux build from source: see [Linux](#linux)).**

![The live expert map while the server decodes at 66 tokens a second: token routes climbing through the 48 layers, resident experts, rescues and background loads](docs/expert-map.png)

*The live expert map (`http://127.0.0.1:8000/expert-map`) during a decode: each token's route through the 48 layers, the experts that are on the card, and the ones rescued or loaded in the background.*

Inkling-2-Mini's routed experts (~14.7 GB across 3,072 native NVFP4 & MXFP8 experts) plus dense weights, KV cache, vision tower, and drafter (~7+ GB) exceed a 16 GB consumer GPU. SmallTide runs the **original Hugging Face checkpoint weights with zero GGUF re-quantization** using [TideLM](https://github.com/raspy135/tideLM)-style adaptive expert residency and custom RDNA 4 HIP kernels on Windows:

- **Native precision, no 3-bit compromise:** keeps the highest-value native NVFP4/MXFP8 experts resident in dedicated GDDR6 VRAM, promotes hot misses asynchronously from host RAM/NVMe, and rescues high-weight misses inline on the GPU.
- **Built-in Vision (0 MB idle VRAM):** full multimodal [image input](#image-input) (PNG/JPEG, OpenAI & Anthropic formats) — the 636 MB vision tower streams layer-by-layer from disk on demand and frees everything immediately after encoding, so vision costs **zero permanent VRAM or RAM**.
- **Lossless Speculative Decoding & Multi-Lane Batching:** accelerates both greedy and sampled generation with [DFlash 2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) or built-in DSpark (87–126 tok/s single-stream, up to 158–195 tok/s aggregate across concurrent lanes).
- **Elastic 32K → 192K/256K Unified Paged KV:** starts compact at 32K in VRAM (0 lost expert slots on normal turns), grows dynamically up to **192K** by default on headless GPUs (configurable up to Inkling-2-Mini's full native **256K**, or **128K** on display GPUs), and offloads idle long-context lanes to pinned host RAM before shrinking back when the server goes idle.

Pre-built Windows releases embed the compiled `.hsaco` GPU kernels (`gfx1201` and `gfx1200`) inside a single `smalltide-cli.exe` — **no Rust toolchain, no ROCm SDK, and no Python environment required** to install and serve.

## Measured Performance (RX 9070 XT 16 GB, Windows 11)

Measured with `bench/bench.py`: median of 3 runs after a warmup (the two games: 1 run each), temperature 0.6, speculative decoding on, AMD Adrenalin 26.10.44 driver, the settings of `.env.headless.example`, and the monitor on the motherboard output so the card holds nothing but the model (see [Monitor on the card or on the motherboard](#monitor-on-the-card-or-on-the-motherboard)). *Decode* is generation speed; *TTFT* is the time to the first token (it includes reading the prompt).

| Workload | Prompt / output | Decode (DFlash 2) | Decode (DSpark) | TTFT |
| --- | ---: | ---: | ---: | ---: |
| `prose` (essay, thinking off) | 58 / 1,024 | **87.0 tok/s** *(peak 90.9)* | 75.6 tok/s | 246 ms |
| `code` (LRU cache + tests, thinking off) | 74 / 1,024 | **91.0 tok/s** *(peak 93.8)* | 89.3 tok/s | 254 ms |
| `edit` (refactor 1.4K-token file, thinking off) | 1,425 / 1,024 | **109.1 tok/s** *(peak 110.3)* | 80.4 tok/s | 0.88 s |
| `code --thinking` (reasoning on) | 76 / 1,024 | **115.2 tok/s** *(peak 124.3)* | — | 259 ms |
| `random` (8K-token random prompt, thinking off) | 8,256 / 1,024 | 68.8 tok/s | **95.0 tok/s** | 4.00 s |
| `angry-birds` (one-shot HTML game, thinking off) | 76 / 8,192 | **114.4 tok/s** | 75.7 tok/s | 384 ms |
| `mario` (one-shot HTML game, thinking off) | 66 / 6,173 *(stop)* | **126.2 tok/s** | 93.8 tok/s | 428 ms |

*DFlash 2 (`incoai/Inkling-2-Mini-DFlash2`, `SMALLTIDE_DRAFTER=auto`) runs 5 sliding-window layers in FP8 (~428 MiB VRAM, 0 global KV) with host-RAM selector codebooks and lossless top-16 stochastic rejection sampling, reaching 4.5–4.8 tokens/step on `edit` and `code --thinking`. On synthetic 8K `random` prompts, DSpark's global attention layer still wins; switch anytime via `SMALLTIDE_DRAFTER=dspark` or `dflash2`. `prose` / `code` held 99.0%–99.1% VRAM hit rate (0.30%–0.49% skipped routing weight).*

Reading a prompt (prefill) runs at about **1,900 tokens a second** on a cold 10.5K-token prompt (`.env.headless`, `PREFILL_CHUNK=1024`, `ATTN_VTILES=all`), ~1,500 tok/s with `PREFILL_CHUNK=512` + `ATTN_VTILES=local`, and ~1,450+ on a 58,000-token one.

**Top-1 agreement with the full model.** Adaptive expert loading skips low-weight experts that are not on the card. Against a reference pass that loads and runs every routed expert, the engine picks the same most likely next token at **95.9%** of positions through draft-verify passes (the path speculative decoding uses) and **96.2%** through single-token steps. Measured with `bench/agreement.py` on four workloads, three prompts each, 1,024 tokens per prompt ([method and how to run it](#top-1-agreement-with-the-full-model)).

| Workload | Positions | Draft-verify passes | Single-token steps | Skipped routing weight (verify / single) |
| --- | ---: | ---: | ---: | ---: |
| `prose` | 3,072 | 96.1% | 95.8% | 0.5% / 0.6% |
| `code` | 3,072 | 96.5% | 96.3% | 1.0% / 0.9% |
| `edit` | 2,870 | 95.4% | 96.2% | 1.1% / 0.6% |
| `long` (9K-token prompt built from this repository's documents) | 3,072 | 95.4% | 96.6% | 1.7% / 0.9% |
| all | 12,086 | **95.9%** | **96.2%** | |

*Same card and settings as the table above. Fed its own reference, the exact configuration agrees at every position of the single-token path (128 positions checked), so the differences come from the skipped experts, in decode and in reading the prompt.*

**Against a lossless native BF16 dense reference (`SMALLTIDE_DENSE_FMT=bf16`, `exact` rescue, 0% skipped experts):** SmallTide's default `fp8` dense + adaptive configuration achieves **95.12%** top-1 agreement through draft-verify passes (`prose` 95.70%, `code` 94.53%; **94.92%** single-step, 512 positions) with elastic 32K → 256K KV (zero slots lost on 0–32K turns). Even with the entire **256K** KV pool allocated in VRAM (`3.42 GiB`, worst-case long conversation), it retains `1,974` resident experts (`95.9%` VRAM hit, `98.9%` Loop Core), holds **93.75%** verify top-1 agreement, and decodes at **60.5 tok/s** with DFlash 2. Setting `SMALLTIDE_DENSE_FMT=bf16` keeps all dense projections and `unembed` in native checkpoint BF16 (+1.05 GiB VRAM, ~30% slower on 16 GB; ideal for 32 GB R9700 lossless runs and parity evals).

The test says how often the top choice changes, not whether the answers are worse.

---

## Requirements

- **GPU:** an AMD Radeon RDNA 4 card with 16 GB+: **RX 9070 XT** (developed & measured here), **RX 9060 XT 16 GB** (verified by community), or **RX 9070 / Radeon AI PRO R9700 32 GB** (expected to work). See [Supported cards](#supported-cards).
- **OS:** Windows 11 (64-bit), or Linux x86-64 with a ROCm HIP runtime (see [Linux](#linux) for its requirements)
- **Driver:** **AMD Software: Adrenalin Edition 26.10.44 or newer** (ships `amdhip64_7.dll` and fixes the 26.5.1–26.9.1 WDDM residency cap bug — no ROCm SDK needed)
- **System RAM:** 32 GB recommended (16 GB of it holds the warm RAM expert tier so misses rescue over PCIe instead of NVMe; 10 GB with the monitor on the motherboard)
- **Disk:** ~36 GB free on NVMe (~21 GB Hugging Face checkpoint + ~14.7 GB packed `experts.stide`)

### Supported cards

The release carries two builds of the GPU kernels and loads the one for the card it finds.

| Card | Kernel build | Status |
| --- | --- | --- |
| Radeon RX 9070 XT (16 GB) | `gfx1201` | Developed and measured on this card. Primary benchmarks above are from it. |
| Radeon RX 9060 XT 16 GB | `gfx1200` | **Verified (community report on Windows 11, monitor on GPU):** **48–71 tok/s** median (**72.1 tok/s** peak on `mario`), 97.3–98.3% VRAM hit rate (2,187/3,072 resident experts at 271 GB/s). See table below. |
| Radeon RX 9070 (16 GB) | `gfx1201` | **Untested.** Same architecture and kernels as the XT, so it is expected to work. |
| Radeon AI PRO R9700 (32 GB) | `gfx1201` | **Untested.** Same `gfx1201` kernels; all 3,072 experts fit 100% in VRAM (even with `SMALLTIDE_DENSE_FMT=bf16` for bit-exact native weights). |

<details>
<summary><strong>Community Benchmark — RX 9060 XT 16 GB (<code>gfx1200</code>, Windows 11, monitor on GPU)</strong></summary>

*Measured with `bench/bench.py` (`thinking: false`, `temp: 0.6`, 3 runs per workload; DXGI budget `13.06–13.16 GiB`, `2,015` NVFP4 + `172` MXFP8 resident slots, `271 GB/s` measured pool copy bandwidth):*

| Workload | TTFT (med) | Decode (med) | Peak Decode | Hit Rate | Skipped Wt | Rescues / tok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `prose` (1K out) | 316.5 ms | **48.36 tok/s** | 48.93 tok/s | 98.30% | 0.62% | 1.88 |
| `code` (1K out) | 402.8 ms | **50.25 tok/s** | 53.99 tok/s | 97.85% | 0.72% | 3.63 |
| `edit` (~1.4K in / 1K out) | 2,236.9 ms | **50.89 tok/s** | 52.38 tok/s | 97.72% | 1.15% | 5.82 |
| `random` (~8.2K in / 1K out) | 9,429.2 ms | **39.50 tok/s** | 43.61 tok/s | 97.31% | 2.47% | 9.97 |
| `angry-birds` (8K out) | 501.7 ms | **63.86 tok/s** | 64.71 tok/s | 97.37% | 1.05% | 4.04 |
| `mario` (8K out) | 396.9 ms | **70.91 tok/s** | 72.11 tok/s | 97.41% | 1.02% | 3.45 |

</details>

`.\smalltide.cmd probe` (the first step of `setup`, before the model download) prints the card, its architecture and the kernel build chosen, and runs a test kernel on the card. If you run an RX 9070 or an R9700, please open an issue with your `probe` output and numbers!

The RX 9070 GRE is `gfx1201` as well and loads the same kernels as the RX 9070 XT. The 8 GB RX 9060 XT and RX 9060 are not a target: the dense weights, KV cache and drafter take about 6 GB before the first expert. RDNA 3 and older cards are not supported. Details in [LIMITATIONS.md](LIMITATIONS.md).

---

## Quick Start (Pre-built Release — No SDKs Needed)

1. **Download and extract** `smalltide-v0.1.1-windows-x64-gfx1201.zip` from [GitHub Releases](../../releases).
2. **Choose your display setup (monitor on the card vs. motherboard / iGPU):** this is the main decision that determines how much of the 16 GB card the model gets and which config you use:
   - **Monitor on the motherboard / CPU iGPU (recommended for best speed):** plug your monitor into the motherboard's video output and reboot. The RX 9070 XT then holds nothing but the model (~2,350–2,550 of 3,072 experts resident with elastic 32K→192K KV, ~98–99% hit rate, zero desktop interference). The numbers in [Measured Performance](#measured-performance-rx-9070-xt-16-gb-windows-11) were taken this way. Use **`.env.headless`** (`SMALLTIDE_RAM_GB=10`).
   - **Monitor plugged into the RX 9070 XT:** Windows and desktop apps hold 3 to 5 GB of VRAM (~1,750 experts resident, ~90% routing weight). Use **`.env.monitor`**, and assign your browser / chat client to integrated graphics in Windows Settings to avoid Edge/Chrome tile glitches and streaming interference.
   - Full comparison and setup steps: [Monitor on the card or on the motherboard](#monitor-on-the-card-or-on-the-motherboard).

3. **Run one-time setup** in PowerShell or Command Prompt inside the extracted folder:

```powershell
.\smalltide.cmd setup
```

`setup` creates `.env.headless`, `.env.monitor`, and `.env` from the `.example` templates, verifies your GPU, its kernels & the HIP driver (`probe`), downloads [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (~21 GB) and [`incoai/Inkling-2-Mini-DFlash2`](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) (~944 MB) via built-in `curl.exe` (resumable; set `$env:HF_TOKEN` only if you want higher Hugging Face rate limits), and packs all 3,072 routed experts into 4 KB-aligned `experts.stide` (~15 s).

4. **Start the server** with the config matching your display setup (do not edit or run `.example` files directly — copy them first if not created by `setup`):

```powershell
# Monitor plugged into the RX 9070 XT:
copy .env.monitor.example .env.monitor
.\smalltide.cmd serve .env.monitor

# Monitor on motherboard / iGPU (headless RX 9070 XT):
copy .env.headless.example .env.headless
.\smalltide.cmd serve .env.headless
```

5. **Use it:**
   - **OpenAI API:** `http://127.0.0.1:8000/v1` (model name: `inkling-2-mini`) — works with Open WebUI, Continue, Cline, OpenCode, and any OpenAI-compatible client.
   - **Live Expert Map:** open **`http://127.0.0.1:8000/expert-map`** in your browser to watch routed experts, VRAM residency, inline rescues, and async promotions live.
   - **One-shot CLI test:**

```powershell
.\smalltide.cmd run -p "Write a concise explanation of speculative decoding." -n 256
```

*(`smalltide.cmd` invokes `smalltide.ps1` with `-ExecutionPolicy Bypass`, so it works out of the box on stock Windows 11 without changing PowerShell execution policies.)*

---

## API Endpoints

| Method & Path | Description |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions (streaming SSE & non-streaming, `reasoning_content`, `reasoning_effort`, tool calls, image input, `cache_prompt`) |
| `POST /v1/completions` | OpenAI Legacy Completions (streaming & non-streaming) |
| `GET  /v1/models` | Lists `inkling-2-mini` |
| `GET  /expert-map` | Live WebGL/Grid expert residency & routing visualizer |
| `GET  /v1/tide/stats` | Live TideLM telemetry (tok/s, hit rates by zone, rescues, skipped weight, VRAM) |
| `POST /v1/tide/config` | Live runtime tuning of `rescue_mode`, `rescue_threshold` and `rescue_auto_gate` |
| `GET  /health` | Health check |

Quick `curl.exe` smoke test:

```powershell
curl.exe -s http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Say hello in 10 words."}],"max_tokens":64}'
```

---

## Image Input

Inkling-2-Mini reads images, and SmallTide runs its vision tower on demand: the tower's weights (636 MB) stay in the checkpoint file and are streamed through the GPU a layer at a time while an image is encoded, then everything is freed. Nothing of it is resident in RAM or VRAM, so text-only use is unaffected.

Send images as content parts of a user (or tool) message, in the OpenAI format with a base64 `data:` URL:

```json
{"role": "user", "content": [
  {"type": "text", "text": "What is in this picture?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}
]}
```

- **Formats:** PNG and JPEG. Anthropic-style parts (`{"type": "image", "source": {"type": "base64", "data": "..."}}`) are accepted too. `http(s)` image URLs are not fetched; send the image data.
- **Cost:** an image is used at its own resolution, one token per 40 × 40 pixel cell: 1024 × 1024 is 676 tokens, 1920 × 1080 is 1,296. Larger images are scaled down to `SMALLTIDE_VISION_MAX_TOKENS`.
- **Time:** encoding takes about 0.3 s for 1024 × 1024 and 0.8 s for 1920 × 1080 (RX 9070 XT), plus the prefill of the image's tokens. It uses 40–240 MB of VRAM for the duration of the encode.
- **Caching:** the prefix cache keys image positions by the image's content, so a follow-up turn or a repeated request does not encode or prefill the image again.

---

## How SmallTide Works

1. **Loop-aware adaptive residency (TideLM).** Inkling-2-Mini runs 24 physical MoE layers as 48 logical layers: 6 Prelude layers, 12 Loop Core layers executed **3 times per token**, and 6 Coda layers. SmallTide tracks every router pick into persistent routing priors (`<MODEL_DIR>/tide_priors.json`), automatically seeded on first boot from `profiles/inkling2-mini-priors.json` (aggregate routing shares only, zero user text). Value-per-byte placement prioritizes the 3× Loop Core first (~98%+ hit rate in VRAM), filling the rest of dedicated VRAM with the hottest Prelude NVFP4 and Coda MXFP8 experts.
2. **Windows DXGI residency-aware sizing.** On Windows WDDM, `hipMemGetInfo` reports shared system RAM as free VRAM, causing oversized allocations to silently drop ~10× in bandwidth over PCIe. With `SMALLTIDE_POOL_BUDGET=resident` (default), SmallTide queries DXGI's local residency budget and probes copy bandwidth across every 256 MiB window at startup (`fit_pools_to_resident`) so 100% of active expert slots stay on dedicated GDDR6.
3. **2-Stage Miss Handling + Fused GPU Spin-Wait Rescue.** Non-VRAM experts live warm in host RAM (`SMALLTIDE_RAM_GB=16`: every expert that is not on the card plus the ones most likely to leave it; a stray one is read back in the background) backed by 4 KB-aligned Win32 `FILE_FLAG_NO_BUFFERING` NVMe reads (`experts.stide`). Misses above each zone's gate, up to per-zone budgets of `33/30/35` per step, are copied over PCIe on a dedicated DMA stream and synchronized on-GPU by a fused HIP spin-wait + route-patch kernel (`SMALLTIDE_RESCUE_WAIT=spin`), while the rest are skipped and renormalized. The gates start at `0.07` and are adjusted while decoding so each zone spends about 80% of its budget (`SMALLTIDE_RESCUE_AUTO_GATE=1`). Hot misses promote asynchronously in the background.
4. **Lossless DFlash 2 & DSpark Speculative Decoding.** SmallTide supports both [`incoai/Inkling-2-Mini-DFlash2`](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) (5 sliding-window layers with 2-tap grouped dynamic convolutions and a pairwise candidate path selector, ~428 MiB VRAM in FP8 with zero global KV and host-RAM codebooks) and Inkling-2-Mini's built-in 3-layer DSpark drafter (`mtp.safetensors`). Both draft up to 7 tokens per step with confidence-gated adaptive depth. Verification uses GPU-side top-k/top-p filtering (`smalltide_logits_filter_rows`) and exact lossless rejection sampling (including sparse top-16 proposal distributions `q` for DFlash 2 at `temperature > 0`), accelerating both greedy and stochastic sampling without altering the target distribution.
5. **Unified Elastic Paged K8/V4 KV & Prefix Caching.** All 40 local sliding-window layers, 8 global layers, and 5 DFlash 2 drafter layers share one unified paged KV pool (`SMALLTIDE_KV_INITIAL_TOKENS=32768` → `SMALLTIDE_KV_TOKENS=196608` headless / `131072` monitor, extendable up to `262144`) with asymmetric **`k8v4`** global quantization (`SMALLTIDE_KV_FMT=k8v4`: FP8 E4M3 keys + per-32-group asymmetric INT4 values, saving ~708 MiB at 192K / ~944 MiB at 256K). Extra serving lanes cost only **~3.7 MiB VRAM and 0 extra host RAM**. On long prompts, the pool grows dynamically by borrowing tail NVFP4 expert slots; after `SMALLTIDE_KV_IDLE_SHRINK_SECS` (default 5 min) of server idleness, blocking lanes are snapshotted into the pinned host-RAM prefix cache first and the pool shrinks back to 32K, returning all borrowed expert slots to VRAM.
6. **Zero-Idle-VRAM Streaming Vision.** Unlike static `mmproj` loaders that permanently occupy ~640 MB of VRAM, SmallTide streams Inkling-2-Mini's 636 MB vision tower directly from the checkpoint file one layer at a time during image encoding (~0.3 s for 1024×1024, ~0.8 s for 1920×1080) and immediately frees all temporary buffers (40–240 MB peak). Encoded image tokens are content-hashed into the prefix cache so follow-up turns reuse them without re-encoding.

---

## Tuning `.env`

Edit `.env` and restart `.\smalltide.cmd serve`. Key knobs:

| Setting | Default | What it controls |
| --- | --- | --- |
| `SMALLTIDE_MAX_CTX` / `SMALLTIDE_KV_TOKENS` | `196608` *(headless)* / `131072` *(monitor)* | Context length & unified paged KV pool ceiling across all lanes (`192K` headless default, extendable up to `262144` / 256K; `128K` on display GPU so desktop VRAM doesn't starve experts). |
| `SMALLTIDE_KV_INITIAL_TOKENS` / `SMALLTIDE_KV_IDLE_SHRINK_SECS` | `32768` / `300` | Elastic KV starting size in VRAM and idle timeout (seconds, `0` = off) before offloading expanded lanes to RAM prefix cache and shrinking back to baseline. |
| `SMALLTIDE_KV_FMT` | `k8v4` | Global KV format: `k8v4` (FP8 keys + Q4 values, saves ~708 MiB @ 192K / ~944 MiB @ 256K) or `fp8` (full FP8 K/V). |
| `SMALLTIDE_DENSE_FMT` | `fp8` | Dense weight precision (attention projections, loop controllers, `unembed`): `fp8` (~1.10 GiB VRAM; recommended on 16 GB) or `bf16` (~2.15 GiB, exact native checkpoint weights for 32 GB R9700 / parity evals). |
| `SMALLTIDE_LANES` | `4` | Concurrent serving lanes (all lanes share the unified paged KV pool; each extra lane costs only ~3.7 MiB VRAM and 0 extra host RAM). |
| `SMALLTIDE_RESCUE_MODE` | `threshold` | `threshold` (balanced), `exact` (copy every miss, bit-exact MoE), or `disabled` (fastest, skip inline misses). |
| `SMALLTIDE_RESCUE_THRESHOLD` / `..._CORE` | `0.07` / `0.07` | Routing-weight gate for inline rescues (Prelude & Coda / Loop Core). Lower gates rescue more: better fidelity, slower decode (a rescue costs about 0.7 ms). |
| `SMALLTIDE_RESCUE_MUST_THRESHOLD` | `0.25` | Missing experts at or above this routing share **always** rescue regardless of zone budgets or auto-gate (`0` = off), preventing sudden domain/language switches from dropping dominant specialist experts. |
| `SMALLTIDE_LOOP_DETECT` | `1` | Repetition loop detector (`1` = on, `0` = off): force-exits thinking loops into a clean answer and hard-stops runaway text loops (skips fuzzy checks inside tool-call JSON). |
| `SMALLTIDE_RESCUE_BUDGET_PRELUDE` / `_CORE` / `_CODA` | `33` / `30` / `35` | Per-token inline rescue caps per architectural zone. |
| `SMALLTIDE_RESCUE_AUTO_GATE` | `1` | `1` adjusts each zone's gate while decoding so the zone uses about 80% of its rescue budget (`SMALLTIDE_RESCUE_AUTO_TARGET`): it starts at the threshold above, falls while budget is left over and rises when the zone goes over. The gate stays within `SMALLTIDE_RESCUE_AUTO_MIN`..`_MAX` (`0.03`..`0.5`). A zone with fewer misses than its budget ends at the lower bound. |
| `SMALLTIDE_PREFILL_CHUNK` | `1024` | Tokens per prefill chunk (64 to 1024). Larger chunks read a prompt faster for more scratch VRAM: 1024 costs about 80 resident experts against 256 and is almost twice as fast. |
| `SMALLTIDE_ATTN_VTILES` | `all` | Prefill attention reads value vectors decoded once per layer pass. `all` spends 256 MiB of VRAM (capped at 128K positions, ~55 resident experts; prompts past 128K fall back to on-the-fly V decode) and reads a 58K-token prompt a third faster; `local` spends 4 MiB; `0` is off. |
| `SMALLTIDE_BATCH_RESCUE` | `1` | Two requests at once are decoded as one batch; `1` rescues missing experts for the batch like a draft-verify pass, `0` skips them (faster, lower fidelity). |
| `SMALLTIDE_IDLE_HEARTBEAT_MS` | `1000` | Idle GPU heartbeat; keep it on when no monitor is plugged into the card (see below). `0` turns it off. |
| `SMALLTIDE_SPEC` | `1` | Enable (`1`) or disable (`0`) speculative decoding. |
| `SMALLTIDE_DRAFTER` | `auto` | Drafter backend: `auto` (DFlash 2 if `models/Inkling-2-Mini-DFlash2` is present, else DSpark), `dflash2`, or `dspark`. Set `SMALLTIDE_DFLASH2_DIR` to use a custom DFlash 2 checkpoint path. |
| `SMALLTIDE_VISION` / `SMALLTIDE_VISION_MAX_TOKENS` | `1` / `1536` | Image input on (`1`) or off (`0`), and the most tokens one image takes (larger images are scaled down). |
| `SMALLTIDE_RAM_PINNED_GB` | `0` | Experimental. This much of the RAM tier in page-locked memory: those experts upload to the card without a copy into the staging buffer (`2`: about 3% faster decode on a 12K-token prompt). Capped at 3; with 5 the server lost the GPU context, see [LIMITATIONS.md](LIMITATIONS.md). |
| `SMALLTIDE_RAM_GB` | `16` | Host RAM tier for non-VRAM experts. It has to hold every expert that is not on the card (about 4.8 MiB each) with room to spare; `16` holds all 3,072. Set `10` when the monitor is on the motherboard (see below). |
| `SMALLTIDE_HOST` / `SMALLTIDE_PORT` | `0.0.0.0` / `8000` | Server bind address (`127.0.0.1` for local-only) and port. |

> **Getting maximum resident experts on Windows:** Closing VRAM-heavy desktop apps (browsers with many tabs, Adobe Creative Cloud, games, hardware-accelerated Electron apps) before launching `.\smalltide.cmd serve` directly increases the number of experts SmallTide can keep resident on the card. Run `.\smalltide.cmd vram-probe` anytime to inspect your card's current resident headroom.

To reset learned routing priors back to the bundled seed, delete `models\Inkling-2-Mini\tide_priors.json` and restart the server.

---

## Monitor on the card or on the motherboard

Where the monitor cable goes decides how much of the 16 GB the model gets. A 4K desktop, a browser and a few Electron apps hold 3 to 5 GB of the card when the monitor is plugged into it; with the cable on the motherboard (the Ryzen's integrated graphics draws the desktop) the card holds nothing but the model.

| | Monitor on the RX 9070 XT | Monitor on the motherboard |
| --- | --- | --- |
| Resident experts (4 lanes, elastic KV from 32K, drafter on) | about 1,750–1,900 of 3,072 *(up to 128K)* | about 2,350–2,550 of 3,072 *(up to 192K, max 256K)* |
| Routing weight they cover | about 91–94% | about 98–99% |
| Desktop apps and the model | compete for VRAM: when the desktop's share grows, Windows moves part of the model to system RAM and decode drops to a quarter of its speed until the server is restarted | separate: the desktop uses system RAM through the integrated graphics |
| A rescue that stalls for more than about 2 s | Windows resets the graphics driver (`hipError_t=719`), the server exits | no driver reset was seen (nothing else asks for the card) |
| An idle card | stays awake (it draws the desktop) | is put to sleep by the driver unless the heartbeat runs: see below |
| System RAM | the desktop's graphics memory is on the card | the desktop's graphics memory comes out of system RAM (1 to 4 GB) |

**Setting it up with the monitor on the motherboard**

1. Move the cable to the motherboard's video output and **reboot**: programs keep using the graphics card they started on, so until then the desktop still sits on the 9070 XT.
2. Copy `.env.headless.example` to `.env.headless` (`copy .env.headless.example .env.headless`) and set `SMALLTIDE_RAM_GB=10`: the desktop now lives in system RAM too.
3. Start the server (`.\smalltide.cmd serve .env.headless`). The startup line should show `VRAM free: … /15.92 GiB` (the engine picks the discrete card by itself; `SMALLTIDE_HIP_DEVICE` overrides it) and a resident count well above 2,000.
4. Leave `SMALLTIDE_IDLE_HEARTBEAT_MS=1000` at its default. Radeon drivers 26.5.1 and later put a card with no display to sleep about 10 seconds after its last GPU work, and Windows first moves everything in the card's memory - the whole model - into system RAM, which exhausts a 32 GB machine within seconds ([ROCm/TheRock#7221](https://github.com/ROCm/TheRock/issues/7221)). The heartbeat is a few milliseconds of GPU work once a second from the moment the server starts, loading included, and keeps the card awake. AMD reports the driver bug fixed in Adrenalin 26.9.2; that was not verified here, and 26.10.44 (build 32.0.31044.16) still has it.
5. Run one server at a time. A second one would load a second copy of the model next to the first; it now stops at once with `cannot listen on …`.

**Keeping the monitor on the RX 9070 XT (avoiding glitches and interference)**

1. Copy `.env.monitor.example` to `.env.monitor` (`copy .env.monitor.example .env.monitor`) and launch with `.\smalltide.cmd serve .env.monitor`. It uses elastic 128K KV starting at 32K (`SMALLTIDE_KV_INITIAL_TOKENS=32768`), `SMALLTIDE_PREFILL_CHUNK=512`, and `SMALLTIDE_ATTN_VTILES=local` to reclaim VRAM for resident experts while leaving headroom for the desktop. (Use `.env.headless.example` → `.env.headless` for elastic 192K — extendable up to 256K — when the monitor is on the motherboard.)
2. **Move your browser / chat client to integrated graphics** if your CPU has an iGPU. Running Edge/Chromium on the 9070 XT alongside HIP compute can cause AMD D3D11/MPO speckled tile glitches inside browser windows and lowers decode tok/s because every streamed token repaints on the same GPU:
   - Open **Windows Settings → System → Display → Graphics**.
   - Select your browser or chat app. If it is not in the list, click **Add desktop app** (or **Browse**) and pick its `.exe` — for Microsoft Edge, `C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`.
   - Set **GPU preference** to **Power saving** (integrated graphics).
   - Fully close the app (for Edge, close background `msedge.exe` processes in Task Manager or turn off *Startup boost*) and reopen it. Verify in `edge://gpu` that `GL_RENDERER` shows your integrated GPU instead of the RX 9070 XT.
   - *(Without an iGPU, turn off **Use graphics acceleration when available** in browser settings, or switch `edge://flags/#use-angle` from `Default` to `OpenGL` / `D3D9`.)*
3. Closing other VRAM-heavy apps before starting the server buys additional resident experts.

---

## Benchmarking

With `.\smalltide.cmd serve` running in another terminal (requires Python 3, standard library only):

```powershell
.\smalltide.cmd bench --workload prose --runs 3 --out results\prose.json
# every workload: prose, code, edit, random (--isl 8192), angry-birds, mario
foreach ($w in "prose","code","edit","random","angry-birds","mario") { .\smalltide.cmd bench --workload $w --out "results\$w.json" }
```

Each run prints decode speed, TTFT, skipped routing weight, inline rescues/token, VRAM hit rates by zone and disk reads; `--out` writes the summary JSON, and the one-shot games are saved to `results/oneshots/`.

### Top-1 agreement with the full model

Adaptive expert loading skips low-weight experts that are not on the card (see [LIMITATIONS.md](LIMITATIONS.md)). `bench/agreement.py` measures how often that changes the most likely next token. A reference pass runs every routed expert (`SMALLTIDE_RESCUE_MODE=exact`, `SMALLTIDE_SPEC=0`) and saves its greedy tokens. A second pass with your settings is fed those tokens one position at a time, and its own top choice is compared with the reference's at each position, once through draft-verify passes (the path speculative decoding uses) and once through single-token steps.

Stop the server first (the test needs the card to itself):

```powershell
python bench\agreement.py                                            # 4 workloads x 3 prompts x 1,024 tokens, about 12 minutes
python bench\agreement.py --workloads prose code --runs 1 --tokens 256   # about 2 minutes
```

The results are in [Measured Performance](#measured-performance-rx-9070-xt-16-gb-windows-11).

---

## Building from Source

Only needed if you want to modify the Rust engine/HIP kernels or choose which architectures the kernels are compiled for (`SMALLTIDE_HIP_ARCH`, comma-separated, default `gfx1201,gfx1200`):

- **Prerequisites:** [Rust](https://rustup.rs/) (MSVC toolchain) + **AMD ROCm HIP SDK for Windows** (`hipcc.exe`, auto-detected under `C:\Program Files\AMD\ROCm\*\bin\` or `ROCM_PATH` / `SMALLTIDE_HIPCC`).

```powershell
git clone https://github.com/raspy135/SmallTide.git
cd SmallTide
.\smalltide.cmd build
.\smalltide.cmd setup
.\smalltide.cmd serve
```

To build a standalone pre-built Windows release zip (for GitHub Releases):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package-release.ps1
```

---

## Linux

SmallTide also builds and runs on Linux from source. Windows and Linux share the same engine and HIP kernels; only the platform layer differs (expert reads use `O_DIRECT`, the HIP runtime is `libamdhip64.so`), and `smalltide.sh` replaces `smalltide.cmd` / `smalltide.ps1` with the same commands and `.env` files.

**Status:** run on one RX 9070 XT (Fedora, ROCm HIP runtime 7.1, monitor on the card). There is no pre-built Linux release yet.

**Requirements:**

- **ROCm HIP runtime** 6.x or 7.x with RDNA 4 support (`libamdhip64.so`, from your distribution's ROCm packages or `/opt/rocm`). Your user needs access to `/dev/kfd` and `/dev/dri` (usually the `render` and `video` groups).
- **`hipcc`** and **[Rust](https://rustup.rs/)** to build. `hipcc` is found via `SMALLTIDE_HIPCC`, `ROCM_PATH` / `HIP_PATH`, `/opt/rocm*/bin`, then `PATH`. Without it the build still succeeds but has no GPU kernels, and `probe` fails.
- **`curl`** for the model download, and **Python 3** only for `bench/` and `tools/`.
- RAM and disk as in [Requirements](#requirements).

```bash
git clone https://github.com/raspy135/SmallTide.git
cd SmallTide
./smalltide.sh build      # or: scripts/build.sh --test
./smalltide.sh setup      # .env files, probe, ~22 GB download, pack experts.stide
./smalltide.sh serve .env.monitor    # monitor on the card
./smalltide.sh serve .env.headless   # monitor on the motherboard / iGPU
```

**Differences from Windows:**

- **VRAM budget:** there is no DXGI on Linux, so `SMALLTIDE_POOL_BUDGET=resident` reads the amdgpu driver's device-wide VRAM counters (`/sys/class/drm/card*/device/mem_info_vram_used` / `_total`). They include the desktop's share, which HIP's own free-memory figure leaves out (about 1.2 GiB on the test machine), so the pools are sized to what is really free and shrunk again once the engine buffers exist. With the monitor on the card, the RX 9070 XT kept about 2,450 of 3,072 experts resident with `.env.monitor`, all at VRAM speed. The templates' `SMALLTIDE_RESIDENT_MARGIN_MB` / `SMALLTIDE_RESIDENT_TAIL_MB` are tuned for Windows; on Linux, `256` / `128` kept image input working and gave about 2,490 experts.
- **btrfs:** reads of compressed btrfs files go through the page cache even with `O_DIRECT`. `scripts/download_inkling2.sh` turns compression off on the model directories before downloading; for a directory you create yourself, run `btrfs property set <dir> compression none` before copying the model into it.
- **Server address:** both `.env` templates set `SMALLTIDE_HOST=0.0.0.0`, which listens on every network interface. Set `SMALLTIDE_HOST=127.0.0.1` to keep the server local.
- **Release bundle:** `scripts/package-release.sh` builds `dist/smalltide-v<version>-linux-x64-<arch>.tar.gz`. It still needs the ROCm HIP runtime on the target machine.

See [LIMITATIONS.md](LIMITATIONS.md) for known constraints, [CREDITS.md](CREDITS.md) for attributions, and [docs/PROGRESS.md](docs/PROGRESS.md) for engineering notes.

## License

Repository code is [MIT-licensed](LICENSE). Inkling-2-Mini model weights are governed by [Thinking Machines' license](https://huggingface.co/thinkingmachines/Inkling-2-Mini).

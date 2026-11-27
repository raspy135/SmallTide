mod chat_format;
mod expert_map;
mod image;
mod prefix_cache;
mod server;
mod settings;

use smalltide_backend_cpu as cpu;
use smalltide_backend_hip::{HipBackend, HipRuntime};
use smalltide_core::{ComputeBackend, DType, Inkling2Config, TideConfig};
use smalltide_io::dense_loader::Inkling2DenseHost;
use smalltide_io::safetensors::inspect_safetensors_header;
use smalltide_io::stide_pack::{ExpertSlabHeader, ExpertSlabLayout, StidePackReader};
use smalltide_io::tokenizer::BpeTokenizer;
use smalltide_io::win_direct_io::{align_up, AlignedBuffer, DirectFile, SECTOR_ALIGN};
use smalltide_model_inkling2::engine::SamplingParams;
use smalltide_model_inkling2::weights::Inkling2DenseGpu;
use smalltide_tide::TideExecutor;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn extract_env_flag(raw: Vec<String>) -> (Option<std::path::PathBuf>, Vec<String>) {
    let mut env_path: Option<std::path::PathBuf> = None;
    let mut out = Vec::with_capacity(raw.len());
    let mut it = raw.into_iter();
    if let Some(prog) = it.next() {
        out.push(prog);
    }
    let mut after_dd = false;
    while let Some(arg) = it.next() {
        if after_dd {
            out.push(arg);
            continue;
        }
        if arg == "--" {
            after_dd = true;
            out.push(arg);
            continue;
        }
        // Never interpret the prompt value following -p/--prompt as a flag.
        if arg == "-p" || arg == "--prompt" {
            out.push(arg);
            if let Some(v) = it.next() {
                out.push(v);
            }
            continue;
        }
        if arg == "--env" || arg == "-e" {
            let Some(p) = it.next() else {
                eprintln!("error: {arg} requires a file path");
                std::process::exit(2);
            };
            env_path = Some(std::path::PathBuf::from(p));
            continue;
        }
        if let Some(p) = arg.strip_prefix("--env=") {
            if p.is_empty() {
                eprintln!("error: --env= requires a file path");
                std::process::exit(2);
            }
            env_path = Some(std::path::PathBuf::from(p));
            continue;
        }
        out.push(arg);
    }
    // Convenience positional form for `serve`: `smalltide-cli serve .env.monitor`
    if env_path.is_none()
        && out.get(1).map(String::as_str) == Some("serve")
        && out.get(2).is_some_and(|a| !a.starts_with('-'))
    {
        env_path = Some(std::path::PathBuf::from(out.remove(2)));
    }
    (env_path, out)
}

fn main() {
    let (env_override, args) = extract_env_flag(env::args().collect());
    let env_file = settings::load_dotenv(env_override.as_deref());
    if matches!(args.get(1).map(String::as_str), Some("run") | Some("serve")) {
        settings::describe_source(&env_file);
    }
    match args.get(1).map(String::as_str) {
        Some("probe") => run_probe(),
        Some("vision-check") => run_vision_check(&args[2..]),
        Some("vram-probe") => run_vram_probe(),
        Some("bench-gpu") => run_bench_gpu(),
        Some("stress-rescue") => {
            let mode = args.get(2).map(String::as_str).unwrap_or("spin");
            let tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(500);
            run_stress_rescue(mode, tokens);
        }
        Some("bench-nvme") => run_bench_nvme(),
        Some("load-test") => {
            #[cfg(windows)]
            let default_dir = r"models\Inkling-2-Mini";
            #[cfg(not(windows))]
            let default_dir = "models/Inkling-2-Mini";
            let dir = args
                .get(2)
                .map(String::as_str)
                .unwrap_or(default_dir);
            run_load_test(Path::new(dir));
        }
        Some("run") => {
            run_generate(&args[2..]);
        }
        Some("serve") => {
            server::run_serve(&args[2..]);
        }
        Some("pack") => {
            let (Some(src), Some(dst)) = (args.get(2), args.get(3)) else {
                eprintln!("Usage: smalltide-cli pack <model.safetensors> <out.stide>");
                std::process::exit(1);
            };
            let cfg = Inkling2Config::default();
            println!("Packing Inkling-2-Mini experts from {} -> {} ...", src, dst);
            let t0 = Instant::now();
            smalltide_io::stide_pack::StidePackReader::build_from_safetensors(
                &cfg,
                Path::new(src),
                Path::new(dst),
            )
            .expect("pack .stide");
            println!("Packed all 3,072 experts in {:.2}s!", t0.elapsed().as_secs_f64());
        }
        Some("render-chat") => {
            // Dev tool: `render-chat <cases.json>`; each case {messages, tools?, reasoning_effort?}.
            let Some(input) = args.get(2) else {
                eprintln!("Usage: smalltide-cli render-chat <cases.json>");
                std::process::exit(1);
            };
            let text = std::fs::read_to_string(input).expect("read input");
            let cases = chat_format::parse_json(&text).expect("parse cases");
            let chat_format::OJson::Arr(cases) = cases else { panic!("cases must be a list") };
            let mut out = Vec::new();
            for c in &cases {
                let msgs = match c.get("messages") {
                    Some(chat_format::OJson::Arr(m)) => m.clone(),
                    _ => Vec::new(),
                };
                let r = chat_format::render_chat(&msgs, c.get("tools"), c.get("reasoning_effort"));
                out.push(match r {
                    // The template's text: without the guards on the caller's text.
                    Ok(r) => serde_json::json!({ "ok": r.prompt.replace(chat_format::TEXT_GUARD, "") }),
                    Err(e) => serde_json::json!({ "err": e }),
                });
            }
            println!("{}", serde_json::to_string(&out).unwrap());
        }
        Some("tokenize") => {
            // Dev tool: `tokenize <tokenizer.json> <in.txt>` -> JSON token ids of the whole file.
            let (Some(tj), Some(input)) = (args.get(2), args.get(3)) else {
                eprintln!("Usage: smalltide-cli tokenize <tokenizer.json> <in.txt>");
                std::process::exit(1);
            };
            let tok = BpeTokenizer::from_file(Path::new(tj)).expect("load tokenizer.json");
            let text = std::fs::read_to_string(input).expect("read input");
            let ids = tok.encode(&text);
            println!("{}", serde_json::to_string(&ids).unwrap());
        }
        Some("inspect") => {
            let Some(path) = args.get(2) else {
                eprintln!("Usage: smalltide-cli inspect <path-to.safetensors> [substring-filter]");
                std::process::exit(1);
            };
            let filter = args.get(3).map(String::as_str);
            run_inspect(Path::new(path), filter);
        }
        _ => {
            println!("SmallTide CLI");
            println!("Commands:");
            println!("  probe                         Probe AMD HIP GPU + Stage 1 H2D DMA microbenchmark");
            println!("  bench-gpu                     Verify GPU kernels vs CPU + benchmark BW / hipGraph / overlap");
            println!("  bench-nvme                    Run Stage 2 Win32 NO_BUFFERING random expert read benchmark");
            println!("  pack <in.safetensors> <out.stide>  Pack 3,072 experts into 4KB-aligned .stide store");
            println!("  load-test [model-dir]         Verify real tokenizer + FP8 dense load + Mode 1 Tide pre-prefill on GPU");
            println!("  run -p <prompt> [-n N] [--temp T] [--effort E] [--raw]   Generate once (config: .env)");
            println!("  serve [env-file]              Serve OpenAI API as inkling-2-mini (default: .env)");
            println!("  vram-probe                    Measure resident vs system-memory-backed VRAM");
            println!("  inspect <file.safetensors>    Inspect tensor names, dtypes, and shapes");
            println!("Options:");
            println!("  --env <path>, -e <path>       Load configuration from <path> (overrides .env and SMALLTIDE_ENV)");
            println!("Configuration: .env in the working directory (or `serve <path>` / `--env <path>` / SMALLTIDE_ENV=path); see .env.headless.example & .env.monitor.example.");
        }
    }
}

fn run_probe() {
    let hip = HipRuntime::load().expect("load HIP runtime");
    let dev = hip.select_device().expect("select HIP device");
    let info = hip.probe_device(dev).expect("probe HIP device");
    println!("Loaded HIP DLL : {}", info.loaded_dll);
    println!("HIP Runtime    : {}", info.runtime_version);
    println!("HIP Driver     : {}", info.driver_version);
    println!("Device Count   : {}", info.device_count);
    println!("Device[{dev}]      : {}", info.device_name);
    println!("Device arch    : {}", info.device_arch.as_deref().unwrap_or("not reported by the runtime"));
    if info.hsaco_bytes == 0 {
        // No kernels for this card (or a build without hipcc): say why and fail the probe.
        let why = hip.kernels_for_device(dev).err().map(|e| e.to_string()).unwrap_or_default();
        println!("Embedded HSACO : 0 bytes");
        eprintln!("error: {why}");
        std::process::exit(1);
    }
    println!("Kernel arch    : {}", info.kernel_arch);
    println!("Embedded HSACO : {} bytes", info.hsaco_bytes);
    match hip.check_kernels(dev) {
        Ok(()) => println!("Kernel check   : module loaded, test kernel ran on the card"),
        Err(e) => {
            println!("Kernel check   : FAILED");
            eprintln!("error: the {} kernels do not run on this card: {e}", info.kernel_arch);
            std::process::exit(1);
        }
    }
    println!(
        "VRAM           : {:.2} GiB free / {:.2} GiB total",
        info.free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        info.total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    for (label, bytes) in [
        ("NVFP4 expert (4.27 MiB)", 4_478_976usize),
        ("MXFP8 expert (7.83 MiB)", 8_211_456usize),
    ] {
        let res = hip.bench_h2d_dma(bytes, 100).expect("DMA bench");
        println!(
            "DMA H2D [{}]: {:.3} ms/expert ({:.2} GB/s)",
            label, res.avg_ms, res.bandwidth_gb_s
        );
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run_bench_gpu() {
    let hip = HipRuntime::load().expect("load HIP runtime");
    let dev = hip.select_device().expect("select HIP device");
    let info = hip.probe_device(dev).expect("probe HIP device");
    println!(
        "=== SmallTide Phase 1 {} Kernel Verify & Benchmark ({}) ===",
        info.kernel_arch, info.device_name
    );
    println!("Embedded HSACO: {} bytes", info.hsaco_bytes);
    let kernels = hip.load_core_kernels(dev).expect("load embedded .hsaco kernels");

    // --- 1. Numerical Parity Checks (GPU vs CPU Reference) ---
    let k_cols = 2304usize;
    let n_rows = 256usize;
    let x: Vec<f32> = (0..k_cols).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    let w_norm: Vec<f32> = (0..k_cols).map(|i| 0.8 + ((i as f32) * 0.07).cos() * 0.2).collect();

    let cpu_rms = cpu::rmsnorm_f32(&x, &w_norm, 1e-6);
    let gpu_rms = hip.run_rmsnorm_f32(&kernels, &x, &w_norm, 1e-6).expect("gpu rmsnorm");
    let err_rms = max_abs_diff(&cpu_rms, &gpu_rms);
    println!("[Parity] RMSNorm (2304)             max_abs_diff = {:.2e}", err_rms);
    assert!(err_rms < 1e-5, "RMSNorm parity failed: {err_rms}");

    // FP8 E4M3 GEMV parity (avoid 0x7F/0xFF NaN encodings)
    let w_fp8: Vec<u8> = (0..(n_rows * k_cols))
        .map(|i| {
            let b = ((i.wrapping_mul(37) ^ (i >> 3)) & 0xFF) as u8;
            if (b & 0x7F) == 0x7F { 0x38 } else { b }
        })
        .collect();
    let row_scale: Vec<f32> = (0..n_rows).map(|r| 0.01 + (r as f32) * 0.0002).collect();
    let cpu_fp8 = cpu::gemv_fp8_e4m3_f32(&x, &w_fp8, &row_scale, n_rows, k_cols);
    let gpu_fp8 = hip
        .run_gemv_fp8_e4m3_f32(&kernels, &x, &w_fp8, &row_scale, n_rows, k_cols)
        .expect("gpu fp8");
    let err_fp8 = max_abs_diff(&cpu_fp8, &gpu_fp8);
    println!("[Parity] FP8 E4M3 GEMV [256x2304]   max_abs_diff = {:.2e}", err_fp8);
    assert!(err_fp8 < 1e-3, "FP8 parity failed: {err_fp8}");

    // NVFP4 GEMV parity
    let w_nvfp4: Vec<u8> = (0..(n_rows * (k_cols / 2)))
        .map(|i| ((i.wrapping_mul(53) ^ (i >> 2)) & 0xFF) as u8)
        .collect();
    let s_nvfp4: Vec<u8> = (0..(n_rows * (k_cols / 16)))
        .map(|i| 0x28 + ((i % 24) as u8)) // valid positive FP8 E4M3 scales
        .collect();
    let cpu_nvfp4 = cpu::gemv_nvfp4_f32(&x, &w_nvfp4, &s_nvfp4, 0.25, n_rows, k_cols);
    let gpu_nvfp4 = hip
        .run_gemv_nvfp4_f32(&kernels, &x, &w_nvfp4, &s_nvfp4, 0.25, n_rows, k_cols)
        .expect("gpu nvfp4");
    let err_nvfp4 = max_abs_diff(&cpu_nvfp4, &gpu_nvfp4);
    println!("[Parity] NVFP4 GEMV    [256x2304]   max_abs_diff = {:.2e}", err_nvfp4);
    assert!(err_nvfp4 < 1e-3, "NVFP4 parity failed: {err_nvfp4}");

    // MXFP8 GEMV parity (verified on-disk [N, K/32] UE8M0 scales)
    let s_mxfp8: Vec<u8> = (0..(n_rows * (k_cols / 32)))
        .map(|i| 120u8 + ((i % 12) as u8)) // UE8M0 scales around 2^-7..2^4
        .collect();
    let cpu_mxfp8 = cpu::gemv_mxfp8_f32(&x, &w_fp8, &s_mxfp8, n_rows, k_cols);
    let gpu_mxfp8 = hip
        .run_gemv_mxfp8_f32(&kernels, &x, &w_fp8, &s_mxfp8, n_rows, k_cols)
        .expect("gpu mxfp8");
    let err_mxfp8 = max_abs_diff(&cpu_mxfp8, &gpu_mxfp8);
    println!("[Parity] MXFP8 GEMV    [256x2304]   max_abs_diff = {:.2e}", err_mxfp8);
    assert!(err_mxfp8 < 1e-2, "MXFP8 parity failed: {err_mxfp8}");

    // --- 2. Windows WDDM Launch Latency: Raw vs HIP Graph (240 kernels = 48 layers x 5 ops) ---
    let g = hip.bench_graph_launch(&kernels).expect("graph bench");
    println!(
        "[WDDM Launch, {} ops] Raw: {:.3} ms | HIP Graph: {:.3} ms ({:.2}x faster)",
        g.kernels_per_iter, g.raw_ms_per_iter, g.graph_ms_per_iter, g.speedup
    );

    // --- 3. Decode Kernel Effective GDDR6 Bandwidth + Async DMA Overlap ---
    let (bw_results, ov) = hip.bench_decode_kernels(&kernels).expect("bw bench");
    for r in bw_results {
        let pct_peak = (r.effective_gb_s / 640.0) * 100.0;
        println!(
            "[GDDR6 BW] {:<45} : {:6.3} ms | {:6.1} GB/s ({:.1}% of 640 GB/s)",
            r.label, r.avg_ms, r.effective_gb_s, pct_peak
        );
    }
    println!(
        "[Async Overlap] Compute: {:.3} ms | DMA (2x NVFP4): {:.3} ms | Concurrent: {:.3} ms (overlap eff: {:.1}%)",
        ov.compute_only_ms, ov.dma_only_ms, ov.overlapped_ms, ov.overlap_efficiency_pct
    );

    // --- 4. Indirect Top-8 MoE SwiGLU Parity & Latency (NVFP4 + MXFP8 + --no-rescue skip!) ---
    let cfg = Inkling2Config::default();
    for (label, dtype, skip_tail) in [
        ("Indirect Top-8 NVFP4 (8/8 hits)", DType::NVFP4, false),
        ("Indirect Top-8 NVFP4 (6/8 hits, 2 skipped+renorm)", DType::NVFP4, true),
        ("Indirect Top-8 MXFP8 (8/8 hits)", DType::MXFP8, false),
    ] {
        let layout = ExpertSlabLayout::for_inkling2(&cfg, dtype);
        let n_slots = 8usize;
        let mut pool = vec![0u8; n_slots * layout.aligned_slab_bytes];
        for s in 0..n_slots {
            let base = s * layout.aligned_slab_bytes;
            let hdr = ExpertSlabHeader {
                w13_outer_scale: 0.125,
                w2_outer_scale: 0.125,
                w13_weight_offset: layout.w13_weight_offset as u32,
                w13_scale_offset: layout.w13_scale_offset as u32,
                w2_weight_offset: layout.w2_weight_offset as u32,
                w2_scale_offset: layout.w2_scale_offset as u32,
                dtype_tag: if dtype == DType::NVFP4 { 0 } else { 1 },
                _reserved: [0; 9],
            };
            let hdr_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    (&hdr as *const ExpertSlabHeader) as *const u8,
                    std::mem::size_of::<ExpertSlabHeader>(),
                )
            };
            pool[base..base + 64].copy_from_slice(hdr_bytes);
            // Fill deterministic weights/scales (avoid FP8 NaN 0x7F/0xFF)
            for i in 0..layout.w13_weight_bytes {
                let mut b = ((i.wrapping_mul(31) ^ (s * 17)) & 0xFF) as u8;
                if dtype == DType::MXFP8 && (b & 0x7F) == 0x7F { b = 0x38; }
                pool[base + layout.w13_weight_offset + i] = b;
            }
            for i in 0..layout.w13_scale_bytes {
                pool[base + layout.w13_scale_offset + i] =
                    if dtype == DType::NVFP4 { 0x2C + ((i + s) % 12) as u8 } else { 121 + ((i + s) % 6) as u8 };
            }
            for i in 0..layout.w2_weight_bytes {
                let mut b = ((i.wrapping_mul(43) ^ (s * 29)) & 0xFF) as u8;
                if dtype == DType::MXFP8 && (b & 0x7F) == 0x7F { b = 0x38; }
                pool[base + layout.w2_weight_offset + i] = b;
            }
            for i in 0..layout.w2_scale_bytes {
                pool[base + layout.w2_scale_offset + i] =
                    if dtype == DType::NVFP4 { 0x2C + ((i + s) % 12) as u8 } else { 121 + ((i + s) % 6) as u8 };
            }
        }

        let mut slot_table = vec![u32::MAX; cfg.n_routed_experts];
        let topk_ids = [5u32, 17, 33, 49, 64, 81, 102, 120];
        let topk_weights = [0.22f32, 0.18, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05];
        for (i, &id) in topk_ids.iter().enumerate() {
            if skip_tail && i >= 6 {
                slot_table[id as usize] = u32::MAX; // Simulate --no-rescue miss!
            } else {
                slot_table[id as usize] = i as u32;
            }
        }

        let cpu_moe = cpu::moe_swiglu_indirect_ref(
            &x,
            &pool,
            &slot_table,
            &topk_ids,
            &topk_weights,
            layout.aligned_slab_bytes,
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.mlp_gate_softcap,
            cfg.mlp_up_softcap,
        );
        let (gpu_moe, moe_ms) = hip
            .run_and_bench_moe_indirect(
                &kernels,
                &x,
                &pool,
                &slot_table,
                &topk_ids,
                &topk_weights,
                layout.aligned_slab_bytes,
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.mlp_gate_softcap,
                cfg.mlp_up_softcap,
                100,
            )
            .expect("gpu moe");
        let err = max_abs_diff(&cpu_moe, &gpu_moe);
        println!(
            "[MoE Parity+Perf] {:<48} : {:6.3} ms/layer | max_abs_diff = {:.2e}",
            label, moe_ms, err
        );
        assert!(err < 1e-2, "MoE parity failed for {label}: {err}");
    }

    // --- 5. FP8 KV Split-K FlashDecoding Benchmark (1K Local + 32K + 256K Global!) ---
    for (label, seq_len, splits) in [
        ("Local 1K  FP8 KV Attn (1 layer)", 1_024usize, 4u32),
        ("Global 32K FP8 KV Attn (1 layer)", 32_768usize, 32u32),
        ("Global 256K FP8 KV Attn (1 layer)", 262_144usize, 64u32),
    ] {
        let (ms, gbs) = hip
            .bench_flash_decode_fp8(&kernels, seq_len, splits, 20)
            .expect("flash decode bench");
        println!(
            "[FP8 Attn] {:<35} : {:6.3} ms/layer | {:6.1} GB/s ({:.1}% of 640 GB/s)",
            label,
            ms,
            gbs,
            (gbs / 640.0) * 100.0
        );
    }

    hip.unload_kernels(kernels);
}

fn run_inspect(path: &Path, filter: Option<&str>) {
    match inspect_safetensors_header(path) {
        Ok(hdr) => {
            println!(
                "Safetensors header: {} bytes, {} tensors",
                hdr.header_len,
                hdr.tensors.len()
            );
            let mut shown = 0usize;
            for (name, entry) in &hdr.tensors {
                if let Some(f) = filter {
                    if !name.contains(f) {
                        continue;
                    }
                } else if shown >= 40 {
                    break;
                }
                let bytes = entry.data_offsets[1].saturating_sub(entry.data_offsets[0]);
                println!(
                    "  {:<62} {:<8} {:?} ({} bytes)",
                    name, entry.dtype, entry.shape, bytes
                );
                shown += 1;
            }
            if filter.is_none() && hdr.tensors.len() > shown {
                println!("  ... and {} more tensors (pass a substring filter to search)", hdr.tensors.len() - shown);
            }
        }
        Err(e) => eprintln!("Failed to inspect {}: {e}", path.display()),
    }
}

fn run_bench_nvme() {
    let temp_path = env::temp_dir().join("smalltide_nvme_bench.tmp");
    let file_size_bytes: usize = 256 * 1024 * 1024;
    println!(
        "Creating 256 MiB aligned benchmark file at {} ...",
        temp_path.display()
    );
    {
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&temp_path)
            .expect("create temp bench file");
        let block = vec![0xA5u8; 4 * 1024 * 1024];
        for _ in 0..(file_size_bytes / block.len()) {
            f.write_all(&block).expect("write bench block");
        }
        f.sync_all().expect("sync bench file");
    }

    let df = DirectFile::open_read(&temp_path).expect("open NO_BUFFERING file");
    for (label, raw_bytes) in [
        ("NVFP4 expert (4.27 MiB aligned)", 4_478_976usize),
        ("MXFP8 expert (7.83 MiB aligned)", 8_211_456usize),
    ] {
        let read_len = align_up(raw_bytes, SECTOR_ALIGN);
        let mut buf = AlignedBuffer::alloc(read_len).expect("alloc aligned buf");
        let max_slots = file_size_bytes / read_len;
        let iterations = 64usize;

        let start = Instant::now();
        let mut state: usize = 0x9E37_79B9;
        for _ in 0..iterations {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let slot = (state >> 16) % max_slots;
            let offset = (slot * read_len) as u64;
            let got = df
                .read_aligned_at(offset, &mut buf, read_len)
                .expect("direct read");
            assert_eq!(got, read_len);
        }
        let secs = start.elapsed().as_secs_f64();
        let avg_ms = (secs * 1000.0) / (iterations as f64);
        let gb_s = (read_len as f64 * iterations as f64 / 1e9) / secs;
        println!("NVMe Direct [{}]: {:.3} ms/expert ({:.2} GB/s)", label, avg_ms, gb_s);
    }

    let _ = std::fs::remove_file(&temp_path);
}

/// Allocates VRAM in 512 MiB chunks and measures read bandwidth of each, revealing where the
/// driver silently backs "device" allocations with system memory (WDDM over-commit).
fn run_vram_probe() {
    let backend = HipBackend::new_auto().expect("init HipBackend");
    let s = backend.create_stream(smalltide_core::QueueKind::Compute).expect("stream");
    let chunk = 512usize << 20;
    let scratch = backend.alloc_device(chunk).expect("scratch");
    let mut bufs = Vec::new();
    #[cfg(windows)]
    let _wddm = {
        let a = smalltide_backend_hip::wddm::Adapter::open_amd();
        if let Some(a) = &a {
            let gib = (1u64 << 30) as f64;
            if let Some(i) = a.local_info() {
                println!("DXGI local: budget {:.2} GiB, usage {:.2}, available-for-reservation {:.2}",
                    i.budget as f64 / gib, i.current_usage as f64 / gib, i.available_for_reservation as f64 / gib);
                let want = std::env::var("SMALLTIDE_RESERVE_GB").ok().and_then(|v| v.parse::<f64>().ok())
                    .map(|g| (g * gib) as u64).unwrap_or(i.available_for_reservation);
                if want > 0 {
                    let ok = a.reserve_local(want);
                    println!("SetVideoMemoryReservation({:.2} GiB) -> {}", want as f64 / gib, if ok { "ok" } else { "failed" });
                }
                if let Some(i) = a.local_info() {
                    println!("DXGI local after: reserved {:.2} GiB", i.current_reservation as f64 / gib);
                }
            }
        }
        a
    };
    let info0 = backend.device_info().expect("info");
    println!("free before: {:.2} GiB", info0.free_vram_bytes as f64 / (1u64 << 30) as f64);
    if let Some(hm_gb) = std::env::var("SMALLTIDE_PROBE_HOSTMAPPED_GB").ok().and_then(|v| v.parse::<usize>().ok()) {
        let (dbuf, pbuf) = backend.alloc_host_mapped(hm_gb << 30).expect("host-mapped alloc");
        backend.zero_f32(dbuf, (hm_gb << 30) / 4, s).expect("zero");
        backend.sync_stream(s).expect("sync");
        for w in 0..(hm_gb * 2) {
            let t0 = Instant::now();
            for _ in 0..4 {
                backend.vec_copy(scratch, dbuf.slice(w * chunk, chunk), chunk / 4, s).expect("copy");
            }
            backend.sync_stream(s).expect("sync");
            let gbs = (4 * chunk) as f64 / t0.elapsed().as_secs_f64() / 1e9;
            println!("host-mapped window {w:2}: read {gbs:7.1} GB/s");
        }
        let _ = backend.free_device(dbuf);
        let _ = backend.free_pinned(pbuf);
        return;
    }
    if let Some(big_gb) = std::env::var("SMALLTIDE_PROBE_BIG_GB").ok().and_then(|v| v.parse::<usize>().ok()) {
        let big = backend.alloc_device(big_gb << 30).expect("big alloc");
        backend.zero_f32(big, (big_gb << 30) / 4, s).expect("zero");
        backend.sync_stream(s).expect("sync");
        for w in 0..(big_gb * 2) {
            let t0 = Instant::now();
            for _ in 0..4 {
                backend.vec_copy(scratch, big.slice(w * chunk, chunk), chunk / 4, s).expect("copy");
            }
            backend.sync_stream(s).expect("sync");
            let gbs = (4 * chunk) as f64 / t0.elapsed().as_secs_f64() / 1e9;
            println!("big alloc window {w:2} ({:5.1} GiB offset): read {gbs:7.1} GB/s", w as f64 * 0.5);
        }
        let _ = backend.free_device(big);
        return;
    }
    #[cfg(windows)]
    let d3d = if std::env::var_os("SMALLTIDE_PROBE_D3D12").is_some() {
        let dev = _wddm.as_ref().and_then(|a| smalltide_backend_hip::d3d12::D3d12Device::create(a.raw()));
        println!("D3D12 device: {}", if dev.is_some() { "ok" } else { "FAILED" });
        dev
    } else {
        None
    };
    #[cfg(windows)]
    let mut d3d_bufs = Vec::new();
    let n_chunks = std::env::var("SMALLTIDE_PROBE_CHUNKS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(30);
    for i in 0..n_chunks {
        #[cfg(windows)]
        let b = if let Some(dev) = &d3d {
            match dev.alloc_shared_buffer(chunk as u64).map_err(|e| e).and_then(|db| {
                let r = backend.import_d3d12_buffer(&db).map_err(|e| format!("{e}"));
                d3d_bufs.push(db);
                r
            }) {
                Ok(b) => b,
                Err(e) => { println!("d3d12 alloc failed at chunk {i}: {e}"); break }
            }
        } else {
            let Ok(b) = backend.alloc_device(chunk) else { println!("alloc failed at chunk {i}"); break };
            b
        };
        #[cfg(not(windows))]
        let Ok(b) = backend.alloc_device(chunk) else { println!("alloc failed at chunk {i}"); break };
        let _ = backend.zero_f32(b, chunk / 4, s);
        let _ = backend.sync_stream(s);
        let t0 = Instant::now();
        for _ in 0..4 {
            backend.vec_copy(scratch, b, chunk / 4, s).expect("copy");
        }
        backend.sync_stream(s).expect("sync");
        let gbs = (4 * chunk) as f64 / t0.elapsed().as_secs_f64() / 1e9;
        let free = backend.device_info().map(|d| d.free_vram_bytes as f64 / (1u64 << 30) as f64).unwrap_or(0.0);
        println!("chunk {i:2} ({:5.1} GiB allocated): read {gbs:7.1} GB/s | reported free {free:.2} GiB", (i + 1) as f64 * 0.5);
        bufs.push(b);
    }
    if std::env::var_os("SMALLTIDE_PROBE_REREAD").is_some() {
        for (i, b) in bufs.iter().enumerate() {
            let t0 = Instant::now();
            for _ in 0..4 {
                backend.vec_copy(scratch, *b, chunk / 4, s).expect("copy");
            }
            backend.sync_stream(s).expect("sync");
            let gbs = (4 * chunk) as f64 / t0.elapsed().as_secs_f64() / 1e9;
            println!("reread chunk {i:2}: {gbs:7.1} GB/s");
        }
    }
    if let Some(secs) = std::env::var("SMALLTIDE_PROBE_HOLD").ok().and_then(|v| v.parse::<u64>().ok()) {
        println!("holding allocations for {secs}s (pid {})", std::process::id());
        std::thread::sleep(std::time::Duration::from_secs(secs));
    }
    for b in bufs {
        let _ = backend.free_device(b);
    }
}

fn run_load_test(model_dir: &Path) {
    let cfg = Inkling2Config::default();
    println!("=== SmallTide Step 2 Live Model Load Test ({}) ===", model_dir.display());

    // 1. Tokenizer check
    let t_tok = Instant::now();
    let tok_path = model_dir.join("tokenizer.json");
    let tok = BpeTokenizer::from_file(&tok_path).expect("load tokenizer.json");
    let sample = "<|message_user|>Hello, Inkling-2-Mini! Explain TideLM in one sentence.<|message_model|>";
    let ids = tok.encode(sample);
    let roundtrip = tok.decode(&ids);
    println!(
        "[Tokenizer] Loaded {} tokens in {:.2}s | encode={} ids | roundtrip_exact={}",
        tok.vocab_size(),
        t_tok.elapsed().as_secs_f64(),
        ids.len(),
        roundtrip == sample
    );
    assert_eq!(roundtrip, sample, "Tokenizer roundtrip mismatch: {roundtrip:?}");

    // 2. Dense --fp8-dense loader
    let t_dense = Instant::now();
    let sf_path = model_dir.join("model-00001-of-00001.safetensors");
    let dense = Inkling2DenseHost::load_from_safetensors(&cfg, &sf_path)
        .expect("load Inkling-2-Mini dense weights");
    let dense_vram_gib = dense.total_vram_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
    let embed_ram_mib = dense.embed_bf16.len() as f64 / (1024.0 * 1024.0);
    println!(
        "[Dense FP8] Loaded all 24 layers + lm_head in {:.2}s | Dense VRAM = {:.2} GiB | CPU embed = {:.1} MiB",
        t_dense.elapsed().as_secs_f64(),
        dense_vram_gib,
        embed_ram_mib
    );

    // 3. Upload FP8 dense weights to RX 9070 XT VRAM + run TideExecutor Mode 1 Pre-Prefill from experts.stide!
    let t_tide = Instant::now();
    let stide_path = model_dir.join("experts.stide");
    let pack = StidePackReader::open(&stide_path).expect("open experts.stide");
    let backend = Arc::new(HipBackend::new_auto().expect("init HipBackend"));
    let info_before = backend.device_info().expect("device_info before");

    let _dense_gpu = Inkling2DenseGpu::upload(backend.clone(), dense)
        .expect("upload FP8 dense weights to VRAM");
    let info_after_dense = backend.device_info().expect("device_info after dense");
    println!(
        "[Dense GPU] Uploaded in {:.2}s | VRAM free after dense: {:.2} GiB / {:.2} GiB",
        t_tide.elapsed().as_secs_f64(),
        info_after_dense.free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        info_after_dense.total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    // Allocate VRAM expert pool: 1,536 Loop Core + 600 outer NVFP4 (~9.13 GiB) + 128 outer MXFP8 (~1.00 GiB) = ~10.13 GiB!
    // Leaves ~4.5 GiB VRAM free for 256K FP8 KV Cache + dspark + WDDM headroom!
    let t_preload = Instant::now();
    let mut tide_cfg = TideConfig::default();
    tide_cfg.pinned_staging_mib = 256;
    let ram_budget = 10 * 1024 * 1024 * 1024usize;
    let mut exec = TideExecutor::new(
        backend.clone(),
        cfg,
        tide_cfg,
        2_136, // NVFP4 resident slots (~9.13 GiB)
        128,   // MXFP8 resident slots (~1.00 GiB)
        ram_budget,
        Some(pack),
    )
    .expect("create TideExecutor");

    let ops = exec.scheduler.plan_pre_prefill(None);
    exec.execute_transfers(&ops, true, false)
        .expect("execute Mode 1 pre-prefill transfers");
    backend.sync_stream(exec.copy_stream).expect("sync copy stream");

    let info_after = backend.device_info().expect("device_info after");
    let (pre, loop_c, coda) = exec.scheduler.cache.count_vram_resident();
    let ram_cnt = exec.scheduler.cache.count_ram_resident();
    println!(
        "[Tide Mode 1] Pre-prefill {} transfers in {:.2}s | VRAM resident: prelude={}, loop_core={}/1536, coda={} | RAM warm: {} experts ({:.2} GiB)",
        ops.len(),
        t_preload.elapsed().as_secs_f64(),
        pre,
        loop_c,
        coda,
        ram_cnt,
        exec.scheduler.cache.ram_used_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    println!(
        "[RX 9070 XT VRAM] Start: {:.2} GiB free -> After Dense+Tide: {:.2} GiB free (total used {:.2} GiB)",
        info_before.free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        info_after.free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        (info_before.free_vram_bytes.saturating_sub(info_after.free_vram_bytes)) as f64 / (1024.0 * 1024.0 * 1024.0),
    );
}

/// Engine/server configuration lives in `.env` (see `.env.headless.example` / `.env.monitor.example`); only per-request inputs
/// are flags. Old configuration flags are rejected with a pointer to the variable.
fn moved_to_env(flag: &str) -> Option<&'static str> {
    Some(match flag {
        "--model" => "SMALLTIDE_MODEL_DIR",
        "--max-ctx" => "SMALLTIDE_MAX_CTX",
        "--rescue-mode" => "SMALLTIDE_RESCUE_MODE",
        "--rescue-threshold" => "SMALLTIDE_RESCUE_THRESHOLD",
        "--nvfp4-slots" => "SMALLTIDE_NVFP4_SLOTS",
        "--mxfp8-slots" => "SMALLTIDE_MXFP8_SLOTS",
        "--mxfp8-host" => "SMALLTIDE_MXFP8_HOST",
        "--ram-gb" => "SMALLTIDE_RAM_GB",
        "--prefill" => "SMALLTIDE_PREFILL",
        "--profile" => "SMALLTIDE_PROFILE",
        "--host" => "SMALLTIDE_HOST",
        "--port" => "SMALLTIDE_PORT",
        _ => return None,
    })
}

pub fn reject_config_flag(flag: &str) -> ! {
    match moved_to_env(flag) {
        Some(var) => eprintln!("{flag} is configured in your .env file now: set {var}=... (see .env.headless.example / .env.monitor.example)"),
        None => eprintln!("unknown flag: {flag}"),
    }
    std::process::exit(2);
}

fn run_generate(args: &[String]) {
    let mut prompt = "Explain in two concise sentences what makes Mixture-of-Experts language models fast.".to_string();
    let mut raw_prompt = false;
    let mut effort = "0".to_string();
    let mut max_tokens: usize = 128;
    let mut temperature: f32 = 0.0;
    // Top-1 agreement test (see the `agree_prompts` block below and bench/agreement.py).
    let (mut agree_prompts, mut agree_ref, mut agree_out): (Option<String>, Option<String>, Option<String>) = (None, None, None);
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" | "-p" if i + 1 < args.len() => { prompt = args[i + 1].clone(); i += 2; }
            "--agree-prompts" if i + 1 < args.len() => { agree_prompts = Some(args[i + 1].clone()); i += 2; }
            "--agree-ref" if i + 1 < args.len() => { agree_ref = Some(args[i + 1].clone()); i += 2; }
            "--agree-out" if i + 1 < args.len() => { agree_out = Some(args[i + 1].clone()); i += 2; }
            // A prompt too long for a command line (Windows: 32,767 characters).
            "--prompt-file" if i + 1 < args.len() => {
                prompt = std::fs::read_to_string(&args[i + 1]).unwrap_or_else(|e| { eprintln!("--prompt-file {}: {e}", args[i + 1]); std::process::exit(2) });
                i += 2;
            }
            "--raw" => { raw_prompt = true; i += 1; }
            "--effort" if i + 1 < args.len() => { effort = args[i + 1].clone(); i += 2; }
            "--max-tokens" | "-n" if i + 1 < args.len() => {
                max_tokens = args[i + 1].parse().unwrap_or_else(|_| { eprintln!("--max-tokens needs a number"); std::process::exit(2) });
                i += 2;
            }
            "--temp" if i + 1 < args.len() => {
                temperature = args[i + 1].parse().unwrap_or_else(|_| { eprintln!("--temp needs a number"); std::process::exit(2) });
                i += 2;
            }
            other => reject_config_flag(other),
        }
    }

    let s = settings::Settings::from_env();
    let max_ctx = s.max_ctx;
    let enable_profile = s.profile;
    let prefill_mode = if s.batched_prefill { "batched" } else { "token" };
    let settings::Loaded { mut engine, tok, backend, nvfp4_slots, mxfp8_slots } = settings::build_engine(&s, if std::env::var_os("SMALLTIDE_BATCH_SELFTEST").is_some() {
        3
    } else {
        std::env::var("SMALLTIDE_BATCH_BENCH").ok().and_then(|v| v.parse::<usize>().ok()).map_or(1, |m| m + 1)
    });

    // `run` is for dev / benchmarks: it does not teach the routing priors (repeated test prompts
    // would skew them) unless SMALLTIDE_PRIORS_UPDATE is set explicitly. `serve` learns from traffic.
    if std::env::var_os("SMALLTIDE_PRIORS_UPDATE").is_none() {
        if let Some(st) = engine.tide.scheduler.priors_store.as_mut() {
            st.update = false;
        }
    }

    if std::env::var_os("SMALLTIDE_BENCH_MOE").is_some() {
        let n4 = nvfp4_slots as u32;
        let m8 = mxfp8_slots as u32;
        let mut sets: Vec<(String, Vec<u32>, bool)> = Vec::new();
        sets.push(("all 8 missing (launch floor)".to_string(), vec![u32::MAX; 8], false));
        for stride in [1u32, 2, 4, 8, 16, 64, 128] {
            if 7 * stride < n4 {
                sets.push((format!("nvfp4 8 slots stride {stride}"), (0..8).map(|i| i * stride).collect(), false));
            }
        }
        for st in (0..m8.saturating_sub(8)).step_by(160) {
            sets.push((format!("mxfp8 slots {st}..{}", st + 8), (st..st + 8).collect(), true));
        }
        for (label, slots, mx) in sets {
            let cold = engine.debug_bench_moe(&slots, mx, 10, true).expect("bench");
            let warm = engine.debug_bench_moe(&slots, mx, 10, false).expect("bench");
            println!("[bench-moe] {label:<24} cold {cold:.3} ms | warm {warm:.3} ms");
        }
    }

    if let Some(pfile) = agree_prompts.as_ref() {
        // Top-1 agreement, in two runs over the prompts of `--agree-prompts` (a JSON list of
        // {"name", "prompt"}), greedy, up to `-n` tokens each:
        //   1. reference (no --agree-ref; run it with SMALLTIDE_RESCUE_MODE=exact SMALLTIDE_SPEC=0,
        //      so every routed expert is loaded and run): writes each prompt's tokens to --agree-out;
        //   2. this configuration (--agree-ref <that file>): the same prompts, fed the reference's
        //      tokens one position at a time (teacher forcing), counting the positions where its
        //      own most likely next token is the reference's. Once through draft-verify passes
        //      (the path speculative decoding uses) and once through single-token steps.
        // Teacher forcing keeps every position comparable: a free-running comparison says
        // nothing after the first difference.
        const EOS_ID: u32 = 200_006;
        const EOT_ID: u32 = 199_999;
        let read_json = |path: &str| -> serde_json::Value {
            let text = std::fs::read_to_string(path).unwrap_or_else(|e| { eprintln!("{path}: {e}"); std::process::exit(2) });
            serde_json::from_str(&text).unwrap_or_else(|e| { eprintln!("{path}: {e}"); std::process::exit(2) })
        };
        let prompts = read_json(pfile);
        let refs_in = agree_ref.as_ref().map(|f| read_json(f));
        let out_path = agree_out.clone().unwrap_or_else(|| { eprintln!("--agree-prompts needs --agree-out <file>"); std::process::exit(2) });
        let width = std::env::var("SMALLTIDE_AGREE_WIDTH").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4).clamp(1, 8);
        let sampling = SamplingParams { temperature: 0.0, ..SamplingParams::default() };
        let skipped = |e: &smalltide_model_inkling2::engine::Inkling2Engine<HipBackend>| {
            let t = &e.tide.scheduler.telemetry;
            (t.routed_weight_skipped, t.routed_weight_total)
        };
        let share = |a: (f64, f64), b: (f64, f64)| if b.1 > a.1 { (b.0 - a.0) / (b.1 - a.1) } else { 0.0 };
        let mut results: Vec<serde_json::Value> = Vec::new();
        for (pi, p) in prompts.as_array().map(|a| a.as_slice()).unwrap_or(&[]).iter().enumerate() {
            let name = p["name"].as_str().unwrap_or("?").to_string();
            let text = p["prompt"].as_str().unwrap_or("");
            let ids = tok.encode(&format!(
                "<|message_system|><|content_text|>Thinking effort level: {effort}<|end_message|><|message_user|><|content_text|>{text}<|end_message|>"
            ));
            let p0 = ids.len();
            engine.reset_sequence().expect("reset");
            let t0 = Instant::now();
            engine.prefill(&ids, 0, true).expect("prefill");
            let _ = engine.rebalance_after_prefill(128).expect("rebalance");
            let first = engine.sample_next_token(&sampling, &[]).expect("sample");
            let prefill_s = t0.elapsed().as_secs_f64();
            match refs_in.as_ref() {
                None => {
                    let mut toks = vec![first];
                    let t1 = Instant::now();
                    while toks.len() <= max_tokens {
                        let t = *toks.last().unwrap();
                        if t == EOS_ID || t == EOT_ID {
                            break;
                        }
                        engine.forward_token(t, p0 + toks.len() - 1, false, true).expect("decode");
                        engine.post_decode_tide_step().expect("post");
                        toks.push(engine.sample_next_token(&sampling, &[]).expect("sample"));
                    }
                    println!("[agree] reference {name}: prompt {p0} tokens, {} tokens in {:.1} s", toks.len(), t1.elapsed().as_secs_f64());
                    results.push(serde_json::json!({ "name": name, "prompt_tokens": p0, "tokens": toks }));
                }
                Some(r) => {
                    let r = &r[pi];
                    let refs: Vec<u32> = r["tokens"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as u32).collect()).unwrap_or_default();
                    if r["name"].as_str() != Some(name.as_str()) || r["prompt_tokens"].as_u64() != Some(p0 as u64) || refs.is_empty() {
                        eprintln!("[agree] {name}: the reference file does not match this prompt");
                        std::process::exit(2);
                    }
                    let n = refs.len() - 1; // positions with a reference next token after the first
                    let mut snap = None;
                    engine.snapshot_state(p0, &mut snap).expect("snapshot");
                    // Draft-verify passes: `width` reference tokens per pass.
                    let s0 = skipped(&engine);
                    let t1 = Instant::now();
                    let (mut v_agree, mut v_first_miss, mut i) = (0usize, None::<usize>, 0usize);
                    while i < n {
                        let w = width.min(n - i);
                        engine.verify(&refs[i..i + w], p0 + i).expect("verify");
                        let am = engine.rows_argmax(w).expect("argmax");
                        for c in 0..w {
                            if am[c] == refs[i + c + 1] {
                                v_agree += 1;
                            } else if v_first_miss.is_none() {
                                v_first_miss = Some(i + c + 1);
                            }
                        }
                        engine.verify_rewind(w, w).expect("rewind");
                        engine.post_batch_tide_step().expect("post batch");
                        i += w;
                    }
                    let (v_secs, s1) = (t1.elapsed().as_secs_f64(), skipped(&engine));
                    // Single-token steps, from the same state after the prompt.
                    engine.restore_state(snap.as_ref().unwrap(), false).expect("restore");
                    let t2 = Instant::now();
                    let (mut s_agree, mut s_first_miss) = (0usize, None::<usize>);
                    for i in 0..n {
                        engine.forward_token(refs[i], p0 + i, false, true).expect("decode");
                        engine.post_decode_tide_step().expect("post");
                        if engine.sample_next_token(&sampling, &[]).expect("sample") == refs[i + 1] {
                            s_agree += 1;
                        } else if s_first_miss.is_none() {
                            s_first_miss = Some(i + 1);
                        }
                    }
                    let (s_secs, s2) = (t2.elapsed().as_secs_f64(), skipped(&engine));
                    if let Some(sn) = snap.take() {
                        engine.free_snapshot(sn).expect("free snapshot");
                    }
                    println!(
                        "[agree] {name}: prompt {p0} tokens (first token {}), {n} positions | verify passes {:.2}% ({:.1} tok/s, skipped weight {:.2}%) | single steps {:.2}% ({:.1} tok/s, skipped weight {:.2}%)",
                        if first == refs[0] { "same" } else { "DIFFERENT" },
                        100.0 * v_agree as f64 / n.max(1) as f64, n as f64 / v_secs, 100.0 * share(s0, s1),
                        100.0 * s_agree as f64 / n.max(1) as f64, n as f64 / s_secs, 100.0 * share(s1, s2),
                    );
                    results.push(serde_json::json!({
                        "name": name, "prompt_tokens": p0, "positions": n, "prefill_s": prefill_s,
                        "first_token_same": first == refs[0],
                        "verify": { "agree": v_agree, "first_miss": v_first_miss, "tok_s": n as f64 / v_secs, "skipped_weight": share(s0, s1), "width": width },
                        "single": { "agree": s_agree, "first_miss": s_first_miss, "tok_s": n as f64 / s_secs, "skipped_weight": share(s1, s2) },
                    }));
                }
            }
        }
        std::fs::write(&out_path, serde_json::to_string_pretty(&serde_json::Value::Array(results)).unwrap()).unwrap_or_else(|e| { eprintln!("{out_path}: {e}"); std::process::exit(2) });
        println!("[agree] wrote {out_path}");
        return;
    }

    let formatted = if raw_prompt {
        prompt.clone()
    } else {
        format!(
            "<|message_system|><|content_text|>Thinking effort level: {effort}<|end_message|><|message_user|><|content_text|>{prompt}<|end_message|>"
        )
    };
    let prompt_ids = tok.encode(&formatted);
    println!("[Prompt] {} tokens", prompt_ids.len());

    // Prefill
    let pf_kt = std::env::var_os("SMALLTIDE_KTIMING_PREFILL").is_some();
    if pf_kt {
        backend.set_kernel_timing(true).expect("kernel timing");
        let _ = backend.take_kernel_times();
    }
    let pf_seg = std::env::var_os("SMALLTIDE_SEGTIME_PREFILL").is_some();
    if pf_seg {
        backend.set_segment_timing(true).expect("segment timing");
        let _ = backend.take_segment_times();
    }
    let t_pf = Instant::now();
    if prefill_mode == "token" {
        for (idx, &tid) in prompt_ids.iter().enumerate() {
            let is_last = idx + 1 == prompt_ids.len();
            engine
                .forward_token(tid, idx, true, is_last)
                .expect("prefill forward_token");
        }
    } else {
        engine.prefill(&prompt_ids, 0, true).expect("batched prefill");
    }
    backend.sync_stream(engine.tide.compute_stream).expect("sync prefill");
    let pf_secs = t_pf.elapsed().as_secs_f64();
    if pf_kt {
        let mut kt = backend.take_kernel_times();
        kt.sort_by(|a, b| b.2.total_cmp(&a.2));
        let total: f64 = kt.iter().map(|k| k.2).sum();
        println!("[GPU kernels, prefill] total {total:.1} ms (wall {:.1} ms)", pf_secs * 1000.0);
        for (name, cnt, ms) in kt.iter().take(14) {
            println!("  {:<32} {:>6}x  {:>9.1} ms  ({:>7.1} us avg)", name, cnt, ms, ms * 1000.0 / *cnt as f64);
        }
        backend.set_kernel_timing(false).expect("kernel timing off");
    }
    if pf_seg {
        let st = backend.take_segment_times();
        let total: f64 = st.iter().map(|k| k.2).sum();
        println!("[GPU segments, prefill] total {total:.1} ms (wall {:.1} ms)", pf_secs * 1000.0);
        for (name, cnt, ms) in st {
            println!("  {:<32} {:>6}x  {:>9.1} ms  ({:>7.1} us avg)", name, cnt, ms, ms * 1000.0 / cnt.max(1) as f64);
        }
        backend.set_segment_timing(false).expect("segment timing off");
    }

    // Mode 2: Prefill -> Decode Boundary Rebalance
    let t_reb = Instant::now();
    let reb_swaps = engine.rebalance_after_prefill(std::env::var("SMALLTIDE_REBALANCE_SWAPS").ok().and_then(|v| v.parse().ok()).unwrap_or(128)).expect("Mode 2 rebalance");
    let reb_ms = t_reb.elapsed().as_secs_f64() * 1000.0;

    let sampling = SamplingParams {
        temperature,
        ..SamplingParams::default()
    };
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut pos = prompt_ids.len();

    print!("\n--- Output ---\n");
    let _ = std::io::stdout().flush();

    let t_dec = Instant::now();
    if enable_profile {
        engine.profile_enabled = true;
    }
    let _ = backend.take_host_stats();
    let mut post_ms = 0.0f64;
    let ktiming = std::env::var_os("SMALLTIDE_KTIMING").is_some();
    let segtiming = std::env::var_os("SMALLTIDE_SEGTIME").is_some();
    if segtiming {
        backend.set_segment_timing(true).expect("segment timing");
        let _ = backend.take_segment_times();
    }
    if ktiming {
        backend.set_kernel_timing(true).expect("kernel timing");
        let _ = backend.take_kernel_times();
    }
    let mut next_tok = engine
        .sample_next_token(&sampling, &generated)
        .expect("sample first token");

    if std::env::var_os("SMALLTIDE_KV_DEBUG").is_some() {
        println!("[kv-debug] pool {:?}", engine.kv_pool_info());
        for (p, (k, sc, v)) in engine.debug_kv_peek().expect("peek").into_iter().enumerate() {
            println!("[kv-debug] pos {p}: K {:?} scale {sc} V {:?}", k, v);
        }
    }

    if std::env::var_os("SMALLTIDE_WAITVALUE_TEST").is_some() {
        // Dev: does a GPU-side wait on a host-written flag work, and what does a release cost?
        let s = engine.tide.compute_stream;
        let (view, host) = backend.alloc_host_mapped(64).expect("mapped");
        let scratch = backend.alloc_device(1 << 20).expect("scratch");
        let flag = |v: u32| unsafe {
            let p = backend.pinned_as_mut_slice(host).unwrap().as_mut_ptr() as *mut u32;
            std::ptr::write_volatile(p, v);
        };
        flag(0);
        let ev = backend.create_event().expect("event");
        backend.zero_f32(scratch, 1 << 18, s).expect("k1");
        match backend.stream_wait_value_ge(s, view.slice(0, 4), 1) {
            Ok(()) => println!("[waitvalue] hipStreamWaitValue32 accepted"),
            Err(e) => { println!("[waitvalue] not available: {e}"); return; }
        }
        backend.zero_f32(scratch, 1 << 18, s).expect("k2");
        backend.record_event(ev, s).expect("rec");
        let _ = backend.query_event(ev); // submit
        std::thread::sleep(std::time::Duration::from_millis(50));
        let blocked = !backend.query_event(ev).expect("q");
        let t0 = Instant::now();
        flag(1);
        while !backend.query_event(ev).expect("q") { std::hint::spin_loop(); }
        println!("[waitvalue] GPU waited for the flag: {blocked}; release -> done {:.1} us", t0.elapsed().as_secs_f64() * 1e6);
        // Throughput: N waits queued up front, released one by one as the host sees progress.
        let n = 200u32;
        flag(0);
        let t0 = Instant::now();
        for i in 1..=n {
            backend.zero_f32(scratch, 1 << 12, s).expect("k");
            backend.stream_wait_value_ge(s, view.slice(0, 4), i).expect("wait");
        }
        backend.zero_f32(scratch, 1 << 12, s).expect("k");
        backend.record_event(ev, s).expect("rec");
        let _ = backend.query_event(ev);
        let t_enq = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        for i in 1..=n { flag(i); }
        while !backend.query_event(ev).expect("q") { std::hint::spin_loop(); }
        println!("[waitvalue] {n} waits: enqueue {t_enq:.2} ms, drain after releases {:.3} ms ({:.1} us per wait)",
            t1.elapsed().as_secs_f64() * 1e3, t1.elapsed().as_secs_f64() * 1e6 / n as f64);
        // Same chain without waits (reference).
        let t2 = Instant::now();
        for _ in 0..=n { backend.zero_f32(scratch, 1 << 12, s).expect("k"); }
        backend.sync_stream(s).expect("sync");
        println!("[waitvalue] {} kernels without waits: {:.3} ms", n + 1, t2.elapsed().as_secs_f64() * 1e3);
        return;
    }

    if let Some(iters) = std::env::var("SMALLTIDE_DRAFT_BENCH").ok().and_then(|v| v.parse::<usize>().ok()) {
        // Dev: the drafter alone (anchor = first sampled token at the prompt end), wall time per
        // draft for 1..7 drafts, then a per-kernel table for 7 drafts.
        let p0 = prompt_ids.len();
        let row = engine.last_tap_rows.max(1) - 1;
        for n in [1usize, 2, 3, 5, 7] {
            for _ in 0..3 { engine.draft(next_tok, p0, row, n).expect("draft"); }
            let t0 = Instant::now();
            for _ in 0..iters { engine.draft(next_tok, p0, row, n).expect("draft"); }
            println!("[draft-bench] {n} drafts: {:.3} ms per draft call", t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
        }
        backend.set_kernel_timing(true).expect("kt");
        let _ = backend.take_kernel_times();
        for _ in 0..iters { engine.draft(next_tok, p0, row, 7).expect("draft"); }
        let mut kt = backend.take_kernel_times();
        backend.set_kernel_timing(false).expect("kt off");
        kt.sort_by(|a, b| b.2.total_cmp(&a.2));
        for (name, cnt, ms) in kt.iter() {
            println!("    {:<30} {:>5.1}x {:>8.3} ms/call ({:.1} us avg)", name, *cnt as f64 / iters as f64, ms / iters as f64, ms * 1000.0 / *cnt as f64);
        }
        return;
    }

    if std::env::var_os("SMALLTIDE_ROWS_CMP").is_some() {
        // Dev: one token from the same state through forward_token, a 1-row batched round and a
        // 1-row verify; max |logit| differences.
        engine.tide.wait_and_publish_all_pending().expect("drain pending");
        let vocab = engine.cfg.unpadded_vocab_size;
        let p0 = prompt_ids.len();
        let read = |eng: &smalltide_model_inkling2::engine::Inkling2Engine<smalltide_backend_hip::HipBackend>, buf: smalltide_core::DeviceBuf| -> Vec<f32> {
            let mut b = vec![0u8; vocab * 4];
            eng.backend.memcpy_d2h_sync(&mut b, buf.slice(0, vocab * 4), 0, eng.tide.compute_stream).expect("d2h");
            b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        let mut snap = None;
        engine.snapshot_state(p0, &mut snap).expect("snapshot");
        let mut t = next_tok;
        for step in 0..4 {
            engine.forward_token(t, p0 + step, false, true).expect("single");
            let a = read(&engine, engine.logits_buf());
            engine.restore_state(snap.as_ref().unwrap(), false).expect("restore");
            for _ in 0..step { /* state at p0 only: compare the first step from the snapshot */ }
            engine.forward_batch(&[(engine.active_lane(), t, p0 + step)]).expect("batch");
            let b = read(&engine, engine.batch_logits_buf().unwrap());
            engine.post_batch_tide_step().expect("post");
            engine.restore_state(snap.as_ref().unwrap(), false).expect("restore");
            if let Ok(d) = std::env::var("SMALLTIDE_DUMP_DIR") {
                let v = std::path::PathBuf::from(d).join("verify");
                let _ = std::fs::create_dir_all(&v);
                engine.set_dump_dir(v);
            }
            engine.verify(&[t], p0 + step).expect("verify");
            let c = read(&engine, engine.batch_logits_buf().unwrap());
            engine.post_batch_tide_step().expect("post");
            engine.restore_state(snap.as_ref().unwrap(), false).expect("restore");
            let d = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
            let am = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0;
            if let Ok(dir) = std::env::var("SMALLTIDE_ROWS_CMP") {
                if dir.len() > 1 {
                    for (name, v) in [("single", &a), ("batch", &b), ("verify", &c)] {
                        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                        let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{name}.f32")), bytes);
                    }
                }
            }
            println!("[rows-cmp] token {t}: single-batch {:.4} single-verify {:.4} batch-verify {:.4} | argmax {} {} {}", d(&a, &b), d(&a, &c), d(&b, &c), am(&a), am(&b), am(&c));
            // Causality: rows must not depend on the drafts after them.
            for off in [0usize, 3, 7, 13] {
                let mut outs = Vec::new();
                for tail in [[11u32, 22, 33, 44, 55, 66, 77], [11u32, 22, 33, 999, 1234, 4321, 8888]] {
                    engine.restore_state(snap.as_ref().unwrap(), false).expect("restore");
                    // advance `off` positions with plain decode so pos0 % 16 varies
                    let mut tt = t;
                    for j in 0..off {
                        engine.forward_token(tt, p0 + j, false, true).expect("adv");
                        tt = 100 + j as u32;
                    }
                    let mut blk = vec![tt];
                    blk.extend_from_slice(&tail);
                    engine.verify(&blk, p0 + off).expect("verify");
                    let bl = engine.batch_logits_buf().unwrap();
                    let rows: Vec<Vec<f32>> = (0..8).map(|r| read(&engine, bl.slice(r * vocab * 4, vocab * 4))).collect();
                    engine.post_batch_tide_step().expect("post");
                    outs.push(rows);
                }
                let diffs: Vec<String> = (0..8).map(|r| format!("{:.3}", d(&outs[0][r], &outs[1][r]))).collect();
                println!("[rows-cmp] causality pos0={} (mod16 {}): row diffs (tails differ from row 3 on) {}", p0 + off, (p0 + off) % 16, diffs.join(" "));
            }
            engine.restore_state(snap.as_ref().unwrap(), false).expect("restore");
            t = am(&a) as u32;
            if step == 0 { break; }
        }
        return;
    }

    if let Some(n) = std::env::var("SMALLTIDE_VERIFY_SELFTEST").ok().and_then(|v| v.parse::<usize>().ok()) {
        // Speculative verify + rewind vs plain greedy decode: reference tokens from forward_token;
        // then the same span through 8-row verify steps whose drafts are the reference tokens with
        // one corrupted per block (forces a rejection and a short-conv rewind every step).
        let p0 = prompt_ids.len();
        // Same resident experts for both runs: land every queued swap first (run with
        // SMALLTIDE_ASYNC_PER_TOKEN=0 so no new ones are queued).
        engine.tide.wait_and_publish_all_pending().expect("drain pending");
        let mut snap = None;
        engine.snapshot_state(p0, &mut snap).expect("snapshot");
        let mut refs = vec![next_tok];
        let t_ref = Instant::now();
        for i in 0..n {
            engine.forward_token(refs[i], p0 + i, false, true).expect("decode");
            engine.post_decode_tide_step().expect("post");
            refs.push(engine.sample_next_token(&sampling, &[]).expect("sample"));
        }
        let ref_ms = t_ref.elapsed().as_secs_f64() * 1000.0 / n as f64;
        engine.restore_state(snap.as_ref().unwrap(), false).expect("restore");
        let vkt = std::env::var_os("SMALLTIDE_KTIMING").is_some();
        if vkt {
            backend.set_kernel_timing(true).expect("kt");
            let _ = backend.take_kernel_times();
        }
        let vseg = std::env::var_os("SMALLTIDE_SEGTIME").is_some();
        if vseg {
            backend.set_segment_timing(true).expect("seg");
            let _ = backend.take_segment_times();
        }
        let t_ver = Instant::now();
        let (mut i, mut steps, mut agree_until) = (0usize, 0usize, None::<usize>);
        let mut out = vec![refs[0]];
        let width = std::env::var("SMALLTIDE_VERIFY_WIDTH").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(8).clamp(2, 8);
        while i + width < refs.len() {
            let mut block: Vec<u32> = refs[i..i + width].to_vec();
            block[0] = *out.last().unwrap();
            let corrupt = 1 + (steps % (width - 1).max(1)).min(width - 2);
            block[corrupt] = (block[corrupt] + 7) % 100_000;
            engine.verify(&block, p0 + i).expect("verify");
            let am = engine.rows_argmax(width).expect("argmax");
            let mut c = 0usize;
            while c < width - 1 && block[c + 1] == am[c] {
                c += 1;
            }
            engine.commit_batch_rows(c + 1);
            engine.post_batch_tide_step().expect("post batch");
            engine.verify_rewind(width, c + 1).expect("rewind");
            out.extend_from_slice(&block[1..=c]);
            out.push(am[c]);
            i += c + 1;
            steps += 1;
            if agree_until.is_none() {
                if let Some(d) = (0..out.len().min(refs.len())).find(|&k| out[k] != refs[k]) {
                    agree_until = Some(d);
                }
            }
        }
        let ver_ms = t_ver.elapsed().as_secs_f64() * 1000.0 / steps.max(1) as f64;
        if vkt {
            let mut kt = backend.take_kernel_times();
            kt.sort_by(|a, b| b.2.total_cmp(&a.2));
            for (name, cnt, ms) in kt.iter().take(40) {
                println!("    {:<32} {:>6.1}x {:>8.3} ms/step ({:.1} us avg)", name, *cnt as f64 / steps.max(1) as f64, ms / steps.max(1) as f64, ms * 1000.0 / *cnt as f64);
            }
            backend.set_kernel_timing(false).expect("kt off");
        }
        if vseg {
            let st = backend.take_segment_times();
            for (name, cnt, ms) in st.iter().take(10) {
                println!("    {:<32} {:>6.1}x {:>8.3} ms/step", name, *cnt as f64 / steps.max(1) as f64, ms / steps.max(1) as f64);
            }
            backend.set_segment_timing(false).expect("seg off");
        }
        if let Some((u, r)) = backend.resident_vram_info() {
            println!("[verify-selftest] process VRAM usage now {:.2} GiB of resident {:.2} GiB", u as f64 / (1u64 << 30) as f64, r as f64 / (1u64 << 30) as f64);
        }
        println!("[verify-selftest] distinct resident experts per verify step {:.1} (all layers)", engine.dev_unique_resident as f64 / engine.dev_rounds.max(1) as f64);
        println!("[verify-selftest] plain decode {ref_ms:.2} ms/token | 8-row verify {ver_ms:.2} ms/step ({:.2}x a decode step)", ver_ms / ref_ms);
        let cmp = out.len().min(refs.len());
        let same = (0..cmp).filter(|&k| out[k] == refs[k]).count();
        println!("[verify-selftest] {steps} verify steps, {} tokens; identical to plain decode: {same}/{cmp}; first divergence at {:?}", out.len(), agree_until);
        return;
    }

    if let Some(mrows) = std::env::var("SMALLTIDE_BATCH_BENCH").ok().and_then(|v| v.parse::<usize>().ok()) {
        // Dev: time batched decode rounds of 1..=M rows (lanes 1..=M, same prompt) with GPU segments.
        for l in 1..=mrows {
            engine.select_lane(l);
            engine.reset_sequence().expect("reset");
            engine.prefill(&prompt_ids, 0, true).expect("prefill lane");
        }
        let first = next_tok;
        for m in 1..=mrows {
            let mut pos = prompt_ids.len();
            let mut toks = vec![first; m];
            let rounds = 24usize;
            for warm in [true, false] {
                if !warm {
                    backend.set_segment_timing(true).expect("seg");
                    let _ = backend.take_segment_times();
                }
                let t0 = Instant::now();
                for _ in 0..(if warm { 4 } else { rounds }) {
                    let rows: Vec<(usize, u32, usize)> = (0..m).map(|i| (i + 1, toks[i], pos)).collect();
                    engine.forward_batch(&rows).expect("batch");
                    for i in 0..m {
                        toks[i] = engine.sample_batch_row(i, i + 1, &sampling, &[]).expect("sample");
                    }
                    engine.post_batch_tide_step().expect("post");
                    pos += 1;
                }
                if !warm {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0 / rounds as f64;
                    let st = backend.take_segment_times();
                    backend.set_segment_timing(false).expect("seg off");
                    println!("[batch-bench] {m} rows: {ms:.2} ms/round ({:.1} tok/s aggregate)", m as f64 * 1000.0 / ms);
                    let total: f64 = st.iter().map(|k| k.2).sum();
                    for (name, cnt, t) in st.iter().filter(|k| k.2 > total * 0.03) {
                        println!("    {:<30} {:>6.1}x {:>8.3} ms/round", name, *cnt as f64 / rounds as f64, t / rounds as f64);
                    }
                }
            }
        }
        return;
    }

    if let Some(n) = std::env::var("SMALLTIDE_BATCH_SELFTEST").ok().and_then(|v| v.parse::<usize>().ok()) {
        // Numerics of batched decode vs the single-row path: lane 0 decodes with forward_token,
        // lanes 1 and 2 (same prompt) together with forward_batch, all teacher-forced on lane 0's
        // greedy tokens.
        let vocab = engine.cfg.unpadded_vocab_size;
        let read = |eng: &smalltide_model_inkling2::engine::Inkling2Engine<smalltide_backend_hip::HipBackend>, buf: smalltide_core::DeviceBuf| -> Vec<f32> {
            let mut b = vec![0u8; vocab * 4];
            eng.backend.memcpy_d2h_sync(&mut b, buf, 0, eng.tide.compute_stream).expect("d2h");
            b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        let argmax = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0;
        // Two different prompts; single-path references first (lane 0), then both in one batch
        // (lanes 1 and 2), teacher-forced on the reference tokens.
        let other = tok.encode("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>Write a haiku about winter rain on a tin roof.<|end_message|><|message_model|>");
        let prompts = [prompt_ids.clone(), other];
        engine.tide.wait_and_publish_all_pending().expect("drain pending");
        let mut refs: Vec<(Vec<u32>, Vec<Vec<f32>>)> = Vec::new();
        for pr in &prompts {
            engine.select_lane(0);
            engine.reset_sequence().expect("reset");
            engine.prefill(pr, 0, true).expect("prefill ref");
            let mut toks = vec![argmax(&read(&engine, engine.logits_buf())) as u32];
            let mut lg = Vec::new();
            for st in 0..n {
                engine.forward_token(toks[st], pr.len() + st, false, true).expect("single");
                engine.post_decode_tide_step().expect("post");
                let a = read(&engine, engine.logits_buf());
                toks.push(argmax(&a) as u32);
                lg.push(a);
            }
            refs.push((toks, lg));
        }
        for (l, pr) in [(1usize, &prompts[0]), (2usize, &prompts[1])] {
            engine.select_lane(l);
            engine.reset_sequence().expect("reset");
            engine.prefill(pr, 0, true).expect("prefill lane");
        }
        let (mut agree, mut worst, mut total) = (0usize, 0f32, 0usize);
        for step in 0..n {
            let rows = [(1usize, refs[0].0[step], prompts[0].len() + step), (2usize, refs[1].0[step], prompts[1].len() + step)];
            engine.forward_batch(&rows).expect("batch");
            let bl = engine.batch_logits_buf().expect("batch logits");
            for r in 0..2 {
                let b = read(&engine, bl.slice(r * vocab * 4, vocab * 4));
                let a = &refs[r].1[step];
                let d = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                let (ia, ib) = (argmax(a), argmax(&b));
                agree += (ia == ib) as usize;
                total += 1;
                worst = worst.max(d);
                if step < 3 || ia != ib {
                    println!("[batch-selftest] step {step:3} row {r}: max|single-batch| {d:.4} argmax {ia} vs {ib}");
                }
            }
            engine.post_batch_tide_step().expect("post batch");
        }
        let n = total;
        println!("[batch-selftest] {n} steps: argmax agreement {agree}/{n}, worst max|dlogit| {worst:.4}");
        return;
    }

    if std::env::var_os("SMALLTIDE_GRAPH_BENCH").is_some() {
        // Dev experiment: GPU time of one decode token, eager vs replayed HIP graph (the graph
        // bakes in this token's position, so its outputs are not used).
        let s = engine.tide.compute_stream;
        let pos0 = prompt_ids.len();
        for _ in 0..3 {
            engine.forward_token(next_tok, pos0, false, true).expect("warm");
            engine.post_decode_tide_step().expect("post");
            backend.sync_stream(s).expect("sync");
        }
        let n = 20;
        let t0 = Instant::now();
        for _ in 0..n {
            engine.forward_token(next_tok, pos0, false, true).expect("eager");
            backend.sync_stream(s).expect("sync");
        }
        let eager = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        engine.post_decode_tide_step().expect("post");
        engine.run_deferred_tide_step().expect("deferred");
        backend.sync_stream(s).expect("sync");
        engine.tide.wait_and_publish_all_pending().expect("drain pending");
        backend.sync_stream(engine.tide.copy_stream).expect("sync copy");
        backend.sync_stream(s).expect("sync");
        let _ = engine.tide.poll_pending();
        engine.clear_route_log_flag();
        backend.begin_graph_capture(s).expect("begin capture");
        let cap = engine.forward_token(next_tok, pos0, false, true);
        let graph = backend.end_graph_capture(s);
        match (cap, graph) {
            (Ok(()), Ok(g)) => {
                for _ in 0..3 {
                    backend.launch_graph(g, s).expect("launch");
                }
                backend.sync_stream(s).expect("sync");
                let t0 = Instant::now();
                for _ in 0..n {
                    backend.launch_graph(g, s).expect("launch");
                    backend.sync_stream(s).expect("sync");
                }
                let replay = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
                println!("[graph-bench] one decode token: eager {eager:.2} ms | graph replay {replay:.2} ms");
            }
            (c, g) => println!("[graph-bench] capture failed: forward={c:?} graph={:?}", g.err()),
        }
        return;
    }

    const EOS_ID: u32 = 200_006;
    const EOT_ID: u32 = 199_999;

    // Speculative decoding (greedy & stochastic rejection sampling): the drafter proposes
    // width - 1 tokens, one verify pass checks them; accepted drafts plus the target's next token
    // are emitted per step.
    let spec = engine.drafter.is_some();
    let spec_greedy = sampling.temperature <= 0.0 && (sampling.repetition_penalty - 1.0).abs() < 1e-6;
    // Width: fixed (SMALLTIDE_SPEC_WIDTH=n) or adaptive (default): per step the width maximizing
    // expected tokens per ms from the recent per-draft acceptance rate.
    let fixed_width = std::env::var("SMALLTIDE_SPEC_WIDTH").ok().and_then(|v| v.parse::<usize>().ok()).map(|w| w.clamp(2, 8));
    let spec_width = fixed_width.unwrap_or(8);
    let mut acc_s = 2.0f64; // decayed accepted drafts
    let mut acc_f = 2.0f64; // decayed rejections
    let mut width_hist = [0usize; 9];
    let (mut spec_steps, mut spec_accepted, mut draft_ms, mut verify_ms) = (0usize, 0usize, 0f64, 0f64);
    let mut spec_hist = [0usize; 8];
    if spec {
        let mut tap_row = engine.last_tap_rows.max(1) - 1;
        let mut done = false;
        let emit = |t: u32, generated: &mut Vec<u32>| -> bool {
            if t == EOS_ID || t == EOT_ID {
                return false;
            }
            generated.push(t);
            print!("{}", tok.decode(&[t]));
            let _ = std::io::stdout().flush();
            generated.len() < max_tokens
        };
        if !emit(next_tok, &mut generated) {
            done = true;
        }
        // Live calibration: ratio of observed acceptance to predicted margin-based probability.
        let mut pred_acc_ema = 3.0f64;
        let mut obs_acc_ema = 2.0f64;
        let mut draft_ema = 3.0f64;
        let mut verify_base_ema = 16.0f64;
        while !done && pos + 2 <= max_ctx {
            let max_w = (max_ctx - pos).min(8);
            let n_draft = match fixed_width {
                Some(w) => w.min(max_w) - 1,
                None => max_w - 1,
            };
            let td = Instant::now();
            let draft_pairs = engine.draft_with_margins(next_tok, pos, tap_row, n_draft).expect("draft");
            let dt_d = td.elapsed().as_secs_f64() * 1000.0;
            draft_ms += dt_d;
            draft_ema = 0.8 * draft_ema + 0.2 * dt_d;

            let calib = (obs_acc_ema / pred_acc_ema.max(0.5)).clamp(0.25, 1.05);
            let mut probs = [0.0f64; 7];
            for (k, &(_, mg)) in draft_pairs.iter().enumerate() {
                let raw_p = 1.0 / (1.0 + (-(mg as f64)).exp());
                probs[k] = (raw_p * calib).clamp(0.05, 0.98);
            }
            let spec_width = fixed_width.map(|w| w.min(max_w)).unwrap_or_else(|| {
                let mut best_w = 2usize;
                let mut best_rate = 0.0f64;
                let mut surv = 1.0f64;
                let mut exp_toks = 1.0f64;
                for w in 2..=max_w {
                    surv *= probs[w - 2];
                    exp_toks += surv;
                    let cost = draft_ema + verify_base_ema + 2.3 * (w - 1) as f64;
                    let rate = exp_toks / cost;
                    if rate > best_rate {
                        best_rate = rate;
                        best_w = w;
                    }
                }
                best_w
            });
            width_hist[spec_width] += 1;
            let n = spec_width - 1;
            let mut block = Vec::with_capacity(spec_width);
            block.push(next_tok);
            for &(t, _) in &draft_pairs[..n] {
                block.push(t);
            }
            let tv = Instant::now();
            engine.verify_for_sampling(&block, pos, spec_greedy).expect("verify");
            let (c, sampled_next) = engine.verify_sample(&block, &sampling, &generated).expect("verify sample");
            engine.commit_batch_rows(c + 1);
            engine.post_batch_tide_step().expect("post verify");
            engine.verify_rewind(spec_width, c + 1).expect("verify rewind");
            let dt_v = tv.elapsed().as_secs_f64() * 1000.0;
            verify_ms += dt_v;
            verify_base_ema = 0.85 * verify_base_ema + 0.15 * (dt_v - 2.3 * n as f64).clamp(10.0, 30.0);
            spec_steps += 1;
            spec_accepted += c;
            spec_hist[c] += 1;
            // Update margin calibration over the evaluated prefix (accepted drafts + first rejection).
            let eval_n = (c + 1).min(n);
            let mut pred_sum = 0.0f64;
            for k in 0..eval_n {
                pred_sum += 1.0 / (1.0 + (-(draft_pairs[k].1 as f64)).exp());
            }
            pred_acc_ema = 0.85 * pred_acc_ema + pred_sum;
            obs_acc_ema = 0.85 * obs_acc_ema + c as f64;
            acc_s = 0.85 * acc_s + c as f64;
            acc_f = 0.85 * acc_f + if c < n { 1.0 } else { 0.0 };
            for &t in &block[1..=c] {
                if !emit(t, &mut generated) {
                    done = true;
                    break;
                }
            }
            pos += c + 1;
            tap_row = c;
            next_tok = sampled_next;
            if !done && !emit(next_tok, &mut generated) {
                done = true;
            }
        }
    }

    for step in 0..max_tokens {
        if spec {
            break;
        }
        if next_tok == EOS_ID || next_tok == EOT_ID {
            break;
        }
        generated.push(next_tok);
        let piece = tok.decode(&[next_tok]);
        print!("{piece}");
        let _ = std::io::stdout().flush();

        if step + 1 >= max_tokens || pos >= max_ctx {
            break;
        }
        engine
            .forward_token(next_tok, pos, false, true)
            .expect("decode forward_token");
        pos += 1;
        next_tok = engine
            .sample_next_token(&sampling, &generated)
            .expect("sample token");
        let tp = Instant::now();
        engine.post_decode_tide_step().expect("post_decode_tide_step");
        post_ms += tp.elapsed().as_secs_f64() * 1000.0;
    }
    let dec_secs = t_dec.elapsed().as_secs_f64();
    let dec_toks = generated.len();
    if engine.profile.rescue_checks > 0 {
        let pr = &engine.profile;
        println!("
[rescue-check] {:.1} checks/token | wait {:.3} ms + plan {:.3} ms per check",
            pr.rescue_checks as f64 / dec_toks.max(1) as f64, pr.rescue_wait_ms / pr.rescue_checks as f64, pr.rescue_plan_ms / pr.rescue_checks as f64);
    }
    {
        // Rescue batches of both paths (single-token checks and verify passes): one batch is one
        // layer's missing experts, copied while the GPU waits at that layer.
        let (n, fill, sync) = engine.tide.rescue_stats;
        if n > 0 {
            println!("[rescue-check] {n} rescue batches: {:.3} ms staging copy + {:.3} ms copy-stream wait each", fill / n as f64, sync / n as f64);
            for (i, (b, f, w)) in engine.tide.rescue_hist.iter().enumerate() {
                if *b > 0 {
                    println!("[rescue-check]   batches of {}{}: {b} | {:.3} ms staging + {:.3} ms wait", i + 1, if i == 3 { "+" } else { "" }, f / *b as f64, w / *b as f64);
                }
            }
            let (d, dn, dnone) = engine.tide.stage_disk;
            if dn > 0 {
                println!("[rescue-check]   read from the disk while staging: {dn} experts ({dnone} of them had no RAM copy), {:.2} ms each", d * 1e3 / dn as f64);
            }
            let (pn, pb) = engine.tide.ram_pinned_stats();
            if pb > 0 {
                println!("[rescue-check]   page-locked RAM copies: {pn} experts in {:.2} GiB; {} experts uploaded straight from them", pb as f64 / (1u64 << 30) as f64, engine.tide.pinned_uploads);
            }
            let kept = engine.tide.scheduler.telemetry.evicted_kept_in_ram;
            if kept > 0 {
                println!("[rescue-check]   {kept} experts left the card with no RAM copy and were read back in the background");
            }
            let (c, k) = engine.tide.stage_copy;
            if k > 0 {
                println!("[rescue-check] {k} experts staged in all (prefill included): {:.3} ms each copying from RAM into the staging buffer", c * 1e3 / k as f64);
            }
        }
    }
    if spec && spec_steps > 0 {
        let hist: Vec<String> = spec_hist[..spec_width].iter().enumerate().map(|(i, c)| format!("{i}:{c}")).collect();
        if fixed_width.is_none() {
            let wh: Vec<String> = (2..=8).filter(|&w| width_hist[w] > 0).map(|w| format!("w{w}:{}", width_hist[w])).collect();
            println!("
[spec] adaptive width per step: {}", wh.join(" "));
        }
        println!(
            "\n[spec] distinct resident experts per verify step {:.1} | width {}: {spec_steps} steps, mean accepted {:.2} -> {:.2} tokens/step | draft {:.2} ms + verify {:.2} ms per step | accepted per step {}",
            engine.dev_unique_resident as f64 / engine.dev_rounds.max(1) as f64,
            fixed_width.map_or("adaptive".to_string(), |w| w.to_string()),
            spec_accepted as f64 / spec_steps as f64,
            1.0 + spec_accepted as f64 / spec_steps as f64,
            draft_ms / spec_steps as f64,
            verify_ms / spec_steps as f64,
            hist.join(" ")
        );
    }
    let hs = backend.take_host_stats();
    let tel = engine.telemetry();
    println!("\n--------------");
    println!(
        "[Perf] Prefill: {} tok in {:.2}s ({:.2} tok/s) | Boundary Rebalance: {} swaps ({:.1} ms) | Decode: {} tok in {:.2}s ({:.2} tok/s)",
        prompt_ids.len(),
        pf_secs,
        prompt_ids.len() as f64 / pf_secs.max(1e-6),
        reb_swaps,
        reb_ms,
        dec_toks,
        dec_secs,
        dec_toks as f64 / dec_secs.max(1e-6),
    );
    let pct = |h: u64, m: u64| -> f64 {
        let tot = h + m;
        if tot == 0 { 100.0 } else { (h as f64 / tot as f64) * 100.0 }
    };
    println!(
        "[Tide] Overall VRAM hit: {:.1}% (Prelude {:.1}%, LoopCore {:.1}%, Coda {:.1}%) | skipped routing weight {:.2}% (prefill {:.2}%) | Stage1 sync={}, async={}, skipped={} | decode async promotions={} | Stage2 disk->RAM={} | RAM-miss disk reads={}",
        tel.total_vram_hit_rate() * 100.0,
        pct(tel.vram_hits_prelude, tel.vram_misses_prelude),
        pct(tel.vram_hits_loop, tel.vram_misses_loop),
        pct(tel.vram_hits_coda, tel.vram_misses_coda),
        tel.skipped_weight_share() * 100.0,
        if tel.prefill_weight_total > 0.0 { 100.0 * tel.prefill_weight_skipped / tel.prefill_weight_total } else { 0.0 },
        tel.stage1_sync_rescues,
        tel.stage1_async_promotions,
        tel.stage1_skipped_tail_misses,
        tel.decode_async_promotions,
        tel.stage2_disk_reads,
        tel.stage1_ram_fallback_reads,
    );
    if let Some(line) = engine.tide.scheduler.finish_request() {
        println!("{line}");
    }
    if dec_toks > 0 {
        let n = dec_toks as f64;
        println!(
            "[Host/tok] {:.2} ms total | {:.0} launches ({:.2} ms CPU in hipModuleLaunchKernel) | {:.1} syncs ({:.2} ms blocked) | {:.2} ms other CPU (post-decode Tide {:.2} ms)",
            dec_secs * 1000.0 / n,
            hs.launches as f64 / n,
            hs.launch_ms / n,
            hs.syncs as f64 / n,
            hs.sync_wait_ms / n,
            (dec_secs * 1000.0 - hs.launch_ms - hs.sync_wait_ms) / n,
            post_ms / n,
        );
    }
    if segtiming && dec_toks > 0 {
        let n = dec_toks as f64;
        let st = backend.take_segment_times();
        let total: f64 = st.iter().map(|k| k.2).sum();
        println!("[GPU segments/tok] total {:.2} ms", total / n);
        for (name, cnt, ms) in st {
            println!("  {:<32} {:>6.1}x  {:>7.3} ms  ({:>6.1} us avg)", name, cnt as f64 / n, ms / n, ms * 1000.0 / cnt as f64);
        }
    }
    if ktiming && dec_toks > 0 {
        let n = dec_toks as f64;
        let kt = backend.take_kernel_times();
        let total: f64 = kt.iter().map(|k| k.2).sum();
        println!("[GPU kernels/tok] total {:.2} ms in {:.0} launches", total / n, kt.iter().map(|k| k.1 as f64).sum::<f64>() / n);
        for (name, cnt, ms) in kt {
            println!("  {:<28} {:>6.1}x  {:>7.3} ms  ({:>5.1} us avg)", name, cnt as f64 / n, ms / n, ms * 1000.0 / cnt as f64);
        }
    }
    if enable_profile && engine.profile.steps > 0 {
        let n = engine.profile.steps as f64;
        let p = &engine.profile;
        println!(
            "[Profile ms/tok avg over {} steps] res+ctrl={:.2} | attn_proj={:.2} | sconv+clip+rel={:.2} | flash_attn={:.2} | attn_out+moe_prep={:.2} | router+d2h={:.2} | tide_rescue={:.2} | moe={:.2} | lm_head={:.2}",
            p.steps,
            p.res_and_ctrl_ms / n,
            p.attn_proj_ms / n,
            p.sconv_clip_rel_ms / n,
            p.flash_attn_ms / n,
            p.attn_out_ms / n,
            p.router_d2h_ms / n,
            p.tide_rescue_ms / n,
            p.moe_ms / n,
            p.lm_head_ms / n,
        );
    }
}

/// Fast (~0.2s boot) GPU stress microbench for GPU-side rescue waits (`spin` vs `waitvalue`),
/// running real `rescue_write_route`, wait+patch, 4.27 MiB SDMA slab copies on `copy_stream`,
/// and real `moe_decode_experts` kernels reading freshly DMA'd slabs across 48 layers/token.
fn run_stress_rescue(mode: &str, tokens: usize) {
    let use_spin = match mode {
        "spin" => true,
        "waitvalue" => false,
        other => {
            eprintln!("Usage: smalltide-cli stress-rescue [spin|waitvalue] [tokens]");
            eprintln!("Unknown mode: {other}");
            std::process::exit(1);
        }
    };
    let t_boot = Instant::now();
    let cfg = Inkling2Config::default();
    let dev = HipRuntime::load().expect("HIP").select_device().expect("select device");
    let backend = Arc::new(HipBackend::new(dev).expect("HipBackend"));
    let compute_stream = backend.create_stream(smalltide_core::QueueKind::Compute).expect("compute stream");
    let copy_stream = backend.create_stream(smalltide_core::QueueKind::AsyncCopyH2D).expect("copy stream");
    let ev = backend.create_event().expect("ev");

    let n_log = cfg.num_logical_layers();
    let n_exp = cfg.n_routed_experts;
    let total_experts = cfg.num_physical_layers * n_exp;
    let layout = ExpertSlabLayout::for_inkling2(&cfg, DType::NVFP4);
    let slab_bytes = layout.aligned_slab_bytes;
    let pool_slots = 48usize; // 40 resident + 8 scratch
    let pool = backend.alloc_device(pool_slots * slab_bytes).expect("pool");
    let slot_table = backend.alloc_device(total_experts * 4).expect("slot_table");

    // Build one valid NVFP4 slab template in pinned staging and initialize all pool slots.
    let staging_slots = 16usize;
    let staging = backend.alloc_pinned(staging_slots * slab_bytes).expect("staging");
    let mut hdr = ExpertSlabHeader {
        w13_outer_scale: 0.25,
        w2_outer_scale: 0.25,
        w13_weight_offset: layout.w13_weight_offset as u32,
        w13_scale_offset: layout.w13_scale_offset as u32,
        w2_weight_offset: layout.w2_weight_offset as u32,
        w2_scale_offset: layout.w2_scale_offset as u32,
        dtype_tag: 0,
        _reserved: [0; 9],
    };
    let hdr_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(&hdr as *const ExpertSlabHeader as *const u8, std::mem::size_of::<ExpertSlabHeader>())
    };
    unsafe {
        let st = backend.pinned_as_mut_slice(staging).expect("staging slice");
        for s in 0..staging_slots {
            let base = s * slab_bytes;
            st[base..base + slab_bytes].fill(0x38);
            st[base..base + hdr_bytes.len()].copy_from_slice(hdr_bytes);
        }
    }
    for s in 0..pool_slots {
        backend.memcpy_h2d_async(pool, s * slab_bytes, staging, 0, slab_bytes, copy_stream).expect("init slab");
    }
    backend.sync_stream(copy_stream).expect("sync init pool");

    // Initial slot_table: experts 0..6 resident in slots 0..6, experts 6..128 missing (u32::MAX).
    let init_tbl = backend.alloc_pinned(total_experts * 4).expect("init_tbl");
    unsafe {
        let p = backend.pinned_as_mut_slice(init_tbl).unwrap().as_mut_ptr() as *mut u32;
        for l in 0..cfg.num_physical_layers {
            for e in 0..n_exp {
                let v = if e < 6 { e as u32 } else { u32::MAX };
                std::ptr::write(p.add(l * n_exp + e), v);
            }
        }
    }
    backend.memcpy_h2d_async(slot_table, 0, init_tbl, 0, total_experts * 4, compute_stream).expect("upload tbl");

    // Routing log: 48 layers x 16 u32s (ids 0..8, weights 0.125 each).
    let route_log = backend.alloc_device(n_log * 16 * 4).expect("route_log");
    let route_init = backend.alloc_pinned(n_log * 16 * 4).expect("route_init");
    unsafe {
        let p = backend.pinned_as_mut_slice(route_init).unwrap().as_mut_ptr() as *mut u32;
        for il in 0..n_log {
            for k in 0..8 {
                std::ptr::write(p.add(il * 16 + k), k as u32);
                std::ptr::write(p.add(il * 16 + 8 + k), 0.125f32.to_bits());
            }
        }
    }
    backend.memcpy_h2d_async(route_log, 0, route_init, 0, n_log * 16 * 4, compute_stream).expect("upload route_log");

    // MoE activation scratch buffers
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let x_buf = backend.alloc_device(h * 4).expect("x");
    let mid_buf = backend.alloc_device(8 * inter * 4).expect("mid");
    let tmp_buf = backend.alloc_device(8 * h * 4).expect("tmp");
    backend.zero_f32(x_buf, h, compute_stream).expect("zero x");

    // Coherent signal buffers matching engine.rs
    const ROUTE_STRIDE: usize = 32;
    const FLAG_STRIDE: usize = 16;
    const PATCH_STRIDE: usize = 512;
    let (rview, rhost) = backend.alloc_host_signal(n_log * ROUTE_STRIDE * 4).expect("route_pinned");
    let (fview, fhost) = backend.alloc_host_signal(n_log * FLAG_STRIDE * 4).expect("wait_flags");
    let (pview, phost) = backend.alloc_host_signal(n_log * PATCH_STRIDE * 4).expect("patches");
    let (ftview, _fthost) = backend.alloc_host_signal(total_experts * 4).expect("full_tbl");
    let (prview, prhost) = backend.alloc_host_signal(8 * 4).expect("probe");
    unsafe {
        for (hbuf, words) in [(rhost, n_log * ROUTE_STRIDE), (fhost, n_log * FLAG_STRIDE), (phost, n_log * PATCH_STRIDE), (prhost, 8)] {
            let p = backend.pinned_as_mut_slice(hbuf).unwrap().as_mut_ptr() as *mut u32;
            for i in 0..words { std::ptr::write_volatile(p.add(i), 0); }
        }
    }
    backend.sync_stream(compute_stream).expect("sync init");
    let boot_ms = t_boot.elapsed().as_secs_f64() * 1000.0;

    let moe_params = smalltide_core::backend::MoeDispatchParams {
        batch_size: 1,
        hidden_size: h,
        intermediate_size: inter,
        top_k: 8,
        weight_dtype: DType::NVFP4,
        expert_stride_bytes: slab_bytes,
    };

    let t0 = Instant::now();
    let mut total_wait_us = 0.0f64;
    let mut total_rescue_us = 0.0f64;
    let mut total_rescues = 0u64;
    let mut total_waits = 0u64;
    let mut scratch_cur = 0usize;
    let mut staging_cur = 0usize;
    let mut prev_rescued_idx: Option<usize> = None;

    for tok in 1..=tokens {
        let ticket = tok as u32;
        unsafe {
            let pp = backend.pinned_as_mut_slice(phost).unwrap().as_mut_ptr() as *mut u32;
            for il in 0..n_log {
                std::ptr::write_volatile(pp.add(il * PATCH_STRIDE), 0);
            }
        }
        // Queue all 48 layers up front (exact engine.rs pattern)
        for il in 0..n_log {
            let ip = il % cfg.num_physical_layers;
            let route_row = route_log.slice(il * 16 * 4, 16 * 4);
            let topk_ids = route_log.slice(il * 16 * 4, 32);
            let layer_tbl = slot_table.slice(ip * n_exp * 4, n_exp * 4);
            backend.rescue_write_route(
                rview.slice(il * ROUTE_STRIDE * 4, ROUTE_STRIDE * 4),
                route_row,
                16,
                16,
                ticket,
                compute_stream,
            ).expect("rescue_write_route");
            let flag_sl = fview.slice(il * FLAG_STRIDE * 4, 4);
            let patch_sl = pview.slice(il * PATCH_STRIDE * 4, PATCH_STRIDE * 4);
            if use_spin {
                backend.rescue_wait_and_patch(
                    slot_table, flag_sl, ticket, patch_sl, ftview, total_experts, Some(prview), il, compute_stream,
                ).expect("rescue_wait_and_patch");
            } else {
                backend.stream_wait_value_ge(compute_stream, flag_sl, ticket).expect("stream_wait_value_ge");
                backend.rescue_apply_patch(
                    slot_table, patch_sl, ftview, total_experts, Some(prview), il, ticket, compute_stream,
                ).expect("rescue_apply_patch");
            }
            backend.moe_decode_experts(mid_buf, tmp_buf, x_buf, pool, layer_tbl, topk_ids, moe_params, compute_stream)
                .expect("moe_decode_experts");
        }
        backend.record_event(ev, compute_stream).expect("record_event");
        let _ = backend.query_event(ev).expect("query_event");

        // Service all 48 checks on host
        for il in 0..n_log {
            let tw = Instant::now();
            let row = unsafe {
                backend.pinned_as_mut_slice(rhost).unwrap()[il * ROUTE_STRIDE * 4..].as_ptr() as *const u32
            };
            let mut spins = 0u64;
            loop {
                if unsafe { std::ptr::read_volatile(row.add(16)) } == ticket {
                    std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
                    break;
                }
                spins += 1;
                if spins % 8192 == 0 {
                    backend.query_event(ev).expect("query_event in spin");
                    if tw.elapsed().as_secs() > 5 {
                        let (pr0, pr1) = unsafe {
                            let p = backend.pinned_as_mut_slice(prhost).unwrap().as_ptr() as *const u32;
                            (std::ptr::read_volatile(p), std::ptr::read_volatile(p.add(1)))
                        };
                        eprintln!(
                            "[stress-rescue] TIMEOUT at tok={tok} il={il} mode={mode}: probe=0x{pr0:08x} (tkt={} il={} st={}) last_n={pr1}",
                            pr0 >> 12, (pr0 >> 4) & 0xFF, pr0 & 0xF
                        );
                        // Release remaining flags so stream doesn't hang on exit
                        unsafe {
                            let fp = backend.pinned_as_mut_slice(fhost).unwrap().as_mut_ptr() as *mut u32;
                            for rem in il..n_log {
                                std::ptr::write_volatile(fp.add(rem * FLAG_STRIDE), ticket);
                            }
                        }
                        std::process::exit(2);
                    }
                }
                std::hint::spin_loop();
            }
            total_wait_us += tw.elapsed().as_secs_f64() * 1e6;
            total_waits += 1;

            // Perform 12 real 4.27 MiB SDMA rescues per token (il % 4 == 0)
            if il % 4 == 0 {
                let tr = Instant::now();
                let ip = il % cfg.num_physical_layers;
                let dst_slot = 40 + (scratch_cur % 8);
                scratch_cur += 1;
                let src_off = (staging_cur % staging_slots) * slab_bytes;
                staging_cur += 1;
                // Rewrite header with distinct outer_scale to verify live DMA + GL2 inv
                hdr.w13_outer_scale = 0.25 + (tok as f32) * 1e-5;
                let hb: &[u8] = unsafe {
                    std::slice::from_raw_parts(&hdr as *const ExpertSlabHeader as *const u8, std::mem::size_of::<ExpertSlabHeader>())
                };
                unsafe {
                    let st = backend.pinned_as_mut_slice(staging).unwrap();
                    st[src_off..src_off + hb.len()].copy_from_slice(hb);
                }
                backend.memcpy_h2d_async(pool, dst_slot * slab_bytes, staging, src_off, slab_bytes, copy_stream)
                    .expect("rescue memcpy_h2d_async");
                backend.sync_stream(copy_stream).expect("rescue sync copy_stream");
                let new_idx = ip * n_exp + 6; // expert 6 rescued into scratch slot
                unsafe {
                    let pp = backend.pinned_as_mut_slice(phost).unwrap().as_mut_ptr() as *mut u32;
                    let dst = pp.add(il * PATCH_STRIDE);
                    let mut n_p = 0u32;
                    if let Some(old) = prev_rescued_idx {
                        std::ptr::write_volatile(dst.add(1 + 2 * n_p as usize), old as u32);
                        std::ptr::write_volatile(dst.add(2 + 2 * n_p as usize), u32::MAX);
                        n_p += 1;
                    }
                    std::ptr::write_volatile(dst.add(1 + 2 * n_p as usize), new_idx as u32);
                    std::ptr::write_volatile(dst.add(2 + 2 * n_p as usize), dst_slot as u32);
                    n_p += 1;
                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                    std::ptr::write_volatile(dst, n_p);
                }
                prev_rescued_idx = Some(new_idx);
                total_rescue_us += tr.elapsed().as_secs_f64() * 1e6;
                total_rescues += 1;
            }
            unsafe {
                let fp = backend.pinned_as_mut_slice(fhost).unwrap().as_mut_ptr() as *mut u32;
                std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                std::ptr::write_volatile(fp.add(il * FLAG_STRIDE), ticket);
            }
        }
        backend.sync_stream(compute_stream).expect("sync token end");
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "[stress-rescue] PASS mode={mode} tokens={tokens} waits={total_waits} rescues={total_rescues} (boot={boot_ms:.0}ms, run={elapsed_ms:.1}ms, {:.2} ms/tok, wait_avg={:.1} us, rescue_dma_avg={:.1} us ({:.1} GB/s))",
        elapsed_ms / tokens as f64,
        total_wait_us / total_waits as f64,
        total_rescue_us / total_rescues.max(1) as f64,
        (slab_bytes as f64 / 1e9) / (total_rescue_us / total_rescues.max(1) as f64 / 1e6),
    );
}

/// Dev: `vision-check --rows R --cols C [--model DIR] [--patches IN.bin] [--patches-out P.bin] --out O.bin`
/// runs the vision tower alone on one image's patches (bf16 `[R * C * 4][1200]`; pseudo-random when
/// no file is given) and writes the feature rows (f32 `[R * C][hidden]`) for tools/inkling2_vision_ref.py.
/// SMALLTIDE_VISION_DUMP=PREFIX also writes the intermediates.
fn run_vision_check(args: &[String]) {
    use smalltide_model_inkling2::vision::{VisionTower, PATCH_K, PER_TOKEN};
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let rows: usize = arg("--rows").and_then(|v| v.parse().ok()).unwrap_or(3);
    let cols: usize = arg("--cols").and_then(|v| v.parse().ok()).unwrap_or(5);
    let model = std::path::PathBuf::from(arg("--model").unwrap_or_else(|| "models/Inkling-2-Mini".into()));
    let hidden: usize = serde_json::from_slice::<serde_json::Value>(&std::fs::read(model.join("config.json")).expect("config.json"))
        .ok()
        .and_then(|v| v["vision_config"]["text_hidden_size"].as_u64())
        .expect("vision_config.text_hidden_size") as usize;
    let n = rows * cols * PER_TOKEN * PATCH_K;
    let patches: Vec<u16> = match arg("--patches") {
        Some(p) => std::fs::read(p).expect("patches").chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
        None => {
            // values in [-1, 1) as bf16, a fixed sequence
            let mut x = 0x2545_F491_4F6C_DD1Du64;
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    let f = ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
                    (f.to_bits() >> 16) as u16
                })
                .collect()
        }
    };
    assert_eq!(patches.len(), n, "patches: {} values, {rows} x {cols} tokens need {n}", patches.len());
    if let Some(p) = arg("--patches-out") {
        std::fs::write(p, patches.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).expect("patches-out");
    }
    let backend = HipBackend::new_auto().expect("init HipBackend");
    let s = backend.create_stream(smalltide_core::QueueKind::Compute).expect("stream");
    let tower = VisionTower::open(&model.join("model-00001-of-00001.safetensors"), hidden).expect("vision tower");
    let reps: usize = arg("--reps").and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut out = Vec::new();
    for _ in 0..reps {
        let (o, st) = tower.encode(&backend, s, &patches, rows, cols).expect("encode");
        println!("vision: {rows} x {cols} = {} tokens in {:.1} ms, {:.1} MB on the device", st.tokens, st.ms, st.device_bytes as f64 / 1e6);
        out = o;
    }
    let finite = out.iter().all(|v| v.is_finite());
    let rms = (out.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / out.len() as f64).sqrt();
    println!("rows: {} x {hidden}, finite {finite}, rms {rms:.4}", rows * cols);
    if let Some(p) = arg("--out") {
        std::fs::write(p, out.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).expect("out");
    }
}

#[cfg(test)]
mod cli_tests {
    use super::extract_env_flag;
    use std::path::PathBuf;

    fn sv(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_env_flag_before_and_after_subcommand() {
        let (p, a) = extract_env_flag(sv(&["smalltide-cli", "--env", ".env.monitor", "serve"]));
        assert_eq!(p, Some(PathBuf::from(".env.monitor")));
        assert_eq!(a, sv(&["smalltide-cli", "serve"]));

        let (p, a) = extract_env_flag(sv(&["smalltide-cli", "serve", "--env=.env.headless"]));
        assert_eq!(p, Some(PathBuf::from(".env.headless")));
        assert_eq!(a, sv(&["smalltide-cli", "serve"]));

        let (p, a) = extract_env_flag(sv(&["smalltide-cli", "run", "-e", "my.env", "-p", "--env not-a-flag"]));
        assert_eq!(p, Some(PathBuf::from("my.env")));
        assert_eq!(a, sv(&["smalltide-cli", "run", "-p", "--env not-a-flag"]));

        let (p, a) = extract_env_flag(sv(&["smalltide-cli", "serve", ".env.monitor"]));
        assert_eq!(p, Some(PathBuf::from(".env.monitor")));
        assert_eq!(a, sv(&["smalltide-cli", "serve"]));

        let (p, a) = extract_env_flag(sv(&["smalltide-cli", "serve", "--unknown"]));
        assert_eq!(p, None);
        assert_eq!(a, sv(&["smalltide-cli", "serve", "--unknown"]));
    }
}

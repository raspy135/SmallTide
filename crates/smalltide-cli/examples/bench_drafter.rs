//! Fast (~2s boot) standalone DSpark drafter benchmark & parity checker.
//! Loads only `mtp.safetensors` (plus synthetic or `--real` `lm_head`), so it never loads the
//! 24-layer model and can run while `smalltide-cli serve` is up.
//!
//! Usage:
//!   cargo run --offline --release --example bench_drafter [-- --real]

use smalltide_backend_hip::{HipBackend, HipRuntime};
use smalltide_core::{ComputeBackend, Inkling2Config, QueueKind};
use smalltide_io::dense_loader::{read_f32_vec, read_fp8_matrix};
use smalltide_io::safetensors::inspect_safetensors_header;
use smalltide_model_inkling2::drafter::Drafter;
use smalltide_model_inkling2::weights::{Fp8MatrixGpu, Inkling2DenseGpu};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn upload_slice(b: &HipBackend, bytes: &[u8], s: smalltide_core::Stream) -> smalltide_core::DeviceBuf {
    let dev = b.alloc_device(bytes.len().max(4)).expect("alloc_device");
    let chunk = 64 * 1024 * 1024;
    let stg = b.alloc_pinned(bytes.len().min(chunk).max(4)).expect("alloc_pinned");
    let mut off = 0;
    while off < bytes.len() {
        let take = (bytes.len() - off).min(stg.size_bytes);
        unsafe {
            b.pinned_as_mut_slice(stg).unwrap()[..take].copy_from_slice(&bytes[off..off + take]);
        }
        b.memcpy_h2d_async(dev, off, stg, 0, take, s).expect("h2d");
        b.sync_stream(s).expect("sync");
        off += take;
    }
    b.free_pinned(stg).expect("free_pinned");
    dev
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_real = args.iter().any(|a| a == "--real");
    let max_seq: usize = 16_384;
    let chunk: usize = 256;
    let cfg = Inkling2Config::default();
    let vocab = cfg.unpadded_vocab_size;
    let d = cfg.hidden_size;

    let t_boot = Instant::now();
    let dev = HipRuntime::load().expect("HIP").select_device().expect("select");
    let backend = Arc::new(HipBackend::new(dev).expect("HipBackend"));
    let s = backend.create_stream(QueueKind::Compute).expect("stream");

    // Quick parity & latency check for GPU `logits_filter_rows` (used in stochastic verify_sample).
    {
        let rows = 8usize;
        let cap = 256usize;
        let margin = 12.0f32;
        let mut host = vec![-5.0f32; rows * vocab];
        let mut expected: Vec<Vec<(u32, f32)>> = vec![Vec::new(); rows];
        for r in 0..rows {
            let peak = 20.0 + r as f32;
            for j in 0..(15 + r * 5) {
                let idx = (j * 4099 + r * 997 + 13) % vocab;
                let val = peak - (j as f32) * 0.25;
                host[r * vocab + idx] = val;
            }
            for i in 0..vocab {
                let v = host[r * vocab + i];
                if v >= peak - margin {
                    expected[r].push((i as u32, v));
                }
            }
            expected[r].sort_by_key(|&(id, _)| id);
        }
        let x_dev = backend.alloc_device(rows * vocab * 4).expect("x_dev");
        let x_pin = backend.alloc_pinned(rows * vocab * 4).expect("x_pin");
        unsafe {
            let dst = backend.pinned_as_mut_slice(x_pin).unwrap();
            let src = std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 4);
            dst[..src.len()].copy_from_slice(src);
        }
        backend.memcpy_h2d_async(x_dev, 0, x_pin, 0, host.len() * 4, s).expect("h2d");
        let buf_bytes = (rows * 40 + rows + 2 * rows * cap) * 4;
        let buf = backend.alloc_device(buf_bytes).expect("buf");
        let row_max = buf.slice(rows * 4, rows * 4);
        let scratch = buf.slice(rows * 8, rows * 32 * 4);
        let d2h_base = rows * 40 * 4;
        let counts = buf.slice(d2h_base, rows * 4);
        let out_idx = buf.slice(d2h_base + rows * 4, rows * cap * 4);
        let out_val = buf.slice(d2h_base + rows * 4 + rows * cap * 4, rows * cap * 4);
        backend.logits_filter_rows(out_idx, out_val, counts, row_max, scratch, x_dev, rows, vocab, margin, cap, s).expect("filter");
        let d2h_bytes = (rows + 2 * rows * cap) * 4;
        let mut raw = vec![0u8; d2h_bytes];
        backend.memcpy_d2h_sync(&mut raw, buf.slice(d2h_base, d2h_bytes), 0, s).expect("d2h");
        let word = |w: usize| u32::from_le_bytes(raw[w * 4..(w + 1) * 4].try_into().unwrap());
        for r in 0..rows {
            let cnt = word(r) as usize;
            assert_eq!(cnt, expected[r].len(), "logits_filter_rows row {r} count mismatch");
            let mut got: Vec<(u32, f32)> = (0..cnt)
                .map(|j| (word(rows + r * cap + j), f32::from_bits(word(rows + rows * cap + r * cap + j))))
                .collect();
            got.sort_by_key(|&(id, _)| id);
            assert_eq!(got, expected[r], "logits_filter_rows row {r} entries mismatch");
        }
        let t0 = Instant::now();
        for _ in 0..20 {
            backend.logits_filter_rows(out_idx, out_val, counts, row_max, scratch, x_dev, rows, vocab, margin, cap, s).expect("filter");
        }
        backend.sync_stream(s).expect("sync");
        println!(
            "[parity] logits_filter_rows [8x{}] EXACT | {:.3} ms/call",
            vocab,
            t0.elapsed().as_secs_f64() * 1000.0 / 20.0
        );
        let _ = backend.free_device(x_dev);
        let _ = backend.free_device(buf);
        let _ = backend.free_pinned(x_pin);
    }

    #[cfg(windows)]
    let model_dir = Path::new(r"models\Inkling-2-Mini");
    #[cfg(not(windows))]
    let model_dir = Path::new("models/Inkling-2-Mini");
    let (unembed_fp8, embed_norm, embed_bf16_host) = if use_real {
        let mpath = model_dir.join("model-00001-of-00001.safetensors");
        let hdr = inspect_safetensors_header(&mpath).expect("inspect model");
        let base = 8u64 + hdr.header_len;
        let mut f = File::open(&mpath).expect("open model");
        let lm = read_fp8_matrix(&mut f, &hdr, base, "model.llm.unembed.weight").expect("unembed");
        let data = upload_slice(&backend, &lm.data, s);
        let row_scales_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(lm.row_scales.as_ptr() as *const u8, lm.row_scales.len() * 4)
        };
        let row_scales = upload_slice(&backend, row_scales_bytes, s);
        let en = read_f32_vec(&mut f, &hdr, base, "model.llm.embed_norm.weight")
            .unwrap_or_else(|_| vec![1.0f32; d]);
        let en_bytes: &[u8] = unsafe { std::slice::from_raw_parts(en.as_ptr() as *const u8, en.len() * 4) };
        let embed_norm = upload_slice(&backend, en_bytes, s);
        // Synthetic embeddings for anchors 0..64 (or load if needed)
        let mut emb = vec![0u8; cfg.vocab_size * d * 2];
        for tok in 0..64usize {
            for i in 0..d {
                let v = (((tok * 131 + i * 17) % 1000) as f32 / 1000.0 - 0.5) * 0.2;
                let bf = (v.to_bits() >> 16) as u16;
                let o = (tok * d + i) * 2;
                emb[o..o + 2].copy_from_slice(&bf.to_le_bytes());
            }
        }
        (
            Fp8MatrixGpu { n_rows: lm.n_rows, k_cols: lm.k_cols, data, row_scales, is_bf16: false },
            embed_norm,
            emb,
        )
    } else {
        // Deterministic FP8 unembed (avoid 0x7F/0xFF NaNs)
        let mut w = vec![0u8; vocab * d];
        for (i, b) in w.iter_mut().enumerate() {
            let x = ((i.wrapping_mul(2654435761) >> 24) & 0xFF) as u8;
            *b = if (x & 0x7F) == 0x7F { 0x38 } else { x };
        }
        let sc: Vec<f32> = (0..vocab).map(|r| 0.005 + (r % 97) as f32 * 0.0001).collect();
        let data = upload_slice(&backend, &w, s);
        let sc_bytes: &[u8] = unsafe { std::slice::from_raw_parts(sc.as_ptr() as *const u8, sc.len() * 4) };
        let row_scales = upload_slice(&backend, sc_bytes, s);
        let en = vec![1.0f32; d];
        let en_bytes: &[u8] = unsafe { std::slice::from_raw_parts(en.as_ptr() as *const u8, en.len() * 4) };
        let embed_norm = upload_slice(&backend, en_bytes, s);
        let mut emb = vec![0u8; cfg.vocab_size * d * 2];
        for tok in 0..64usize {
            for i in 0..d {
                let v = (((tok * 131 + i * 17) % 1000) as f32 / 1000.0 - 0.5) * 0.2;
                let bf = (v.to_bits() >> 16) as u16;
                let o = (tok * d + i) * 2;
                emb[o..o + 2].copy_from_slice(&bf.to_le_bytes());
            }
        }
        (Fp8MatrixGpu { n_rows: vocab, k_cols: d, data, row_scales, is_bf16: false }, embed_norm, emb)
    };

    let dummy = backend.alloc_device(4).expect("dummy");
    let dense = Inkling2DenseGpu {
        backend: backend.clone(),
        embed_bf16_host,
        embed_norm,
        final_norm: dummy,
        output_res_query: dummy,
        middle_cycle_cross_pass_queries: dummy,
        middle_cycle_cross_pass_kv_queries: dummy,
        unembed_fp8,
        layers: Vec::new(),
    };

    let v_q4 = std::env::var("SMALLTIDE_KV_FMT").map(|v| v.eq_ignore_ascii_case("k8v4")).unwrap_or(false);
    let mut drafter = Drafter::load(
        backend.clone(),
        &model_dir.join("mtp.safetensors"),
        &dense,
        vocab,
        max_seq,
        chunk,
        v_q4,
        1,
        max_seq,
    )
    .expect("Drafter::load");
    println!(
        "[bench-drafter] loaded mtp.safetensors in {:.2}s | drafter VRAM = {:.1} MiB (max_seq={max_seq}, real_lm_head={use_real})",
        t_boot.elapsed().as_secs_f64(),
        drafter.vram_bytes() as f64 / (1024.0 * 1024.0),
    );

    // Deterministic taps [4][chunk][D]
    let mut taps_host = vec![0.0f32; 4 * chunk * d];
    for t in 0..4 {
        for r in 0..chunk {
            for i in 0..d {
                let idx = (t * chunk + r) * d + i;
                taps_host[idx] = (((idx.wrapping_mul(1103515245).wrapping_add(12345)) >> 16) as f32 / 32768.0 - 1.0) * 0.4;
            }
        }
    }
    let taps_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(taps_host.as_ptr() as *const u8, taps_host.len() * 4)
    };
    let taps = upload_slice(&backend, taps_bytes, s);

    // Populate context KV up to 10,240 tokens in chunks of 256
    let t_ing = Instant::now();
    let mut cur_p = 0usize;
    let target_p = 10_240usize;
    while cur_p < target_p {
        let m = (target_p - cur_p).min(chunk);
        drafter.ingest(taps, chunk, cur_p, m, s).expect("ingest");
        cur_p += m;
    }
    backend.sync_stream(s).expect("sync ingest");
    println!("[bench-drafter] ingested {target_p} ctx tokens in {:.1} ms", t_ing.elapsed().as_secs_f64() * 1000.0);

    // Parity signature: 8 anchors x 7 drafts at P=1024
    println!("[parity] 8 anchors x 7 drafts @ P=1024:");
    for a in 0..8u32 {
        let toks = drafter.draft(&dense, a, 1024, taps, chunk, a as usize, 7, s).expect("draft");
        println!("  anchor={a}: {:?}", toks);
    }

    // Benchmark across context lengths P and draft counts n
    for &p in &[128usize, 2048, 10240] {
        for &n in &[3usize, 5, 7] {
            // Warmup
            for _ in 0..3 {
                let _ = drafter.draft(&dense, 7, p, taps, chunk, 0, n, s).expect("warm");
            }
            let iters = 15;
            let t0 = Instant::now();
            for it in 0..iters {
                let _ = drafter.draft(&dense, (it % 16) as u32, p, taps, chunk, 0, n, s).expect("bench");
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            println!("[bench] P={p:5} n={n}: {ms:.3} ms/draft");
        }
    }

    // Per-kernel GPU breakdown at P=10240, n=7
    println!("\n[kernel-timing] P=10240, n=7 (avg over 5 drafts):");
    backend.set_kernel_timing(true).expect("ktiming on");
    let _ = backend.take_kernel_times();
    let k_iters = 5;
    for it in 0..k_iters {
        let _ = drafter.draft(&dense, (it % 16) as u32, 10240, taps, chunk, 0, 7, s).expect("ktimed");
    }
    backend.sync_stream(s).expect("sync");
    let mut kt = backend.take_kernel_times();
    backend.set_kernel_timing(false).expect("ktiming off");
    kt.sort_by(|a, b| b.2.total_cmp(&a.2));
    let total_ms: f64 = kt.iter().map(|x| x.2).sum();
    println!("  TOTAL GPU kernel time: {:.3} ms/draft", total_ms / k_iters as f64);
    for (name, cnt, ms) in kt {
        println!(
            "  {:<32} {:>5.1}x  {:>7.3} ms/draft  ({:>6.1} us/call)",
            name,
            cnt as f64 / k_iters as f64,
            ms / k_iters as f64,
            ms * 1000.0 / cnt as f64
        );
    }
}

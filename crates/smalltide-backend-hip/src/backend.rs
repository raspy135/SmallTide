//! `HipBackend` implementing the backend-agnostic `smalltide_core::ComputeBackend` trait.
//!
//! All ROCm/HIP handles (`*mut c_void`, `HipStream`, `HipEvent`, `HipGraphExec`) remain strictly
//! internal to this crate behind opaque integer handles (`DeviceBuf`, `PinnedBuf`, `Stream`, `Event`, `Graph`).

use crate::ffi::{
    check_hip, CoreKernels, HipEvent, HipGraph, HipGraphExec, HipRuntime, HipStream,
    HIP_ERROR_NOT_READY, HIP_EVENT_DISABLE_TIMING, HIP_HOST_MALLOC_DEFAULT,
    HIP_MEMCPY_DEVICE_TO_DEVICE, HIP_MEMCPY_DEVICE_TO_HOST,
    HIP_MEMCPY_HOST_TO_DEVICE, HIP_STREAM_CAPTURE_MODE_GLOBAL, HIP_STREAM_NON_BLOCKING,
};
use smalltide_core::backend::{AttnPrepParams, FlashDecodeParams, FusedMixParams, GemvSegment, GroupedMoeParams, MoeDispatchParams, PrefillAttnParams};
use smalltide_core::{
    ComputeBackend, DType, DeviceBuf, DeviceInfo, Event, Graph, PinnedBuf, QueueKind, Result,
    SmallTideError, Stream,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

struct DeviceAlloc {
    ptr: *mut c_void,
    size_bytes: usize,
    /// Device view of host-mapped pinned memory (owned by a PinnedBuf; never hipFree'd).
    borrowed: bool,
}
unsafe impl Send for DeviceAlloc {}

struct PinnedAlloc {
    ptr: *mut c_void,
    size_bytes: usize,
}
unsafe impl Send for PinnedAlloc {}

/// Reusable pinned bounce buffer for `memcpy_d2h_sync` so small/medium readbacks do one
/// stream-ordered `hipMemcpyAsync` + `hipStreamSynchronize` instead of two GPU syncs.
struct D2hBounce {
    ptr: *mut c_void,
    cap: usize,
}
unsafe impl Send for D2hBounce {}

struct GraphEntry {
    graph: HipGraph,
    exec: HipGraphExec,
}
unsafe impl Send for GraphEntry {}

struct StreamEntry(HipStream);
unsafe impl Send for StreamEntry {}

struct EventEntry(HipEvent);
unsafe impl Send for EventEntry {}

pub struct HipBackend {
    rt: HipRuntime,
    kernels: CoreKernels,
    device_id: i32,
    next_id: AtomicU64,
    dev_bufs: Mutex<HashMap<u64, DeviceAlloc>>,
    pinned_bufs: Mutex<HashMap<u64, PinnedAlloc>>,
    streams: Mutex<HashMap<u64, StreamEntry>>,
    events: Mutex<HashMap<u64, EventEntry>>,
    graphs: Mutex<HashMap<u64, GraphEntry>>,
    /// f16 copy of a many-row GEMM's activations (`gemm_fp8_rows`), grown on demand.
    gemm_x16: Mutex<Option<DeviceBuf>>,
    /// Reusable pinned bounce buffer for single-sync `memcpy_d2h_sync` (up to 1 MiB).
    d2h_bounce: Mutex<Option<D2hBounce>>,
    /// Host-side instrumentation (see `HipBackend::take_host_stats`).
    stat_launches: AtomicU64,
    /// Kernel launches since the backend was created (never reset: `launch_epoch`).
    launch_epoch: AtomicU64,
    seg_kernels: bool,
    stat_launch_ns: AtomicU64,
    stat_sync_ns: AtomicU64,
    stat_syncs: AtomicU64,
    kernel_timer: Mutex<Option<KernelTimer>>,
    seg_timer: Mutex<Option<SegTimer>>,
    /// Zero (touch) every new device allocation so it is committed - and counted by the
    /// residency query - right away (load phase; see `set_touch_on_alloc`).
    touch_on_alloc: std::sync::atomic::AtomicBool,
    /// Physical allocations for paged memory: handle id -> (hip handle, bytes).
    vmm_handles: Mutex<HashMap<u64, (usize, usize)>>,
    #[cfg(windows)]
    wddm: Option<crate::wddm::Adapter>,
}

/// Coarse GPU timeline segments (timing events at `seg_mark` points); dev profiling only.
struct SegTimer {
    events: Vec<HipEvent>,
    marks: Vec<&'static str>,
    totals: HashMap<&'static str, (u64, f64)>,
}
unsafe impl Send for SegTimer {}

/// Per-kernel GPU timing (timing events around every launch); dev/profiling only.
struct KernelTimer {
    events: Vec<HipEvent>,
    cursor: usize,
    pending: Vec<(&'static str, usize)>,
    totals: HashMap<&'static str, (u64, f64)>,
}
unsafe impl Send for KernelTimer {}

/// Host-side cost counters since the last `take_host_stats`.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostStats {
    pub launches: u64,
    pub launch_ms: f64,
    pub syncs: u64,
    pub sync_wait_ms: f64,
}

unsafe impl Send for HipBackend {}
unsafe impl Sync for HipBackend {}

#[repr(C)]
struct CtrlMultiC {
    out: [*mut c_void; 4],
    cur: [*mut c_void; 4],
    msg: [*mut c_void; 4],
    gain: [*mut c_void; 4],
    forget: [*mut c_void; 4],
    norm_w: [*mut c_void; 4],
    norm_out: [*mut c_void; 4],
    hidden: u32,
    eps: f32,
}

impl HipBackend {
    fn vmm_prop(&self) -> crate::ffi::HipMemAllocationProp {
        crate::ffi::HipMemAllocationProp {
            kind: 1, // hipMemAllocationTypePinned
            requested_handle_type: 0,
            location_type: 1, // hipMemLocationTypeDevice
            location_id: self.device_id,
            win32_handle_meta_data: ptr::null_mut(),
            compression_type: 0,
            gpu_direct_rdma_capable: 0,
            usage: 0,
        }
    }

    fn ctrl_args(&self, hidden: usize, eps: f32) -> CtrlMultiC {
        let z = [ptr::null_mut(); 4];
        CtrlMultiC { out: z, cur: z, msg: z, gain: z, forget: z, norm_w: z, norm_out: z, hidden: hidden as u32, eps }
    }

    /// Whether new device allocations are zeroed (committed) immediately. On during model load
    /// so residency measurements see every buffer; turn off once the engine is built.
    pub fn set_touch_on_alloc(&self, on: bool) {
        self.touch_on_alloc.store(on, Ordering::Relaxed);
    }

    fn gemv_multi_impl(&self, segs: &[GemvSegment], rows: Option<u32>, stream: Stream) -> Result<()> {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct SegC {
            w: *const c_void,
            scale: *const c_void,
            x: *const c_void,
            y: *mut c_void,
            n_rows: u32,
            k_cols: u32,
            is_f32: u32,
            row_start: u32,
        }
        #[repr(C)]
        struct ArgsC {
            seg: [SegC; 8],
            n_seg: u32,
            total_rows: u32,
        }
        let empty = SegC { w: ptr::null(), scale: ptr::null(), x: ptr::null(), y: ptr::null_mut(), n_rows: 0, k_cols: 0, is_f32: 0, row_start: u32::MAX };
        for chunk in segs.chunks(8) {
            let mut a = ArgsC { seg: [empty; 8], n_seg: chunk.len() as u32, total_rows: 0 };
            for (i, sg) in chunk.iter().enumerate() {
                if sg.k_cols % 16 != 0 {
                    return Err(SmallTideError::Backend { backend: "hip", message: format!("gemv_multi: k_cols {} not a multiple of 16", sg.k_cols) });
                }
                a.seg[i] = SegC {
                    w: self.dev_ptr(sg.w)? as *const c_void,
                    scale: match sg.row_scales { Some(r) => self.dev_ptr(r)? as *const c_void, None => ptr::null() },
                    x: self.dev_ptr(sg.x)? as *const c_void,
                    y: self.dev_ptr(sg.y)?,
                    n_rows: sg.n_rows as u32,
                    k_cols: sg.k_cols as u32,
                    is_f32: if sg.is_bf16 { 2 } else { sg.row_scales.is_none() as u32 },
                    row_start: a.total_rows,
                };
                a.total_rows += sg.n_rows as u32;
            }
            match rows {
                None => {
                    let mut args = [p(&mut a)];
                    self.launch(self.kernels.gemv_multi, (a.total_rows.div_ceil(4), 1, 1), 128, &mut args, stream, "launch(gemv_multi)")?;
                }
                Some(r) => {
                    let mut rr = r;
                    let mut args = [p(&mut a), p(&mut rr)];
                    let no_wmma = std::env::var_os("SMALLTIDE_NO_WMMA_MULTI").is_some();
                    let can_wmma_fp8 = r >= 4
                        && !no_wmma
                        && chunk.iter().all(|sg| !sg.is_bf16 && sg.row_scales.is_some() && sg.n_rows % 16 == 0 && sg.k_cols % 32 == 0);
                    let can_wmma_bf16 = r >= 4
                        && !no_wmma
                        && chunk.iter().all(|sg| sg.is_bf16 && sg.n_rows % 16 == 0 && sg.k_cols % 32 == 0);
                    if can_wmma_fp8 {
                        let kfn = if r <= 16 {
                            self.kernels.gemv_multi_wmma
                        } else if r <= 32 {
                            self.kernels.gemv_multi_wmma_m32
                        } else {
                            self.kernels.gemv_multi_wmma_m64
                        };
                        self.launch(kfn, (a.total_rows.div_ceil(32), 1, 1), 64, &mut args, stream, "launch(gemv_multi_wmma)")?;
                    } else if can_wmma_bf16 {
                        let kfn = if r <= 16 {
                            self.kernels.gemv_multi_wmma_bf16
                        } else if r <= 32 {
                            self.kernels.gemv_multi_wmma_bf16_m32
                        } else {
                            self.kernels.gemv_multi_wmma_bf16_m64
                        };
                        self.launch(kfn, (a.total_rows.div_ceil(32), 1, 1), 64, &mut args, stream, "launch(gemv_multi_wmma_bf16)")?;
                    } else {
                        self.launch(self.kernels.gemv_multi_rows, (a.total_rows.div_ceil(4), 1, 1), 128, &mut args, stream, "launch(gemv_multi_rows)")?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Imports a shared D3D12 buffer (NT handle) as device memory. The returned buffer is
    /// borrowed: the D3D12 resource must outlive it.
    #[cfg(windows)]
    pub fn import_d3d12_buffer(&self, buf: &crate::d3d12::D3d12Buffer) -> Result<DeviceBuf> {
        #[repr(C)]
        struct HandleDesc {
            ty: i32,
            handle: *mut c_void,
            name: *const c_void,
            size: u64,
            flags: u32,
            reserved: [u32; 16],
        }
        #[repr(C)]
        struct BufferDesc {
            offset: u64,
            size: u64,
            flags: u32,
            reserved: [u32; 16],
        }
        type FnImport = unsafe extern "C" fn(*mut *mut c_void, *const HandleDesc) -> i32;
        type FnMap = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const BufferDesc) -> i32;
        let lib = unsafe { libloading::Library::new("amdhip64_7.dll") }
            .map_err(|e| SmallTideError::Backend { backend: "hip", message: format!("reload amdhip64_7.dll: {e}") })?;
        let (import, map): (FnImport, FnMap) = unsafe {
            let a = lib.get::<FnImport>(b"hipImportExternalMemory\0").map_err(|e| SmallTideError::Backend { backend: "hip", message: format!("{e}") })?;
            let b = lib.get::<FnMap>(b"hipExternalMemoryGetMappedBuffer\0").map_err(|e| SmallTideError::Backend { backend: "hip", message: format!("{e}") })?;
            (*a, *b)
        };
        std::mem::forget(lib); // already loaded by the runtime; keep the symbols valid
        const D3D12_RESOURCE: i32 = 5;
        const DEDICATED: u32 = 1;
        let hd = HandleDesc { ty: D3D12_RESOURCE, handle: buf.handle, name: ptr::null(), size: buf.size, flags: DEDICATED, reserved: [0; 16] };
        let mut ext: *mut c_void = ptr::null_mut();
        unsafe { check_hip(import(&mut ext, &hd), "hipImportExternalMemory")? };
        let bd = BufferDesc { offset: 0, size: buf.size, flags: 0, reserved: [0; 16] };
        let mut dptr: *mut c_void = ptr::null_mut();
        unsafe { check_hip(map(&mut dptr, ext, &bd), "hipExternalMemoryGetMappedBuffer")? };
        let size_bytes = buf.size as usize;
        let did = self.next_handle();
        self.dev_bufs.lock().unwrap().insert(did, DeviceAlloc { ptr: dptr, size_bytes, borrowed: true });
        Ok(DeviceBuf { id: did, offset_bytes: 0, size_bytes })
    }

    /// Backend on the device chosen by `HipRuntime::select_device` (env `SMALLTIDE_HIP_DEVICE`).
    /// Kernel launches so far, from any thread: unchanged between two reads means the engine
    /// queued no GPU work in between.
    pub fn launch_epoch(&self) -> u64 {
        self.launch_epoch.load(Ordering::Relaxed)
    }

    pub fn new_auto() -> Result<Self> {
        let rt = HipRuntime::load()?;
        let d = rt.select_device()?;
        drop(rt);
        Self::new(d)
    }

    pub fn new(device_id: i32) -> Result<Self> {
        let rt = HipRuntime::load()?;
        let _ = rt.probe_device(device_id)?;
        let kernels = rt.load_core_kernels(device_id)?;
        Ok(Self {
            rt,
            kernels,
            device_id,
            next_id: AtomicU64::new(1),
            dev_bufs: Mutex::new(HashMap::new()),
            pinned_bufs: Mutex::new(HashMap::new()),
            streams: Mutex::new(HashMap::new()),
            events: Mutex::new(HashMap::new()),
            graphs: Mutex::new(HashMap::new()),
            gemm_x16: Mutex::new(None),
            d2h_bounce: Mutex::new(None),
            stat_launches: AtomicU64::new(0),
            launch_epoch: AtomicU64::new(0),
            seg_kernels: std::env::var_os("SMALLTIDE_SEGTIME_KERNELS").is_some(),
            stat_launch_ns: AtomicU64::new(0),
            stat_sync_ns: AtomicU64::new(0),
            stat_syncs: AtomicU64::new(0),
            kernel_timer: Mutex::new(None),
            seg_timer: Mutex::new(None),
            touch_on_alloc: std::sync::atomic::AtomicBool::new(true),
            vmm_handles: Mutex::new(HashMap::new()),
            #[cfg(windows)]
            wddm: {
                let a = crate::wddm::Adapter::open_amd();
                // Ask the OS to keep what it can of our working set resident (best effort).
                if let Some(a) = &a {
                    if let Some(i) = a.local_info() {
                        let _ = a.reserve_local(i.available_for_reservation);
                    }
                }
                a
            },
        })
    }

    /// Windows: `(process local usage, how much this process can keep resident)` in bytes.
    ///
    /// Allocations past the resident amount are silently backed by system memory (GPU reads over
    /// PCIe, ~28 GB/s) although `hipMemGetInfo` counts them as VRAM. The resident amount is the
    /// dedicated size minus what other processes hold resident (adapter-wide segment statistics),
    /// capped by the DXGI budget. Radeon drivers 26.5.1-26.9.1 had a KMD segment bug that capped
    /// residency near half the card (ROCm/TheRock#7221); there `SMALLTIDE_RESIDENT_CAP=reservation`
    /// restores the old DXGI-reservation estimate.
    pub fn resident_vram_info(&self) -> Option<(u64, u64)> {
        #[cfg(windows)]
        {
            let a = self.wddm.as_ref()?;
            let i = a.local_info()?;
            let reservation = i.current_reservation.max(i.available_for_reservation);
            if std::env::var("SMALLTIDE_RESIDENT_CAP").ok().as_deref() == Some("reservation") {
                return Some((i.current_usage, reservation));
            }
            let resident = match a.adapter_resident_bytes() {
                Some(all) => {
                    let others = all.saturating_sub(i.current_usage);
                    (a.dedicated_bytes as u64).saturating_sub(others).min(i.budget)
                }
                None => reservation,
            };
            Some((i.current_usage, resident))
        }
        // Linux: (device-wide VRAM used, total) from amdgpu sysfs, see `linux::amdgpu_vram_used_total`.
        #[cfg(not(windows))]
        {
            let total = self.device_info().ok()?.total_vram_bytes as u64;
            crate::linux::amdgpu_vram_used_total(total)
        }
    }

    /// Enables coarse segment timing (see `ComputeBackend::seg_mark`).
    pub fn set_segment_timing(&self, on: bool) -> Result<()> {
        let mut g = self.seg_timer.lock().unwrap();
        if on && g.is_none() {
            let mut events = Vec::with_capacity(4096);
            for _ in 0..4096 {
                let mut e: HipEvent = ptr::null_mut();
                unsafe { check_hip((self.rt.hip_event_create_with_flags)(&mut e, 0), "hipEventCreateWithFlags")? };
                events.push(e);
            }
            *g = Some(SegTimer { events, marks: Vec::new(), totals: HashMap::new() });
        }
        Ok(())
    }

    fn flush_segments(&self, t: &mut SegTimer) {
        let n = t.marks.len();
        if n >= 2 {
            unsafe { let _ = (self.rt.hip_event_synchronize)(t.events[n - 1]); }
            for i in 0..n - 1 {
                let mut ms = 0.0f32;
                unsafe { let _ = (self.rt.hip_event_elapsed_time)(&mut ms, t.events[i], t.events[i + 1]); }
                let e = t.totals.entry(t.marks[i]).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += ms as f64;
            }
        }
        // keep the last mark as the start of the next batch
        if n > 0 {
            let last = t.marks[n - 1];
            t.events.swap(0, n - 1);
            t.marks.clear();
            t.marks.push(last);
        }
    }

    /// Returns and resets `(segment, count, total GPU ms)` sorted by time (drops the open segment).
    pub fn take_segment_times(&self) -> Vec<(&'static str, u64, f64)> {
        let mut g = self.seg_timer.lock().unwrap();
        let Some(t) = g.as_mut() else { return Vec::new() };
        self.flush_segments(t);
        t.marks.clear();
        let mut v: Vec<_> = t.totals.drain().map(|(k, (n, ms))| (k, n, ms)).collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        v
    }

    fn kernel_name(&self, f: crate::ffi::HipFunction) -> &'static str {
        let k = &self.kernels;
        macro_rules! names {
            ($($field:ident),*) => {{ $( if f == k.$field { return stringify!($field); } )* "unknown" }};
        }
        names!(noop, rmsnorm_f32, gemv_fp8_e4m3_f32, gemv_nvfp4_f32, gemv_mxfp8_f32, moe_w13_swiglu_indirect, moe_w2_accum_indirect, router_topk8_f32, sconv_k4_decode_f32, kv_store_fp8_decode, flash_decode_fp8_split, flash_decode_fp8_reduce, head_l2_clip_f32, value_norm_no_gamma_f32, gate_attn_sigmoid_mul_f32, attn_res_mix_f32, gemv_f32, rel_table_f32, loop_control_prep_f32, loop_control_apply_f32, vec_add_f32, vec_scale_f32, vec_copy_f32, zero_f32, rescue_write_route, rescue_apply_patch, rescue_wait_and_patch, final_logit_softcap_f32, argmax_partial_f32, gemm_fp8_e4m3_batched_f32, gemm_f32_rows, attn_res_mix_rows_f32, head_l2_clip_rows_f32, sconv_k4_rows_f32, kv_store_fp8_rows, attn_prefill_fp8, router_topk8_rows, loop_control_prep_rows_f32, moe_w13_grouped, moe_w2_grouped, moe_reduce_rows, moe_w13_decode, moe_w2_decode, moe_post_decode, moe_w13_decode3, moe_w2_decode3, gemv_multi, attn_prep_decode, fused_mix_decode, moe_w13_rows, moe_w13_grouped_wmma, moe_w2_grouped_wmma, moe_w13_grouped_wmma2, moe_w2_grouped_wmma2, moe_w13_grouped_wmma_m64, moe_w2_grouped_wmma_m64, f32_to_f16, attn_prefill_wmma, gemm_fp8_rows_wmma, gemm_fp8_rows_wmma_x16, gemm_fp8_rows_wmma_x16_m32, gemm_fp8_rows_wmma_x16_m64, gemm_fp8_rows_wmma_small, gemv_fp8_rows, sconv_rewind, argmax_rows, moe_group_rows, moe_post_rows, gemv_fp8_rows2, moe_group_rows2, gemv_multi_rows, gemv_multi_wmma, gemv_multi_wmma_m32, gemv_multi_wmma_m64, attn_prep_rows, moe_post_sconv_rows, mix_rows, ctrl_prep_multi, ctrl_apply_norm_multi, moe_group_rows3, dspark_mix_rms, dspark_head_norm_rope, dspark_ctx_store, dspark_attn, silu_mul, dspark_z, dspark_cell, dspark_logits_argmax, dspark_argmax_final, dspark_argmax_final_margin, argmax_rows_part, argmax_rows_reduce, attn_prefill_wmma_split, attn_prefill_wmma_reduce, attn_prefill_wmma_vt, kv_v_tiles_f16, dspark_quant_fold_fp8, dspark_logits_argmax_fp8, dspark_rnn_step, dspark_ctx_store_fp8, dspark_attn_fp8_split, dspark_attn_reduce, dspark_embed_prep, add_rmsnorm_rows, dspark_qk_norm_rope, moe_w2_rows, dflash2_attn_fp8_split, dflash2_conv_prepare, dflash2_conv_finish_add_rmsnorm, topk16_rows_part, topk16_rows_reduce, vit_gemm_bf16, vit_epilogue, vit_ln, vit_rope, vit_attn)
    }

    /// Enables/disables per-kernel GPU timing (adds 2 timing events per launch).
    pub fn set_kernel_timing(&self, on: bool) -> Result<()> {
        let mut g = self.kernel_timer.lock().unwrap();
        if on && g.is_none() {
            let mut events = Vec::with_capacity(8192);
            for _ in 0..8192 {
                let mut e: HipEvent = ptr::null_mut();
                unsafe { check_hip((self.rt.hip_event_create_with_flags)(&mut e, 0), "hipEventCreateWithFlags")? };
                events.push(e);
            }
            *g = Some(KernelTimer { events, cursor: 0, pending: Vec::new(), totals: HashMap::new() });
        } else if !on {
            if let Some(t) = g.take() {
                for e in t.events {
                    unsafe { let _ = (self.rt.hip_event_destroy)(e); }
                }
            }
        }
        Ok(())
    }

    fn flush_timer(&self, t: &mut KernelTimer) {
        if let Some(&(_, last)) = t.pending.last() {
            unsafe { let _ = (self.rt.hip_event_synchronize)(t.events[last + 1]); }
        }
        for (name, i) in t.pending.drain(..) {
            let mut ms = 0.0f32;
            unsafe { let _ = (self.rt.hip_event_elapsed_time)(&mut ms, t.events[i], t.events[i + 1]); }
            let e = t.totals.entry(name).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ms as f64;
        }
        t.cursor = 0;
    }

    /// Returns and resets accumulated `(kernel, launches, total GPU ms)`, sorted by time.
    pub fn take_kernel_times(&self) -> Vec<(&'static str, u64, f64)> {
        let mut g = self.kernel_timer.lock().unwrap();
        let Some(t) = g.as_mut() else { return Vec::new() };
        self.flush_timer(t);
        let mut v: Vec<_> = t.totals.drain().map(|(k, (n, ms))| (k, n, ms)).collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        v
    }

    /// Launches `f` with the given argument pointers (each must point at the argument value).
    fn launch(
        &self,
        f: crate::ffi::HipFunction,
        grid: (u32, u32, u32),
        block: u32,
        args: &mut [*mut c_void],
        stream: Stream,
        what: &'static str,
    ) -> Result<()> {
        let s = self.raw_stream(stream)?;
        unsafe {
            check_hip(
                self.launch_raw(f, grid.0, grid.1, grid.2, block, 1, 1, 0, s, args.as_mut_ptr(), ptr::null_mut()),
                what,
            )
        }
    }

    /// Returns and resets host-side launch / sync-wait counters.
    pub fn take_host_stats(&self) -> HostStats {
        HostStats {
            launches: self.stat_launches.swap(0, Ordering::Relaxed),
            launch_ms: self.stat_launch_ns.swap(0, Ordering::Relaxed) as f64 * 1e-6,
            syncs: self.stat_syncs.swap(0, Ordering::Relaxed),
            sync_wait_ms: self.stat_sync_ns.swap(0, Ordering::Relaxed) as f64 * 1e-6,
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    unsafe fn launch_raw(
        &self,
        f: crate::ffi::HipFunction,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        shmem: u32,
        s: HipStream,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> crate::ffi::HipError {
        self.bind();
        if self.seg_kernels {
            // Dev (SMALLTIDE_SEGTIME_KERNELS): one timeline mark per kernel, so the segment
            // timer reports in-stream time per kernel with no sync per launch.
            let mut g = self.seg_timer.lock().unwrap();
            if let Some(t) = g.as_mut() {
                if t.marks.len() + 1 >= t.events.len() {
                    self.flush_segments(t);
                }
                let i = t.marks.len();
                let _ = (self.rt.hip_event_record)(t.events[i], s);
                t.marks.push(self.kernel_name(f));
            }
        }
        let t0 = std::time::Instant::now();
        let mut timer = self.kernel_timer.lock().unwrap();
        let slot = timer.as_mut().map(|t| {
            if t.cursor + 2 > t.events.len() {
                self.flush_timer(t);
            }
            let i = t.cursor;
            t.cursor += 2;
            let _ = (self.rt.hip_event_record)(t.events[i], s);
            i
        });
        let r = (self.rt.hip_module_launch_kernel)(f, gx, gy, gz, bx, by, bz, shmem, s, params, extra);
        if let (Some(t), Some(i)) = (timer.as_mut(), slot) {
            let _ = (self.rt.hip_event_record)(t.events[i + 1], s);
            t.pending.push((self.kernel_name(f), i));
        }
        drop(timer);
        self.stat_launch_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.stat_launches.fetch_add(1, Ordering::Relaxed);
        self.launch_epoch.fetch_add(1, Ordering::Relaxed);
        r
    }

    fn next_handle(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// HIP's current device is per thread: make sure this thread targets our device before any
    /// call that uses it implicitly (allocation, stream/event creation, launches).
    #[inline]
    fn bind(&self) {
        thread_local! {
            static CUR_DEVICE: std::cell::Cell<i32> = const { std::cell::Cell::new(-1) };
        }
        CUR_DEVICE.with(|c| {
            if c.get() != self.device_id {
                unsafe {
                    let _ = (self.rt.hip_set_device)(self.device_id);
                }
                c.set(self.device_id);
            }
        });
    }

    fn dev_ptr(&self, buf: DeviceBuf) -> Result<*mut c_void> {
        self.bind();
        let guard = self.dev_bufs.lock().unwrap();
        let a = guard.get(&buf.id).ok_or_else(|| SmallTideError::Backend {
            backend: "hip",
            message: format!("invalid DeviceBuf id={}", buf.id),
        })?;
        if buf.offset_bytes + buf.size_bytes > a.size_bytes {
            return Err(SmallTideError::Backend {
                backend: "hip",
                message: format!(
                    "DeviceBuf slice out of bounds: {} + {} > {}",
                    buf.offset_bytes, buf.size_bytes, a.size_bytes
                ),
            });
        }
        Ok(unsafe { (a.ptr as *mut u8).add(buf.offset_bytes) as *mut c_void })
    }

    fn opt_dev_ptr(&self, buf: Option<DeviceBuf>) -> Result<*mut c_void> {
        match buf {
            Some(b) => self.dev_ptr(b),
            None => Ok(ptr::null_mut()),
        }
    }

    fn pinned_entry(&self, buf: PinnedBuf) -> Result<(*mut c_void, usize)> {
        let guard = self.pinned_bufs.lock().unwrap();
        guard
            .get(&buf.id)
            .map(|a| (a.ptr, a.size_bytes))
            .ok_or_else(|| SmallTideError::Backend {
                backend: "hip",
                message: format!("invalid PinnedBuf id={}", buf.id),
            })
    }

    fn raw_stream(&self, stream: Stream) -> Result<HipStream> {
        if stream.0 == 0 {
            return Ok(ptr::null_mut());
        }
        let guard = self.streams.lock().unwrap();
        guard
            .get(&stream.0)
            .map(|s| s.0)
            .ok_or_else(|| SmallTideError::Backend {
                backend: "hip",
                message: format!("invalid Stream id={}", stream.0),
            })
    }

    fn raw_event(&self, event: Event) -> Result<HipEvent> {
        let guard = self.events.lock().unwrap();
        guard
            .get(&event.0)
            .map(|e| e.0)
            .ok_or_else(|| SmallTideError::Backend {
                backend: "hip",
                message: format!("invalid Event id={}", event.0),
            })
    }
}

impl Drop for HipBackend {
    fn drop(&mut self) {
        for (_, g) in self.graphs.get_mut().unwrap().drain() {
            unsafe {
                let _ = (self.rt.hip_graph_exec_destroy)(g.exec);
                let _ = (self.rt.hip_graph_destroy)(g.graph);
            }
        }
        for (_, e) in self.events.get_mut().unwrap().drain() {
            unsafe { let _ = (self.rt.hip_event_destroy)(e.0); }
        }
        for (_, s) in self.streams.get_mut().unwrap().drain() {
            unsafe { let _ = (self.rt.hip_stream_destroy)(s.0); }
        }
        if let Some(b) = self.d2h_bounce.get_mut().unwrap().take() {
            unsafe { let _ = (self.rt.hip_host_free)(b.ptr); }
        }
        for (_, p) in self.pinned_bufs.get_mut().unwrap().drain() {
            unsafe { let _ = (self.rt.hip_host_free)(p.ptr); }
        }
        for (_, d) in self.dev_bufs.get_mut().unwrap().drain() {
            if !d.borrowed {
                unsafe { let _ = (self.rt.hip_free)(d.ptr); }
            }
        }
        unsafe {
            let _ = (self.rt.hip_module_unload)(self.kernels.module);
        }
    }
}

impl ComputeBackend for HipBackend {
    fn device_info(&self) -> Result<DeviceInfo> {
        self.bind();
        let r = self.rt.probe_device(self.device_id)?;
        Ok(DeviceInfo {
            backend_name: "hip",
            device_name: r.device_name,
            arch_name: r.kernel_arch,
            total_vram_bytes: r.total_vram_bytes,
            free_vram_bytes: r.free_vram_bytes,
            supports_fp8_wmma: true,
            supports_graphs: true,
        })
    }

    fn alloc_device(&self, size_bytes: usize) -> Result<DeviceBuf> {
        self.bind();
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe { check_hip((self.rt.hip_malloc)(&mut ptr, size_bytes), "hipMalloc")? };
        let id = self.next_handle();
        self.dev_bufs
            .lock()
            .unwrap()
            .insert(id, DeviceAlloc { ptr, size_bytes, borrowed: false });
        let buf = DeviceBuf { id, offset_bytes: 0, size_bytes };
        if size_bytes >= 4 && self.touch_on_alloc.load(Ordering::Relaxed) {
            self.zero_f32(buf, size_bytes / 4, Stream(0))?;
            unsafe { check_hip((self.rt.hip_stream_synchronize)(ptr::null_mut()), "hipStreamSynchronize(touch)")? };
        }
        Ok(buf)
    }

    fn free_device(&self, buf: DeviceBuf) -> Result<()> {
        if buf.offset_bytes != 0 {
            return Err(SmallTideError::Backend {
                backend: "hip",
                message: "cannot free a DeviceBuf sub-slice".into(),
            });
        }
        if let Some(a) = self.dev_bufs.lock().unwrap().remove(&buf.id) {
            if !a.borrowed {
                unsafe { check_hip((self.rt.hip_free)(a.ptr), "hipFree")? };
            }
        }
        Ok(())
    }

    fn vmm_granularity(&self) -> Option<usize> {
        let v = self.rt.vmm?;
        if std::env::var("SMALLTIDE_KV_PAGED").map_or(false, |x| x.trim() == "0") {
            return None;
        }
        self.bind();
        let prop = self.vmm_prop();
        let mut g = 0usize;
        let r = unsafe { (v.granularity)(&mut g, &prop, 0) };
        (r == 0 && g > 0).then_some(g)
    }

    fn vmm_reserve(&self, size: usize) -> Result<DeviceBuf> {
        let v = self.rt.vmm.ok_or_else(|| SmallTideError::Unsupported("hip vmm".into()))?;
        self.bind();
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe { check_hip((v.reserve)(&mut ptr, size, 0, ptr::null_mut(), 0), "hipMemAddressReserve")? };
        let id = self.next_handle();
        // Address space only: never hipFree'd (it lives as long as the process).
        self.dev_bufs.lock().unwrap().insert(id, DeviceAlloc { ptr, size_bytes: size, borrowed: true });
        Ok(DeviceBuf { id, offset_bytes: 0, size_bytes: size })
    }

    fn vmm_create(&self, size: usize) -> Result<u64> {
        let v = self.rt.vmm.ok_or_else(|| SmallTideError::Unsupported("hip vmm".into()))?;
        self.bind();
        let prop = self.vmm_prop();
        let mut h: *mut c_void = ptr::null_mut();
        unsafe { check_hip((v.create)(&mut h, size, &prop, 0), "hipMemCreate")? };
        let id = self.next_handle();
        self.vmm_handles.lock().unwrap().insert(id, (h as usize, size));
        Ok(id)
    }

    fn vmm_map(&self, va: DeviceBuf, offset: usize, handle: u64) -> Result<()> {
        let v = self.rt.vmm.ok_or_else(|| SmallTideError::Unsupported("hip vmm".into()))?;
        let (h, size) = *self.vmm_handles.lock().unwrap().get(&handle).ok_or_else(|| SmallTideError::Backend {
            backend: "hip",
            message: format!("vmm_map: unknown handle {handle}"),
        })?;
        let base = self.dev_ptr(va)? as usize + offset;
        let desc = crate::ffi::HipMemAccessDesc { location_type: 1, location_id: self.device_id, flags: 3 };
        unsafe {
            check_hip((v.map)(base as *mut c_void, size, 0, h as *mut c_void, 0), "hipMemMap")?;
            check_hip((v.set_access)(base as *mut c_void, size, &desc, 1), "hipMemSetAccess")?;
        }
        Ok(())
    }

    fn vmm_unmap(&self, va: DeviceBuf, offset: usize, size: usize) -> Result<()> {
        let v = self.rt.vmm.ok_or_else(|| SmallTideError::Unsupported("hip vmm".into()))?;
        let base = self.dev_ptr(va)? as usize + offset;
        unsafe { check_hip((v.unmap)(base as *mut c_void, size), "hipMemUnmap") }
    }

    fn stream_wait_value_ge(&self, stream: Stream, flag: DeviceBuf, value: u32) -> Result<()> {
        const HIP_STREAM_WAIT_VALUE_GTE: u32 = 0x0;
        let f = self.rt.hip_stream_wait_value32.ok_or_else(|| SmallTideError::Unsupported("hipStreamWaitValue32".into()))?;
        let s = self.raw_stream(stream)?;
        let ptr = self.dev_ptr(flag)?;
        unsafe { check_hip(f(s, ptr, value, HIP_STREAM_WAIT_VALUE_GTE, 0xFFFF_FFFF), "hipStreamWaitValue32") }
    }

    fn alloc_host_signal(&self, size_bytes: usize) -> Result<(DeviceBuf, PinnedBuf)> {
        self.bind();
        const HIP_HOST_MALLOC_PORTABLE: u32 = 0x1;
        const HIP_HOST_MALLOC_MAPPED: u32 = 0x2;
        const HIP_HOST_MALLOC_COHERENT: u32 = 0x4000_0000;
        let mut hptr: *mut c_void = ptr::null_mut();
        let r = unsafe {
            (self.rt.hip_host_malloc)(
                &mut hptr,
                size_bytes,
                HIP_HOST_MALLOC_PORTABLE | HIP_HOST_MALLOC_MAPPED | HIP_HOST_MALLOC_COHERENT,
            )
        };
        if r != 0 || hptr.is_null() {
            static LOGGED_FALLBACK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !LOGGED_FALLBACK.swap(true, Ordering::Relaxed) {
                eprintln!("[hip] alloc_host_signal: HIP_HOST_MALLOC_COHERENT failed (hipError_t={r}); falling back to non-coherent mapped host memory");
            }
            return self.alloc_host_mapped(size_bytes);
        }
        static LOGGED_COHERENT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED_COHERENT.swap(true, Ordering::Relaxed) {
            eprintln!("[hip] alloc_host_signal: coherent mapped host memory active");
        }
        let mut dptr: *mut c_void = ptr::null_mut();
        let r2 = unsafe { (self.rt.hip_host_get_device_pointer)(&mut dptr, hptr, 0) };
        if r2 != 0 || dptr.is_null() {
            unsafe { let _ = (self.rt.hip_host_free)(hptr); }
            check_hip(r2, "hipHostGetDevicePointer(signal)")?;
            return Err(SmallTideError::Backend { backend: "hip", message: "null device pointer for coherent signal memory".into() });
        }
        let pid = self.next_handle();
        self.pinned_bufs.lock().unwrap().insert(pid, PinnedAlloc { ptr: hptr, size_bytes });
        let did = self.next_handle();
        self.dev_bufs
            .lock()
            .unwrap()
            .insert(did, DeviceAlloc { ptr: dptr, size_bytes, borrowed: true });
        Ok((DeviceBuf { id: did, offset_bytes: 0, size_bytes }, PinnedBuf { id: pid, size_bytes }))
    }

    fn rescue_write_route(
        &self,
        dst: DeviceBuf,
        src: DeviceBuf,
        n_words: usize,
        sentinel_idx: usize,
        ticket: u32,
        stream: Stream,
    ) -> Result<()> {
        let mut d_dst = self.dev_ptr(dst)?;
        let mut d_src = self.dev_ptr(src)?;
        let mut n_arg = n_words as u32;
        let mut s_arg = sentinel_idx as u32;
        let mut t_arg = ticket;
        let mut args = [p(&mut d_dst), p(&mut d_src), p(&mut n_arg), p(&mut s_arg), p(&mut t_arg)];
        self.launch(self.kernels.rescue_write_route, (1, 1, 1), 32, &mut args, stream, "launch(rescue_write_route)")
    }

    fn rescue_apply_patch(
        &self,
        slot_table: DeviceBuf,
        patch_hdr: DeviceBuf,
        full_table: DeviceBuf,
        total_entries: usize,
        probe: Option<DeviceBuf>,
        layer_id: usize,
        ticket: u32,
        stream: Stream,
    ) -> Result<()> {
        let mut d_tbl = self.dev_ptr(slot_table)?;
        let mut d_pat = self.dev_ptr(patch_hdr)?;
        let mut d_full = self.dev_ptr(full_table)?;
        let mut tot = total_entries as u32;
        let mut d_probe = self.opt_dev_ptr(probe)?;
        let mut lid = layer_id as u32;
        let mut tkt = ticket;
        let mut args = [
            p(&mut d_tbl),
            p(&mut d_pat),
            p(&mut d_full),
            p(&mut tot),
            p(&mut d_probe),
            p(&mut lid),
            p(&mut tkt),
        ];
        self.launch(self.kernels.rescue_apply_patch, (1, 1, 1), 256, &mut args, stream, "launch(rescue_apply_patch)")
    }

    fn rescue_wait_and_patch(
        &self,
        slot_table: DeviceBuf,
        flag: DeviceBuf,
        ticket: u32,
        patch_hdr: DeviceBuf,
        full_table: DeviceBuf,
        total_entries: usize,
        probe: Option<DeviceBuf>,
        layer_id: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_tbl = self.dev_ptr(slot_table)?;
        let mut d_flag = self.dev_ptr(flag)?;
        let mut tkt = ticket;
        let mut d_pat = self.dev_ptr(patch_hdr)?;
        let mut d_full = self.dev_ptr(full_table)?;
        let mut tot = total_entries as u32;
        let mut d_probe = self.opt_dev_ptr(probe)?;
        let mut lid = layer_id as u32;
        let mut args = [
            p(&mut d_tbl),
            p(&mut d_flag),
            p(&mut tkt),
            p(&mut d_pat),
            p(&mut d_full),
            p(&mut tot),
            p(&mut d_probe),
            p(&mut lid),
        ];
        self.launch(self.kernels.rescue_wait_and_patch, (1, 1, 1), 256, &mut args, stream, "launch(rescue_wait_and_patch)")
    }

    fn alloc_host_mapped(&self, size_bytes: usize) -> Result<(DeviceBuf, PinnedBuf)> {
        self.bind();
        const HIP_HOST_MALLOC_PORTABLE: u32 = 0x1;
        const HIP_HOST_MALLOC_MAPPED: u32 = 0x2;
        let mut hptr: *mut c_void = ptr::null_mut();
        unsafe {
            check_hip(
                (self.rt.hip_host_malloc)(&mut hptr, size_bytes, HIP_HOST_MALLOC_PORTABLE | HIP_HOST_MALLOC_MAPPED),
                "hipHostMalloc(mapped)",
            )?;
        }
        let mut dptr: *mut c_void = ptr::null_mut();
        let r = unsafe { (self.rt.hip_host_get_device_pointer)(&mut dptr, hptr, 0) };
        if r != 0 || dptr.is_null() {
            unsafe { let _ = (self.rt.hip_host_free)(hptr); }
            check_hip(r, "hipHostGetDevicePointer")?;
            return Err(SmallTideError::Backend { backend: "hip", message: "null device pointer for mapped host memory".into() });
        }
        let pid = self.next_handle();
        self.pinned_bufs.lock().unwrap().insert(pid, PinnedAlloc { ptr: hptr, size_bytes });
        let did = self.next_handle();
        self.dev_bufs
            .lock()
            .unwrap()
            .insert(did, DeviceAlloc { ptr: dptr, size_bytes, borrowed: true });
        Ok((DeviceBuf { id: did, offset_bytes: 0, size_bytes }, PinnedBuf { id: pid, size_bytes }))
    }

    fn alloc_pinned(&self, size_bytes: usize) -> Result<PinnedBuf> {
        self.bind();
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe {
            check_hip(
                (self.rt.hip_host_malloc)(&mut ptr, size_bytes, HIP_HOST_MALLOC_DEFAULT),
                "hipHostMalloc",
            )?;
        }
        let id = self.next_handle();
        self.pinned_bufs
            .lock()
            .unwrap()
            .insert(id, PinnedAlloc { ptr, size_bytes });
        Ok(PinnedBuf { id, size_bytes })
    }

    fn free_pinned(&self, buf: PinnedBuf) -> Result<()> {
        if let Some(a) = self.pinned_bufs.lock().unwrap().remove(&buf.id) {
            unsafe { check_hip((self.rt.hip_host_free)(a.ptr), "hipHostFree")? };
        }
        Ok(())
    }

    unsafe fn pinned_as_mut_slice(&self, buf: PinnedBuf) -> Result<&mut [u8]> {
        let (ptr, len) = self.pinned_entry(buf)?;
        Ok(std::slice::from_raw_parts_mut(ptr as *mut u8, len))
    }

    fn create_stream(&self, _kind: QueueKind) -> Result<Stream> {
        self.bind();
        let mut s: HipStream = ptr::null_mut();
        unsafe {
            check_hip(
                (self.rt.hip_stream_create_with_flags)(&mut s, HIP_STREAM_NON_BLOCKING),
                "hipStreamCreateWithFlags",
            )?;
        }
        let id = self.next_handle();
        self.streams.lock().unwrap().insert(id, StreamEntry(s));
        Ok(Stream(id))
    }

    fn destroy_stream(&self, stream: Stream) -> Result<()> {
        if let Some(s) = self.streams.lock().unwrap().remove(&stream.0) {
            unsafe { check_hip((self.rt.hip_stream_destroy)(s.0), "hipStreamDestroy")? };
        }
        Ok(())
    }

    fn create_event(&self) -> Result<Event> {
        self.bind();
        let mut e: HipEvent = ptr::null_mut();
        unsafe {
            check_hip(
                (self.rt.hip_event_create_with_flags)(&mut e, HIP_EVENT_DISABLE_TIMING),
                "hipEventCreateWithFlags",
            )?;
        }
        let id = self.next_handle();
        self.events.lock().unwrap().insert(id, EventEntry(e));
        Ok(Event(id))
    }

    fn destroy_event(&self, event: Event) -> Result<()> {
        if let Some(e) = self.events.lock().unwrap().remove(&event.0) {
            unsafe { check_hip((self.rt.hip_event_destroy)(e.0), "hipEventDestroy")? };
        }
        Ok(())
    }

    fn record_event(&self, event: Event, stream: Stream) -> Result<()> {
        let e = self.raw_event(event)?;
        let s = self.raw_stream(stream)?;
        unsafe { check_hip((self.rt.hip_event_record)(e, s), "hipEventRecord") }
    }

    fn stream_wait_event(&self, stream: Stream, event: Event) -> Result<()> {
        let e = self.raw_event(event)?;
        let s = self.raw_stream(stream)?;
        unsafe { check_hip((self.rt.hip_stream_wait_event)(s, e, 0), "hipStreamWaitEvent") }
    }

    fn sync_event(&self, event: Event) -> Result<()> {
        let e = self.raw_event(event)?;
        unsafe { check_hip((self.rt.hip_event_synchronize)(e), "hipEventSynchronize") }
    }

    fn query_event(&self, event: Event) -> Result<bool> {
        let e = self.raw_event(event)?;
        let code = unsafe { (self.rt.hip_event_query)(e) };
        match code {
            0 => Ok(true),
            HIP_ERROR_NOT_READY => Ok(false),
            other => check_hip(other, "hipEventQuery").map(|_| false),
        }
    }

    fn seg_mark(&self, label: &'static str, stream: Stream) {
        let mut g = self.seg_timer.lock().unwrap();
        let Some(t) = g.as_mut() else { return };
        if t.marks.len() + 1 >= t.events.len() {
            self.flush_segments(t);
        }
        if let Ok(s) = self.raw_stream(stream) {
            let i = t.marks.len();
            unsafe { let _ = (self.rt.hip_event_record)(t.events[i], s); }
            t.marks.push(label);
        }
    }

    fn sync_stream(&self, stream: Stream) -> Result<()> {
        self.bind();
        let s = self.raw_stream(stream)?;
        let t0 = std::time::Instant::now();
        let r = unsafe { check_hip((self.rt.hip_stream_synchronize)(s), "hipStreamSynchronize") };
        self.stat_sync_ns.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.stat_syncs.fetch_add(1, Ordering::Relaxed);
        r
    }

    fn memcpy_d2h_async(
        &self,
        dst: PinnedBuf,
        dst_offset: usize,
        src: DeviceBuf,
        src_offset: usize,
        len: usize,
        stream: Stream,
    ) -> Result<()> {
        let d = self.dev_ptr(src)?;
        let (h, h_len) = self.pinned_entry(dst)?;
        if src_offset + len > src.size_bytes || dst_offset + len > h_len {
            return Err(SmallTideError::Backend {
                backend: "hip",
                message: "memcpy_d2h_async out of bounds".into(),
            });
        }
        let s = self.raw_stream(stream)?;
        unsafe {
            let src_p = (d as *const u8).add(src_offset) as *const c_void;
            let dst_p = (h as *mut u8).add(dst_offset) as *mut c_void;
            check_hip(
                (self.rt.hip_memcpy_async)(dst_p, src_p, len, HIP_MEMCPY_DEVICE_TO_HOST, s),
                "hipMemcpyAsync(D2H)",
            )
        }
    }

    fn memcpy_d2d_async(
        &self,
        dst: DeviceBuf,
        dst_offset: usize,
        src: DeviceBuf,
        src_offset: usize,
        len: usize,
        stream: Stream,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let d = self.dev_ptr(dst)?;
        let sp = self.dev_ptr(src)?;
        if src_offset + len > src.size_bytes || dst_offset + len > dst.size_bytes {
            return Err(SmallTideError::Backend {
                backend: "hip",
                message: "memcpy_d2d_async out of bounds".into(),
            });
        }
        let s = self.raw_stream(stream)?;
        unsafe {
            let src_p = (sp as *const u8).add(src_offset) as *const c_void;
            let dst_p = (d as *mut u8).add(dst_offset) as *mut c_void;
            check_hip(
                (self.rt.hip_memcpy_async)(dst_p, src_p, len, HIP_MEMCPY_DEVICE_TO_DEVICE, s),
                "hipMemcpyAsync(D2D)",
            )
        }
    }

    fn memcpy_h2d_async(
        &self,
        dst: DeviceBuf,
        dst_offset: usize,
        src: PinnedBuf,
        src_offset: usize,
        len: usize,
        stream: Stream,
    ) -> Result<()> {
        let d = self.dev_ptr(dst)?;
        let (h, h_len) = self.pinned_entry(src)?;
        if dst_offset + len > dst.size_bytes || src_offset + len > h_len {
            return Err(SmallTideError::Backend {
                backend: "hip",
                message: "memcpy_h2d_async out of bounds".into(),
            });
        }
        let s = self.raw_stream(stream)?;
        unsafe {
            let dst_p = (d as *mut u8).add(dst_offset) as *mut c_void;
            let src_p = (h as *const u8).add(src_offset) as *const c_void;
            check_hip(
                (self.rt.hip_memcpy_async)(dst_p, src_p, len, HIP_MEMCPY_HOST_TO_DEVICE, s),
                "hipMemcpyAsync(H2D)",
            )
        }
    }

    fn memcpy_d2h_sync(
        &self,
        dst: &mut [u8],
        src: DeviceBuf,
        src_offset: usize,
        stream: Stream,
    ) -> Result<()> {
        if dst.is_empty() {
            return self.sync_stream(stream);
        }
        let d = self.dev_ptr(src)?;
        if src_offset + dst.len() > src.size_bytes {
            return Err(SmallTideError::Backend {
                backend: "hip",
                message: "memcpy_d2h_sync out of bounds".into(),
            });
        }
        let src_p = unsafe { (d as *const u8).add(src_offset) as *const c_void };
        // Single-sync path for decode-sized readbacks (drafts, argmax, filtered logits, route logs):
        // stream-ordered async D2H into a reusable pinned bounce buffer + one hipStreamSynchronize,
        // avoiding a second blocking default-stream hipMemcpy round-trip on WDDM.
        const MAX_BOUNCE_BYTES: usize = 1 << 20;
        if dst.len() <= MAX_BOUNCE_BYTES {
            let s = self.raw_stream(stream)?;
            let mut guard = self.d2h_bounce.lock().unwrap();
            let need = dst.len().max(64 * 1024);
            if guard.as_ref().map_or(true, |b| b.cap < dst.len()) {
                let mut hptr: *mut c_void = ptr::null_mut();
                let r = unsafe { (self.rt.hip_host_malloc)(&mut hptr, need, HIP_HOST_MALLOC_DEFAULT) };
                if r == 0 && !hptr.is_null() {
                    if let Some(old) = guard.replace(D2hBounce { ptr: hptr, cap: need }) {
                        unsafe { let _ = (self.rt.hip_host_free)(old.ptr); }
                    }
                }
            }
            if let Some(b) = guard.as_ref().filter(|b| b.cap >= dst.len()) {
                unsafe {
                    check_hip(
                        (self.rt.hip_memcpy_async)(b.ptr, src_p, dst.len(), HIP_MEMCPY_DEVICE_TO_HOST, s),
                        "hipMemcpyAsync(D2H sync)",
                    )?;
                }
                self.sync_stream(stream)?;
                unsafe {
                    ptr::copy_nonoverlapping(b.ptr as *const u8, dst.as_mut_ptr(), dst.len());
                }
                return Ok(());
            }
        }
        self.sync_stream(stream)?;
        unsafe {
            check_hip(
                (self.rt.hip_memcpy)(
                    dst.as_mut_ptr() as *mut c_void,
                    src_p,
                    dst.len(),
                    HIP_MEMCPY_DEVICE_TO_HOST,
                ),
                "hipMemcpy(D2H)",
            )
        }
    }

    fn begin_graph_capture(&self, stream: Stream) -> Result<()> {
        self.bind();
        let s = self.raw_stream(stream)?;
        unsafe {
            check_hip(
                (self.rt.hip_stream_begin_capture)(s, HIP_STREAM_CAPTURE_MODE_GLOBAL),
                "hipStreamBeginCapture",
            )
        }
    }

    fn end_graph_capture(&self, stream: Stream) -> Result<Graph> {
        let s = self.raw_stream(stream)?;
        let mut graph: HipGraph = ptr::null_mut();
        let mut exec: HipGraphExec = ptr::null_mut();
        unsafe {
            check_hip((self.rt.hip_stream_end_capture)(s, &mut graph), "hipStreamEndCapture")?;
            check_hip(
                (self.rt.hip_graph_instantiate)(&mut exec, graph, ptr::null_mut(), ptr::null_mut(), 0),
                "hipGraphInstantiate",
            )?;
        }
        let id = self.next_handle();
        self.graphs
            .lock()
            .unwrap()
            .insert(id, GraphEntry { graph, exec });
        Ok(Graph(id))
    }

    fn launch_graph(&self, graph: Graph, stream: Stream) -> Result<()> {
        let s = self.raw_stream(stream)?;
        let exec = {
            let guard = self.graphs.lock().unwrap();
            guard
                .get(&graph.0)
                .map(|g| g.exec)
                .ok_or_else(|| SmallTideError::Backend {
                    backend: "hip",
                    message: format!("invalid Graph id={}", graph.0),
                })?
        };
        unsafe { check_hip((self.rt.hip_graph_launch)(exec, s), "hipGraphLaunch") }
    }

    fn destroy_graph(&self, graph: Graph) -> Result<()> {
        if let Some(g) = self.graphs.lock().unwrap().remove(&graph.0) {
            unsafe {
                let _ = (self.rt.hip_graph_exec_destroy)(g.exec);
                let _ = (self.rt.hip_graph_destroy)(g.graph);
            }
        }
        Ok(())
    }

    fn rms_norm(
        &self,
        out: DeviceBuf,
        x: DeviceBuf,
        weight: DeviceBuf,
        rows: usize,
        hidden: usize,
        eps: f32,
        stream: Stream,
    ) -> Result<()> {
        let mut d_o = self.dev_ptr(out)?;
        let mut d_x = self.dev_ptr(x)?;
        let mut d_w = self.dev_ptr(weight)?;
        let s = self.raw_stream(stream)?;
        let mut h_arg = hidden as u32;
        let mut eps_arg = eps;
        let mut params: [*mut c_void; 5] = [
            &mut d_o as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_w as *mut _ as *mut c_void,
            &mut h_arg as *mut _ as *mut c_void,
            &mut eps_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.rmsnorm_f32, rows as u32, 1, 1, if rows == 1 { 1024 } else { 256 }, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(rmsnorm_f32)",
            )
        }
    }

    fn matmul(
        &self,
        out: DeviceBuf,
        x: DeviceBuf,
        weight: DeviceBuf,
        weight_dtype: DType,
        m: usize,
        n: usize,
        k: usize,
        stream: Stream,
    ) -> Result<()> {
        if m != 1 || weight_dtype != DType::FP8E4M3 {
            return Err(SmallTideError::Unsupported(format!(
                "HipBackend::matmul currently implements m=1 FP8E4M3 (got m={m}, dtype={weight_dtype:?})"
            )));
        }
        let mut d_y = self.dev_ptr(out)?;
        let mut d_x = self.dev_ptr(x)?;
        let mut d_w = self.dev_ptr(weight)?;
        let mut d_s: *mut c_void = ptr::null_mut();
        let s = self.raw_stream(stream)?;
        let mut n_arg = n as u32;
        let mut k_arg = k as u32;
        let mut params: [*mut c_void; 6] = [
            &mut d_y as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_w as *mut _ as *mut c_void,
            &mut d_s as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
            &mut k_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.gemv_fp8_e4m3_f32, n_arg.div_ceil(4), 1, 1, 128, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(gemv_fp8_e4m3_f32)",
            )
        }
    }

    fn gemv_fp8(
        &self,
        out: DeviceBuf,
        x: DeviceBuf,
        w_fp8: DeviceBuf,
        row_scales: DeviceBuf,
        n_rows: usize,
        k_cols: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_y = self.dev_ptr(out)?;
        let mut d_x = self.dev_ptr(x)?;
        let mut d_w = self.dev_ptr(w_fp8)?;
        let mut d_s = self.dev_ptr(row_scales)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n_rows as u32;
        let mut k_arg = k_cols as u32;
        let mut params: [*mut c_void; 6] = [
            &mut d_y as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_w as *mut _ as *mut c_void,
            &mut d_s as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
            &mut k_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.gemv_fp8_e4m3_f32, n_arg.div_ceil(4), 1, 1, 128, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(gemv_fp8_e4m3_f32)",
            )
        }
    }

    fn gemv_f32(
        &self,
        out: DeviceBuf,
        x: DeviceBuf,
        w_f32: DeviceBuf,
        n_rows: usize,
        k_cols: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_y = self.dev_ptr(out)?;
        let mut d_x = self.dev_ptr(x)?;
        let mut d_w = self.dev_ptr(w_f32)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n_rows as u32;
        let mut k_arg = k_cols as u32;
        let mut params: [*mut c_void; 5] = [
            &mut d_y as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_w as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
            &mut k_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.gemv_f32, n_arg.div_ceil(4), 1, 1, 128, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(gemv_f32)",
            )
        }
    }

    fn moe_swiglu_indirect(
        &self,
        out: DeviceBuf,
        x: DeviceBuf,
        expert_pool: DeviceBuf,
        slot_table: DeviceBuf,
        topk_ids: DeviceBuf,
        topk_weights: DeviceBuf,
        params: MoeDispatchParams,
        stream: Stream,
    ) -> Result<()> {
        let mid_buf = self.alloc_device(params.top_k * params.intermediate_size * 4)?;
        let res = self.moe_swiglu_indirect_with_scratch(
            out, mid_buf, x, expert_pool, slot_table, topk_ids, topk_weights, params, stream,
        );
        let _ = self.sync_stream(stream);
        let _ = self.free_device(mid_buf);
        res
    }

    fn moe_swiglu_indirect_with_scratch(
        &self,
        out: DeviceBuf,
        mid_scratch: DeviceBuf,
        x: DeviceBuf,
        expert_pool: DeviceBuf,
        slot_table: DeviceBuf,
        topk_ids: DeviceBuf,
        topk_weights: DeviceBuf,
        params: MoeDispatchParams,
        stream: Stream,
    ) -> Result<()> {
        if params.batch_size != 1 {
            return Err(SmallTideError::Unsupported(
                "batched MoE GEMM (prefill/dspark) lands in Phase 2b".into(),
            ));
        }
        let mut d_mid = self.dev_ptr(mid_scratch)?;
        let mut d_out = self.dev_ptr(out)?;
        let mut d_x = self.dev_ptr(x)?;
        let mut d_pool = self.dev_ptr(expert_pool)?;
        let mut d_slots = self.dev_ptr(slot_table)?;
        let mut d_ids = self.dev_ptr(topk_ids)?;
        let mut d_w = self.dev_ptr(topk_weights)?;
        let s = self.raw_stream(stream)?;

        let mut stride_arg = params.expert_stride_bytes as u64;
        let mut h_arg = params.hidden_size as u32;
        let mut ff_arg = params.intermediate_size as u32;
        let mut gcap_arg: f32 = 4.0;
        let mut ucap_arg: f32 = 25.0;
        let mut k_arg = params.top_k as u32;

        let mut p1: [*mut c_void; 10] = [
            &mut d_mid as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_pool as *mut _ as *mut c_void,
            &mut d_slots as *mut _ as *mut c_void,
            &mut d_ids as *mut _ as *mut c_void,
            &mut stride_arg as *mut _ as *mut c_void,
            &mut h_arg as *mut _ as *mut c_void,
            &mut ff_arg as *mut _ as *mut c_void,
            &mut gcap_arg as *mut _ as *mut c_void,
            &mut ucap_arg as *mut _ as *mut c_void,
        ];
        let mut p2: [*mut c_void; 10] = [
            &mut d_out as *mut _ as *mut c_void,
            &mut d_mid as *mut _ as *mut c_void,
            &mut d_pool as *mut _ as *mut c_void,
            &mut d_slots as *mut _ as *mut c_void,
            &mut d_ids as *mut _ as *mut c_void,
            &mut d_w as *mut _ as *mut c_void,
            &mut stride_arg as *mut _ as *mut c_void,
            &mut h_arg as *mut _ as *mut c_void,
            &mut ff_arg as *mut _ as *mut c_void,
            &mut k_arg as *mut _ as *mut c_void,
        ];

        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.moe_w13_swiglu_indirect, ff_arg.div_ceil(4), k_arg, 1, 128, 1, 1, 0, s, p1.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(moe_w13_swiglu_indirect)",
            )?;
            check_hip(
                self.launch_raw(
                    self.kernels.moe_w2_accum_indirect, h_arg.div_ceil(4), 1, 1, 128, 1, 1, 0, s, p2.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(moe_w2_accum_indirect)",
            )
        }
    }

    fn router_topk8(
        &self,
        topk_ids: DeviceBuf,
        topk_weights: DeviceBuf,
        x: DeviceBuf,
        gate_w: DeviceBuf,
        gate_b: Option<DeviceBuf>,
        hidden_size: usize,
        route_scale: f32,
        stream: Stream,
    ) -> Result<()> {
        let mut d_ids = self.dev_ptr(topk_ids)?;
        let mut d_w = self.dev_ptr(topk_weights)?;
        let mut d_x = self.dev_ptr(x)?;
        let mut d_gw = self.dev_ptr(gate_w)?;
        let mut d_gb = self.opt_dev_ptr(gate_b)?;
        let s = self.raw_stream(stream)?;
        let mut h_arg = hidden_size as u32;
        let mut rs_arg = route_scale;
        let mut params: [*mut c_void; 7] = [
            &mut d_ids as *mut _ as *mut c_void,
            &mut d_w as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_gw as *mut _ as *mut c_void,
            &mut d_gb as *mut _ as *mut c_void,
            &mut h_arg as *mut _ as *mut c_void,
            &mut rs_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.router_topk8_f32, 1, 1, 1, 128, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(router_topk8_f32)",
            )
        }
    }

    fn sconv_k4_decode(
        &self,
        out: DeviceBuf,
        state: DeviceBuf,
        x: DeviceBuf,
        weight: DeviceBuf,
        dyn_taps: Option<DeviceBuf>,
        dim: usize,
        head_dim: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_o = self.dev_ptr(out)?;
        let mut d_st = self.dev_ptr(state)?;
        let mut d_x = self.dev_ptr(x)?;
        let mut d_w = self.dev_ptr(weight)?;
        let mut d_dyn = self.opt_dev_ptr(dyn_taps)?;
        let s = self.raw_stream(stream)?;
        let mut dim_arg = dim as u32;
        let mut hd_arg = head_dim as u32;
        let mut params: [*mut c_void; 7] = [
            &mut d_o as *mut _ as *mut c_void,
            &mut d_st as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut d_w as *mut _ as *mut c_void,
            &mut d_dyn as *mut _ as *mut c_void,
            &mut dim_arg as *mut _ as *mut c_void,
            &mut hd_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.sconv_k4_decode_f32, dim_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(sconv_k4_decode_f32)",
            )
        }
    }

    fn head_l2_clip(
        &self,
        x: DeviceBuf,
        scale: Option<DeviceBuf>,
        n_heads: usize,
        head_dim: usize,
        threshold: f32,
        stream: Stream,
    ) -> Result<()> {
        let mut d_x = self.dev_ptr(x)?;
        let mut d_sc = self.opt_dev_ptr(scale)?;
        let s = self.raw_stream(stream)?;
        let mut nh_arg = n_heads as u32;
        let mut hd_arg = head_dim as u32;
        let mut thr_arg = threshold;
        let mut params: [*mut c_void; 5] = [
            &mut d_x as *mut _ as *mut c_void,
            &mut d_sc as *mut _ as *mut c_void,
            &mut nh_arg as *mut _ as *mut c_void,
            &mut hd_arg as *mut _ as *mut c_void,
            &mut thr_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.head_l2_clip_f32, nh_arg, 1, 1, 128, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(head_l2_clip_f32)",
            )
        }
    }

    fn value_norm_no_gamma(
        &self,
        v: DeviceBuf,
        n_kv_heads: usize,
        head_dim: usize,
        eps: f32,
        stream: Stream,
    ) -> Result<()> {
        let mut d_v = self.dev_ptr(v)?;
        let s = self.raw_stream(stream)?;
        let mut nh_arg = n_kv_heads as u32;
        let mut hd_arg = head_dim as u32;
        let mut eps_arg = eps;
        let mut params: [*mut c_void; 4] = [
            &mut d_v as *mut _ as *mut c_void,
            &mut nh_arg as *mut _ as *mut c_void,
            &mut hd_arg as *mut _ as *mut c_void,
            &mut eps_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.value_norm_no_gamma_f32, nh_arg, 1, 1, 128, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(value_norm_no_gamma_f32)",
            )
        }
    }

    fn rel_table(
        &self,
        table: DeviceBuf,
        r: DeviceBuf,
        rel_proj: DeviceBuf,
        n_heads: usize,
        rel_extent: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_t = self.dev_ptr(table)?;
        let mut d_r = self.dev_ptr(r)?;
        let mut d_p = self.dev_ptr(rel_proj)?;
        let s = self.raw_stream(stream)?;
        let mut nh_arg = n_heads as u32;
        let mut ext_arg = rel_extent as u32;
        let mut params: [*mut c_void; 5] = [
            &mut d_t as *mut _ as *mut c_void,
            &mut d_r as *mut _ as *mut c_void,
            &mut d_p as *mut _ as *mut c_void,
            &mut nh_arg as *mut _ as *mut c_void,
            &mut ext_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.rel_table_f32, ext_arg.div_ceil(128), nh_arg, 1, 128, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(rel_table_f32)",
            )
        }
    }

    fn kv_store_fp8_decode(
        &self,
        k_cache: DeviceBuf,
        k_scales: DeviceBuf,
        v_cache: DeviceBuf,
        v_scales: DeviceBuf,
        k_in: DeviceBuf,
        v_in: DeviceBuf,
        n_kv_heads: usize,
        max_seq: usize,
        head_dim: usize,
        slot_pos: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_kc = self.dev_ptr(k_cache)?;
        let mut d_ks = self.dev_ptr(k_scales)?;
        let mut d_vc = self.dev_ptr(v_cache)?;
        let mut d_vs = self.dev_ptr(v_scales)?;
        let mut d_ki = self.dev_ptr(k_in)?;
        let mut d_vi = self.dev_ptr(v_in)?;
        let s = self.raw_stream(stream)?;
        let mut nkv_arg = n_kv_heads as u32;
        let mut max_arg = max_seq as u32;
        let mut hd_arg = head_dim as u32;
        let mut slot_arg = slot_pos as u32;
        let mut params: [*mut c_void; 10] = [
            &mut d_kc as *mut _ as *mut c_void,
            &mut d_ks as *mut _ as *mut c_void,
            &mut d_vc as *mut _ as *mut c_void,
            &mut d_vs as *mut _ as *mut c_void,
            &mut d_ki as *mut _ as *mut c_void,
            &mut d_vi as *mut _ as *mut c_void,
            &mut nkv_arg as *mut _ as *mut c_void,
            &mut max_arg as *mut _ as *mut c_void,
            &mut hd_arg as *mut _ as *mut c_void,
            &mut slot_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.kv_store_fp8_decode, nkv_arg, 1, 1, hd_arg, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(kv_store_fp8_decode)",
            )
        }
    }

    fn flash_decode_fp8(
        &self,
        out: DeviceBuf,
        partial_out: DeviceBuf,
        partial_m: DeviceBuf,
        partial_l: DeviceBuf,
        q: DeviceBuf,
        k_cache: DeviceBuf,
        k_scales: DeviceBuf,
        v_cache: DeviceBuf,
        v_scales: DeviceBuf,
        rel_table: Option<DeviceBuf>,
        sink_logit: Option<DeviceBuf>,
        params: FlashDecodeParams,
        stream: Stream,
    ) -> Result<()> {
        let mut d_out = self.dev_ptr(out)?;
        let mut d_po = self.dev_ptr(partial_out)?;
        let mut d_pm = self.dev_ptr(partial_m)?;
        let mut d_pl = self.dev_ptr(partial_l)?;
        let mut d_q = self.dev_ptr(q)?;
        let mut d_kc = self.dev_ptr(k_cache)?;
        let mut d_ks = self.dev_ptr(k_scales)?;
        let mut d_vc = self.dev_ptr(v_cache)?;
        let mut d_vs = self.dev_ptr(v_scales)?;
        let mut d_rel = self.opt_dev_ptr(rel_table)?;
        let mut d_sink = self.opt_dev_ptr(sink_logit)?;
        let s = self.raw_stream(stream)?;

        let mut nq_arg = params.n_q_heads as u32;
        let mut nkv_arg = params.n_kv_heads as u32;
        let mut max_arg = params.max_seq as u32;
        let mut qpos_arg = params.q_pos as u32;
        let mut rext_arg = params.rel_extent as u32;
        let mut loc_arg: u32 = if params.is_local { 1 } else { 0 };
        let mut splits_arg = params.num_splits as u32;
        let mut inv_sqrt_d = params.inv_sqrt_d;
        let mut window_arg = params.window as u32;
        let mut d_pt = self.opt_dev_ptr(params.page_table)?;
        let mut vq4_arg: u32 = params.v_q4 as u32;

        let mut p1: [*mut c_void; 20] = [
            &mut d_po as *mut _ as *mut c_void,
            &mut d_pm as *mut _ as *mut c_void,
            &mut d_pl as *mut _ as *mut c_void,
            &mut d_q as *mut _ as *mut c_void,
            &mut d_kc as *mut _ as *mut c_void,
            &mut d_ks as *mut _ as *mut c_void,
            &mut d_vc as *mut _ as *mut c_void,
            &mut d_vs as *mut _ as *mut c_void,
            &mut d_rel as *mut _ as *mut c_void,
            &mut nq_arg as *mut _ as *mut c_void,
            &mut nkv_arg as *mut _ as *mut c_void,
            &mut max_arg as *mut _ as *mut c_void,
            &mut qpos_arg as *mut _ as *mut c_void,
            &mut rext_arg as *mut _ as *mut c_void,
            &mut loc_arg as *mut _ as *mut c_void,
            &mut splits_arg as *mut _ as *mut c_void,
            &mut inv_sqrt_d as *mut _ as *mut c_void,
            &mut window_arg as *mut _ as *mut c_void,
            &mut d_pt as *mut _ as *mut c_void,
            &mut vq4_arg as *mut _ as *mut c_void,
        ];
        let mut p2: [*mut c_void; 7] = [
            &mut d_out as *mut _ as *mut c_void,
            &mut d_po as *mut _ as *mut c_void,
            &mut d_pm as *mut _ as *mut c_void,
            &mut d_pl as *mut _ as *mut c_void,
            &mut d_sink as *mut _ as *mut c_void,
            &mut nq_arg as *mut _ as *mut c_void,
            &mut splits_arg as *mut _ as *mut c_void,
        ];

        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.flash_decode_fp8_split, nkv_arg, splits_arg, 1, 128, 1, 1, 0, s, p1.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(flash_decode_fp8_split)",
            )?;
            check_hip(
                self.launch_raw(
                    self.kernels.flash_decode_fp8_reduce, nq_arg, 1, 1, 32, 1, 1, 0, s, p2.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(flash_decode_fp8_reduce)",
            )
        }
    }

    fn gate_attn_sigmoid_mul(
        &self,
        out: DeviceBuf,
        attn: DeviceBuf,
        gate: DeviceBuf,
        n: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_o = self.dev_ptr(out)?;
        let mut d_a = self.dev_ptr(attn)?;
        let mut d_g = self.dev_ptr(gate)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n as u32;
        let mut params: [*mut c_void; 4] = [
            &mut d_o as *mut _ as *mut c_void,
            &mut d_a as *mut _ as *mut c_void,
            &mut d_g as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.gate_attn_sigmoid_mul_f32, n_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(gate_attn_sigmoid_mul_f32)",
            )
        }
    }

    fn attn_res_mix(
        &self,
        out: DeviceBuf,
        blocks: Option<DeviceBuf>,
        cur: DeviceBuf,
        queries: DeviceBuf,
        num_blocks: usize,
        num_queries: usize,
        hidden: usize,
        eps: f32,
        normalize_values: bool,
        stream: Stream,
    ) -> Result<()> {
        let mut d_o = self.dev_ptr(out)?;
        let mut d_b = self.opt_dev_ptr(blocks)?;
        let mut d_c = self.dev_ptr(cur)?;
        let mut d_q = self.dev_ptr(queries)?;
        let s = self.raw_stream(stream)?;
        let mut nb_arg = num_blocks as u32;
        let mut nq_arg = num_queries as u32;
        let mut h_arg = hidden as u32;
        let mut eps_arg = eps;
        let mut nv_arg: u32 = if normalize_values { 1 } else { 0 };
        let mut params: [*mut c_void; 9] = [
            &mut d_o as *mut _ as *mut c_void,
            &mut d_b as *mut _ as *mut c_void,
            &mut d_c as *mut _ as *mut c_void,
            &mut d_q as *mut _ as *mut c_void,
            &mut nb_arg as *mut _ as *mut c_void,
            &mut nq_arg as *mut _ as *mut c_void,
            &mut h_arg as *mut _ as *mut c_void,
            &mut eps_arg as *mut _ as *mut c_void,
            &mut nv_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.attn_res_mix_f32, nq_arg, 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(attn_res_mix_f32)",
            )
        }
    }

    fn loop_control_prep(
        &self,
        out: DeviceBuf,
        cur: DeviceBuf,
        msg: DeviceBuf,
        hidden: usize,
        eps: f32,
        stream: Stream,
    ) -> Result<()> {
        let mut d_o = self.dev_ptr(out)?;
        let mut d_c = self.dev_ptr(cur)?;
        let mut d_m = self.dev_ptr(msg)?;
        let s = self.raw_stream(stream)?;
        let mut h_arg = hidden as u32;
        let mut eps_arg = eps;
        let mut params: [*mut c_void; 5] = [
            &mut d_o as *mut _ as *mut c_void,
            &mut d_c as *mut _ as *mut c_void,
            &mut d_m as *mut _ as *mut c_void,
            &mut h_arg as *mut _ as *mut c_void,
            &mut eps_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.loop_control_prep_f32, 2, 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(loop_control_prep_f32)",
            )
        }
    }

    fn loop_control_apply(
        &self,
        out: DeviceBuf,
        cur: DeviceBuf,
        msg: DeviceBuf,
        gain_raw: DeviceBuf,
        forget_raw: DeviceBuf,
        n: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_o = self.dev_ptr(out)?;
        let mut d_c = self.dev_ptr(cur)?;
        let mut d_m = self.dev_ptr(msg)?;
        let mut d_g = self.dev_ptr(gain_raw)?;
        let mut d_f = self.dev_ptr(forget_raw)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n as u32;
        let mut params: [*mut c_void; 6] = [
            &mut d_o as *mut _ as *mut c_void,
            &mut d_c as *mut _ as *mut c_void,
            &mut d_m as *mut _ as *mut c_void,
            &mut d_g as *mut _ as *mut c_void,
            &mut d_f as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.loop_control_apply_f32, n_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(loop_control_apply_f32)",
            )
        }
    }

    fn vec_add(
        &self,
        out: DeviceBuf,
        a: DeviceBuf,
        b: DeviceBuf,
        n: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_o = self.dev_ptr(out)?;
        let mut d_a = self.dev_ptr(a)?;
        let mut d_b = self.dev_ptr(b)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n as u32;
        let mut params: [*mut c_void; 4] = [
            &mut d_o as *mut _ as *mut c_void,
            &mut d_a as *mut _ as *mut c_void,
            &mut d_b as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.vec_add_f32, n_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(vec_add_f32)",
            )
        }
    }

    fn vec_scale(&self, x: DeviceBuf, alpha: f32, n: usize, stream: Stream) -> Result<()> {
        let mut d_x = self.dev_ptr(x)?;
        let s = self.raw_stream(stream)?;
        let mut a_arg = alpha;
        let mut n_arg = n as u32;
        let mut params: [*mut c_void; 3] = [
            &mut d_x as *mut _ as *mut c_void,
            &mut a_arg as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.vec_scale_f32, n_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(vec_scale_f32)",
            )
        }
    }

    fn vec_copy(&self, dst: DeviceBuf, src: DeviceBuf, n: usize, stream: Stream) -> Result<()> {
        let mut d_dst = self.dev_ptr(dst)?;
        let mut d_src = self.dev_ptr(src)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n as u32;
        let mut params: [*mut c_void; 3] = [
            &mut d_dst as *mut _ as *mut c_void,
            &mut d_src as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.vec_copy_f32, n_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(vec_copy_f32)",
            )
        }
    }

    fn zero_f32(&self, dst: DeviceBuf, n: usize, stream: Stream) -> Result<()> {
        let mut d_dst = self.dev_ptr(dst)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n as u32;
        let mut params: [*mut c_void; 2] = [
            &mut d_dst as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.zero_f32, n_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(zero_f32)",
            )
        }
    }

    fn final_logit_softcap(
        &self,
        logits: DeviceBuf,
        vocab_size: usize,
        cap: f32,
        stream: Stream,
    ) -> Result<()> {
        let mut d_l = self.dev_ptr(logits)?;
        let s = self.raw_stream(stream)?;
        let mut v_arg = vocab_size as u32;
        let mut c_arg = cap;
        let mut params: [*mut c_void; 3] = [
            &mut d_l as *mut _ as *mut c_void,
            &mut v_arg as *mut _ as *mut c_void,
            &mut c_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.final_logit_softcap_f32, v_arg.div_ceil(256), 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(final_logit_softcap_f32)",
            )
        }
    }

    fn argmax_partial_f32(
        &self,
        out_vals: DeviceBuf,
        out_idx: DeviceBuf,
        x: DeviceBuf,
        n: usize,
        stream: Stream,
    ) -> Result<()> {
        let mut d_v = self.dev_ptr(out_vals)?;
        let mut d_i = self.dev_ptr(out_idx)?;
        let mut d_x = self.dev_ptr(x)?;
        let s = self.raw_stream(stream)?;
        let mut n_arg = n as u32;
        let mut params: [*mut c_void; 4] = [
            &mut d_v as *mut _ as *mut c_void,
            &mut d_i as *mut _ as *mut c_void,
            &mut d_x as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe {
            check_hip(
                self.launch_raw(
                    self.kernels.argmax_partial_f32, 64, 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(argmax_partial_f32)",
            )
        }
    }
    fn gemm_fp8_rows(&self, y: DeviceBuf, x: DeviceBuf, w_fp8: DeviceBuf, row_scales: DeviceBuf,
        rows: usize, n_rows: usize, k_cols: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c, mut d) = (self.dev_ptr(y)?, self.dev_ptr(x)?, self.dev_ptr(w_fp8)?, self.dev_ptr(row_scales)?);
        let (mut m, mut n, mut k) = (rows as u32, n_rows as u32, k_cols as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut m), p(&mut n), p(&mut k)];
        if k_cols % 32 == 0 && std::env::var_os("SMALLTIDE_GEMM_SCALAR").is_none() {
            // Both WMMA kernels compute every output element identically, so a row's result does
            // not depend on how many rows share the call (prefill segmentation invariance).
            if rows <= 16 {
                return self.launch(self.kernels.gemm_fp8_rows_wmma_small, (n.div_ceil(256), 1, 1), 256, &mut args, stream, "launch(gemm_fp8_rows_wmma_small)");
            }
            if std::env::var_os("SMALLTIDE_GEMM_NO_X16").is_none() {
                // The activations to f16 once, then the GEMM reads them as WMMA fragments.
                let need = rows * k_cols * 2;
                let x16 = {
                    let mut g = self.gemm_x16.lock().unwrap();
                    if g.map_or(true, |buf| buf.size_bytes < need) {
                        if let Some(old) = g.take() {
                            self.sync_stream(stream)?;
                            self.free_device(old)?;
                        }
                        *g = Some(self.alloc_device(need)?);
                    }
                    g.unwrap()
                };
                self.f32_to_f16(x16, x, rows * k_cols, stream)?;
                let mut xh = self.dev_ptr(x16)?;
                let mut args16 = [p(&mut a), p(&mut xh), p(&mut c), p(&mut d), p(&mut m), p(&mut n), p(&mut k)];
                if rows <= 32 {
                    return self.launch(self.kernels.gemm_fp8_rows_wmma_x16_m32, (n.div_ceil(32), 1, 1), 64, &mut args16, stream, "launch(gemm_fp8_rows_wmma_x16_m32)");
                }
                if rows <= 64 || n_rows <= 512 {
                    return self.launch(self.kernels.gemm_fp8_rows_wmma_x16_m64, (n.div_ceil(32), m.div_ceil(64), 1), 64, &mut args16, stream, "launch(gemm_fp8_rows_wmma_x16_m64)");
                }
                return self.launch(self.kernels.gemm_fp8_rows_wmma_x16, (n.div_ceil(64), m.div_ceil(64), 1), 64, &mut args16, stream, "launch(gemm_fp8_rows_wmma_x16)");
            }
            return self.launch(self.kernels.gemm_fp8_rows_wmma, (n.div_ceil(64), m.div_ceil(128), 1), 256, &mut args, stream, "launch(gemm_fp8_rows_wmma)");
        }
        self.launch(self.kernels.gemm_fp8_e4m3_batched_f32, (n.div_ceil(4), m.div_ceil(8), 1), 128, &mut args, stream, "launch(gemm_fp8_rows)")
    }

    fn gemm_fp8_rows_decode(&self, y: DeviceBuf, x: DeviceBuf, w_fp8: DeviceBuf, row_scales: DeviceBuf,
        rows: usize, n_rows: usize, k_cols: usize, stream: Stream) -> Result<()> {
        if rows > 8 || k_cols % 16 != 0 {
            return self.gemm_fp8_rows(y, x, w_fp8, row_scales, rows, n_rows, k_cols, stream);
        }
        let (mut a, mut b, mut c, mut d) = (self.dev_ptr(y)?, self.dev_ptr(x)?, self.dev_ptr(w_fp8)?, self.dev_ptr(row_scales)?);
        let (mut m, mut n, mut k) = (rows as u32, n_rows as u32, k_cols as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut m), p(&mut n), p(&mut k)];
        if rows >= 4 && k_cols % 32 == 0 && std::env::var_os("SMALLTIDE_NO_WMMA_DECODE").is_none() {
            return self.launch(self.kernels.gemm_fp8_rows_wmma_small, (n.div_ceil(256), 1, 1), 256, &mut args, stream, "launch(gemm_fp8_rows_wmma_small)");
        }
        // (The LDS-staged smalltide_gemv_fp8_rows2 measured slower for 1..8 rows.)
        self.launch(self.kernels.gemv_fp8_rows, (n.div_ceil(4), 1, 1), 128, &mut args, stream, "launch(gemv_fp8_rows)")
    }

    fn gemv_bf16(&self, out: DeviceBuf, x: DeviceBuf, w_bf16: DeviceBuf,
        n_rows: usize, k_cols: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c) = (self.dev_ptr(out)?, self.dev_ptr(x)?, self.dev_ptr(w_bf16)?);
        let (mut n, mut k) = (n_rows as u32, k_cols as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut n), p(&mut k)];
        self.launch(self.kernels.gemv_bf16_f32, (n.div_ceil(4), 1, 1), 128, &mut args, stream, "launch(gemv_bf16_f32)")
    }

    fn gemm_bf16_rows(&self, y: DeviceBuf, x: DeviceBuf, w_bf16: DeviceBuf,
        rows: usize, n_rows: usize, k_cols: usize, stream: Stream) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let (mut a, mut b, mut c) = (self.dev_ptr(y)?, self.dev_ptr(x)?, self.dev_ptr(w_bf16)?);
        let (mut m, mut n, mut k) = (rows as u32, n_rows as u32, k_cols as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut m), p(&mut n), p(&mut k)];
        let wmma_ok = k_cols % 32 == 0 && std::env::var_os("SMALLTIDE_GEMM_SCALAR").is_none();
        if wmma_ok && rows >= 4 {
            if rows <= 16 {
                return self.launch(self.kernels.gemm_bf16_rows_wmma_small, (n.div_ceil(256), 1, 1), 256, &mut args, stream, "launch(gemm_bf16_rows_wmma_small)");
            }
            if rows <= 32 {
                return self.launch(self.kernels.gemm_bf16_rows_wmma_m32, (n.div_ceil(32), 1, 1), 64, &mut args, stream, "launch(gemm_bf16_rows_wmma_m32)");
            }
            if rows <= 64 {
                return self.launch(self.kernels.gemm_bf16_rows_wmma_m64, (n.div_ceil(32), 1, 1), 64, &mut args, stream, "launch(gemm_bf16_rows_wmma_m64)");
            }
            return self.launch(self.kernels.vit_gemm_bf16, (n.div_ceil(64), m.div_ceil(128), 1), 256, &mut args, stream, "launch(vit_gemm_bf16)");
        }
        let mut off = 0usize;
        while off < rows {
            let take = (rows - off).min(8);
            let ys = y.slice(off * n_rows * 4, take * n_rows * 4);
            let xs = x.slice(off * k_cols * 4, take * k_cols * 4);
            let (mut a, mut b, mut c) = (self.dev_ptr(ys)?, self.dev_ptr(xs)?, self.dev_ptr(w_bf16)?);
            let (mut m, mut n, mut k) = (take as u32, n_rows as u32, k_cols as u32);
            let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut m), p(&mut n), p(&mut k)];
            self.launch(self.kernels.gemv_bf16_rows, (n.div_ceil(4), 1, 1), 128, &mut args, stream, "launch(gemv_bf16_rows)")?;
            off += take;
        }
        Ok(())
    }

    fn gemm_f32_rows(&self, y: DeviceBuf, x: DeviceBuf, w: DeviceBuf, rows: usize, n_rows: usize,
        k_cols: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c) = (self.dev_ptr(y)?, self.dev_ptr(x)?, self.dev_ptr(w)?);
        let (mut m, mut n, mut k) = (rows as u32, n_rows as u32, k_cols as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut m), p(&mut n), p(&mut k)];
        self.launch(self.kernels.gemm_f32_rows, (n.div_ceil(4), m.div_ceil(8), 1), 128, &mut args, stream, "launch(gemm_f32_rows)")
    }

    fn attn_res_mix_rows(&self, out: DeviceBuf, blocks: Option<DeviceBuf>, cur: DeviceBuf,
        queries: DeviceBuf, num_blocks: usize, num_queries: usize, hidden: usize, eps: f32,
        rows: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c, mut d) = (self.dev_ptr(out)?, self.opt_dev_ptr(blocks)?, self.dev_ptr(cur)?, self.dev_ptr(queries)?);
        let (mut nb, mut nq, mut h, mut e, mut nv, mut m) = (num_blocks as u32, num_queries as u32, hidden as u32, eps, 1u32, rows as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut nb), p(&mut nq), p(&mut h), p(&mut e), p(&mut nv), p(&mut m)];
        let threads = if m == 1 { 1024 } else { 256 };
        self.launch(self.kernels.attn_res_mix_rows_f32, (nq, m, 1), threads, &mut args, stream, "launch(attn_res_mix_rows)")
    }

    fn head_l2_clip_rows(&self, x: DeviceBuf, scale: Option<DeviceBuf>, total_heads: usize,
        head_dim: usize, heads_per_row: usize, threshold: f32, tau: Option<(usize, f32, f32)>,
        stream: Stream) -> Result<()> {
        let (mut a, mut b) = (self.dev_ptr(x)?, self.opt_dev_ptr(scale)?);
        let (pos0, alpha, floor) = tau.unwrap_or((0, 0.0, 1.0));
        let (mut th, mut hd, mut hpr, mut thr, mut p0, mut al, mut fl) =
            (total_heads as u32, head_dim as u32, heads_per_row as u32, threshold, pos0 as u32, alpha, floor);
        let mut args = [p(&mut a), p(&mut b), p(&mut th), p(&mut hd), p(&mut hpr), p(&mut thr), p(&mut p0), p(&mut al), p(&mut fl)];
        self.launch(self.kernels.head_l2_clip_rows_f32, (th, 1, 1), 128, &mut args, stream, "launch(head_l2_clip_rows)")
    }

    fn sconv_k4_rows(&self, out: DeviceBuf, state: DeviceBuf, x: DeviceBuf, weight: DeviceBuf,
        dyn_taps: Option<DeviceBuf>, dim: usize, head_dim: usize, rows: usize, dyn_stride: usize,
        stream: Stream) -> Result<()> {
        self.sconv_k4_rows_hist(out, state, x, weight, dyn_taps, dim, head_dim, rows, dyn_stride, None, stream)
    }

    fn sconv_rewind(&self, entries: DeviceBuf, n: usize, max_dim: usize, keep: usize, stream: Stream) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let (mut a, mut k) = (self.dev_ptr(entries)?, keep as u32);
        let mut args = [p(&mut a), p(&mut k)];
        self.launch(self.kernels.sconv_rewind, (n as u32, (max_dim as u32).div_ceil(256), 1), 256, &mut args, stream, "launch(sconv_rewind)")
    }

    fn moe_rows_grouped(&self, mid: DeviceBuf, tmp: DeviceBuf, x: DeviceBuf, xscratch: DeviceBuf, route: DeviceBuf,
        slot_table: DeviceBuf, pool: DeviceBuf, desc: DeviceBuf, rows: usize, mp: MoeDispatchParams,
        stream: Stream) -> Result<()> {
        if rows == 0 || rows > 64 {
            return Err(SmallTideError::Backend { backend: "hip", message: format!("moe_rows_grouped: {rows} rows") });
        }
        // 1. plan on the GPU (up to 64 rows -> <= 160 work items = 2560 B; pairs start at 4096 B)
        let (mut w, mut pr, mut rt, mut stb, mut nr) = (self.dev_ptr(desc)?, self.dev_ptr(desc.slice(4096, desc.size_bytes - 4096))?,
            self.dev_ptr(route)?, self.dev_ptr(slot_table)?, rows as u32);
        // 2. x -> f16 in the same launch
        let n = (rows * mp.hidden_size) as u32;
        let (mut xo, mut xi, mut nn) = (self.dev_ptr(xscratch)?, self.dev_ptr(x)?, n);
        let mut args = [p(&mut w), p(&mut pr), p(&mut rt), p(&mut stb), p(&mut nr), p(&mut xo), p(&mut xi), p(&mut nn)];
        self.launch(self.kernels.moe_group_rows3, (1, 1, 1), 256, &mut args, stream, "launch(moe_group_rows3)")?;
        // 3. WMMA grouped w13 / w2 over up to n_work items (unused ones exit immediately)
        let n_work = if rows <= 16 { (rows * 8) as u32 } else { 160u32 };
        let (mut a, mut x16, mut c, mut d, mut e) = (self.dev_ptr(mid)?, self.dev_ptr(xscratch)?, self.dev_ptr(pool)?, w, pr);
        let (mut st, mut h, mut i, mut gc, mut uc) = (mp.expert_stride_bytes as u64, mp.hidden_size as u32, mp.intermediate_size as u32, 4.0f32, 25.0f32);
        let mut args = [p(&mut a), p(&mut x16), p(&mut c), p(&mut d), p(&mut e), p(&mut st), p(&mut h), p(&mut i), p(&mut gc), p(&mut uc)];
        self.launch(self.kernels.moe_w13_grouped_wmma2, (n_work, (2 * i).div_ceil(128), 1), 256, &mut args, stream, "launch(moe_w13_grouped_wmma2)")?;
        let (mut a2, mut bm, mut c2, mut d2, mut e2) = (self.dev_ptr(tmp)?, self.dev_ptr(mid)?, self.dev_ptr(pool)?, w, pr);
        let mut f2: *mut c_void = ptr::null_mut();
        let mut args = [p(&mut a2), p(&mut bm), p(&mut c2), p(&mut d2), p(&mut e2), p(&mut f2), p(&mut st), p(&mut h), p(&mut i)];
        self.launch(self.kernels.moe_w2_grouped_wmma2, (n_work, h.div_ceil(128), 1), 256, &mut args, stream, "launch(moe_w2_grouped_wmma2)")
    }

    fn moe_post_rows(&self, out: DeviceBuf, tmp: DeviceBuf, slot_table: DeviceBuf, route: DeviceBuf,
        rows: usize, hidden: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c, mut d, mut h) = (self.dev_ptr(out)?, self.dev_ptr(tmp)?, self.dev_ptr(slot_table)?, self.dev_ptr(route)?, hidden as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut h)];
        self.launch(self.kernels.moe_post_rows, (h.div_ceil(256), rows as u32, 1), 256, &mut args, stream, "launch(moe_post_rows)")
    }

    fn dspark_mix_rms(&self, out: DeviceBuf, taps: DeviceBuf, tap_stride: usize, alpha: [f32; 4], w: DeviceBuf,
        rows: usize, d: usize, eps: f32, stream: Stream) -> Result<()> {
        let (mut o, mut t, mut ts, mut a0, mut a1, mut a2, mut a3, mut ww, mut dd, mut e) = (self.dev_ptr(out)?, self.dev_ptr(taps)?,
            tap_stride as u32, alpha[0], alpha[1], alpha[2], alpha[3], self.dev_ptr(w)?, d as u32, eps);
        let mut args = [p(&mut o), p(&mut t), p(&mut ts), p(&mut a0), p(&mut a1), p(&mut a2), p(&mut a3), p(&mut ww), p(&mut dd), p(&mut e)];
        self.launch(self.kernels.dspark_mix_rms, (rows as u32, 1, 1), 1024, &mut args, stream, "launch(dspark_mix_rms)")
    }

    fn dspark_head_norm_rope(&self, x: DeviceBuf, w: DeviceBuf, nh: usize, rows: usize, pos0: usize, theta: f32,
        eps: f32, stream: Stream) -> Result<()> {
        let (mut xx, mut ww, mut n, mut p0, mut lt, mut e) = (self.dev_ptr(x)?, self.dev_ptr(w)?, nh as u32, pos0 as u32, theta.log2(), eps);
        let mut args = [p(&mut xx), p(&mut ww), p(&mut n), p(&mut p0), p(&mut lt), p(&mut e)];
        self.launch(self.kernels.dspark_head_norm_rope, (nh as u32, rows as u32, 1), 64, &mut args, stream, "launch(dspark_head_norm_rope)")
    }

    fn dspark_ctx_store(&self, kc: DeviceBuf, vc: DeviceBuf, k: DeviceBuf, v: DeviceBuf, pos0: usize, rows: usize,
        ring: usize, dim: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c, mut d, mut p0, mut rg, mut dm) = (self.dev_ptr(kc)?, self.dev_ptr(vc)?, self.dev_ptr(k)?,
            self.dev_ptr(v)?, pos0 as u32, ring as u32, dim as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut p0), p(&mut rg), p(&mut dm)];
        self.launch(self.kernels.dspark_ctx_store, (rows as u32, 1, 1), 256, &mut args, stream, "launch(dspark_ctx_store)")
    }

    fn dspark_attn(&self, out: DeviceBuf, q: DeviceBuf, kb: DeviceBuf, vb: DeviceBuf, kc: DeviceBuf, vc: DeviceBuf,
        rows: usize, pp: usize, lo: usize, ring: usize, nh: usize, nkv: usize, nblk: usize, scale: f32,
        stream: Stream) -> Result<()> {
        let (mut o, mut qq, mut a, mut b, mut c, mut d) = (self.dev_ptr(out)?, self.dev_ptr(q)?, self.dev_ptr(kb)?, self.dev_ptr(vb)?,
            self.dev_ptr(kc)?, self.dev_ptr(vc)?);
        let (mut pv, mut lv, mut rg, mut n1, mut n2, mut nb, mut sc) = (pp as u32, lo as u32, ring as u32, nh as u32, nkv as u32, nblk as u32, scale);
        let mut args = [p(&mut o), p(&mut qq), p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut pv), p(&mut lv), p(&mut rg),
            p(&mut n1), p(&mut n2), p(&mut nb), p(&mut sc)];
        self.launch(self.kernels.dspark_attn, (nh as u32, rows as u32, 1), 256, &mut args, stream, "launch(dspark_attn)")
    }

    fn moe_post_sconv_rows(&self, cur: DeviceBuf, ffn_inp: DeviceBuf, tmp: DeviceBuf, slot_table: DeviceBuf, route: DeviceBuf,
        state: DeviceBuf, weight: DeviceBuf, hist: Option<DeviceBuf>, hidden: usize, rows: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c, mut d, mut e) = (self.dev_ptr(cur)?, self.dev_ptr(ffn_inp)?, self.dev_ptr(tmp)?, self.dev_ptr(slot_table)?, self.dev_ptr(route)?);
        let (mut f, mut g, mut hh) = (self.dev_ptr(state)?, self.dev_ptr(weight)?, self.opt_dev_ptr(hist)?);
        let (mut hd, mut r) = (hidden as u32, rows as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut e), p(&mut f), p(&mut g), p(&mut hh), p(&mut hd), p(&mut r)];
        self.launch(self.kernels.moe_post_sconv_rows, (hd.div_ceil(256), 1, 1), 256, &mut args, stream, "launch(moe_post_sconv_rows)")
    }

    fn mix_rows(&self, pp: smalltide_core::backend::MixRowsParams, stream: Stream) -> Result<()> {
        #[repr(C)]
        struct ArgsC {
            cur: *mut c_void,
            add_src: *mut c_void,
            residual: *mut c_void,
            blocks: *mut c_void,
            copy_out: *mut c_void,
            queries: *mut c_void,
            mix_out: *mut c_void,
            norm_w0: *mut c_void,
            norm_w1: *mut c_void,
            norm_w2: *mut c_void,
            norm_out: *mut c_void,
            nb: u32,
            boundary: u32,
            nq: u32,
            hidden: u32,
            rows: u32,
            eps: f32,
        }
        if pp.hidden > 3072 || pp.nq == 0 || pp.nq > 3 || pp.nb + 1 > 15 || pp.rows == 0 {
            return Err(SmallTideError::Backend { backend: "hip", message: "mix_rows: unsupported shape".into() });
        }
        let mut a = ArgsC {
            cur: self.dev_ptr(pp.cur)?,
            add_src: self.opt_dev_ptr(pp.add_src)?,
            residual: self.opt_dev_ptr(pp.residual)?,
            blocks: self.opt_dev_ptr(pp.blocks)?,
            copy_out: self.opt_dev_ptr(pp.copy_out)?,
            queries: self.opt_dev_ptr(pp.queries)?,
            mix_out: self.opt_dev_ptr(pp.mix_out)?,
            norm_w0: self.opt_dev_ptr(pp.norm_w[0])?,
            norm_w1: self.opt_dev_ptr(pp.norm_w[1])?,
            norm_w2: self.opt_dev_ptr(pp.norm_w[2])?,
            norm_out: self.opt_dev_ptr(pp.norm_out)?,
            nb: pp.nb as u32,
            boundary: pp.boundary as u32,
            nq: pp.nq as u32,
            hidden: pp.hidden as u32,
            rows: pp.rows as u32,
            eps: pp.eps,
        };
        let mut args = [p(&mut a)];
        self.launch(self.kernels.mix_rows, (pp.rows as u32, 1, 1), 1024, &mut args, stream, "launch(mix_rows)")
    }

    fn ctrl_prep_multi(&self, items: &[(DeviceBuf, DeviceBuf, DeviceBuf)], hidden: usize, eps: f32, rows: usize, stream: Stream) -> Result<()> {
        let mut a = self.ctrl_args(hidden, eps);
        if items.is_empty() || items.len() > 4 {
            return Err(SmallTideError::Backend { backend: "hip", message: "ctrl_prep_multi: 1..=4 items".into() });
        }
        for (i, &(o, c, m)) in items.iter().enumerate() {
            a.out[i] = self.dev_ptr(o)?;
            a.cur[i] = self.dev_ptr(c)?;
            a.msg[i] = self.dev_ptr(m)?;
        }
        let mut args = [p(&mut a)];
        self.launch(self.kernels.ctrl_prep_multi, (2, rows as u32, items.len() as u32), 256, &mut args, stream, "launch(ctrl_prep_multi)")
    }

    fn ctrl_apply_norm_multi(&self, items: &[(DeviceBuf, DeviceBuf, DeviceBuf, DeviceBuf, DeviceBuf, Option<(DeviceBuf, DeviceBuf)>)],
        hidden: usize, eps: f32, rows: usize, stream: Stream) -> Result<()> {
        let mut a = self.ctrl_args(hidden, eps);
        if items.is_empty() || items.len() > 4 || hidden > 3072 {
            return Err(SmallTideError::Backend { backend: "hip", message: "ctrl_apply_norm_multi: bad shape".into() });
        }
        for (i, &(o, c, m, g, f, n)) in items.iter().enumerate() {
            a.out[i] = self.dev_ptr(o)?;
            a.cur[i] = self.dev_ptr(c)?;
            a.msg[i] = self.dev_ptr(m)?;
            a.gain[i] = self.dev_ptr(g)?;
            a.forget[i] = self.dev_ptr(f)?;
            if let Some((w, no)) = n {
                a.norm_w[i] = self.dev_ptr(w)?;
                a.norm_out[i] = self.dev_ptr(no)?;
            }
        }
        let mut args = [p(&mut a)];
        self.launch(self.kernels.ctrl_apply_norm_multi, (rows as u32, items.len() as u32, 1), 1024, &mut args, stream, "launch(ctrl_apply_norm_multi)")
    }

    fn vit_gemm_bf16(&self, y: DeviceBuf, x: DeviceBuf, w: DeviceBuf, m: usize, n: usize, k: usize,
        stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c) = (self.dev_ptr(y)?, self.dev_ptr(x)?, self.dev_ptr(w)?);
        let (mut mm, mut nn, mut kk) = (m as u32, n as u32, k as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut mm), p(&mut nn), p(&mut kk)];
        self.launch(self.kernels.vit_gemm_bf16, (nn.div_ceil(64), mm.div_ceil(128), 1), 256, &mut args, stream, "launch(vit_gemm_bf16)")
    }

    fn vit_epilogue(&self, out: DeviceBuf, y: DeviceBuf, bias: Option<DeviceBuf>, gamma: Option<DeviceBuf>,
        residual: Option<DeviceBuf>, m: usize, n: usize, gelu: bool, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c, mut d, mut e) = (self.dev_ptr(out)?, self.dev_ptr(y)?, self.opt_dev_ptr(bias)?,
            self.opt_dev_ptr(gamma)?, self.opt_dev_ptr(residual)?);
        let (mut mm, mut nn, mut g) = (m as u32, n as u32, gelu as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut e), p(&mut mm), p(&mut nn), p(&mut g)];
        self.launch(self.kernels.vit_epilogue, (nn.div_ceil(256), mm, 1), 256, &mut args, stream, "launch(vit_epilogue)")
    }

    fn vit_ln(&self, out: DeviceBuf, x: DeviceBuf, w: DeviceBuf, b: DeviceBuf, rows: usize, n: usize, eps: f32,
        stream: Stream) -> Result<()> {
        let (mut a, mut bb, mut c, mut d) = (self.dev_ptr(out)?, self.dev_ptr(x)?, self.dev_ptr(w)?, self.dev_ptr(b)?);
        let (mut r, mut nn, mut e) = (rows as u32, n as u32, eps);
        let mut args = [p(&mut a), p(&mut bb), p(&mut c), p(&mut d), p(&mut r), p(&mut nn), p(&mut e)];
        self.launch(self.kernels.vit_ln, (r, 1, 1), 256, &mut args, stream, "launch(vit_ln)")
    }

    fn vit_rope(&self, qkv: DeviceBuf, rows: usize, heads: usize, cs: DeviceBuf, sn: DeviceBuf,
        stream: Stream) -> Result<()> {
        let (mut a, mut c, mut d) = (self.dev_ptr(qkv)?, self.dev_ptr(cs)?, self.dev_ptr(sn)?);
        let (mut r, mut h) = (rows as u32, heads as u32);
        let mut args = [p(&mut a), p(&mut r), p(&mut h), p(&mut c), p(&mut d)];
        self.launch(self.kernels.vit_rope, (r, 1, 1), 256, &mut args, stream, "launch(vit_rope)")
    }

    fn vit_attn(&self, out: DeviceBuf, qkv: DeviceBuf, rows: usize, heads: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b) = (self.dev_ptr(out)?, self.dev_ptr(qkv)?);
        let (mut r, mut h) = (rows as u32, heads as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut r), p(&mut h)];
        self.launch(self.kernels.vit_attn, (r.div_ceil(64), h, 1), 64, &mut args, stream, "launch(vit_attn)")
    }

    fn f32_to_f16(&self, out: DeviceBuf, x: DeviceBuf, n: usize, stream: Stream) -> Result<()> {
        let (mut xo, mut xi, mut nn) = (self.dev_ptr(out)?, self.dev_ptr(x)?, n as u32);
        let mut args = [p(&mut xo), p(&mut xi), p(&mut nn)];
        self.launch(self.kernels.f32_to_f16, (nn.div_ceil(1024), 1, 1), 256, &mut args, stream, "launch(f32_to_f16)")
    }

    fn silu_mul(&self, out: DeviceBuf, g: DeviceBuf, u: DeviceBuf, n: usize, stream: Stream) -> Result<()> {
        let (mut o, mut gg, mut uu, mut nn) = (self.dev_ptr(out)?, self.dev_ptr(g)?, self.dev_ptr(u)?, n as u32);
        let mut args = [p(&mut o), p(&mut gg), p(&mut uu), p(&mut nn)];
        self.launch(self.kernels.silu_mul, (nn.div_ceil(256), 1, 1), 256, &mut args, stream, "launch(silu_mul)")
    }

    fn dspark_markov_step(&self, z: DeviceBuf, raw: DeviceBuf, state: DeviceBuf, latent: DeviceBuf, w1: DeviceBuf,
        drafts: DeviceBuf, step: usize, anchor: u32, h: DeviceBuf, frnn: DeviceBuf, rnn_w: DeviceBuf, rnn_b: DeviceBuf,
        base: DeviceBuf, fold_b: DeviceBuf, fold16: DeviceBuf, pmax: DeviceBuf, pidx: DeviceBuf, r: usize, d: usize,
        v: usize, cap: f32, stream: Stream) -> Result<()> {
        let (mut zz, mut st, mut w1p, mut dr, mut sp, mut an, mut hh, mut fr, mut rr, mut dd) = (self.dev_ptr(z)?, self.dev_ptr(state)?,
            self.dev_ptr(w1)?, self.dev_ptr(drafts)?, step as u32, anchor, self.dev_ptr(h)?, self.dev_ptr(frnn)?, r as u32, d as u32);
        let nz = (2 * r + 2 * d) as u32;
        let mut args = [p(&mut zz), p(&mut st), p(&mut w1p), p(&mut dr), p(&mut sp), p(&mut an), p(&mut hh), p(&mut fr), p(&mut rr), p(&mut dd)];
        self.launch(self.kernels.dspark_z, (nz.div_ceil(256), 1, 1), 256, &mut args, stream, "launch(dspark_z)")?;
        self.gemv_f32(raw, z, rnn_w, 3 * r, 2 * r + 2 * d, stream)?;
        let (mut st2, mut la, mut rw, mut rb, mut r2) = (self.dev_ptr(state)?, self.dev_ptr(latent)?, self.dev_ptr(raw)?, self.dev_ptr(rnn_b)?, r as u32);
        let mut args = [p(&mut st2), p(&mut la), p(&mut rw), p(&mut rb), p(&mut r2)];
        self.launch(self.kernels.dspark_cell, ((r as u32).div_ceil(256), 1, 1), 256, &mut args, stream, "launch(dspark_cell)")?;
        let nparts = (v as u32).div_ceil(256);
        let (mut pm, mut pi, mut bs, mut fb, mut f16p, mut la2, mut vv, mut r3, mut cp) = (self.dev_ptr(pmax)?, self.dev_ptr(pidx)?,
            self.dev_ptr(base)?, self.dev_ptr(fold_b)?, self.dev_ptr(fold16)?, self.dev_ptr(latent)?, v as u32, r as u32, cap);
        let mut args = [p(&mut pm), p(&mut pi), p(&mut bs), p(&mut fb), p(&mut f16p), p(&mut la2), p(&mut vv), p(&mut r3), p(&mut cp)];
        self.launch(self.kernels.dspark_logits_argmax, (nparts, 1, 1), 256, &mut args, stream, "launch(dspark_logits_argmax)")?;
        let (mut dr2, mut sp2, mut pm2, mut pi2, mut np) = (self.dev_ptr(drafts)?, step as u32, self.dev_ptr(pmax)?, self.dev_ptr(pidx)?, nparts);
        let mut args = [p(&mut dr2), p(&mut sp2), p(&mut pm2), p(&mut pi2), p(&mut np)];
        self.launch(self.kernels.dspark_argmax_final, (1, 1, 1), 1024, &mut args, stream, "launch(dspark_argmax_final)")
    }

    fn dspark_quant_fold_fp8(&self, dst: DeviceBuf, scales: DeviceBuf, src: DeviceBuf,
        r: usize, v: usize, v_pad: usize, stream: Stream) -> Result<()> {
        let (mut d, mut sc, mut s, mut vv, mut vp) = (self.dev_ptr(dst)?, self.dev_ptr(scales)?, self.dev_ptr(src)?, v as u32, v_pad as u32);
        let mut args = [p(&mut d), p(&mut sc), p(&mut s), p(&mut vv), p(&mut vp)];
        self.launch(self.kernels.dspark_quant_fold_fp8, (r as u32, 1, 1), 1024, &mut args, stream, "launch(dspark_quant_fold_fp8)")
    }

    fn dspark_markov_step_fast(&self, state: DeviceBuf, latent: DeviceBuf, w1: DeviceBuf,
        drafts: DeviceBuf, margins: DeviceBuf, step: usize, anchor: u32, w_dyn: DeviceBuf,
        frnn_proj: DeviceBuf, hb_proj: DeviceBuf, rnn_b: DeviceBuf, base: DeviceBuf,
        fold_b: DeviceBuf, fold_fp8: DeviceBuf, fold_scales: DeviceBuf,
        pmax: DeviceBuf, pidx: DeviceBuf, pmax2: DeviceBuf,
        r: usize, v: usize, v_pad: usize, stream: Stream) -> Result<()> {
        let (mut st, mut la, mut w1p, mut dr, mut sp, mut an) = (self.dev_ptr(state)?, self.dev_ptr(latent)?,
            self.dev_ptr(w1)?, self.dev_ptr(drafts)?, step as u32, anchor);
        let (mut wd, mut fp, mut hp, mut rb, mut rr) = (self.dev_ptr(w_dyn)?, self.dev_ptr(frnn_proj)?,
            self.dev_ptr(hb_proj)?, self.dev_ptr(rnn_b)?, r as u32);
        let mut args = [p(&mut st), p(&mut la), p(&mut w1p), p(&mut dr), p(&mut sp), p(&mut an),
            p(&mut wd), p(&mut fp), p(&mut hp), p(&mut rb), p(&mut rr)];
        self.launch(self.kernels.dspark_rnn_step, (rr.div_ceil(8), 1, 1), 256, &mut args, stream, "launch(dspark_rnn_step)")?;

        let nparts = (v as u32).div_ceil(1024);
        let (mut pm, mut pi, mut pm2, mut bs, mut fb, mut ff, mut fs, mut la2, mut vv, mut vp, mut r2) = (
            self.dev_ptr(pmax)?, self.dev_ptr(pidx)?, self.dev_ptr(pmax2)?, self.dev_ptr(base)?,
            self.dev_ptr(fold_b)?, self.dev_ptr(fold_fp8)?, self.dev_ptr(fold_scales)?,
            self.dev_ptr(latent)?, v as u32, v_pad as u32, r as u32);
        let mut args = [p(&mut pm), p(&mut pi), p(&mut pm2), p(&mut bs), p(&mut fb), p(&mut ff), p(&mut fs), p(&mut la2), p(&mut vv), p(&mut vp), p(&mut r2)];
        self.launch(self.kernels.dspark_logits_argmax_fp8, (nparts, 1, 1), 256, &mut args, stream, "launch(dspark_logits_argmax_fp8)")?;

        let (mut dr2, mut mg, mut sp2, mut pm_b, mut pi_b, mut pm2_b, mut np) = (
            self.dev_ptr(drafts)?, self.dev_ptr(margins)?, step as u32,
            self.dev_ptr(pmax)?, self.dev_ptr(pidx)?, self.dev_ptr(pmax2)?, nparts);
        let mut args = [p(&mut dr2), p(&mut mg), p(&mut sp2), p(&mut pm_b), p(&mut pi_b), p(&mut pm2_b), p(&mut np)];
        self.launch(self.kernels.dspark_argmax_final_margin, (1, 1, 1), 256, &mut args, stream, "launch(dspark_argmax_final_margin)")
    }

    fn argmax_rows_fast(&self, out: DeviceBuf, scratch: DeviceBuf, x: DeviceBuf, rows: usize, n: usize, stream: Stream) -> Result<()> {
        const PARTS: u32 = 16;
        let pbytes = rows * (PARTS as usize) * 4;
        let (mut pv, mut pi, mut xx, mut nn) = (
            self.dev_ptr(scratch.slice(0, pbytes))?,
            self.dev_ptr(scratch.slice(pbytes, pbytes))?,
            self.dev_ptr(x)?,
            n as u32,
        );
        let mut args1 = [p(&mut pv), p(&mut pi), p(&mut xx), p(&mut nn)];
        self.launch(self.kernels.argmax_rows_part, (PARTS, rows as u32, 1), 256, &mut args1, stream, "launch(argmax_rows_part)")?;
        let (mut oo, mut pv2, mut pi2, mut np) = (self.dev_ptr(out)?, pv, pi, PARTS);
        let mut args2 = [p(&mut oo), p(&mut pv2), p(&mut pi2), p(&mut np)];
        self.launch(self.kernels.argmax_rows_reduce, (rows as u32, 1, 1), 32, &mut args2, stream, "launch(argmax_rows_reduce)")
    }

    fn logits_filter_rows(
        &self,
        out_idx: DeviceBuf,
        out_val: DeviceBuf,
        counts: DeviceBuf,
        row_max: DeviceBuf,
        scratch: DeviceBuf,
        x: DeviceBuf,
        rows: usize,
        n: usize,
        margin: f32,
        cap: usize,
        stream: Stream,
    ) -> Result<()> {
        const PARTS: u32 = 16;
        let pbytes = rows * (PARTS as usize) * 4;
        let (mut pv, mut pi, mut xx, mut nn) = (
            self.dev_ptr(scratch.slice(0, pbytes))?,
            self.dev_ptr(scratch.slice(pbytes, pbytes))?,
            self.dev_ptr(x)?,
            n as u32,
        );
        let mut args1 = [p(&mut pv), p(&mut pi), p(&mut xx), p(&mut nn)];
        self.launch(self.kernels.argmax_rows_part, (PARTS, rows as u32, 1), 256, &mut args1, stream, "launch(argmax_rows_part)")?;
        let (mut rm, mut cnt, mut pv2, mut np) = (self.dev_ptr(row_max)?, self.dev_ptr(counts)?, pv, PARTS);
        let mut args2 = [p(&mut rm), p(&mut cnt), p(&mut pv2), p(&mut np)];
        self.launch(self.kernels.argmax_rows_reduce_max, (rows as u32, 1, 1), 32, &mut args2, stream, "launch(argmax_rows_reduce_max)")?;
        let (mut oi, mut ov, mut cnt2, mut rm2, mut xx2, mut nn2, mut mg, mut cp) = (
            self.dev_ptr(out_idx)?,
            self.dev_ptr(out_val)?,
            cnt,
            rm,
            xx,
            n as u32,
            margin,
            cap as u32,
        );
        let mut args3 = [p(&mut oi), p(&mut ov), p(&mut cnt2), p(&mut rm2), p(&mut xx2), p(&mut nn2), p(&mut mg), p(&mut cp)];
        self.launch(self.kernels.logits_filter_rows, ((n as u32).div_ceil(1024), rows as u32, 1), 256, &mut args3, stream, "launch(logits_filter_rows)")
    }

    fn attn_prefill_fp8_split(&self, out: DeviceBuf, part_o: DeviceBuf, part_m: DeviceBuf, part_l: DeviceBuf,
        q: DeviceBuf, k_cache: DeviceBuf, k_scales: DeviceBuf, v_cache: DeviceBuf, v_scales: DeviceBuf,
        rel_table: Option<DeviceBuf>, sinks: Option<DeviceBuf>, pp: PrefillAttnParams, stream: Stream) -> Result<()> {
        let g = pp.n_q_heads / pp.n_kv_heads;
        let lo0 = if pp.window != 0 && pp.pos0 + 1 > pp.window { pp.pos0 + 1 - pp.window } else { 0 };
        let key_lo = lo0 & !15;
        let key_hi = pp.pos0 + pp.rows.saturating_sub(1);
        let n_tiles = (key_hi.saturating_sub(key_lo)) / 16 + 1;
        let q_blks = (pp.rows * g).div_ceil(16);
        let num_splits = (n_tiles / 8).clamp(1, 16);
        if pp.rows > 8 || num_splits <= 1 || std::env::var_os("SMALLTIDE_ATTN_NOSPLIT").is_some() {
            return self.attn_prefill_fp8(out, q, k_cache, k_scales, v_cache, v_scales, rel_table, sinks, pp, stream);
        }
        let (mut po, mut pm, mut pl) = (self.dev_ptr(part_o)?, self.dev_ptr(part_m)?, self.dev_ptr(part_l)?);
        let (mut qq, mut kc, mut ks, mut vc, mut vs) = (self.dev_ptr(q)?, self.dev_ptr(k_cache)?, self.dev_ptr(k_scales)?,
            self.dev_ptr(v_cache)?, self.dev_ptr(v_scales)?);
        let (mut rt, mut pt) = (self.opt_dev_ptr(rel_table)?, self.opt_dev_ptr(pp.page_table)?);
        let (mut nqh, mut nkv, mut ms, mut win, mut p0, mut re, mut isd, mut rr, mut ns, mut vq4) = (
            pp.n_q_heads as u32, pp.n_kv_heads as u32, pp.max_seq as u32, pp.window as u32,
            pp.pos0 as u32, pp.rel_extent as u32, pp.inv_sqrt_d, pp.rows as u32, num_splits as u32, pp.v_q4 as u32);
        let mut args1 = [p(&mut po), p(&mut pm), p(&mut pl), p(&mut qq), p(&mut kc), p(&mut ks), p(&mut vc), p(&mut vs),
            p(&mut rt), p(&mut nqh), p(&mut nkv), p(&mut ms), p(&mut win), p(&mut p0), p(&mut re), p(&mut isd),
            p(&mut rr), p(&mut ns), p(&mut pt), p(&mut vq4)];
        self.launch(self.kernels.attn_prefill_wmma_split, (nkv, q_blks as u32, ns), 128, &mut args1, stream, "launch(attn_prefill_wmma_split)")?;
        let (mut oo, mut po2, mut pm2, mut pl2, mut sk, mut nqh2, mut ns2) = (
            self.dev_ptr(out)?, po, pm, pl, self.opt_dev_ptr(sinks)?, pp.n_q_heads as u32, num_splits as u32);
        let mut args2 = [p(&mut oo), p(&mut po2), p(&mut pm2), p(&mut pl2), p(&mut sk), p(&mut nqh2), p(&mut ns2)];
        self.launch(self.kernels.attn_prefill_wmma_reduce, (nqh2, rr, 1), 32, &mut args2, stream, "launch(attn_prefill_wmma_reduce)")
    }

    fn dspark_ctx_store_fp8(&self, kc: DeviceBuf, ks: DeviceBuf, vc: DeviceBuf, vs: DeviceBuf,
        k: DeviceBuf, v: DeviceBuf, pos0: usize, rows: usize, ring: usize, nkv: usize,
        v_q4: bool, page_table: Option<DeviceBuf>, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c, mut d, mut kk, mut vv) = (self.dev_ptr(kc)?, self.dev_ptr(ks)?,
            self.dev_ptr(vc)?, self.dev_ptr(vs)?, self.dev_ptr(k)?, self.dev_ptr(v)?);
        let (mut p0, mut rg, mut nk, mut vq4) = (pos0 as u32, ring as u32, nkv as u32, v_q4 as u32);
        let mut pt = self.opt_dev_ptr(page_table)?;
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut kk), p(&mut vv), p(&mut p0), p(&mut rg), p(&mut nk), p(&mut vq4), p(&mut pt)];
        self.launch(self.kernels.dspark_ctx_store_fp8, (rows as u32, nk, 1), 128, &mut args, stream, "launch(dspark_ctx_store_fp8)")
    }

    fn dspark_attn_fp8(&self, out: DeviceBuf, part_o: DeviceBuf, part_m: DeviceBuf, part_l: DeviceBuf,
        q: DeviceBuf, kb: DeviceBuf, vb: DeviceBuf, kc: DeviceBuf, ks: DeviceBuf,
        vc: DeviceBuf, vs: DeviceBuf, rows: usize, pp: usize, lo: usize, ring: usize,
        nh: usize, nkv: usize, nblk: usize, scale: f32, v_q4: bool, page_table: Option<DeviceBuf>, stream: Stream) -> Result<()> {
        let nctx = pp.saturating_sub(lo);
        let mut pt = self.opt_dev_ptr(page_table)?;
        let num_splits = nctx.div_ceil(128).clamp(1, 16) as u32;
        let (mut o, mut po, mut pm, mut pl) = (self.dev_ptr(out)?, self.dev_ptr(part_o)?, self.dev_ptr(part_m)?, self.dev_ptr(part_l)?);
        let (mut qq, mut a, mut b, mut c, mut cs, mut d, mut ds) = (self.dev_ptr(q)?, self.dev_ptr(kb)?, self.dev_ptr(vb)?,
            self.dev_ptr(kc)?, self.dev_ptr(ks)?, self.dev_ptr(vc)?, self.dev_ptr(vs)?);
        let (mut pv, mut lv, mut rg, mut n1, mut n2, mut nb, mut ns, mut sc, mut vq4) =
            (pp as u32, lo as u32, ring as u32, nh as u32, nkv as u32, nblk as u32, num_splits, scale, v_q4 as u32);
        let mut args = [p(&mut o), p(&mut po), p(&mut pm), p(&mut pl), p(&mut qq), p(&mut a), p(&mut b),
            p(&mut c), p(&mut cs), p(&mut d), p(&mut ds), p(&mut pv), p(&mut lv), p(&mut rg),
            p(&mut n1), p(&mut n2), p(&mut nb), p(&mut ns), p(&mut sc), p(&mut vq4), p(&mut pt)];
        self.launch(self.kernels.dspark_attn_fp8_split, (n2, ns, 1), 256, &mut args, stream, "launch(dspark_attn_fp8_split)")?;
        if num_splits > 1 {
            let (mut o2, mut po2, mut pm2, mut pl2, mut n1b, mut ns2) = (o, po, pm, pl, nh as u32, num_splits);
            let mut rargs = [p(&mut o2), p(&mut po2), p(&mut pm2), p(&mut pl2), p(&mut n1b), p(&mut ns2)];
            self.launch(self.kernels.dspark_attn_reduce, (n1b, rows as u32, 1), 32, &mut rargs, stream, "launch(dspark_attn_reduce)")?;
        }
        Ok(())
    }

    fn dspark_embed_prep(&self, res: DeviceBuf, xn: DeviceBuf, anchor_in: DeviceBuf,
        embed_norm: DeviceBuf, mask_embed: DeviceBuf, in_ln: DeviceBuf, d: usize, eps: f32,
        stream: Stream) -> Result<()> {
        let (mut r, mut x, mut ai, mut en, mut me, mut ln) = (self.dev_ptr(res)?, self.dev_ptr(xn)?,
            self.dev_ptr(anchor_in)?, self.dev_ptr(embed_norm)?, self.dev_ptr(mask_embed)?, self.dev_ptr(in_ln)?);
        let (mut dd, mut e) = (d as u32, eps);
        let mut args = [p(&mut r), p(&mut x), p(&mut ai), p(&mut en), p(&mut me), p(&mut ln), p(&mut dd), p(&mut e)];
        self.launch(self.kernels.dspark_embed_prep, (8, 1, 1), 1024, &mut args, stream, "launch(dspark_embed_prep)")
    }

    fn add_rmsnorm_rows(&self, res: DeviceBuf, xn: DeviceBuf, tmp: DeviceBuf, w: DeviceBuf,
        rows: usize, d: usize, eps: f32, stream: Stream) -> Result<()> {
        let (mut r, mut x, mut t, mut ww, mut dd, mut e) = (self.dev_ptr(res)?, self.dev_ptr(xn)?,
            self.dev_ptr(tmp)?, self.dev_ptr(w)?, d as u32, eps);
        let mut args = [p(&mut r), p(&mut x), p(&mut t), p(&mut ww), p(&mut dd), p(&mut e)];
        self.launch(self.kernels.add_rmsnorm_rows, (rows as u32, 1, 1), 1024, &mut args, stream, "launch(add_rmsnorm_rows)")
    }

    fn dspark_qk_norm_rope(&self, q: DeviceBuf, k: DeviceBuf, qw: DeviceBuf, kw: DeviceBuf,
        hq: usize, hk: usize, rows: usize, pos0: usize, theta: f32, eps: f32,
        stream: Stream) -> Result<()> {
        let (mut qp, mut kp, mut qwp, mut kwp) = (self.dev_ptr(q)?, self.dev_ptr(k)?, self.dev_ptr(qw)?, self.dev_ptr(kw)?);
        let (mut nq, mut nk, mut p0, mut lt, mut e) = (hq as u32, hk as u32, pos0 as u32, theta.log2(), eps);
        let mut args = [p(&mut qp), p(&mut kp), p(&mut qwp), p(&mut kwp), p(&mut nq), p(&mut nk), p(&mut p0), p(&mut lt), p(&mut e)];
        self.launch(self.kernels.dspark_qk_norm_rope, (nq + nk, rows as u32, 1), 64, &mut args, stream, "launch(dspark_qk_norm_rope)")
    }

    fn dflash2_attn_fp8(&self, out: DeviceBuf, part_o: DeviceBuf, part_m: DeviceBuf, part_l: DeviceBuf,
        q: DeviceBuf, kb: DeviceBuf, vb: DeviceBuf, kc: DeviceBuf, ks: DeviceBuf,
        vc: DeviceBuf, vs: DeviceBuf, rows: usize, pp: usize, window: usize, ring: usize,
        nh: usize, nkv: usize, nblk: usize, scale: f32, v_q4: bool, page_table: Option<DeviceBuf>,
        stream: Stream) -> Result<()> {
        let lo = pp.saturating_sub(window.saturating_sub(1));
        let nctx = pp.saturating_sub(lo);
        let num_splits = nctx.div_ceil(128).clamp(1, 16) as u32;
        let (mut o, mut po, mut pm, mut pl) = (self.dev_ptr(out)?, self.dev_ptr(part_o)?, self.dev_ptr(part_m)?, self.dev_ptr(part_l)?);
        let (mut qq, mut a, mut b, mut c, mut cs, mut d, mut ds) = (self.dev_ptr(q)?, self.dev_ptr(kb)?, self.dev_ptr(vb)?,
            self.dev_ptr(kc)?, self.dev_ptr(ks)?, self.dev_ptr(vc)?, self.dev_ptr(vs)?);
        let (mut pv, mut lv, mut wv, mut rg, mut n1, mut n2, mut nb, mut ns, mut sc, mut vq4) =
            (pp as u32, lo as u32, window as u32, ring as u32, nh as u32, nkv as u32, nblk as u32, num_splits, scale, v_q4 as u32);
        let mut pt = self.opt_dev_ptr(page_table)?;
        let mut args = [p(&mut o), p(&mut po), p(&mut pm), p(&mut pl), p(&mut qq), p(&mut a), p(&mut b),
            p(&mut c), p(&mut cs), p(&mut d), p(&mut ds), p(&mut pv), p(&mut lv), p(&mut wv), p(&mut rg),
            p(&mut n1), p(&mut n2), p(&mut nb), p(&mut ns), p(&mut sc), p(&mut vq4), p(&mut pt)];
        self.launch(self.kernels.dflash2_attn_fp8_split, (n2, ns, 1), 256, &mut args, stream, "launch(dflash2_attn_fp8_split)")?;
        if num_splits > 1 {
            let (mut o2, mut po2, mut pm2, mut pl2, mut n1b, mut ns2) = (o, po, pm, pl, nh as u32, num_splits);
            let mut rargs = [p(&mut o2), p(&mut po2), p(&mut pm2), p(&mut pl2), p(&mut n1b), p(&mut ns2)];
            self.launch(self.kernels.dspark_attn_reduce, (n1b, rows as u32, 1), 32, &mut rargs, stream, "launch(dspark_attn_reduce)")?;
        }
        Ok(())
    }

    fn dflash2_conv_prepare(&self, out: DeviceBuf, x: DeviceBuf, coeff: DeviceBuf, base: DeviceBuf,
        rows: usize, d: usize, group_size: usize, stream: Stream) -> Result<()> {
        let (mut o, mut xx, mut cc, mut bb) = (self.dev_ptr(out)?, self.dev_ptr(x)?, self.dev_ptr(coeff)?, self.dev_ptr(base)?);
        let (mut dd, mut gs) = (d as u32, group_size as u32);
        let mut args = [p(&mut o), p(&mut xx), p(&mut cc), p(&mut bb), p(&mut dd), p(&mut gs)];
        self.launch(self.kernels.dflash2_conv_prepare, (rows as u32, 1, 1), 256, &mut args, stream, "launch(dflash2_conv_prepare)")
    }

    fn dflash2_conv_finish_add_rmsnorm(&self, res: DeviceBuf, xn: DeviceBuf, tmp: DeviceBuf,
        coeff: DeviceBuf, base: DeviceBuf, w: DeviceBuf, rows: usize, d: usize,
        group_size: usize, eps: f32, stream: Stream) -> Result<()> {
        let (mut r, mut x, mut t, mut cc, mut bb, mut ww) = (self.dev_ptr(res)?, self.dev_ptr(xn)?,
            self.dev_ptr(tmp)?, self.dev_ptr(coeff)?, self.dev_ptr(base)?, self.dev_ptr(w)?);
        let (mut dd, mut gs, mut e) = (d as u32, group_size as u32, eps);
        let mut args = [p(&mut r), p(&mut x), p(&mut t), p(&mut cc), p(&mut bb), p(&mut ww), p(&mut dd), p(&mut gs), p(&mut e)];
        self.launch(self.kernels.dflash2_conv_finish_add_rmsnorm, (rows as u32, 1, 1), 1024, &mut args, stream, "launch(dflash2_conv_finish_add_rmsnorm)")
    }

    fn topk16_rows(&self, out_val: DeviceBuf, out_idx: DeviceBuf,
        part_val: DeviceBuf, part_idx: DeviceBuf, x: DeviceBuf,
        rows: usize, n: usize, stream: Stream) -> Result<()> {
        let num_blocks = n.div_ceil(4096).max(1) as u32;
        let (mut pv, mut pi, mut xx) = (self.dev_ptr(part_val)?, self.dev_ptr(part_idx)?, self.dev_ptr(x)?);
        let (mut nn, mut nb) = (n as u32, num_blocks);
        let mut args1 = [p(&mut pv), p(&mut pi), p(&mut xx), p(&mut nn), p(&mut nb)];
        self.launch(self.kernels.topk16_rows_part, (num_blocks, rows as u32, 1), 256, &mut args1, stream, "launch(topk16_rows_part)")?;
        let (mut ov, mut oi, mut pv2, mut pi2, mut nb2) = (self.dev_ptr(out_val)?, self.dev_ptr(out_idx)?, pv, pi, num_blocks);
        let mut args2 = [p(&mut ov), p(&mut oi), p(&mut pv2), p(&mut pi2), p(&mut nb2)];
        self.launch(self.kernels.topk16_rows_reduce, (rows as u32, 1, 1), 256, &mut args2, stream, "launch(topk16_rows_reduce)")
    }

    fn device_address(&self, buf: DeviceBuf) -> Result<u64> {
        Ok(self.dev_ptr(buf)? as u64)
    }

    fn argmax_rows(&self, out: DeviceBuf, x: DeviceBuf, rows: usize, n: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut nn) = (self.dev_ptr(out)?, self.dev_ptr(x)?, n as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut nn)];
        self.launch(self.kernels.argmax_rows, (rows as u32, 1, 1), 1024, &mut args, stream, "launch(argmax_rows)")
    }

    fn sconv_k4_rows_hist(&self, out: DeviceBuf, state: DeviceBuf, x: DeviceBuf, weight: DeviceBuf,
        dyn_taps: Option<DeviceBuf>, dim: usize, head_dim: usize, rows: usize, dyn_stride: usize,
        hist: Option<DeviceBuf>, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c, mut d, mut e) =
            (self.dev_ptr(out)?, self.dev_ptr(state)?, self.dev_ptr(x)?, self.dev_ptr(weight)?, self.opt_dev_ptr(dyn_taps)?);
        let (mut di, mut hd, mut m, mut ds) = (dim as u32, head_dim as u32, rows as u32, dyn_stride as u32);
        let mut hi = self.opt_dev_ptr(hist)?;
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut e), p(&mut di), p(&mut hd), p(&mut m), p(&mut ds), p(&mut hi)];
        // One block per 256 channels and 16 rows (ST_SCONV_ROWS in the kernel).
        self.launch(self.kernels.sconv_k4_rows_f32, (di.div_ceil(256), m.div_ceil(16), 1), 256, &mut args, stream, "launch(sconv_k4_rows)")
    }

    fn kv_store_fp8_rows(&self, k_cache: DeviceBuf, k_scales: DeviceBuf, v_cache: DeviceBuf,
        v_scales: DeviceBuf, k_in: DeviceBuf, v_in: DeviceBuf, n_kv_heads: usize, max_seq: usize,
        head_dim: usize, pos0: usize, rows: usize, is_local: bool, page_table: Option<DeviceBuf>,
        v_q4: bool, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c, mut d, mut e, mut f) = (self.dev_ptr(k_cache)?, self.dev_ptr(k_scales)?,
            self.dev_ptr(v_cache)?, self.dev_ptr(v_scales)?, self.dev_ptr(k_in)?, self.dev_ptr(v_in)?);
        let (mut nkv, mut ms, mut hd, mut p0, mut loc, mut vq4) = (n_kv_heads as u32, max_seq as u32, head_dim as u32, pos0 as u32, is_local as u32, v_q4 as u32);
        let mut pt = self.opt_dev_ptr(page_table)?;
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut e), p(&mut f), p(&mut nkv), p(&mut ms), p(&mut hd), p(&mut p0), p(&mut loc), p(&mut pt), p(&mut vq4)];
        self.launch(self.kernels.kv_store_fp8_rows, (nkv, rows as u32, 1), hd, &mut args, stream, "launch(kv_store_fp8_rows)")
    }

    fn attn_prefill_fp8(&self, out: DeviceBuf, q: DeviceBuf, k_cache: DeviceBuf, k_scales: DeviceBuf,
        v_cache: DeviceBuf, v_scales: DeviceBuf, rel_table: Option<DeviceBuf>, sinks: Option<DeviceBuf>,
        pp: PrefillAttnParams, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h) = (self.dev_ptr(out)?, self.dev_ptr(q)?,
            self.dev_ptr(k_cache)?, self.dev_ptr(k_scales)?, self.dev_ptr(v_cache)?, self.dev_ptr(v_scales)?,
            self.opt_dev_ptr(rel_table)?, self.opt_dev_ptr(sinks)?);
        let (mut nq, mut nkv, mut ms, mut win, mut p0, mut re, mut isd, mut vq4) = (pp.n_q_heads as u32, pp.n_kv_heads as u32,
            pp.max_seq as u32, pp.window as u32, pp.pos0 as u32, pp.rel_extent as u32, pp.inv_sqrt_d, pp.v_q4 as u32);
        let mut pt = self.opt_dev_ptr(pp.page_table)?;
        if std::env::var_os("SMALLTIDE_ATTN_SCALAR").is_none() && pp.n_q_heads % pp.n_kv_heads == 0 {
            let mut rows = pp.rows as u32;
            let nqv = rows * (nq / nkv);
            if let Some(vt) = pp.v_tiles {
                // Pre-decode the V tiles of every key the rows can see, then attend reading them.
                let lo0 = if pp.window != 0 && pp.pos0 + 1 > pp.window { pp.pos0 + 1 - pp.window } else { 0 };
                let (mut key_lo, mut key_hi) = ((lo0 & !15) as u32, (pp.pos0 + pp.rows - 1) as u32);
                let n_tiles = (key_hi - key_lo) / 16 + 1;
                let mut vtp = self.dev_ptr(vt)?;
                let mut targs = [p(&mut vtp), p(&mut e), p(&mut f), p(&mut nkv), p(&mut ms), p(&mut win), p(&mut key_lo), p(&mut key_hi), p(&mut pt), p(&mut vq4)];
                self.launch(self.kernels.kv_v_tiles_f16, (nkv, n_tiles, 1), 128, &mut targs, stream, "launch(kv_v_tiles_f16)")?;
                let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut e), p(&mut f), p(&mut g), p(&mut h),
                    p(&mut nq), p(&mut nkv), p(&mut ms), p(&mut win), p(&mut p0), p(&mut re), p(&mut isd), p(&mut rows), p(&mut pt), p(&mut vq4), p(&mut vtp)];
                return self.launch(self.kernels.attn_prefill_wmma_vt, (nkv, nqv.div_ceil(16), 1), 128, &mut args, stream, "launch(attn_prefill_wmma_vt)");
            }
            let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut e), p(&mut f), p(&mut g), p(&mut h),
                p(&mut nq), p(&mut nkv), p(&mut ms), p(&mut win), p(&mut p0), p(&mut re), p(&mut isd), p(&mut rows), p(&mut pt), p(&mut vq4)];
            return self.launch(self.kernels.attn_prefill_wmma, (nkv, nqv.div_ceil(16), 1), 128, &mut args, stream, "launch(attn_prefill_wmma)");
        }
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut e), p(&mut f), p(&mut g), p(&mut h),
            p(&mut nq), p(&mut nkv), p(&mut ms), p(&mut win), p(&mut p0), p(&mut re), p(&mut isd), p(&mut pt), p(&mut vq4)];
        self.launch(self.kernels.attn_prefill_fp8, (nkv, pp.rows as u32, 1), 128, &mut args, stream, "launch(attn_prefill_fp8)")
    }

    fn router_topk8_rows(&self, route: DeviceBuf, logits: DeviceBuf, bias: Option<DeviceBuf>,
        route_scale: f32, rows: usize, row_stride_words: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c) = (self.dev_ptr(route)?, self.dev_ptr(logits)?, self.opt_dev_ptr(bias)?);
        let (mut rs, mut st) = (route_scale, row_stride_words as u32);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut rs), p(&mut st)];
        self.launch(self.kernels.router_topk8_rows, (rows as u32, 1, 1), 128, &mut args, stream, "launch(router_topk8_rows)")
    }

    fn loop_control_prep_rows(&self, out: DeviceBuf, cur: DeviceBuf, msg: DeviceBuf, hidden: usize,
        eps: f32, rows: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b, mut c) = (self.dev_ptr(out)?, self.dev_ptr(cur)?, self.dev_ptr(msg)?);
        let (mut h, mut e) = (hidden as u32, eps);
        let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut h), p(&mut e)];
        let threads = if rows == 1 { 1024 } else { 256 };
        self.launch(self.kernels.loop_control_prep_rows_f32, (2, rows as u32, 1), threads, &mut args, stream, "launch(loop_control_prep_rows)")
    }

    fn moe_grouped(&self, out: DeviceBuf, mid: DeviceBuf, tmp: DeviceBuf, x: DeviceBuf, route: DeviceBuf,
        pool: DeviceBuf, desc: DeviceBuf, gp: GroupedMoeParams, reduce: bool, stream: Stream) -> Result<()> {
        if gp.n_work > 0 {
            let work = desc.slice(0, gp.n_work * 16);
            let pairs = desc.slice(gp.pairs_offset_bytes, desc.size_bytes - gp.pairs_offset_bytes);
            let (mut a, mut b, mut c, mut d, mut e) =
                (self.dev_ptr(mid)?, self.dev_ptr(x)?, self.dev_ptr(pool)?, self.dev_ptr(work)?, self.dev_ptr(pairs)?);
            let (mut st, mut h, mut i, mut gc, mut uc) = (gp.expert_stride_bytes as u64, gp.hidden_size as u32, gp.intermediate_size as u32, 4.0f32, 25.0f32);
            let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut e), p(&mut st), p(&mut h), p(&mut i), p(&mut gc), p(&mut uc)];
            let wmma = std::env::var_os("SMALLTIDE_MOE_SCALAR").is_none();
            let v1 = std::env::var_os("SMALLTIDE_MOE_WMMA1").is_some();
            if wmma && !v1 {
                // x -> f16 once into `out` (free until the final reduce), f16 `mid`.
                let n = (gp.rows * gp.hidden_size) as u32;
                let (mut xo, mut xi, mut nn) = (self.dev_ptr(out)?, self.dev_ptr(x)?, n);
                let mut cargs = [p(&mut xo), p(&mut xi), p(&mut nn)];
                self.launch(self.kernels.f32_to_f16, (n.div_ceil(1024), 1, 1), 256, &mut cargs, stream, "launch(f32_to_f16)")?;
                let mut x16 = self.dev_ptr(out)?;
                let mut args = [p(&mut a), p(&mut x16), p(&mut c), p(&mut d), p(&mut e), p(&mut st), p(&mut h), p(&mut i), p(&mut gc), p(&mut uc)];
                let use_m64 = std::env::var_os("SMALLTIDE_MOE_M16").is_none();
                let (k_w13, n_w13, k_w2, n_w2) = if use_m64 {
                    (self.kernels.moe_w13_grouped_wmma_m64, "launch(moe_w13_grouped_wmma_m64)", self.kernels.moe_w2_grouped_wmma_m64, "launch(moe_w2_grouped_wmma_m64)")
                } else {
                    (self.kernels.moe_w13_grouped_wmma2, "launch(moe_w13_grouped_wmma2)", self.kernels.moe_w2_grouped_wmma2, "launch(moe_w2_grouped_wmma2)")
                };
                self.launch(k_w13, (gp.n_work as u32, (2 * i).div_ceil(128), 1), 256, &mut args, stream, n_w13)?;
                let (mut a2, mut bm, mut c2, mut d2, mut e2, mut f2) = (self.dev_ptr(tmp)?, self.dev_ptr(mid)?, self.dev_ptr(pool)?,
                    self.dev_ptr(work)?, self.dev_ptr(pairs)?, self.dev_ptr(route)?);
                let mut args = [p(&mut a2), p(&mut bm), p(&mut c2), p(&mut d2), p(&mut e2), p(&mut f2), p(&mut st), p(&mut h), p(&mut i)];
                self.launch(k_w2, (gp.n_work as u32, h.div_ceil(128), 1), 256, &mut args, stream, n_w2)?;
                if reduce {
                    self.moe_reduce_rows(out, tmp, gp.hidden_size, gp.rows, stream)?;
                }
                return Ok(());
            }
            if wmma {
                self.launch(self.kernels.moe_w13_grouped_wmma, (gp.n_work as u32, (2 * i).div_ceil(128), 1), 256, &mut args, stream, "launch(moe_w13_grouped_wmma)")?;
            } else {
                self.launch(self.kernels.moe_w13_grouped, (gp.n_work as u32, i.div_ceil(64), 1), 256, &mut args, stream, "launch(moe_w13_grouped)")?;
            }
            let (mut a, mut b, mut c, mut d, mut e, mut f) = (self.dev_ptr(tmp)?, self.dev_ptr(mid)?, self.dev_ptr(pool)?,
                self.dev_ptr(work)?, self.dev_ptr(pairs)?, self.dev_ptr(route)?);
            let mut args = [p(&mut a), p(&mut b), p(&mut c), p(&mut d), p(&mut e), p(&mut f), p(&mut st), p(&mut h), p(&mut i)];
            if wmma {
                self.launch(self.kernels.moe_w2_grouped_wmma, (gp.n_work as u32, h.div_ceil(128), 1), 256, &mut args, stream, "launch(moe_w2_grouped_wmma)")?;
            } else {
                self.launch(self.kernels.moe_w2_grouped, (gp.n_work as u32, h.div_ceil(64), 1), 256, &mut args, stream, "launch(moe_w2_grouped)")?;
            }
        }
        if reduce {
            self.moe_reduce_rows(out, tmp, gp.hidden_size, gp.rows, stream)?;
        }
        Ok(())
    }

    fn moe_decode_experts(&self, mid: DeviceBuf, tmp: DeviceBuf, x: DeviceBuf, pool: DeviceBuf,
        slot_table: DeviceBuf, topk_ids: DeviceBuf, mp: MoeDispatchParams, stream: Stream) -> Result<()> {
        let (mut a, mut bx, mut c, mut d, mut e) =
            (self.dev_ptr(mid)?, self.dev_ptr(x)?, self.dev_ptr(pool)?, self.dev_ptr(slot_table)?, self.dev_ptr(topk_ids)?);
        let (mut st, mut h, mut i, mut gc, mut uc) =
            (mp.expert_stride_bytes as u64, mp.hidden_size as u32, mp.intermediate_size as u32, 4.0f32, 25.0f32);
        let k = mp.top_k as u32;
        let mut args = [p(&mut a), p(&mut bx), p(&mut c), p(&mut d), p(&mut e), p(&mut st), p(&mut h), p(&mut i), p(&mut gc), p(&mut uc)];
        let v2 = std::env::var_os("SMALLTIDE_MOE_V3").is_none();
        if v2 {
            self.launch(self.kernels.moe_w13_decode, (i.div_ceil(8), k, 1), 256, &mut args, stream, "launch(moe_w13_decode)")?;
        } else {
            self.launch(self.kernels.moe_w13_decode3, (i.div_ceil(32), k, 1), 256, &mut args, stream, "launch(moe_w13_decode3)")?;
        }
        let (mut a, mut bm) = (self.dev_ptr(tmp)?, self.dev_ptr(mid)?);
        let mut args = [p(&mut a), p(&mut bm), p(&mut c), p(&mut d), p(&mut e), p(&mut st), p(&mut h), p(&mut i)];
        if v2 {
            self.launch(self.kernels.moe_w2_decode, (h.div_ceil(8), k, 1), 256, &mut args, stream, "launch(moe_w2_decode)")
        } else {
            self.launch(self.kernels.moe_w2_decode3, (h.div_ceil(32), k, 1), 256, &mut args, stream, "launch(moe_w2_decode3)")
        }
    }

    fn moe_decode_rows(&self, mid: DeviceBuf, tmp: DeviceBuf, x: DeviceBuf, pool: DeviceBuf,
        slot_table: DeviceBuf, route: DeviceBuf, rows: usize, mp: MoeDispatchParams, stream: Stream) -> Result<()> {
        if rows == 0 || rows > 8 {
            return Err(SmallTideError::Backend { backend: "hip", message: format!("moe_decode_rows: {rows} rows (1..=8)") });
        }
        let (mut a, mut bx, mut c, mut d, mut e) =
            (self.dev_ptr(mid)?, self.dev_ptr(x)?, self.dev_ptr(pool)?, self.dev_ptr(slot_table)?, self.dev_ptr(route)?);
        let mut nr = rows as u32;
        let (mut st, mut h, mut i, mut gc, mut uc) =
            (mp.expert_stride_bytes as u64, mp.hidden_size as u32, mp.intermediate_size as u32, 4.0f32, 25.0f32);
        let pairs = nr * 8;
        let mut args = [p(&mut a), p(&mut bx), p(&mut c), p(&mut d), p(&mut e), p(&mut nr), p(&mut st), p(&mut h), p(&mut i), p(&mut gc), p(&mut uc)];
        self.launch(self.kernels.moe_w13_rows, (i.div_ceil(8), pairs, 1), 256, &mut args, stream, "launch(moe_w13_rows)")?;
        let (mut a, mut bm) = (self.dev_ptr(tmp)?, self.dev_ptr(mid)?);
        let mut args = [p(&mut a), p(&mut bm), p(&mut c), p(&mut d), p(&mut e), p(&mut nr), p(&mut st), p(&mut h), p(&mut i)];
        self.launch(self.kernels.moe_w2_rows, (h.div_ceil(8), pairs, 1), 256, &mut args, stream, "launch(moe_w2_rows)")
    }

    fn moe_decode_post(&self, out: DeviceBuf, tmp: DeviceBuf, slot_table: DeviceBuf, topk_ids: DeviceBuf,
        topk_weights: DeviceBuf, top_k: usize, sconv: Option<(DeviceBuf, DeviceBuf, DeviceBuf)>,
        hidden: usize, stream: Stream) -> Result<()> {
        let (mut a, mut bt, mut c, mut d, mut e) = (self.dev_ptr(out)?, self.dev_ptr(tmp)?, self.dev_ptr(slot_table)?,
            self.dev_ptr(topk_ids)?, self.dev_ptr(topk_weights)?);
        let (mut f, mut g, mut r) = match sconv {
            Some((state, w, residual)) => (self.dev_ptr(state)?, self.dev_ptr(w)?, self.dev_ptr(residual)?),
            None => (ptr::null_mut(), ptr::null_mut(), ptr::null_mut()),
        };
        let (mut k, mut h) = (top_k as u32, hidden as u32);
        let mut args = [p(&mut a), p(&mut bt), p(&mut c), p(&mut d), p(&mut e), p(&mut k), p(&mut f), p(&mut g), p(&mut r), p(&mut h)];
        self.launch(self.kernels.moe_post_decode, (h.div_ceil(256), 1, 1), 256, &mut args, stream, "launch(moe_post_decode)")
    }

    fn attn_prep_decode(&self, pp: AttnPrepParams, stream: Stream) -> Result<()> {
        #[repr(C)]
        struct ArgsC {
            q: *mut c_void,
            r: *mut c_void,
            k_raw: *mut c_void,
            v_raw: *mut c_void,
            k_dyn: *mut c_void,
            v_dyn: *mut c_void,
            sconv_k: *mut c_void,
            sconv_v: *mut c_void,
            k_w: *mut c_void,
            v_w: *mut c_void,
            q_scale: *mut c_void,
            r_scale: *mut c_void,
            rel_proj: *mut c_void,
            rel_table: *mut c_void,
            k_cache: *mut c_void,
            k_scales: *mut c_void,
            v_cache: *mut c_void,
            v_scales: *mut c_void,
            n_q: u32,
            n_kv: u32,
            max_seq: u32,
            slot: u32,
            rel_extent: u32,
            qk_thr: f32,
            rel_thr: f32,
            tau: f32,
            eps: f32,
            page_table: *mut c_void,
            v_q4: u32,
        }
        if pp.n_q_heads == 0 || pp.n_kv_heads == 0 {
            return Err(SmallTideError::Backend { backend: "hip", message: "attn_prep_decode: bad head counts".into() });
        }
        let mut a = ArgsC {
            q: self.dev_ptr(pp.q)?,
            r: self.dev_ptr(pp.r)?,
            k_raw: self.dev_ptr(pp.k_raw)?,
            v_raw: self.dev_ptr(pp.v_raw)?,
            k_dyn: self.opt_dev_ptr(pp.k_dyn)?,
            v_dyn: self.opt_dev_ptr(pp.v_dyn)?,
            sconv_k: self.dev_ptr(pp.sconv_k)?,
            sconv_v: self.dev_ptr(pp.sconv_v)?,
            k_w: self.dev_ptr(pp.k_w)?,
            v_w: self.dev_ptr(pp.v_w)?,
            q_scale: self.dev_ptr(pp.q_scale)?,
            r_scale: self.dev_ptr(pp.r_scale)?,
            rel_proj: self.dev_ptr(pp.rel_proj)?,
            rel_table: self.dev_ptr(pp.rel_table)?,
            k_cache: self.dev_ptr(pp.k_cache)?,
            k_scales: self.dev_ptr(pp.k_scales)?,
            v_cache: self.dev_ptr(pp.v_cache)?,
            v_scales: self.dev_ptr(pp.v_scales)?,
            n_q: pp.n_q_heads as u32,
            n_kv: pp.n_kv_heads as u32,
            max_seq: pp.max_seq as u32,
            slot: pp.slot as u32,
            rel_extent: pp.rel_extent as u32,
            qk_thr: pp.qk_threshold,
            rel_thr: pp.rel_threshold,
            tau: pp.tau,
            eps: pp.eps,
            page_table: self.opt_dev_ptr(pp.page_table)?,
            v_q4: pp.v_q4 as u32,
        };
        let blocks = (pp.n_kv_heads + 2 * pp.n_q_heads) as u32;
        let mut args = [p(&mut a)];
        self.launch(self.kernels.attn_prep_decode, (blocks, 1, 1), 128, &mut args, stream, "launch(attn_prep_decode)")
    }

    fn attn_prep_rows(&self, pp: AttnPrepParams, hist: Option<(DeviceBuf, DeviceBuf)>, pos0: usize, is_local: bool,
        rows: usize, tau: Option<(f32, f32)>, stream: Stream) -> Result<()> {
        #[repr(C)]
        struct ExtC {
            hist_k: *mut c_void,
            hist_v: *mut c_void,
            pos0: u32,
            is_local: u32,
            rows: u32,
            tau_alpha: f32,
            tau_floor: f32,
        }
        #[repr(C)]
        struct ArgsC {
            q: *mut c_void,
            r: *mut c_void,
            k_raw: *mut c_void,
            v_raw: *mut c_void,
            k_dyn: *mut c_void,
            v_dyn: *mut c_void,
            sconv_k: *mut c_void,
            sconv_v: *mut c_void,
            k_w: *mut c_void,
            v_w: *mut c_void,
            q_scale: *mut c_void,
            r_scale: *mut c_void,
            rel_proj: *mut c_void,
            rel_table: *mut c_void,
            k_cache: *mut c_void,
            k_scales: *mut c_void,
            v_cache: *mut c_void,
            v_scales: *mut c_void,
            n_q: u32,
            n_kv: u32,
            max_seq: u32,
            slot: u32,
            rel_extent: u32,
            qk_thr: f32,
            rel_thr: f32,
            tau: f32,
            eps: f32,
            page_table: *mut c_void,
            v_q4: u32,
        }
        if pp.n_q_heads == 0 || pp.n_kv_heads == 0 {
            return Err(SmallTideError::Backend { backend: "hip", message: "attn_prep_decode: bad head counts".into() });
        }
        let mut a = ArgsC {
            q: self.dev_ptr(pp.q)?,
            r: self.dev_ptr(pp.r)?,
            k_raw: self.dev_ptr(pp.k_raw)?,
            v_raw: self.dev_ptr(pp.v_raw)?,
            k_dyn: self.opt_dev_ptr(pp.k_dyn)?,
            v_dyn: self.opt_dev_ptr(pp.v_dyn)?,
            sconv_k: self.dev_ptr(pp.sconv_k)?,
            sconv_v: self.dev_ptr(pp.sconv_v)?,
            k_w: self.dev_ptr(pp.k_w)?,
            v_w: self.dev_ptr(pp.v_w)?,
            q_scale: self.dev_ptr(pp.q_scale)?,
            r_scale: self.dev_ptr(pp.r_scale)?,
            rel_proj: self.dev_ptr(pp.rel_proj)?,
            rel_table: self.dev_ptr(pp.rel_table)?,
            k_cache: self.dev_ptr(pp.k_cache)?,
            k_scales: self.dev_ptr(pp.k_scales)?,
            v_cache: self.dev_ptr(pp.v_cache)?,
            v_scales: self.dev_ptr(pp.v_scales)?,
            n_q: pp.n_q_heads as u32,
            n_kv: pp.n_kv_heads as u32,
            max_seq: pp.max_seq as u32,
            slot: pp.slot as u32,
            rel_extent: pp.rel_extent as u32,
            qk_thr: pp.qk_threshold,
            rel_thr: pp.rel_threshold,
            tau: pp.tau,
            eps: pp.eps,
            page_table: self.opt_dev_ptr(pp.page_table)?,
            v_q4: pp.v_q4 as u32,
        };
        let (ta, tf) = tau.unwrap_or((0.0, 1.0));
        let mut e = ExtC {
            hist_k: match hist { Some((k, _)) => self.dev_ptr(k)?, None => std::ptr::null_mut() },
            hist_v: match hist { Some((_, v)) => self.dev_ptr(v)?, None => std::ptr::null_mut() },
            pos0: pos0 as u32,
            is_local: is_local as u32,
            rows: rows as u32,
            tau_alpha: ta,
            tau_floor: tf,
        };
        let blocks = (pp.n_kv_heads + 2 * pp.n_q_heads * rows) as u32;
        let mut args = [p(&mut a), p(&mut e)];
        self.launch(self.kernels.attn_prep_rows, (blocks, 1, 1), 128, &mut args, stream, "launch(attn_prep_rows)")
    }

    fn fused_mix_decode(&self, pp: FusedMixParams, stream: Stream) -> Result<()> {
        #[repr(C)]
        struct ArgsC {
            cur: *mut c_void,
            o_raw: *mut c_void,
            sconv_state: *mut c_void,
            sconv_w: *mut c_void,
            residual: *mut c_void,
            blocks: *mut c_void,
            copy_out: *mut c_void,
            queries: *mut c_void,
            mix_out: *mut c_void,
            norm_w0: *mut c_void,
            norm_w1: *mut c_void,
            norm_w2: *mut c_void,
            norm_out: *mut c_void,
            nb: u32,
            boundary: u32,
            nq: u32,
            hidden: u32,
            eps: f32,
        }
        if pp.hidden > 3 * 1024 || pp.nq == 0 || pp.nq > 3 || pp.nb + 1 >= 16 {
            return Err(SmallTideError::Backend {
                backend: "hip",
                message: format!("fused_mix_decode: unsupported hidden={} nq={} nb={}", pp.hidden, pp.nq, pp.nb),
            });
        }
        let (o_raw, st, w) = match pp.sconv {
            Some((x, st, w)) => (Some(x), Some(st), Some(w)),
            None => (None, None, None),
        };
        let nw = if pp.norm_w[0].is_some() { pp.norm_w } else { [None; 3] };
        let mut a = ArgsC {
            cur: self.dev_ptr(pp.cur)?,
            o_raw: self.opt_dev_ptr(o_raw)?,
            sconv_state: self.opt_dev_ptr(st)?,
            sconv_w: self.opt_dev_ptr(w)?,
            residual: self.dev_ptr(pp.residual)?,
            blocks: self.dev_ptr(pp.blocks)?,
            copy_out: self.opt_dev_ptr(pp.copy_out)?,
            queries: self.opt_dev_ptr(pp.queries)?,
            mix_out: self.opt_dev_ptr(pp.mix_out)?,
            norm_w0: self.opt_dev_ptr(nw[0])?,
            norm_w1: self.opt_dev_ptr(nw[1])?,
            norm_w2: self.opt_dev_ptr(nw[2])?,
            norm_out: self.dev_ptr(pp.norm_out)?,
            nb: pp.nb as u32,
            boundary: pp.boundary as u32,
            nq: pp.nq as u32,
            hidden: pp.hidden as u32,
            eps: pp.eps,
        };
        let mut args = [p(&mut a)];
        self.launch(self.kernels.fused_mix_decode, (1, 1, 1), 1024, &mut args, stream, "launch(fused_mix_decode)")
    }

    fn gemv_multi_rows(&self, segs: &[GemvSegment], rows: usize, stream: Stream) -> Result<()> {
        if rows == 0 {
            return Err(SmallTideError::Backend { backend: "hip", message: format!("gemv_multi_rows: {rows} rows") });
        }
        let no_wmma = std::env::var_os("SMALLTIDE_NO_WMMA_MULTI").is_some();
        let all_wmma = rows >= 4
            && !no_wmma
            && (segs.iter().all(|sg| !sg.is_bf16 && sg.row_scales.is_some() && sg.n_rows % 16 == 0 && sg.k_cols % 32 == 0)
                || segs.iter().all(|sg| sg.is_bf16 && sg.n_rows % 16 == 0 && sg.k_cols % 32 == 0));
        let tile = if all_wmma { 64 } else { 8 };
        if rows <= tile {
            return self.gemv_multi_impl(segs, Some(rows as u32), stream);
        }
        let mut off = 0usize;
        while off < rows {
            let take = (rows - off).min(tile);
            let sliced: Vec<GemvSegment> = segs
                .iter()
                .map(|sg| GemvSegment {
                    w: sg.w,
                    row_scales: sg.row_scales,
                    x: sg.x.slice(off * sg.k_cols * 4, take * sg.k_cols * 4),
                    y: sg.y.slice(off * sg.n_rows * 4, take * sg.n_rows * 4),
                    n_rows: sg.n_rows,
                    k_cols: sg.k_cols,
                    is_bf16: sg.is_bf16,
                })
                .collect();
            self.gemv_multi_impl(&sliced, Some(take as u32), stream)?;
            off += take;
        }
        Ok(())
    }

    fn gemv_multi(&self, segs: &[GemvSegment], stream: Stream) -> Result<()> {
        self.gemv_multi_impl(segs, None, stream)
    }



    fn moe_reduce_rows(&self, out: DeviceBuf, tmp: DeviceBuf, hidden: usize, rows: usize, stream: Stream) -> Result<()> {
        let (mut a, mut b) = (self.dev_ptr(out)?, self.dev_ptr(tmp)?);
        let mut h = hidden as u32;
        let mut args = [p(&mut a), p(&mut b), p(&mut h)];
        self.launch(self.kernels.moe_reduce_rows, (h.div_ceil(256), rows as u32, 1), 256, &mut args, stream, "launch(moe_reduce_rows)")
    }
}

/// Kernel-argument pointer helper.
#[inline]
fn p<T>(v: &mut T) -> *mut c_void {
    v as *mut T as *mut c_void
}

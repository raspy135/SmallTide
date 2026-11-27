//! ABI-stable dynamic FFI bindings to AMD HIP runtime (`amdhip64_7.dll` / `amdhip64_6.dll`)
//! plus embedded `.hsaco` module loading, kernel launch, HIP Graph capture, and Phase 0/1 benchmarks.

use libloading::Library;
use smalltide_core::{Result, SmallTideError};
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr;
use std::time::Instant;

// `EMBEDDED_HSACOS`: one compiled kernel module per GPU architecture, as (architecture, code
// object), in the order of `SMALLTIDE_HIP_ARCH` at build time (default: gfx1201, gfx1200).
include!(concat!(env!("OUT_DIR"), "/hsaco_table.rs"));

pub type HipError = c_int;
pub type HipStream = *mut c_void;
pub type HipEvent = *mut c_void;
pub type HipModule = *mut c_void;
pub type HipFunction = *mut c_void;
pub type HipGraph = *mut c_void;
pub type HipGraphExec = *mut c_void;

pub const HIP_SUCCESS: HipError = 0;
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
pub const HIP_ERROR_NOT_READY: HipError = 600;
pub const HIP_STREAM_NON_BLOCKING: c_uint = 0x01;
pub const HIP_HOST_MALLOC_DEFAULT: c_uint = 0x00;
pub const HIP_EVENT_DISABLE_TIMING: c_uint = 0x02;
pub const HIP_STREAM_CAPTURE_MODE_GLOBAL: c_int = 0;

type FnHipGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> HipError;
type FnHipSetDevice = unsafe extern "C" fn(c_int) -> HipError;
type FnHipDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> HipError;
/// `hipGetDeviceProperties(hipDeviceProp_t*, int)`; the struct is only read as raw bytes.
type FnHipGetDeviceProperties = unsafe extern "C" fn(*mut c_void, c_int) -> HipError;
type FnHipMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> HipError;
type FnHipRuntimeGetVersion = unsafe extern "C" fn(*mut c_int) -> HipError;
type FnHipDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> HipError;
type FnHipMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> HipError;
type FnHipFree = unsafe extern "C" fn(*mut c_void) -> HipError;
type FnHipHostMalloc = unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> HipError;
type FnHipHostFree = unsafe extern "C" fn(*mut c_void) -> HipError;
type FnHipHostGetDevicePointer = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, c_uint) -> HipError;
type FnHipStreamCreateWithFlags = unsafe extern "C" fn(*mut HipStream, c_uint) -> HipError;
type FnHipStreamDestroy = unsafe extern "C" fn(HipStream) -> HipError;
type FnHipStreamSynchronize = unsafe extern "C" fn(HipStream) -> HipError;
type FnHipStreamWaitEvent = unsafe extern "C" fn(HipStream, HipEvent, c_uint) -> HipError;
type FnHipEventCreateWithFlags = unsafe extern "C" fn(*mut HipEvent, c_uint) -> HipError;
type FnHipEventDestroy = unsafe extern "C" fn(HipEvent) -> HipError;
type FnHipEventRecord = unsafe extern "C" fn(HipEvent, HipStream) -> HipError;
type FnHipEventSynchronize = unsafe extern "C" fn(HipEvent) -> HipError;
type FnHipEventQuery = unsafe extern "C" fn(HipEvent) -> HipError;
type FnHipStreamWaitValue32 = unsafe extern "C" fn(HipStream, *mut c_void, u32, u32, u32) -> HipError;

/// hipMemAllocationProp (HIP virtual memory management).
#[repr(C)]
pub struct HipMemAllocationProp {
    pub kind: i32,
    pub requested_handle_type: i32,
    pub location_type: i32,
    pub location_id: i32,
    pub win32_handle_meta_data: *mut c_void,
    pub compression_type: u8,
    pub gpu_direct_rdma_capable: u8,
    pub usage: u16,
}
/// hipMemAccessDesc.
#[repr(C)]
pub struct HipMemAccessDesc {
    pub location_type: i32,
    pub location_id: i32,
    pub flags: i32,
}
pub type FnHipMemGetAllocationGranularity = unsafe extern "C" fn(*mut usize, *const HipMemAllocationProp, u32) -> HipError;
pub type FnHipMemAddressReserve = unsafe extern "C" fn(*mut *mut c_void, usize, usize, *mut c_void, u64) -> HipError;
pub type FnHipMemCreate = unsafe extern "C" fn(*mut *mut c_void, usize, *const HipMemAllocationProp, u64) -> HipError;
pub type FnHipMemMap = unsafe extern "C" fn(*mut c_void, usize, usize, *mut c_void, u64) -> HipError;
pub type FnHipMemSetAccess = unsafe extern "C" fn(*mut c_void, usize, *const HipMemAccessDesc, usize) -> HipError;
pub type FnHipMemUnmap = unsafe extern "C" fn(*mut c_void, usize) -> HipError;

/// Optional virtual-memory entry points (paged KV); all present or the set is `None`.
#[derive(Clone, Copy)]
pub struct HipVmm {
    pub granularity: FnHipMemGetAllocationGranularity,
    pub reserve: FnHipMemAddressReserve,
    pub create: FnHipMemCreate,
    pub map: FnHipMemMap,
    pub set_access: FnHipMemSetAccess,
    pub unmap: FnHipMemUnmap,
}
type FnHipEventElapsedTime = unsafe extern "C" fn(*mut f32, HipEvent, HipEvent) -> HipError;
type FnHipMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> HipError;
type FnHipMemcpyAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, HipStream) -> HipError;
type FnHipModuleLoadData = unsafe extern "C" fn(*mut HipModule, *const c_void) -> HipError;
type FnHipModuleUnload = unsafe extern "C" fn(HipModule) -> HipError;
type FnHipModuleGetFunction =
    unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> HipError;
type FnHipModuleLaunchKernel = unsafe extern "C" fn(
    HipFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    HipStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> HipError;
type FnHipStreamBeginCapture = unsafe extern "C" fn(HipStream, c_int) -> HipError;
type FnHipStreamEndCapture = unsafe extern "C" fn(HipStream, *mut HipGraph) -> HipError;
type FnHipGraphInstantiate = unsafe extern "C" fn(
    *mut HipGraphExec,
    HipGraph,
    *mut *mut c_void,
    *mut c_char,
    usize,
) -> HipError;
type FnHipGraphLaunch = unsafe extern "C" fn(HipGraphExec, HipStream) -> HipError;
type FnHipGraphExecDestroy = unsafe extern "C" fn(HipGraphExec) -> HipError;
type FnHipGraphDestroy = unsafe extern "C" fn(HipGraph) -> HipError;

#[derive(Debug, Clone)]
pub struct HipProbeReport {
    pub loaded_dll: String,
    pub runtime_version: i32,
    pub driver_version: i32,
    pub device_count: i32,
    pub device_name: String,
    /// The architecture the runtime reports for the device ("gfx1201"), if it reports one.
    pub device_arch: Option<String>,
    /// The architecture of the embedded kernels chosen for the device; empty if none fits.
    pub kernel_arch: String,
    pub free_vram_bytes: usize,
    pub total_vram_bytes: usize,
    /// Size of the chosen embedded kernel module (0 if none fits or the build had no hipcc).
    pub hsaco_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct DmaBenchResult {
    pub chunk_bytes: usize,
    pub iterations: usize,
    pub avg_ms: f64,
    pub bandwidth_gb_s: f64,
}

#[derive(Debug, Clone)]
pub struct GraphLaunchBenchResult {
    pub kernels_per_iter: usize,
    pub raw_ms_per_iter: f64,
    pub graph_ms_per_iter: f64,
    pub speedup: f64,
}

#[derive(Debug, Clone)]
pub struct KernelBwResult {
    pub label: String,
    pub bytes_read: usize,
    pub avg_ms: f64,
    pub effective_gb_s: f64,
}

#[derive(Debug, Clone)]
pub struct OverlapBenchResult {
    pub compute_only_ms: f64,
    pub dma_only_ms: f64,
    pub overlapped_ms: f64,
    pub overlap_efficiency_pct: f64,
}

pub struct CoreKernels {
    pub module: HipModule,
    pub noop: HipFunction,
    pub rmsnorm_f32: HipFunction,
    pub gemv_fp8_e4m3_f32: HipFunction,
    pub gemv_nvfp4_f32: HipFunction,
    pub gemv_mxfp8_f32: HipFunction,
    pub moe_w13_swiglu_indirect: HipFunction,
    pub moe_w2_accum_indirect: HipFunction,
    pub router_topk8_f32: HipFunction,
    pub sconv_k4_decode_f32: HipFunction,
    pub kv_store_fp8_decode: HipFunction,
    pub flash_decode_fp8_split: HipFunction,
    pub flash_decode_fp8_reduce: HipFunction,
    pub head_l2_clip_f32: HipFunction,
    pub value_norm_no_gamma_f32: HipFunction,
    pub gate_attn_sigmoid_mul_f32: HipFunction,
    pub attn_res_mix_f32: HipFunction,
    pub gemv_f32: HipFunction,
    pub rel_table_f32: HipFunction,
    pub loop_control_prep_f32: HipFunction,
    pub loop_control_apply_f32: HipFunction,
    pub vec_add_f32: HipFunction,
    pub vec_scale_f32: HipFunction,
    pub vec_copy_f32: HipFunction,
    pub zero_f32: HipFunction,
    pub rescue_write_route: HipFunction,
    pub rescue_apply_patch: HipFunction,
    pub rescue_wait_and_patch: HipFunction,
    pub final_logit_softcap_f32: HipFunction,
    pub argmax_partial_f32: HipFunction,
    pub gemm_fp8_e4m3_batched_f32: HipFunction,
    pub gemm_f32_rows: HipFunction,
    pub attn_res_mix_rows_f32: HipFunction,
    pub head_l2_clip_rows_f32: HipFunction,
    pub sconv_k4_rows_f32: HipFunction,
    pub kv_store_fp8_rows: HipFunction,
    pub attn_prefill_fp8: HipFunction,
    pub router_topk8_rows: HipFunction,
    pub loop_control_prep_rows_f32: HipFunction,
    pub moe_w13_grouped: HipFunction,
    pub moe_w2_grouped: HipFunction,
    pub moe_reduce_rows: HipFunction,
    pub moe_w13_decode: HipFunction,
    pub moe_w2_decode: HipFunction,
    pub moe_post_decode: HipFunction,
    pub moe_w13_decode3: HipFunction,
    pub moe_w2_decode3: HipFunction,
    pub gemv_multi: HipFunction,
    pub attn_prep_decode: HipFunction,
    pub fused_mix_decode: HipFunction,
    pub moe_w13_rows: HipFunction,
    pub moe_w13_grouped_wmma: HipFunction,
    pub moe_w2_grouped_wmma: HipFunction,
    pub moe_w13_grouped_wmma2: HipFunction,
    pub moe_w2_grouped_wmma2: HipFunction,
    pub moe_w13_grouped_wmma_m64: HipFunction,
    pub moe_w2_grouped_wmma_m64: HipFunction,
    pub f32_to_f16: HipFunction,
    pub vit_gemm_bf16: HipFunction,
    pub vit_epilogue: HipFunction,
    pub vit_ln: HipFunction,
    pub vit_rope: HipFunction,
    pub vit_attn: HipFunction,
    pub attn_prefill_wmma: HipFunction,
    pub attn_prefill_wmma_vt: HipFunction,
    pub kv_v_tiles_f16: HipFunction,
    pub gemm_fp8_rows_wmma: HipFunction,
    pub gemm_fp8_rows_wmma_x16: HipFunction,
    pub gemm_fp8_rows_wmma_x16_m32: HipFunction,
    pub gemm_fp8_rows_wmma_x16_m64: HipFunction,
    pub gemm_fp8_rows_wmma_small: HipFunction,
    pub gemv_fp8_rows: HipFunction,
    pub sconv_rewind: HipFunction,
    pub argmax_rows: HipFunction,
    pub moe_group_rows: HipFunction,
    pub moe_post_rows: HipFunction,
    pub gemv_fp8_rows2: HipFunction,
    pub moe_group_rows2: HipFunction,
    pub gemv_multi_rows: HipFunction,
    pub attn_prep_rows: HipFunction,
    pub moe_post_sconv_rows: HipFunction,
    pub mix_rows: HipFunction,
    pub ctrl_prep_multi: HipFunction,
    pub ctrl_apply_norm_multi: HipFunction,
    pub moe_group_rows3: HipFunction,
    pub dspark_mix_rms: HipFunction,
    pub dspark_head_norm_rope: HipFunction,
    pub dspark_ctx_store: HipFunction,
    pub dspark_attn: HipFunction,
    pub silu_mul: HipFunction,
    pub dspark_z: HipFunction,
    pub dspark_cell: HipFunction,
    pub dspark_logits_argmax: HipFunction,
    pub dspark_argmax_final: HipFunction,
    pub dspark_argmax_final_margin: HipFunction,
    pub argmax_rows_part: HipFunction,
    pub argmax_rows_reduce: HipFunction,
    pub argmax_rows_reduce_max: HipFunction,
    pub logits_filter_rows: HipFunction,
    pub attn_prefill_wmma_split: HipFunction,
    pub attn_prefill_wmma_reduce: HipFunction,
    pub gemv_multi_wmma: HipFunction,
    pub gemv_multi_wmma_m32: HipFunction,
    pub gemv_multi_wmma_m64: HipFunction,
    pub gemv_bf16_f32: HipFunction,
    pub gemv_bf16_rows: HipFunction,
    pub gemm_bf16_rows_wmma_small: HipFunction,
    pub gemm_bf16_rows_wmma_m32: HipFunction,
    pub gemm_bf16_rows_wmma_m64: HipFunction,
    pub gemv_multi_wmma_bf16: HipFunction,
    pub gemv_multi_wmma_bf16_m32: HipFunction,
    pub gemv_multi_wmma_bf16_m64: HipFunction,
    pub dspark_quant_fold_fp8: HipFunction,
    pub dspark_logits_argmax_fp8: HipFunction,
    pub dspark_rnn_step: HipFunction,
    pub dspark_ctx_store_fp8: HipFunction,
    pub dspark_attn_fp8_split: HipFunction,
    pub dspark_attn_reduce: HipFunction,
    pub dspark_embed_prep: HipFunction,
    pub add_rmsnorm_rows: HipFunction,
    pub dspark_qk_norm_rope: HipFunction,
    pub moe_w2_rows: HipFunction,
    pub dflash2_attn_fp8_split: HipFunction,
    pub dflash2_conv_prepare: HipFunction,
    pub dflash2_conv_finish_add_rmsnorm: HipFunction,
    pub topk16_rows_part: HipFunction,
    pub topk16_rows_reduce: HipFunction,
}

pub struct HipRuntime {
    _lib: Library,
    loaded_dll: String,
    pub hip_get_device_count: FnHipGetDeviceCount,
    pub hip_set_device: FnHipSetDevice,
    pub hip_device_get_name: FnHipDeviceGetName,
    /// Optional: only used to read the device's architecture name.
    pub hip_get_device_properties: Option<FnHipGetDeviceProperties>,
    pub hip_mem_get_info: FnHipMemGetInfo,
    pub hip_runtime_get_version: FnHipRuntimeGetVersion,
    pub hip_driver_get_version: FnHipDriverGetVersion,
    pub hip_malloc: FnHipMalloc,
    pub hip_free: FnHipFree,
    pub hip_host_malloc: FnHipHostMalloc,
    pub hip_host_free: FnHipHostFree,
    pub hip_host_get_device_pointer: FnHipHostGetDevicePointer,
    pub hip_stream_create_with_flags: FnHipStreamCreateWithFlags,
    pub hip_stream_destroy: FnHipStreamDestroy,
    pub hip_stream_synchronize: FnHipStreamSynchronize,
    pub hip_stream_wait_event: FnHipStreamWaitEvent,
    pub hip_event_create_with_flags: FnHipEventCreateWithFlags,
    pub hip_event_destroy: FnHipEventDestroy,
    pub hip_event_record: FnHipEventRecord,
    pub hip_event_synchronize: FnHipEventSynchronize,
    pub hip_event_query: FnHipEventQuery,
    /// Optional (older runtimes may lack it): GPU-side wait on a 32-bit value in memory.
    pub hip_stream_wait_value32: Option<FnHipStreamWaitValue32>,
    pub vmm: Option<HipVmm>,
    pub hip_event_elapsed_time: FnHipEventElapsedTime,
    pub hip_memcpy: FnHipMemcpy,
    pub hip_memcpy_async: FnHipMemcpyAsync,
    pub hip_module_load_data: FnHipModuleLoadData,
    pub hip_module_unload: FnHipModuleUnload,
    pub hip_module_get_function: FnHipModuleGetFunction,
    pub hip_module_launch_kernel: FnHipModuleLaunchKernel,
    pub hip_stream_begin_capture: FnHipStreamBeginCapture,
    pub hip_stream_end_capture: FnHipStreamEndCapture,
    pub hip_graph_instantiate: FnHipGraphInstantiate,
    pub hip_graph_launch: FnHipGraphLaunch,
    pub hip_graph_exec_destroy: FnHipGraphExecDestroy,
    pub hip_graph_destroy: FnHipGraphDestroy,
}

unsafe impl Send for HipRuntime {}
unsafe impl Sync for HipRuntime {}

pub fn check_hip(code: HipError, op: &'static str) -> Result<()> {
    if code == HIP_SUCCESS {
        Ok(())
    } else {
        Err(SmallTideError::Backend {
            backend: "hip",
            message: format!("{op} failed with hipError_t={code}"),
        })
    }
}

/// The first "gfx..." token in a `hipDeviceProp_t` read as bytes: "gfx1201" from "gfx1201",
/// "gfx90a" from "gfx90a:sramecc+:xnack-".
fn parse_gcn_arch(bytes: &[u8]) -> Option<String> {
    let start = bytes.windows(3).position(|w| w == b"gfx")?;
    let len = bytes[start..].iter().take_while(|b| b.is_ascii_alphanumeric() || **b == b'-').count();
    (len > 3).then(|| String::from_utf8_lossy(&bytes[start..start + len]).into_owned())
}

#[cfg(test)]
mod arch_tests {
    use super::{parse_gcn_arch, EMBEDDED_HSACOS};

    #[test]
    fn gcn_arch_is_found_anywhere_in_the_struct() {
        let mut prop = vec![0u8; 2048];
        prop[..21].copy_from_slice(b"AMD Radeon RX 9060 XT");
        prop[1400..1407].copy_from_slice(b"gfx1200");
        assert_eq!(parse_gcn_arch(&prop).as_deref(), Some("gfx1200"));
        prop[1400..1422].copy_from_slice(b"gfx90a:sramecc+:xnack-");
        assert_eq!(parse_gcn_arch(&prop).as_deref(), Some("gfx90a"));
        assert_eq!(parse_gcn_arch(&[0u8; 2048]), None);
        assert_eq!(parse_gcn_arch(b"AMD Radeon RX 9070 XT "), None);
    }

    #[test]
    fn kernel_table_has_distinct_architectures() {
        assert!(!EMBEDDED_HSACOS.is_empty());
        for (i, (a, _)) in EMBEDDED_HSACOS.iter().enumerate() {
            assert!(EMBEDDED_HSACOS[..i].iter().all(|(b, _)| b != a), "{a} listed twice");
        }
    }
}

impl HipRuntime {
    pub fn load() -> Result<Self> {
        #[cfg(windows)]
        let candidates: &[&str] = &["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"];
        #[cfg(not(windows))]
        let candidates: &[&str] = crate::linux::HIP_RUNTIME_CANDIDATES;
        let mut last_err = String::new();
        for &name in candidates {
            match unsafe { Library::new(name) } {
                Ok(lib) => return Self::from_library(lib, name.to_string()),
                Err(e) => last_err = format!("{name}: {e}"),
            }
        }
        #[cfg(windows)]
        return Err(SmallTideError::BackendUnavailable(format!(
            "failed to load any AMD HIP runtime DLL ({last_err}). Install AMD Software: Adrenalin Edition (26.10.44+ recommended)."
        )));
        #[cfg(not(windows))]
        return Err(SmallTideError::BackendUnavailable(format!(
            "failed to load any AMD HIP runtime library ({last_err}). {}",
            crate::linux::HIP_RUNTIME_HINT
        )));
    }

    fn from_library(lib: Library, loaded_dll: String) -> Result<Self> {
        macro_rules! load_sym {
            ($sym:literal, $ty:ty) => {{
                let f: libloading::Symbol<$ty> = unsafe { lib.get($sym) }.map_err(|e| {
                    SmallTideError::BackendUnavailable(format!(
                        "missing symbol {:?} in {loaded_dll}: {e}",
                        std::str::from_utf8($sym).unwrap_or("?")
                    ))
                })?;
                *f
            }};
        }

        Ok(Self {
            hip_get_device_count: load_sym!(b"hipGetDeviceCount\0", FnHipGetDeviceCount),
            hip_set_device: load_sym!(b"hipSetDevice\0", FnHipSetDevice),
            hip_device_get_name: load_sym!(b"hipDeviceGetName\0", FnHipDeviceGetName),
            hip_get_device_properties: unsafe {
                lib.get::<FnHipGetDeviceProperties>(b"hipGetDevicePropertiesR0600\0")
                    .or_else(|_| lib.get::<FnHipGetDeviceProperties>(b"hipGetDeviceProperties\0"))
                    .ok()
                    .map(|f| *f)
            },
            hip_mem_get_info: load_sym!(b"hipMemGetInfo\0", FnHipMemGetInfo),
            hip_runtime_get_version: load_sym!(b"hipRuntimeGetVersion\0", FnHipRuntimeGetVersion),
            hip_driver_get_version: load_sym!(b"hipDriverGetVersion\0", FnHipDriverGetVersion),
            hip_malloc: load_sym!(b"hipMalloc\0", FnHipMalloc),
            hip_free: load_sym!(b"hipFree\0", FnHipFree),
            hip_host_malloc: load_sym!(b"hipHostMalloc\0", FnHipHostMalloc),
            hip_host_free: load_sym!(b"hipHostFree\0", FnHipHostFree),
            hip_host_get_device_pointer: load_sym!(b"hipHostGetDevicePointer\0", FnHipHostGetDevicePointer),
            hip_stream_create_with_flags: load_sym!(
                b"hipStreamCreateWithFlags\0",
                FnHipStreamCreateWithFlags
            ),
            hip_stream_destroy: load_sym!(b"hipStreamDestroy\0", FnHipStreamDestroy),
            hip_stream_synchronize: load_sym!(b"hipStreamSynchronize\0", FnHipStreamSynchronize),
            hip_stream_wait_event: load_sym!(b"hipStreamWaitEvent\0", FnHipStreamWaitEvent),
            hip_event_create_with_flags: load_sym!(
                b"hipEventCreateWithFlags\0",
                FnHipEventCreateWithFlags
            ),
            hip_event_destroy: load_sym!(b"hipEventDestroy\0", FnHipEventDestroy),
            hip_event_record: load_sym!(b"hipEventRecord\0", FnHipEventRecord),
            hip_event_synchronize: load_sym!(b"hipEventSynchronize\0", FnHipEventSynchronize),
            hip_event_query: load_sym!(b"hipEventQuery\0", FnHipEventQuery),
            hip_stream_wait_value32: unsafe { lib.get::<FnHipStreamWaitValue32>(b"hipStreamWaitValue32\0").ok().map(|f| *f) },
            vmm: unsafe {
                (|| {
                    Some(HipVmm {
                        granularity: *lib.get::<FnHipMemGetAllocationGranularity>(b"hipMemGetAllocationGranularity\0").ok()?,
                        reserve: *lib.get::<FnHipMemAddressReserve>(b"hipMemAddressReserve\0").ok()?,
                        create: *lib.get::<FnHipMemCreate>(b"hipMemCreate\0").ok()?,
                        map: *lib.get::<FnHipMemMap>(b"hipMemMap\0").ok()?,
                        set_access: *lib.get::<FnHipMemSetAccess>(b"hipMemSetAccess\0").ok()?,
                        unmap: *lib.get::<FnHipMemUnmap>(b"hipMemUnmap\0").ok()?,
                    })
                })()
            },
            hip_event_elapsed_time: load_sym!(b"hipEventElapsedTime\0", FnHipEventElapsedTime),
            hip_memcpy: load_sym!(b"hipMemcpy\0", FnHipMemcpy),
            hip_memcpy_async: load_sym!(b"hipMemcpyAsync\0", FnHipMemcpyAsync),
            hip_module_load_data: load_sym!(b"hipModuleLoadData\0", FnHipModuleLoadData),
            hip_module_unload: load_sym!(b"hipModuleUnload\0", FnHipModuleUnload),
            hip_module_get_function: load_sym!(b"hipModuleGetFunction\0", FnHipModuleGetFunction),
            hip_module_launch_kernel: load_sym!(
                b"hipModuleLaunchKernel\0",
                FnHipModuleLaunchKernel
            ),
            hip_stream_begin_capture: load_sym!(
                b"hipStreamBeginCapture\0",
                FnHipStreamBeginCapture
            ),
            hip_stream_end_capture: load_sym!(b"hipStreamEndCapture\0", FnHipStreamEndCapture),
            hip_graph_instantiate: load_sym!(b"hipGraphInstantiate\0", FnHipGraphInstantiate),
            hip_graph_launch: load_sym!(b"hipGraphLaunch\0", FnHipGraphLaunch),
            hip_graph_exec_destroy: load_sym!(b"hipGraphExecDestroy\0", FnHipGraphExecDestroy),
            hip_graph_destroy: load_sym!(b"hipGraphDestroy\0", FnHipGraphDestroy),
            _lib: lib,
            loaded_dll,
        })
    }

    /// The architecture name the runtime reports for a device ("gfx1201", "gfx1200"), or `None`
    /// if it does not say. `hipDeviceProp_t` changes layout between HIP versions, so its
    /// `gcnArchName` field is found by scanning the returned bytes for "gfx" instead of by offset
    /// (the device name, the only other text in the struct, is a product name).
    pub fn device_arch(&self, device_id: i32) -> Option<String> {
        let get_props = self.hip_get_device_properties?;
        // Far larger than any hipDeviceProp_t (about 1.5 KiB); u64 for alignment.
        let mut buf = vec![0u64; 2048];
        if unsafe { get_props(buf.as_mut_ptr() as *mut c_void, device_id) } != HIP_SUCCESS {
            return None;
        }
        let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 8) };
        parse_gcn_arch(bytes)
    }

    /// The embedded kernel module for a device, as (architecture, code object).
    /// `SMALLTIDE_KERNEL_ARCH` forces one; otherwise the device's architecture picks it, and a
    /// runtime that does not name the architecture gets the first one built.
    pub fn kernels_for_device(&self, device_id: i32) -> Result<(&'static str, &'static [u8])> {
        let built = || EMBEDDED_HSACOS.iter().map(|(a, _)| *a).collect::<Vec<_>>().join(", ");
        let forced = std::env::var("SMALLTIDE_KERNEL_ARCH").ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let (arch, code) = if let Some(want) = forced {
            *EMBEDDED_HSACOS.iter().find(|(a, _)| *a == want).ok_or_else(|| {
                SmallTideError::BackendUnavailable(format!(
                    "SMALLTIDE_KERNEL_ARCH={want}: this build has kernels for {} only",
                    built()
                ))
            })?
        } else {
            match self.device_arch(device_id) {
                Some(dev_arch) => *EMBEDDED_HSACOS.iter().find(|(a, _)| *a == dev_arch).ok_or_else(|| {
                    SmallTideError::BackendUnavailable(format!(
                        "this GPU is {dev_arch}; this build has kernels for {} only (RDNA 4: RX 9070 / 9060 families). \
                         Build from source with SMALLTIDE_HIP_ARCH={dev_arch} if the kernels compile for it, or set \
                         SMALLTIDE_KERNEL_ARCH to try one of the built ones",
                        built()
                    ))
                })?,
                None => *EMBEDDED_HSACOS.first().ok_or_else(|| {
                    SmallTideError::BackendUnavailable("no kernel module was built into this binary".into())
                })?,
            }
        };
        if code.is_empty() {
            return Err(SmallTideError::BackendUnavailable(format!(
                "embedded {arch} .hsaco is empty (rebuild after installing ROCm HIP SDK)"
            )));
        }
        Ok((arch, code))
    }

    /// Load the kernel module built for `device_id`'s architecture (see `kernels_for_device`).
    pub fn load_core_kernels(&self, device_id: i32) -> Result<CoreKernels> {
        let (arch, code) = self.kernels_for_device(device_id)?;
        let mut module: HipModule = ptr::null_mut();
        unsafe {
            check_hip((self.hip_set_device)(device_id), "hipSetDevice")?;
            let rc = (self.hip_module_load_data)(&mut module, code.as_ptr() as *const c_void);
            if rc != HIP_SUCCESS {
                return Err(SmallTideError::Backend {
                    backend: "hip",
                    message: format!("hipModuleLoadData failed with hipError_t={rc} loading the {arch} kernels"),
                });
            }
        }
        let get_fn = |name: &str| -> Result<HipFunction> {
            let c = CString::new(name).unwrap();
            let mut f: HipFunction = ptr::null_mut();
            unsafe {
                check_hip(
                    (self.hip_module_get_function)(&mut f, module, c.as_ptr()),
                    "hipModuleGetFunction",
                )?;
            }
            Ok(f)
        };
        Ok(CoreKernels {
            module,
            noop: get_fn("smalltide_noop")?,
            rmsnorm_f32: get_fn("smalltide_rmsnorm_f32")?,
            gemv_fp8_e4m3_f32: get_fn("smalltide_gemv_fp8_e4m3_f32")?,
            gemv_nvfp4_f32: get_fn("smalltide_gemv_nvfp4_f32")?,
            gemv_mxfp8_f32: get_fn("smalltide_gemv_mxfp8_f32")?,
            moe_w13_swiglu_indirect: get_fn("smalltide_moe_w13_swiglu_indirect")?,
            moe_w2_accum_indirect: get_fn("smalltide_moe_w2_accum_indirect")?,
            router_topk8_f32: get_fn("smalltide_router_topk8_f32")?,
            sconv_k4_decode_f32: get_fn("smalltide_sconv_k4_decode_f32")?,
            kv_store_fp8_decode: get_fn("smalltide_kv_store_fp8_decode")?,
            flash_decode_fp8_split: get_fn("smalltide_flash_decode_fp8_split")?,
            flash_decode_fp8_reduce: get_fn("smalltide_flash_decode_fp8_reduce")?,
            head_l2_clip_f32: get_fn("smalltide_head_l2_clip_f32")?,
            value_norm_no_gamma_f32: get_fn("smalltide_value_norm_no_gamma_f32")?,
            gate_attn_sigmoid_mul_f32: get_fn("smalltide_gate_attn_sigmoid_mul_f32")?,
            attn_res_mix_f32: get_fn("smalltide_attn_res_mix_f32")?,
            gemv_f32: get_fn("smalltide_gemv_f32")?,
            rel_table_f32: get_fn("smalltide_rel_table_f32")?,
            loop_control_prep_f32: get_fn("smalltide_loop_control_prep_f32")?,
            loop_control_apply_f32: get_fn("smalltide_loop_control_apply_f32")?,
            vec_add_f32: get_fn("smalltide_vec_add_f32")?,
            vec_scale_f32: get_fn("smalltide_vec_scale_f32")?,
            vec_copy_f32: get_fn("smalltide_vec_copy_f32")?,
            zero_f32: get_fn("smalltide_zero_f32")?,
            rescue_write_route: get_fn("smalltide_rescue_write_route")?,
            rescue_apply_patch: get_fn("smalltide_rescue_apply_patch")?,
            rescue_wait_and_patch: get_fn("smalltide_rescue_wait_and_patch")?,
            final_logit_softcap_f32: get_fn("smalltide_final_logit_softcap_f32")?,
            argmax_partial_f32: get_fn("smalltide_argmax_partial_f32")?,
            gemm_fp8_e4m3_batched_f32: get_fn("smalltide_gemm_fp8_e4m3_batched_f32")?,
            gemm_f32_rows: get_fn("smalltide_gemm_f32_rows")?,
            attn_res_mix_rows_f32: get_fn("smalltide_attn_res_mix_rows_f32")?,
            head_l2_clip_rows_f32: get_fn("smalltide_head_l2_clip_rows_f32")?,
            sconv_k4_rows_f32: get_fn("smalltide_sconv_k4_rows_f32")?,
            kv_store_fp8_rows: get_fn("smalltide_kv_store_fp8_rows")?,
            attn_prefill_fp8: get_fn("smalltide_attn_prefill_fp8")?,
            router_topk8_rows: get_fn("smalltide_router_topk8_rows")?,
            loop_control_prep_rows_f32: get_fn("smalltide_loop_control_prep_rows_f32")?,
            moe_w13_grouped: get_fn("smalltide_moe_w13_grouped")?,
            moe_w2_grouped: get_fn("smalltide_moe_w2_grouped")?,
            moe_reduce_rows: get_fn("smalltide_moe_reduce_rows")?,
            moe_w13_decode: get_fn("smalltide_moe_w13_decode")?,
            moe_w2_decode: get_fn("smalltide_moe_w2_decode")?,
            moe_post_decode: get_fn("smalltide_moe_post_decode")?,
            moe_w13_decode3: get_fn("smalltide_moe_w13_decode3")?,
            moe_w2_decode3: get_fn("smalltide_moe_w2_decode3")?,
            gemv_multi: get_fn("smalltide_gemv_multi")?,
            attn_prep_decode: get_fn("smalltide_attn_prep_decode")?,
            fused_mix_decode: get_fn("smalltide_fused_mix_decode")?,
            moe_w13_rows: get_fn("smalltide_moe_w13_rows")?,
            moe_w13_grouped_wmma: get_fn("smalltide_moe_w13_grouped_wmma")?,
            moe_w2_grouped_wmma: get_fn("smalltide_moe_w2_grouped_wmma")?,
            moe_w13_grouped_wmma2: get_fn("smalltide_moe_w13_grouped_wmma2")?,
            moe_w2_grouped_wmma2: get_fn("smalltide_moe_w2_grouped_wmma2")?,
            moe_w13_grouped_wmma_m64: get_fn("smalltide_moe_w13_grouped_wmma_m64")?,
            moe_w2_grouped_wmma_m64: get_fn("smalltide_moe_w2_grouped_wmma_m64")?,
            f32_to_f16: get_fn("smalltide_f32_to_f16")?,
            vit_gemm_bf16: get_fn("smalltide_vit_gemm_bf16")?,
            vit_epilogue: get_fn("smalltide_vit_epilogue")?,
            vit_ln: get_fn("smalltide_vit_ln")?,
            vit_rope: get_fn("smalltide_vit_rope")?,
            vit_attn: get_fn("smalltide_vit_attn")?,
            attn_prefill_wmma: get_fn("smalltide_attn_prefill_wmma")?,
            attn_prefill_wmma_vt: get_fn("smalltide_attn_prefill_wmma_vt")?,
            kv_v_tiles_f16: get_fn("smalltide_kv_v_tiles_f16")?,
            gemm_fp8_rows_wmma: get_fn("smalltide_gemm_fp8_rows_wmma")?,
            gemm_fp8_rows_wmma_x16: get_fn("smalltide_gemm_fp8_rows_wmma_x16")?,
            gemm_fp8_rows_wmma_x16_m32: get_fn("smalltide_gemm_fp8_rows_wmma_x16_m32")?,
            gemm_fp8_rows_wmma_x16_m64: get_fn("smalltide_gemm_fp8_rows_wmma_x16_m64")?,
            gemm_fp8_rows_wmma_small: get_fn("smalltide_gemm_fp8_rows_wmma_small")?,
            gemv_fp8_rows: get_fn("smalltide_gemv_fp8_rows")?,
            sconv_rewind: get_fn("smalltide_sconv_rewind")?,
            argmax_rows: get_fn("smalltide_argmax_rows")?,
            moe_group_rows: get_fn("smalltide_moe_group_rows")?,
            moe_post_rows: get_fn("smalltide_moe_post_rows")?,
            gemv_fp8_rows2: get_fn("smalltide_gemv_fp8_rows2")?,
            moe_group_rows2: get_fn("smalltide_moe_group_rows2")?,
            gemv_multi_rows: get_fn("smalltide_gemv_multi_rows")?,
            attn_prep_rows: get_fn("smalltide_attn_prep_rows")?,
            moe_post_sconv_rows: get_fn("smalltide_moe_post_sconv_rows")?,
            mix_rows: get_fn("smalltide_mix_rows")?,
            ctrl_prep_multi: get_fn("smalltide_ctrl_prep_multi")?,
            ctrl_apply_norm_multi: get_fn("smalltide_ctrl_apply_norm_multi")?,
            moe_group_rows3: get_fn("smalltide_moe_group_rows3")?,
            dspark_mix_rms: get_fn("smalltide_dspark_mix_rms")?,
            dspark_head_norm_rope: get_fn("smalltide_dspark_head_norm_rope")?,
            dspark_ctx_store: get_fn("smalltide_dspark_ctx_store")?,
            dspark_attn: get_fn("smalltide_dspark_attn")?,
            silu_mul: get_fn("smalltide_silu_mul")?,
            dspark_z: get_fn("smalltide_dspark_z")?,
            dspark_cell: get_fn("smalltide_dspark_cell")?,
            dspark_logits_argmax: get_fn("smalltide_dspark_logits_argmax")?,
            dspark_argmax_final: get_fn("smalltide_dspark_argmax_final")?,
            dspark_argmax_final_margin: get_fn("smalltide_dspark_argmax_final_margin")?,
            argmax_rows_part: get_fn("smalltide_argmax_rows_part")?,
            argmax_rows_reduce: get_fn("smalltide_argmax_rows_reduce")?,
            argmax_rows_reduce_max: get_fn("smalltide_argmax_rows_reduce_max")?,
            logits_filter_rows: get_fn("smalltide_logits_filter_rows")?,
            attn_prefill_wmma_split: get_fn("smalltide_attn_prefill_wmma_split")?,
            attn_prefill_wmma_reduce: get_fn("smalltide_attn_prefill_wmma_reduce")?,
            gemv_multi_wmma: get_fn("smalltide_gemv_multi_wmma")?,
            gemv_multi_wmma_m32: get_fn("smalltide_gemv_multi_wmma_m32")?,
            gemv_multi_wmma_m64: get_fn("smalltide_gemv_multi_wmma_m64")?,
            gemv_bf16_f32: get_fn("smalltide_gemv_bf16_f32")?,
            gemv_bf16_rows: get_fn("smalltide_gemv_bf16_rows")?,
            gemm_bf16_rows_wmma_small: get_fn("smalltide_gemm_bf16_rows_wmma_small")?,
            gemm_bf16_rows_wmma_m32: get_fn("smalltide_gemm_bf16_rows_wmma_m32")?,
            gemm_bf16_rows_wmma_m64: get_fn("smalltide_gemm_bf16_rows_wmma_m64")?,
            gemv_multi_wmma_bf16: get_fn("smalltide_gemv_multi_wmma_bf16")?,
            gemv_multi_wmma_bf16_m32: get_fn("smalltide_gemv_multi_wmma_bf16_m32")?,
            gemv_multi_wmma_bf16_m64: get_fn("smalltide_gemv_multi_wmma_bf16_m64")?,
            dspark_quant_fold_fp8: get_fn("smalltide_dspark_quant_fold_fp8")?,
            dspark_logits_argmax_fp8: get_fn("smalltide_dspark_logits_argmax_fp8")?,
            dspark_rnn_step: get_fn("smalltide_dspark_rnn_step")?,
            dspark_ctx_store_fp8: get_fn("smalltide_dspark_ctx_store_fp8")?,
            dspark_attn_fp8_split: get_fn("smalltide_dspark_attn_fp8_split")?,
            dspark_attn_reduce: get_fn("smalltide_dspark_attn_reduce")?,
            dspark_embed_prep: get_fn("smalltide_dspark_embed_prep")?,
            add_rmsnorm_rows: get_fn("smalltide_add_rmsnorm_rows")?,
            dspark_qk_norm_rope: get_fn("smalltide_dspark_qk_norm_rope")?,
            moe_w2_rows: get_fn("smalltide_moe_w2_rows")?,
            dflash2_attn_fp8_split: get_fn("smalltide_dflash2_attn_fp8_split")?,
            dflash2_conv_prepare: get_fn("smalltide_dflash2_conv_prepare")?,
            dflash2_conv_finish_add_rmsnorm: get_fn("smalltide_dflash2_conv_finish_add_rmsnorm")?,
            topk16_rows_part: get_fn("smalltide_topk16_rows_part")?,
            topk16_rows_reduce: get_fn("smalltide_topk16_rows_reduce")?,
        })
    }

    /// Load the device's kernel module, resolve every kernel and run one on the card (a counter
    /// increment on 4 bytes of VRAM). `probe` runs it, so a card whose kernels do not load or
    /// run is reported by `setup` before the model download.
    pub fn check_kernels(&self, device_id: i32) -> Result<()> {
        let kernels = self.load_core_kernels(device_id)?;
        let run = || -> Result<()> {
            let mut d_counter: *mut c_void = ptr::null_mut();
            unsafe { check_hip((self.hip_malloc)(&mut d_counter, 4), "hipMalloc(counter)")? };
            let counted = (|| -> Result<u32> {
                let mut count: u32 = 0;
                let mut arg_ptr = d_counter;
                let mut params: [*mut c_void; 1] = [&mut arg_ptr as *mut *mut c_void as *mut c_void];
                unsafe {
                    check_hip(
                        (self.hip_memcpy)(d_counter, &count as *const u32 as *const c_void, 4, HIP_MEMCPY_HOST_TO_DEVICE),
                        "hipMemcpy(counter)",
                    )?;
                    check_hip(
                        (self.hip_module_launch_kernel)(
                            kernels.noop, 1, 1, 1, 32, 1, 1, 0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut(),
                        ),
                        "launch(noop)",
                    )?;
                    check_hip((self.hip_stream_synchronize)(ptr::null_mut()), "hipStreamSynchronize(noop)")?;
                    check_hip(
                        (self.hip_memcpy)(&mut count as *mut u32 as *mut c_void, d_counter, 4, HIP_MEMCPY_DEVICE_TO_HOST),
                        "hipMemcpy(counter back)",
                    )?;
                }
                Ok(count)
            })();
            unsafe { let _ = (self.hip_free)(d_counter); }
            match counted? {
                1 => Ok(()),
                n => Err(SmallTideError::Backend {
                    backend: "hip",
                    message: format!("test kernel ran but wrote {n} instead of 1"),
                }),
            }
        };
        let res = run();
        self.unload_kernels(kernels);
        res
    }

    pub fn unload_kernels(&self, kernels: CoreKernels) {
        unsafe {
            let _ = (self.hip_module_unload)(kernels.module);
        }
    }

    /// The HIP device to use: `SMALLTIDE_HIP_DEVICE` if set, otherwise the discrete GPU with the
    /// most memory. Device order is not stable across driver versions (an integrated Radeon can
    /// enumerate first), so never assume device 0.
    pub fn select_device(&self) -> Result<i32> {
        if let Some(v) = std::env::var("SMALLTIDE_HIP_DEVICE").ok().filter(|v| !v.trim().is_empty()) {
            return v.trim().parse::<i32>().map_err(|_| SmallTideError::Backend {
                backend: "hip",
                message: format!("SMALLTIDE_HIP_DEVICE={v} is not a device index"),
            });
        }
        let mut count = 0;
        unsafe { check_hip((self.hip_get_device_count)(&mut count), "hipGetDeviceCount")? };
        let mut best: Option<(bool, usize, i32)> = None;
        for d in 0..count {
            let mut name_buf = [0 as c_char; 256];
            let (mut free, mut total) = (0usize, 0usize);
            unsafe {
                if (self.hip_set_device)(d) != 0
                    || (self.hip_device_get_name)(name_buf.as_mut_ptr(), name_buf.len() as c_int, d) != 0
                    || (self.hip_mem_get_info)(&mut free, &mut total) != 0
                {
                    continue;
                }
            }
            let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }.to_string_lossy().into_owned();
            // Integrated Radeon parts report "AMD Radeon(TM) Graphics" (memory = shared system RAM).
            let discrete = !name.contains("(TM) Graphics");
            let key = (discrete, total, d);
            if best.map_or(true, |b| (key.0, key.1) > (b.0, b.1)) {
                best = Some(key);
            }
        }
        best.map(|b| b.2).ok_or_else(|| SmallTideError::Backend { backend: "hip", message: "no usable HIP device".into() })
    }

    pub fn probe_device(&self, device_id: i32) -> Result<HipProbeReport> {
        let mut device_count = 0;
        let mut runtime_version = 0;
        let mut driver_version = 0;
        let mut free_vram = 0usize;
        let mut total_vram = 0usize;
        let mut name_buf = [0 as c_char; 256];

        unsafe {
            check_hip((self.hip_get_device_count)(&mut device_count), "hipGetDeviceCount")?;
            check_hip((self.hip_runtime_get_version)(&mut runtime_version), "hipRuntimeGetVersion")?;
            check_hip((self.hip_driver_get_version)(&mut driver_version), "hipDriverGetVersion")?;
            check_hip((self.hip_set_device)(device_id), "hipSetDevice")?;
            check_hip(
                (self.hip_device_get_name)(name_buf.as_mut_ptr(), name_buf.len() as c_int, device_id),
                "hipDeviceGetName",
            )?;
            check_hip((self.hip_mem_get_info)(&mut free_vram, &mut total_vram), "hipMemGetInfo")?;
        }

        let device_name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let (kernel_arch, hsaco_bytes) = match self.kernels_for_device(device_id) {
            Ok((arch, code)) => (arch.to_string(), code.len()),
            Err(_) => (String::new(), 0),
        };

        Ok(HipProbeReport {
            loaded_dll: self.loaded_dll.clone(),
            runtime_version,
            driver_version,
            device_count,
            device_name,
            device_arch: self.device_arch(device_id),
            kernel_arch,
            free_vram_bytes: free_vram,
            total_vram_bytes: total_vram,
            hsaco_bytes,
        })
    }

    pub fn bench_h2d_dma(&self, chunk_bytes: usize, iterations: usize) -> Result<DmaBenchResult> {
        let mut d_ptr: *mut c_void = ptr::null_mut();
        let mut h_ptr: *mut c_void = ptr::null_mut();
        let mut stream: HipStream = ptr::null_mut();

        unsafe {
            check_hip((self.hip_malloc)(&mut d_ptr, chunk_bytes), "hipMalloc")?;
            check_hip(
                (self.hip_host_malloc)(&mut h_ptr, chunk_bytes, HIP_HOST_MALLOC_DEFAULT),
                "hipHostMalloc",
            )?;
            ptr::write_bytes(h_ptr as *mut u8, 0x5A, chunk_bytes);
            check_hip(
                (self.hip_stream_create_with_flags)(&mut stream, HIP_STREAM_NON_BLOCKING),
                "hipStreamCreateWithFlags",
            )?;

            for _ in 0..4 {
                check_hip(
                    (self.hip_memcpy_async)(d_ptr, h_ptr, chunk_bytes, HIP_MEMCPY_HOST_TO_DEVICE, stream),
                    "hipMemcpyAsync(warmup)",
                )?;
            }
            check_hip((self.hip_stream_synchronize)(stream), "hipStreamSynchronize(warmup)")?;

            let start = Instant::now();
            for _ in 0..iterations {
                check_hip(
                    (self.hip_memcpy_async)(d_ptr, h_ptr, chunk_bytes, HIP_MEMCPY_HOST_TO_DEVICE, stream),
                    "hipMemcpyAsync",
                )?;
            }
            check_hip((self.hip_stream_synchronize)(stream), "hipStreamSynchronize")?;
            let secs = start.elapsed().as_secs_f64();

            let _ = (self.hip_stream_destroy)(stream);
            let _ = (self.hip_host_free)(h_ptr);
            let _ = (self.hip_free)(d_ptr);

            Ok(DmaBenchResult {
                chunk_bytes,
                iterations,
                avg_ms: (secs * 1000.0) / (iterations as f64),
                bandwidth_gb_s: (chunk_bytes as f64 * iterations as f64 / 1e9) / secs,
            })
        }
    }

    /// Benchmarks raw Windows WDDM kernel dispatch vs HIP Graph replay for 240 launches (48 layers x 5 ops).
    pub fn bench_graph_launch(&self, kernels: &CoreKernels) -> Result<GraphLaunchBenchResult> {
        let kernels_per_iter = 240usize;
        let iters = 50usize;
        let mut d_counter: *mut c_void = ptr::null_mut();
        let mut stream: HipStream = ptr::null_mut();

        unsafe {
            check_hip((self.hip_malloc)(&mut d_counter, 4), "hipMalloc(counter)")?;
            let zero: u32 = 0;
            check_hip(
                (self.hip_memcpy)(d_counter, &zero as *const u32 as *const c_void, 4, HIP_MEMCPY_HOST_TO_DEVICE),
                "hipMemcpy(zero)",
            )?;
            check_hip(
                (self.hip_stream_create_with_flags)(&mut stream, HIP_STREAM_NON_BLOCKING),
                "hipStreamCreateWithFlags",
            )?;

            let mut arg_ptr = d_counter;
            let mut params: [*mut c_void; 1] = [&mut arg_ptr as *mut *mut c_void as *mut c_void];

            // Warmup raw
            for _ in 0..10 {
                check_hip(
                    (self.hip_module_launch_kernel)(
                        kernels.noop, 1, 1, 1, 32, 1, 1, 0, stream, params.as_mut_ptr(), ptr::null_mut(),
                    ),
                    "launch(noop warmup)",
                )?;
            }
            check_hip((self.hip_stream_synchronize)(stream), "sync(raw warmup)")?;

            // Measure raw dispatch
            let t0 = Instant::now();
            for _ in 0..iters {
                for _ in 0..kernels_per_iter {
                    check_hip(
                        (self.hip_module_launch_kernel)(
                            kernels.noop, 1, 1, 1, 32, 1, 1, 0, stream, params.as_mut_ptr(), ptr::null_mut(),
                        ),
                        "launch(noop raw)",
                    )?;
                }
                check_hip((self.hip_stream_synchronize)(stream), "sync(raw)")?;
            }
            let raw_ms = (t0.elapsed().as_secs_f64() * 1000.0) / (iters as f64);

            // Capture HIP Graph
            let mut graph: HipGraph = ptr::null_mut();
            let mut graph_exec: HipGraphExec = ptr::null_mut();
            check_hip(
                (self.hip_stream_begin_capture)(stream, HIP_STREAM_CAPTURE_MODE_GLOBAL),
                "hipStreamBeginCapture",
            )?;
            for _ in 0..kernels_per_iter {
                check_hip(
                    (self.hip_module_launch_kernel)(
                        kernels.noop, 1, 1, 1, 32, 1, 1, 0, stream, params.as_mut_ptr(), ptr::null_mut(),
                    ),
                    "launch(noop capture)",
                )?;
            }
            check_hip((self.hip_stream_end_capture)(stream, &mut graph), "hipStreamEndCapture")?;
            check_hip(
                (self.hip_graph_instantiate)(&mut graph_exec, graph, ptr::null_mut(), ptr::null_mut(), 0),
                "hipGraphInstantiate",
            )?;

            // Warmup graph
            for _ in 0..5 {
                check_hip((self.hip_graph_launch)(graph_exec, stream), "hipGraphLaunch(warmup)")?;
            }
            check_hip((self.hip_stream_synchronize)(stream), "sync(graph warmup)")?;

            // Measure graph replay
            let t1 = Instant::now();
            for _ in 0..iters {
                check_hip((self.hip_graph_launch)(graph_exec, stream), "hipGraphLaunch")?;
                check_hip((self.hip_stream_synchronize)(stream), "sync(graph)")?;
            }
            let graph_ms = (t1.elapsed().as_secs_f64() * 1000.0) / (iters as f64);

            let _ = (self.hip_graph_exec_destroy)(graph_exec);
            let _ = (self.hip_graph_destroy)(graph);
            let _ = (self.hip_stream_destroy)(stream);
            let _ = (self.hip_free)(d_counter);

            Ok(GraphLaunchBenchResult {
                kernels_per_iter,
                raw_ms_per_iter: raw_ms,
                graph_ms_per_iter: graph_ms,
                speedup: raw_ms / graph_ms.max(1e-6),
            })
        }
    }

    /// Benchmarks effective GDDR6 bandwidth of FP8, NVFP4, and MXFP8 decode GEMV kernels
    /// on exact Inkling-2-Mini shapes + verifies concurrent Compute/H2D DMA overlap.
    pub fn bench_decode_kernels(
        &self,
        kernels: &CoreKernels,
    ) -> Result<(Vec<KernelBwResult>, OverlapBenchResult)> {
        let mut results = Vec::new();
        let mut stream: HipStream = ptr::null_mut();
        let mut copy_stream: HipStream = ptr::null_mut();

        unsafe {
            check_hip(
                (self.hip_stream_create_with_flags)(&mut stream, HIP_STREAM_NON_BLOCKING),
                "hipStreamCreate(compute)",
            )?;
            check_hip(
                (self.hip_stream_create_with_flags)(&mut copy_stream, HIP_STREAM_NON_BLOCKING),
                "hipStreamCreate(copy)",
            )?;

            // 1. FP8 E4M3 lm_head [201024, 2304] (~463 MB — defeats 64MB L3 completely!)
            {
                let n: u32 = 201_024;
                let k: u32 = 2_304;
                let w_bytes = (n as usize) * (k as usize);
                let s_bytes = (n as usize) * 4;
                let x_bytes = (k as usize) * 4;
                let y_bytes = (n as usize) * 4;

                let (mut d_w, mut d_s, mut d_x, mut d_y) =
                    (ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
                check_hip((self.hip_malloc)(&mut d_w, w_bytes), "malloc(fp8 w)")?;
                check_hip((self.hip_malloc)(&mut d_s, s_bytes), "malloc(fp8 s)")?;
                check_hip((self.hip_malloc)(&mut d_x, x_bytes), "malloc(fp8 x)")?;
                check_hip((self.hip_malloc)(&mut d_y, y_bytes), "malloc(fp8 y)")?;

                let grid_x = n.div_ceil(4);
                let mut n_arg = n;
                let mut k_arg = k;
                let mut params: [*mut c_void; 6] = [
                    &mut d_y as *mut _ as *mut c_void,
                    &mut d_x as *mut _ as *mut c_void,
                    &mut d_w as *mut _ as *mut c_void,
                    &mut d_s as *mut _ as *mut c_void,
                    &mut n_arg as *mut _ as *mut c_void,
                    &mut k_arg as *mut _ as *mut c_void,
                ];
                for _ in 0..3 {
                    check_hip(
                        (self.hip_module_launch_kernel)(
                            kernels.gemv_fp8_e4m3_f32, grid_x, 1, 1, 128, 1, 1, 0, stream, params.as_mut_ptr(), ptr::null_mut(),
                        ),
                        "launch(fp8 warmup)",
                    )?;
                }
                check_hip((self.hip_stream_synchronize)(stream), "sync(fp8 warmup)")?;

                let iters = 20usize;
                let t0 = Instant::now();
                for _ in 0..iters {
                    check_hip(
                        (self.hip_module_launch_kernel)(
                            kernels.gemv_fp8_e4m3_f32, grid_x, 1, 1, 128, 1, 1, 0, stream, params.as_mut_ptr(), ptr::null_mut(),
                        ),
                        "launch(fp8)",
                    )?;
                }
                check_hip((self.hip_stream_synchronize)(stream), "sync(fp8)")?;
                let secs = t0.elapsed().as_secs_f64();
                let bytes = w_bytes + s_bytes + x_bytes + y_bytes;
                results.push(KernelBwResult {
                    label: "FP8 E4M3 lm_head [201024x2304]".into(),
                    bytes_read: bytes,
                    avg_ms: secs * 1000.0 / (iters as f64),
                    effective_gb_s: (bytes as f64 * iters as f64 / 1e9) / secs,
                });

                let _ = (self.hip_free)(d_w);
                let _ = (self.hip_free)(d_s);
                let _ = (self.hip_free)(d_x);
                let _ = (self.hip_free)(d_y);
            }

            // 2. NVFP4 large matrix (64 stacked experts worth = ~192 MB to exceed 64MB L3!)
            //    [N=147456, K=2304] = 64 x 2304 rows (equivalent to 64 w13 projections)
            let overlap_res = {
                let n: u32 = 147_456;
                let k: u32 = 2_304;
                let w_bytes = (n as usize) * (k as usize / 2);
                let s_bytes = (n as usize) * (k as usize / 16);
                let x_bytes = (k as usize) * 4;
                let y_bytes = (n as usize) * 4;

                let (mut d_w, mut d_s, mut d_x, mut d_y) =
                    (ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
                check_hip((self.hip_malloc)(&mut d_w, w_bytes), "malloc(nvfp4 w)")?;
                check_hip((self.hip_malloc)(&mut d_s, s_bytes), "malloc(nvfp4 s)")?;
                check_hip((self.hip_malloc)(&mut d_x, x_bytes), "malloc(nvfp4 x)")?;
                check_hip((self.hip_malloc)(&mut d_y, y_bytes), "malloc(nvfp4 y)")?;

                let grid_x = n.div_ceil(4);
                let mut outer_scale: f32 = 1.0;
                let mut n_arg = n;
                let mut k_arg = k;
                let mut params: [*mut c_void; 7] = [
                    &mut d_y as *mut _ as *mut c_void,
                    &mut d_x as *mut _ as *mut c_void,
                    &mut d_w as *mut _ as *mut c_void,
                    &mut d_s as *mut _ as *mut c_void,
                    &mut outer_scale as *mut _ as *mut c_void,
                    &mut n_arg as *mut _ as *mut c_void,
                    &mut k_arg as *mut _ as *mut c_void,
                ];
                for _ in 0..3 {
                    check_hip(
                        (self.hip_module_launch_kernel)(
                            kernels.gemv_nvfp4_f32, grid_x, 1, 1, 128, 1, 1, 0, stream, params.as_mut_ptr(), ptr::null_mut(),
                        ),
                        "launch(nvfp4 warmup)",
                    )?;
                }
                check_hip((self.hip_stream_synchronize)(stream), "sync(nvfp4 warmup)")?;

                let iters = 25usize;
                let t0 = Instant::now();
                for _ in 0..iters {
                    check_hip(
                        (self.hip_module_launch_kernel)(
                            kernels.gemv_nvfp4_f32, grid_x, 1, 1, 128, 1, 1, 0, stream, params.as_mut_ptr(), ptr::null_mut(),
                        ),
                        "launch(nvfp4)",
                    )?;
                }
                check_hip((self.hip_stream_synchronize)(stream), "sync(nvfp4)")?;
                let secs = t0.elapsed().as_secs_f64();
                let bytes = w_bytes + s_bytes + x_bytes + y_bytes;
                let compute_only_ms = secs * 1000.0 / (iters as f64);
                results.push(KernelBwResult {
                    label: "NVFP4 GEMV [147456x2304] (~192 MB, L3-bypass)".into(),
                    bytes_read: bytes,
                    avg_ms: compute_only_ms,
                    effective_gb_s: (bytes as f64 * iters as f64 / 1e9) / secs,
                });

                // Measure concurrent Compute + Stage 1 H2D DMA (2 x 4.27 MiB experts per compute pass)
                let dma_bytes = 2 * 4_478_976usize;
                let mut d_dma: *mut c_void = ptr::null_mut();
                let mut h_dma: *mut c_void = ptr::null_mut();
                check_hip((self.hip_malloc)(&mut d_dma, dma_bytes), "malloc(dma)")?;
                check_hip(
                    (self.hip_host_malloc)(&mut h_dma, dma_bytes, HIP_HOST_MALLOC_DEFAULT),
                    "host_malloc(dma)",
                )?;

                let td = Instant::now();
                for _ in 0..iters {
                    check_hip(
                        (self.hip_memcpy_async)(d_dma, h_dma, dma_bytes, HIP_MEMCPY_HOST_TO_DEVICE, copy_stream),
                        "dma_only",
                    )?;
                }
                check_hip((self.hip_stream_synchronize)(copy_stream), "sync(dma_only)")?;
                let dma_only_ms = td.elapsed().as_secs_f64() * 1000.0 / (iters as f64);

                let to = Instant::now();
                for _ in 0..iters {
                    check_hip(
                        (self.hip_module_launch_kernel)(
                            kernels.gemv_nvfp4_f32, grid_x, 1, 1, 128, 1, 1, 0, stream, params.as_mut_ptr(), ptr::null_mut(),
                        ),
                        "overlap(compute)",
                    )?;
                    check_hip(
                        (self.hip_memcpy_async)(d_dma, h_dma, dma_bytes, HIP_MEMCPY_HOST_TO_DEVICE, copy_stream),
                        "overlap(dma)",
                    )?;
                }
                check_hip((self.hip_stream_synchronize)(stream), "sync(overlap compute)")?;
                check_hip((self.hip_stream_synchronize)(copy_stream), "sync(overlap copy)")?;
                let overlapped_ms = to.elapsed().as_secs_f64() * 1000.0 / (iters as f64);

                let serial_ms = compute_only_ms + dma_only_ms;
                let ideal_ms = compute_only_ms.max(dma_only_ms);
                let hidden = (serial_ms - overlapped_ms).max(0.0);
                let max_hideable = (serial_ms - ideal_ms).max(1e-6);
                let overlap_efficiency_pct = (hidden / max_hideable * 100.0).min(100.0);

                let _ = (self.hip_host_free)(h_dma);
                let _ = (self.hip_free)(d_dma);
                let _ = (self.hip_free)(d_w);
                let _ = (self.hip_free)(d_s);
                let _ = (self.hip_free)(d_x);
                let _ = (self.hip_free)(d_y);

                OverlapBenchResult {
                    compute_only_ms,
                    dma_only_ms,
                    overlapped_ms,
                    overlap_efficiency_pct,
                }
            };

            // 3. MXFP8 large matrix [N=73728, K=2304] (~175 MB, L3-bypass, [N, K/32] scales)
            {
                let n: u32 = 73_728;
                let k: u32 = 2_304;
                let w_bytes = (n as usize) * (k as usize);
                let s_bytes = (n as usize) * (k as usize / 32);
                let x_bytes = (k as usize) * 4;
                let y_bytes = (n as usize) * 4;

                let (mut d_w, mut d_s, mut d_x, mut d_y) =
                    (ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
                check_hip((self.hip_malloc)(&mut d_w, w_bytes), "malloc(mxfp8 w)")?;
                check_hip((self.hip_malloc)(&mut d_s, s_bytes), "malloc(mxfp8 s)")?;
                check_hip((self.hip_malloc)(&mut d_x, x_bytes), "malloc(mxfp8 x)")?;
                check_hip((self.hip_malloc)(&mut d_y, y_bytes), "malloc(mxfp8 y)")?;

                let grid_x = n.div_ceil(4);
                let mut n_arg = n;
                let mut k_arg = k;
                let mut params: [*mut c_void; 6] = [
                    &mut d_y as *mut _ as *mut c_void,
                    &mut d_x as *mut _ as *mut c_void,
                    &mut d_w as *mut _ as *mut c_void,
                    &mut d_s as *mut _ as *mut c_void,
                    &mut n_arg as *mut _ as *mut c_void,
                    &mut k_arg as *mut _ as *mut c_void,
                ];
                for _ in 0..3 {
                    check_hip(
                        (self.hip_module_launch_kernel)(
                            kernels.gemv_mxfp8_f32, grid_x, 1, 1, 128, 1, 1, 0, stream, params.as_mut_ptr(), ptr::null_mut(),
                        ),
                        "launch(mxfp8 warmup)",
                    )?;
                }
                check_hip((self.hip_stream_synchronize)(stream), "sync(mxfp8 warmup)")?;

                let iters = 25usize;
                let t0 = Instant::now();
                for _ in 0..iters {
                    check_hip(
                        (self.hip_module_launch_kernel)(
                            kernels.gemv_mxfp8_f32, grid_x, 1, 1, 128, 1, 1, 0, stream, params.as_mut_ptr(), ptr::null_mut(),
                        ),
                        "launch(mxfp8)",
                    )?;
                }
                check_hip((self.hip_stream_synchronize)(stream), "sync(mxfp8)")?;
                let secs = t0.elapsed().as_secs_f64();
                let bytes = w_bytes + s_bytes + x_bytes + y_bytes;
                results.push(KernelBwResult {
                    label: "MXFP8 GEMV [73728x2304] (~175 MB, L3-bypass)".into(),
                    bytes_read: bytes,
                    avg_ms: secs * 1000.0 / (iters as f64),
                    effective_gb_s: (bytes as f64 * iters as f64 / 1e9) / secs,
                });

                let _ = (self.hip_free)(d_w);
                let _ = (self.hip_free)(d_s);
                let _ = (self.hip_free)(d_x);
                let _ = (self.hip_free)(d_y);
            }

            let _ = (self.hip_stream_destroy)(copy_stream);
            let _ = (self.hip_stream_destroy)(stream);

            Ok((results, overlap_res))
        }
    }

    /// Runs numerical parity tests on GPU vs host slices for RMSNorm, FP8, NVFP4, and MXFP8.
    pub fn run_rmsnorm_f32(&self, kernels: &CoreKernels, x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>> {
        let hidden = x.len() as u32;
        let bytes = x.len() * 4;
        let mut out = vec![0.0f32; x.len()];
        unsafe {
            let (mut d_x, mut d_w, mut d_o) = (ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
            check_hip((self.hip_malloc)(&mut d_x, bytes), "malloc(x)")?;
            check_hip((self.hip_malloc)(&mut d_w, bytes), "malloc(w)")?;
            check_hip((self.hip_malloc)(&mut d_o, bytes), "malloc(o)")?;
            check_hip((self.hip_memcpy)(d_x, x.as_ptr() as *const c_void, bytes, HIP_MEMCPY_HOST_TO_DEVICE), "h2d(x)")?;
            check_hip((self.hip_memcpy)(d_w, w.as_ptr() as *const c_void, bytes, HIP_MEMCPY_HOST_TO_DEVICE), "h2d(w)")?;

            let mut h_arg = hidden;
            let mut eps_arg = eps;
            let mut params: [*mut c_void; 5] = [
                &mut d_o as *mut _ as *mut c_void,
                &mut d_x as *mut _ as *mut c_void,
                &mut d_w as *mut _ as *mut c_void,
                &mut h_arg as *mut _ as *mut c_void,
                &mut eps_arg as *mut _ as *mut c_void,
            ];
            check_hip(
                (self.hip_module_launch_kernel)(
                    kernels.rmsnorm_f32, 1, 1, 1, 256, 1, 1, 0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(rmsnorm)",
            )?;
            check_hip((self.hip_memcpy)(out.as_mut_ptr() as *mut c_void, d_o, bytes, HIP_MEMCPY_DEVICE_TO_HOST), "d2h(o)")?;
            let _ = (self.hip_free)(d_x);
            let _ = (self.hip_free)(d_w);
            let _ = (self.hip_free)(d_o);
        }
        Ok(out)
    }

    pub fn run_gemv_fp8_e4m3_f32(
        &self,
        kernels: &CoreKernels,
        x: &[f32],
        w_fp8: &[u8],
        row_scale: &[f32],
        n_rows: usize,
        k_cols: usize,
    ) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; n_rows];
        unsafe {
            let (mut d_x, mut d_w, mut d_s, mut d_y) =
                (ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
            check_hip((self.hip_malloc)(&mut d_x, k_cols * 4), "malloc")?;
            check_hip((self.hip_malloc)(&mut d_w, w_fp8.len()), "malloc")?;
            check_hip((self.hip_malloc)(&mut d_s, n_rows * 4), "malloc")?;
            check_hip((self.hip_malloc)(&mut d_y, n_rows * 4), "malloc")?;
            check_hip((self.hip_memcpy)(d_x, x.as_ptr() as *const _, k_cols * 4, HIP_MEMCPY_HOST_TO_DEVICE), "h2d")?;
            check_hip((self.hip_memcpy)(d_w, w_fp8.as_ptr() as *const _, w_fp8.len(), HIP_MEMCPY_HOST_TO_DEVICE), "h2d")?;
            check_hip((self.hip_memcpy)(d_s, row_scale.as_ptr() as *const _, n_rows * 4, HIP_MEMCPY_HOST_TO_DEVICE), "h2d")?;

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
            check_hip(
                (self.hip_module_launch_kernel)(
                    kernels.gemv_fp8_e4m3_f32, n_arg.div_ceil(4), 1, 1, 128, 1, 1, 0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(gemv_fp8)",
            )?;
            check_hip((self.hip_memcpy)(out.as_mut_ptr() as *mut _, d_y, n_rows * 4, HIP_MEMCPY_DEVICE_TO_HOST), "d2h")?;
            let _ = (self.hip_free)(d_x);
            let _ = (self.hip_free)(d_w);
            let _ = (self.hip_free)(d_s);
            let _ = (self.hip_free)(d_y);
        }
        Ok(out)
    }

    pub fn run_gemv_nvfp4_f32(
        &self,
        kernels: &CoreKernels,
        x: &[f32],
        w_packed: &[u8],
        w_scales: &[u8],
        outer_scale: f32,
        n_rows: usize,
        k_cols: usize,
    ) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; n_rows];
        unsafe {
            let (mut d_x, mut d_w, mut d_s, mut d_y) =
                (ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
            check_hip((self.hip_malloc)(&mut d_x, k_cols * 4), "malloc")?;
            check_hip((self.hip_malloc)(&mut d_w, w_packed.len()), "malloc")?;
            check_hip((self.hip_malloc)(&mut d_s, w_scales.len()), "malloc")?;
            check_hip((self.hip_malloc)(&mut d_y, n_rows * 4), "malloc")?;
            check_hip((self.hip_memcpy)(d_x, x.as_ptr() as *const _, k_cols * 4, HIP_MEMCPY_HOST_TO_DEVICE), "h2d")?;
            check_hip((self.hip_memcpy)(d_w, w_packed.as_ptr() as *const _, w_packed.len(), HIP_MEMCPY_HOST_TO_DEVICE), "h2d")?;
            check_hip((self.hip_memcpy)(d_s, w_scales.as_ptr() as *const _, w_scales.len(), HIP_MEMCPY_HOST_TO_DEVICE), "h2d")?;

            let mut os_arg = outer_scale;
            let mut n_arg = n_rows as u32;
            let mut k_arg = k_cols as u32;
            let mut params: [*mut c_void; 7] = [
                &mut d_y as *mut _ as *mut c_void,
                &mut d_x as *mut _ as *mut c_void,
                &mut d_w as *mut _ as *mut c_void,
                &mut d_s as *mut _ as *mut c_void,
                &mut os_arg as *mut _ as *mut c_void,
                &mut n_arg as *mut _ as *mut c_void,
                &mut k_arg as *mut _ as *mut c_void,
            ];
            check_hip(
                (self.hip_module_launch_kernel)(
                    kernels.gemv_nvfp4_f32, n_arg.div_ceil(4), 1, 1, 128, 1, 1, 0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(gemv_nvfp4)",
            )?;
            check_hip((self.hip_memcpy)(out.as_mut_ptr() as *mut _, d_y, n_rows * 4, HIP_MEMCPY_DEVICE_TO_HOST), "d2h")?;
            let _ = (self.hip_free)(d_x);
            let _ = (self.hip_free)(d_w);
            let _ = (self.hip_free)(d_s);
            let _ = (self.hip_free)(d_y);
        }
        Ok(out)
    }

    pub fn run_gemv_mxfp8_f32(
        &self,
        kernels: &CoreKernels,
        x: &[f32],
        w_fp8: &[u8],
        w_scales: &[u8],
        n_rows: usize,
        k_cols: usize,
    ) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; n_rows];
        unsafe {
            let (mut d_x, mut d_w, mut d_s, mut d_y) =
                (ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut());
            check_hip((self.hip_malloc)(&mut d_x, k_cols * 4), "malloc")?;
            check_hip((self.hip_malloc)(&mut d_w, w_fp8.len()), "malloc")?;
            check_hip((self.hip_malloc)(&mut d_s, w_scales.len()), "malloc")?;
            check_hip((self.hip_malloc)(&mut d_y, n_rows * 4), "malloc")?;
            check_hip((self.hip_memcpy)(d_x, x.as_ptr() as *const _, k_cols * 4, HIP_MEMCPY_HOST_TO_DEVICE), "h2d")?;
            check_hip((self.hip_memcpy)(d_w, w_fp8.as_ptr() as *const _, w_fp8.len(), HIP_MEMCPY_HOST_TO_DEVICE), "h2d")?;
            check_hip((self.hip_memcpy)(d_s, w_scales.as_ptr() as *const _, w_scales.len(), HIP_MEMCPY_HOST_TO_DEVICE), "h2d")?;

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
            check_hip(
                (self.hip_module_launch_kernel)(
                    kernels.gemv_mxfp8_f32, n_arg.div_ceil(4), 1, 1, 128, 1, 1, 0, ptr::null_mut(), params.as_mut_ptr(), ptr::null_mut(),
                ),
                "launch(gemv_mxfp8)",
            )?;
            check_hip((self.hip_memcpy)(out.as_mut_ptr() as *mut _, d_y, n_rows * 4, HIP_MEMCPY_DEVICE_TO_HOST), "d2h")?;
            let _ = (self.hip_free)(d_x);
            let _ = (self.hip_free)(d_w);
            let _ = (self.hip_free)(d_s);
            let _ = (self.hip_free)(d_y);
        }
        Ok(out)
    }

    /// Runs and optionally benchmarks our indirect Top-K MoE SwiGLU (`w13` + `w2` accum)
    /// using the GPU `slot_table` indirection layout. Returns `(output_vec, avg_layer_ms)`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_and_bench_moe_indirect(
        &self,
        kernels: &CoreKernels,
        x: &[f32],
        expert_pool: &[u8],
        slot_table: &[u32],
        topk_ids: &[u32],
        topk_weights: &[f32],
        slab_stride_bytes: usize,
        hidden_size: usize,
        intermediate_size: usize,
        gate_softcap: f32,
        up_softcap: f32,
        bench_iters: usize,
    ) -> Result<(Vec<f32>, f64)> {
        let top_k = topk_ids.len();
        let mut out = vec![0.0f32; hidden_size];
        let mut avg_ms = 0.0;

        unsafe {
            let mut stream: HipStream = ptr::null_mut();
            check_hip(
                (self.hip_stream_create_with_flags)(&mut stream, HIP_STREAM_NON_BLOCKING),
                "stream(moe)",
            )?;

            let (mut d_x, mut d_pool, mut d_slots, mut d_ids, mut d_w, mut d_mid, mut d_out) = (
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            );
            check_hip((self.hip_malloc)(&mut d_x, hidden_size * 4), "malloc(x)")?;
            check_hip((self.hip_malloc)(&mut d_pool, expert_pool.len()), "malloc(pool)")?;
            check_hip((self.hip_malloc)(&mut d_slots, slot_table.len() * 4), "malloc(slots)")?;
            check_hip((self.hip_malloc)(&mut d_ids, top_k * 4), "malloc(ids)")?;
            check_hip((self.hip_malloc)(&mut d_w, top_k * 4), "malloc(weights)")?;
            check_hip((self.hip_malloc)(&mut d_mid, top_k * intermediate_size * 4), "malloc(mid)")?;
            check_hip((self.hip_malloc)(&mut d_out, hidden_size * 4), "malloc(out)")?;

            check_hip((self.hip_memcpy)(d_x, x.as_ptr() as *const _, hidden_size * 4, HIP_MEMCPY_HOST_TO_DEVICE), "h2d(x)")?;
            check_hip((self.hip_memcpy)(d_pool, expert_pool.as_ptr() as *const _, expert_pool.len(), HIP_MEMCPY_HOST_TO_DEVICE), "h2d(pool)")?;
            check_hip((self.hip_memcpy)(d_slots, slot_table.as_ptr() as *const _, slot_table.len() * 4, HIP_MEMCPY_HOST_TO_DEVICE), "h2d(slots)")?;
            check_hip((self.hip_memcpy)(d_ids, topk_ids.as_ptr() as *const _, top_k * 4, HIP_MEMCPY_HOST_TO_DEVICE), "h2d(ids)")?;
            check_hip((self.hip_memcpy)(d_w, topk_weights.as_ptr() as *const _, top_k * 4, HIP_MEMCPY_HOST_TO_DEVICE), "h2d(w)")?;

            let mut stride_arg = slab_stride_bytes as u64;
            let mut h_arg = hidden_size as u32;
            let mut ff_arg = intermediate_size as u32;
            let mut gcap_arg = gate_softcap;
            let mut ucap_arg = up_softcap;
            let mut k_arg = top_k as u32;

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

            let g1_x = ff_arg.div_ceil(4);
            let g2_x = h_arg.div_ceil(4);

            // Warmup + parity execution
            for _ in 0..3 {
                check_hip(
                    (self.hip_module_launch_kernel)(
                        kernels.moe_w13_swiglu_indirect, g1_x, k_arg, 1, 128, 1, 1, 0, stream, p1.as_mut_ptr(), ptr::null_mut(),
                    ),
                    "launch(moe_w13)",
                )?;
                check_hip(
                    (self.hip_module_launch_kernel)(
                        kernels.moe_w2_accum_indirect, g2_x, 1, 1, 128, 1, 1, 0, stream, p2.as_mut_ptr(), ptr::null_mut(),
                    ),
                    "launch(moe_w2)",
                )?;
            }
            check_hip((self.hip_stream_synchronize)(stream), "sync(moe warmup)")?;
            check_hip((self.hip_memcpy)(out.as_mut_ptr() as *mut _, d_out, hidden_size * 4, HIP_MEMCPY_DEVICE_TO_HOST), "d2h(out)")?;

            if bench_iters > 0 {
                let t0 = Instant::now();
                for _ in 0..bench_iters {
                    check_hip(
                        (self.hip_module_launch_kernel)(
                            kernels.moe_w13_swiglu_indirect, g1_x, k_arg, 1, 128, 1, 1, 0, stream, p1.as_mut_ptr(), ptr::null_mut(),
                        ),
                        "launch(moe_w13 bench)",
                    )?;
                    check_hip(
                        (self.hip_module_launch_kernel)(
                            kernels.moe_w2_accum_indirect, g2_x, 1, 1, 128, 1, 1, 0, stream, p2.as_mut_ptr(), ptr::null_mut(),
                        ),
                        "launch(moe_w2 bench)",
                    )?;
                }
                check_hip((self.hip_stream_synchronize)(stream), "sync(moe bench)")?;
                avg_ms = (t0.elapsed().as_secs_f64() * 1000.0) / (bench_iters as f64);
            }

            let _ = (self.hip_stream_destroy)(stream);
            let _ = (self.hip_free)(d_x);
            let _ = (self.hip_free)(d_pool);
            let _ = (self.hip_free)(d_slots);
            let _ = (self.hip_free)(d_ids);
            let _ = (self.hip_free)(d_w);
            let _ = (self.hip_free)(d_mid);
            let _ = (self.hip_free)(d_out);
        }

        Ok((out, avg_ms))
    }

    /// Benchmarks 256K-context (or arbitrary `seq_len`) FP8 Split-K FlashDecoding GQA attention
    /// for 1 Inkling-2-Mini global layer (`24 Q heads : 8 KV heads`, `head_dim = 128`).
    /// Returns `(avg_ms_per_layer, effective_gb_s)`.
    pub fn bench_flash_decode_fp8(
        &self,
        kernels: &CoreKernels,
        seq_len: usize,
        num_splits: u32,
        iters: usize,
    ) -> Result<(f64, f64)> {
        let n_q: u32 = 24;
        let n_kv: u32 = 8;
        let hd: usize = 128;
        let kv_bytes = (n_kv as usize) * seq_len * hd; // 1 byte FP8
        let sc_bytes = (n_kv as usize) * seq_len * 4;  // f32 scales
        let q_bytes = (n_q as usize) * hd * 4;
        let p_out_bytes = (n_q as usize) * (num_splits as usize) * hd * 4;
        let p_meta_bytes = (n_q as usize) * (num_splits as usize) * 4;

        unsafe {
            let mut stream: HipStream = ptr::null_mut();
            check_hip(
                (self.hip_stream_create_with_flags)(&mut stream, HIP_STREAM_NON_BLOCKING),
                "stream(flash)",
            )?;

            let (mut d_q, mut d_k, mut d_ks, mut d_v, mut d_vs, mut d_po, mut d_pm, mut d_pl, mut d_out) = (
                ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut(),
                ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut(),
            );
            check_hip((self.hip_malloc)(&mut d_q, q_bytes), "malloc(q)")?;
            check_hip((self.hip_malloc)(&mut d_k, kv_bytes), "malloc(k)")?;
            check_hip((self.hip_malloc)(&mut d_ks, sc_bytes), "malloc(ks)")?;
            check_hip((self.hip_malloc)(&mut d_v, kv_bytes), "malloc(v)")?;
            check_hip((self.hip_malloc)(&mut d_vs, sc_bytes), "malloc(vs)")?;
            check_hip((self.hip_malloc)(&mut d_po, p_out_bytes), "malloc(po)")?;
            check_hip((self.hip_malloc)(&mut d_pm, p_meta_bytes), "malloc(pm)")?;
            check_hip((self.hip_malloc)(&mut d_pl, p_meta_bytes), "malloc(pl)")?;
            check_hip((self.hip_malloc)(&mut d_out, q_bytes), "malloc(out)")?;

            let mut nq_arg = n_q;
            let mut nkv_arg = n_kv;
            let mut max_seq_arg = seq_len as u32;
            let mut q_pos_arg = (seq_len.saturating_sub(1)) as u32;
            let mut rel_extent_arg: u32 = 1024;
            let mut is_local_arg: u32 = 0;
            let mut splits_arg = num_splits;
            let mut inv_sqrt_d: f32 = 1.0 / (hd as f32).sqrt();
            let mut window_arg: u32 = 0;
            let mut vq4_arg: u32 = 0;
            let mut null_rel: *mut c_void = ptr::null_mut();
            let mut null_page_table: *mut c_void = ptr::null_mut();
            let mut null_sink: *mut c_void = ptr::null_mut();

            let mut p1: [*mut c_void; 20] = [
                &mut d_po as *mut _ as *mut c_void,
                &mut d_pm as *mut _ as *mut c_void,
                &mut d_pl as *mut _ as *mut c_void,
                &mut d_q as *mut _ as *mut c_void,
                &mut d_k as *mut _ as *mut c_void,
                &mut d_ks as *mut _ as *mut c_void,
                &mut d_v as *mut _ as *mut c_void,
                &mut d_vs as *mut _ as *mut c_void,
                &mut null_rel as *mut _ as *mut c_void,
                &mut nq_arg as *mut _ as *mut c_void,
                &mut nkv_arg as *mut _ as *mut c_void,
                &mut max_seq_arg as *mut _ as *mut c_void,
                &mut q_pos_arg as *mut _ as *mut c_void,
                &mut rel_extent_arg as *mut _ as *mut c_void,
                &mut is_local_arg as *mut _ as *mut c_void,
                &mut splits_arg as *mut _ as *mut c_void,
                &mut inv_sqrt_d as *mut _ as *mut c_void,
                &mut window_arg as *mut _ as *mut c_void,
                &mut null_page_table as *mut _ as *mut c_void,
                &mut vq4_arg as *mut _ as *mut c_void,
            ];
            let mut p2: [*mut c_void; 7] = [
                &mut d_out as *mut _ as *mut c_void,
                &mut d_po as *mut _ as *mut c_void,
                &mut d_pm as *mut _ as *mut c_void,
                &mut d_pl as *mut _ as *mut c_void,
                &mut null_sink as *mut _ as *mut c_void,
                &mut nq_arg as *mut _ as *mut c_void,
                &mut splits_arg as *mut _ as *mut c_void,
            ];

            for _ in 0..3 {
                check_hip(
                    (self.hip_module_launch_kernel)(
                        kernels.flash_decode_fp8_split, n_kv, num_splits, 1, 128, 1, 1, 0, stream, p1.as_mut_ptr(), ptr::null_mut(),
                    ),
                    "flash_split(warmup)",
                )?;
                check_hip(
                    (self.hip_module_launch_kernel)(
                        kernels.flash_decode_fp8_reduce, n_q, 1, 1, 32, 1, 1, 0, stream, p2.as_mut_ptr(), ptr::null_mut(),
                    ),
                    "flash_reduce(warmup)",
                )?;
            }
            check_hip((self.hip_stream_synchronize)(stream), "sync(flash warmup)")?;

            let t0 = Instant::now();
            for _ in 0..iters {
                check_hip(
                    (self.hip_module_launch_kernel)(
                        kernels.flash_decode_fp8_split, n_kv, num_splits, 1, 128, 1, 1, 0, stream, p1.as_mut_ptr(), ptr::null_mut(),
                    ),
                    "flash_split",
                )?;
                check_hip(
                    (self.hip_module_launch_kernel)(
                        kernels.flash_decode_fp8_reduce, n_q, 1, 1, 32, 1, 1, 0, stream, p2.as_mut_ptr(), ptr::null_mut(),
                    ),
                    "flash_reduce",
                )?;
            }
            check_hip((self.hip_stream_synchronize)(stream), "sync(flash)")?;
            let secs = t0.elapsed().as_secs_f64();
            let avg_ms = (secs * 1000.0) / (iters as f64);
            // Unique KV bytes read from GDDR6 (K + V + scales; GQA 3:1 reuses KV in L2/L3)
            let bytes = (kv_bytes * 2 + sc_bytes * 2) as f64;
            let gb_s = (bytes * (iters as f64) / 1e9) / secs;

            let _ = (self.hip_stream_destroy)(stream);
            let _ = (self.hip_free)(d_q);
            let _ = (self.hip_free)(d_k);
            let _ = (self.hip_free)(d_ks);
            let _ = (self.hip_free)(d_v);
            let _ = (self.hip_free)(d_vs);
            let _ = (self.hip_free)(d_po);
            let _ = (self.hip_free)(d_pm);
            let _ = (self.hip_free)(d_pl);
            let _ = (self.hip_free)(d_out);

            Ok((avg_ms, gb_s))
        }
    }
}

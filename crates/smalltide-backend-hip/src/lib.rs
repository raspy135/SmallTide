//! Dynamic ROCm/HIP backend (`amdhip64_7.dll` + offline-compiled `.hsaco` code objects).
//!
//! All ROCm/HIP symbols are isolated inside this crate behind `smalltide_core::ComputeBackend`.
//! Loading `amdhip64_7.dll` dynamically via `libloading` avoids hard DLL dependencies so the
//! same SmallTide binary can gracefully select CUDA or Vulkan on non-AMD GPUs in Phase 5.

pub mod backend;
pub mod ffi;
#[cfg(windows)]
pub mod d3d12;
#[cfg(windows)]
pub mod wddm;
#[cfg(not(windows))]
pub mod linux;

pub use backend::HipBackend;
pub use ffi::{HipProbeReport, HipRuntime};

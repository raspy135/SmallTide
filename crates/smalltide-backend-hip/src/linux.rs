//! Linux-only pieces of the HIP backend (the Windows equivalents live inline in `ffi.rs`,
//! `wddm.rs` and `d3d12.rs`).

/// HIP runtime shared objects tried in order by `HipRuntime::load`: the ROCm 7 / 6 sonames on
/// the loader path (distro packages, `LD_LIBRARY_PATH`), then the default `/opt/rocm` install.
pub const HIP_RUNTIME_CANDIDATES: &[&str] = &[
    "libamdhip64.so.7",
    "libamdhip64.so.6",
    "libamdhip64.so",
    "/opt/rocm/lib/libamdhip64.so.7",
    "/opt/rocm/lib/libamdhip64.so.6",
    "/opt/rocm/lib/libamdhip64.so",
];

/// Install hint appended when no HIP runtime could be loaded.
pub const HIP_RUNTIME_HINT: &str =
    "Install the ROCm HIP runtime (6.x/7.x with RDNA 4 support) or add its lib directory to LD_LIBRARY_PATH.";

/// Device-wide VRAM `(used, total)` in bytes from the amdgpu driver's sysfs counters
/// (`/sys/class/drm/card*/device/mem_info_vram_{used,total}`), for the AMD card whose VRAM size
/// matches `hip_total` (the HIP device's total, within 1/16).
///
/// `used` counts every client of the card (the desktop, other processes and this one), so
/// `total - used` is what can still be allocated. Returned as `HipBackend::resident_vram_info`'s
/// `(usage, resident)` on Linux, it makes the pool sizing's `resident - usage` the real free VRAM.
/// `None` when no card matches, or when two different cards do (it cannot tell which is ours).
pub fn amdgpu_vram_used_total(hip_total: u64) -> Option<(u64, u64)> {
    let read_u64 = |p: std::path::PathBuf| -> Option<u64> { std::fs::read_to_string(p).ok()?.trim().parse().ok() };
    let mut best: Option<(u64, std::path::PathBuf, u64, u64)> = None; // (size diff, device, used, total)
    let mut ambiguous = false;
    for entry in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let dev = entry.path().join("device");
        let vendor = std::fs::read_to_string(dev.join("vendor")).unwrap_or_default();
        if vendor.trim() != "0x1002" {
            continue;
        }
        let (Some(total), Some(used)) = (read_u64(dev.join("mem_info_vram_total")), read_u64(dev.join("mem_info_vram_used"))) else {
            continue;
        };
        let diff = total.abs_diff(hip_total);
        if diff > hip_total / 16 {
            continue;
        }
        // card1 and renderD128 are the same device: compare the resolved PCI device paths.
        let canon = std::fs::canonicalize(&dev).unwrap_or(dev);
        match &best {
            Some((_, path, ..)) if *path == canon => {}
            Some((d, ..)) if diff == *d => ambiguous = true,
            Some((d, ..)) if diff > *d => {}
            _ => {
                best = Some((diff, canon, used, total));
                ambiguous = false;
            }
        }
    }
    if ambiguous {
        return None;
    }
    best.map(|(_, _, used, total)| (used, total))
}

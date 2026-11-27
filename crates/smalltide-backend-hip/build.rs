use std::env;
use std::fmt::Write as _;
#[cfg_attr(not(windows), allow(unused_imports))] // `Path` is used by the Windows search only
use std::path::{Path, PathBuf};
use std::process::Command;

/// Kernel targets built when `SMALLTIDE_HIP_ARCH` is not set: the two RDNA 4 chips.
/// `gfx1201` = RX 9070 XT / 9070 / 9070 GRE / Radeon AI PRO R9700, `gfx1200` = RX 9060 XT / 9060.
/// The first entry is the fallback when the runtime cannot name the card's architecture.
const DEFAULT_ARCHS: &str = "gfx1201,gfx1200";

// Linux/Unix hosts: `hipcc` search in its own file; the function below is the Windows one.
#[cfg(not(windows))]
#[path = "build_linux.rs"]
mod build_linux;
#[cfg(not(windows))]
use build_linux::find_hipcc;

#[cfg(windows)]
fn find_hipcc() -> Option<PathBuf> {
    if let Ok(p) = env::var("SMALLTIDE_HIPCC") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    if let Ok(rocm) = env::var("ROCM_PATH").or_else(|_| env::var("HIP_PATH")) {
        let candidate = Path::new(&rocm).join("bin").join("hipcc.exe");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    // Search standard AMD ROCm Windows install paths (newest version first)
    let rocm_root = Path::new(r"C:\Program Files\AMD\ROCm");
    if let Ok(entries) = std::fs::read_dir(rocm_root) {
        let mut versions: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        versions.sort();
        versions.reverse();
        for dir in versions {
            let candidate = dir.join("bin").join("hipcc.exe");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    None
}

fn main() {
    println!("cargo:rerun-if-changed=kernels/core_ops.hip");
    println!("cargo:rerun-if-changed=kernels/moe_ops.hip");
    println!("cargo:rerun-if-changed=kernels/attn_ops.hip");
    println!("cargo:rerun-if-changed=kernels/inkling2_ops.hip");
    println!("cargo:rerun-if-changed=kernels/batched_ops.hip");
    println!("cargo:rerun-if-changed=kernels/prefill_ops.hip");
    println!("cargo:rerun-if-changed=kernels/vision_ops.hip");
    println!("cargo:rerun-if-env-changed=SMALLTIDE_HIPCC");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=SMALLTIDE_HIP_ARCH");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    // One .hsaco per architecture, comma-separated (e.g. SMALLTIDE_HIP_ARCH=gfx1201 for one).
    let arch_list = env::var("SMALLTIDE_HIP_ARCH")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_ARCHS.to_string());
    let mut archs: Vec<String> = Vec::new();
    for a in arch_list.split(',').map(str::trim).filter(|a| !a.is_empty()) {
        if !a.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            panic!("SMALLTIDE_HIP_ARCH: {a:?} is not an architecture name (expected e.g. gfx1201,gfx1200)");
        }
        if !archs.iter().any(|x| x == a) {
            archs.push(a.to_string());
        }
    }
    if archs.is_empty() {
        panic!("SMALLTIDE_HIP_ARCH lists no architecture");
    }

    // `hsaco_table.rs` is included by src/ffi.rs: (architecture, code object) in list order.
    let mut table = String::from("pub static EMBEDDED_HSACOS: &[(&str, &[u8])] = &[\n");
    for a in &archs {
        writeln!(table, "    (\"{a}\", include_bytes!(concat!(env!(\"OUT_DIR\"), \"/smalltide_{a}.hsaco\"))),").unwrap();
    }
    table.push_str("];\n");
    std::fs::write(out_dir.join("hsaco_table.rs"), table).expect("write hsaco_table.rs");

    let Some(hipcc) = find_hipcc() else {
        // Allow compiling the workspace even on machines without ROCm SDK installed;
        // emit empty marker files and fail gracefully at runtime if kernels are invoked.
        println!("cargo:warning=hipcc not found; building smalltide-backend-hip without compiled .hsaco kernels");
        for a in &archs {
            std::fs::write(out_dir.join(format!("smalltide_{a}.hsaco")), []).expect("write empty hsaco placeholder");
        }
        return;
    };

    for arch in &archs {
        let hsaco_out = out_dir.join(format!("smalltide_{arch}.hsaco"));
        let status = Command::new(&hipcc)
            .args([
                "--genco",
                &format!("--offload-arch={arch}"),
                "-O3",
                "-ffast-math",
                "kernels/core_ops.hip",
                "-o",
            ])
            .arg(&hsaco_out)
            .status()
            .unwrap_or_else(|e| panic!("failed to execute {}: {e}", hipcc.display()));

        if !status.success() {
            panic!("hipcc failed with status {status} compiling kernels/core_ops.hip for {arch}");
        }
    }
}

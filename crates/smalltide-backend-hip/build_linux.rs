// Linux (non-Windows host) `hipcc` discovery for build.rs; the Windows search lives in build.rs.

use std::env;
use std::path::{Path, PathBuf};

/// `SMALLTIDE_HIPCC`, then `$ROCM_PATH`/`$HIP_PATH`/bin/hipcc, then `/opt/rocm/bin/hipcc` and
/// versioned `/opt/rocm-*` installs (newest first), then the first `hipcc` on `PATH`.
pub fn find_hipcc() -> Option<PathBuf> {
    if let Ok(p) = env::var("SMALLTIDE_HIPCC") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    if let Ok(rocm) = env::var("ROCM_PATH").or_else(|_| env::var("HIP_PATH")) {
        let candidate = Path::new(&rocm).join("bin").join("hipcc");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    let default = Path::new("/opt/rocm/bin/hipcc");
    if default.exists() {
        return Some(default.to_path_buf());
    }
    if let Ok(entries) = std::fs::read_dir("/opt") {
        let mut versions: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir() && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("rocm-")))
            .collect();
        versions.sort();
        versions.reverse();
        for dir in versions {
            let candidate = dir.join("bin").join("hipcc");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    println!("cargo:rerun-if-env-changed=PATH");
    let path = env::var_os("PATH")?;
    env::split_paths(&path).map(|d| d.join("hipcc")).find(|c| c.is_file())
}

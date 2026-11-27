//! Linux unbuffered direct NVMe I/O (`O_DIRECT` + positional reads).
//!
//! Linux counterpart of `win_direct_io.rs` (`FILE_FLAG_NO_BUFFERING`): bypasses the page cache
//! so Stage 2 (Disk -> RAM) expert streaming never competes with the Tier 1 RAM pool.
//!
//! On non-Windows targets `lib.rs` compiles this file *as* `win_direct_io`, so it must keep the
//! exact public API of the Windows module (`SECTOR_ALIGN`, `align_up`, `AlignedBuffer`,
//! `DirectFile`) and callers need no change.

use smalltide_core::{Result, SmallTideError};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::Path;

pub const SECTOR_ALIGN: usize = 4096;

pub const fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

/// `O_DIRECT` from `<fcntl.h>`; its value differs per architecture.
#[cfg(all(target_os = "linux", any(target_arch = "aarch64", target_arch = "arm")))]
const O_DIRECT: i32 = 0o200000;
#[cfg(all(target_os = "linux", not(any(target_arch = "aarch64", target_arch = "arm"))))]
const O_DIRECT: i32 = 0o40000;
/// Other Unix targets: plain buffered reads (no portable `O_DIRECT`).
#[cfg(not(target_os = "linux"))]
const O_DIRECT: i32 = 0;

/// `EINVAL`: returned by `open` when the filesystem does not support `O_DIRECT` (e.g. tmpfs).
const EINVAL: i32 = 22;

/// Page-aligned (4 KB) host buffer required by `O_DIRECT`.
pub struct AlignedBuffer {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    pub fn alloc(size_bytes: usize) -> Result<Self> {
        let len = align_up(size_bytes.max(SECTOR_ALIGN), SECTOR_ALIGN);
        let layout = Self::layout(len)?;
        // Zeroed like VirtualAlloc'd pages; large sizes are lazily zeroed by mmap.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(SmallTideError::Io(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!("AlignedBuffer: cannot allocate {len} bytes"),
            )));
        }
        Ok(Self { ptr, len })
    }

    fn layout(len: usize) -> Result<Layout> {
        Layout::from_size_align(len, SECTOR_ALIGN).map_err(|e| {
            SmallTideError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))
        })
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if let Ok(layout) = Self::layout(self.len) {
            unsafe { dealloc(self.ptr, layout) };
        }
    }
}

/// Unbuffered file handle (`O_DIRECT`, falling back to buffered reads where unsupported).
pub struct DirectFile {
    file: File,
}

impl DirectFile {
    pub fn open_read(path: &Path) -> Result<Self> {
        let direct = OpenOptions::new().read(true).custom_flags(O_DIRECT).open(path);
        let file = match direct {
            Ok(f) => f,
            Err(e) if e.raw_os_error() == Some(EINVAL) => {
                eprintln!(
                    "[io] {}: filesystem does not support O_DIRECT; using buffered reads",
                    path.display()
                );
                File::open(path).map_err(SmallTideError::Io)?
            }
            Err(e) => return Err(SmallTideError::Io(e)),
        };
        Ok(Self { file })
    }

    /// Reads `len` bytes (must be a multiple of 4096) at `offset` (multiple of 4096)
    /// into an aligned buffer. Returns fewer bytes only at end of file.
    pub fn read_aligned_at(&self, offset: u64, buf: &mut AlignedBuffer, len: usize) -> Result<usize> {
        if offset % (SECTOR_ALIGN as u64) != 0 || len % SECTOR_ALIGN != 0 || len > buf.len() {
            return Err(SmallTideError::InvalidModel(format!(
                "direct read requires 4KB alignment: offset={offset}, len={len}, cap={}",
                buf.len()
            )));
        }
        let dst = &mut buf.as_mut_slice()[..len];
        let mut done = 0usize;
        while done < len {
            match self.file.read_at(&mut dst[done..], offset + done as u64) {
                Ok(0) => break, // end of file
                Ok(n) => done += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(SmallTideError::Io(e)),
            }
        }
        Ok(done)
    }
}

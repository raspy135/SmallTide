//! Stage 2 storage I/O: Win32 `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` / IoRing
//! async expert reader, Safetensors & GGUF parsers, and optional 4 KB-aligned expert packer.

pub mod dense_loader;
pub mod quant;
pub mod safetensors;
pub mod stide_pack;
pub mod tokenizer;
pub mod unicode_cats;
#[cfg(windows)]
pub mod win_direct_io;
// Linux build: same API under the same module path, so callers are identical on both platforms.
#[cfg(not(windows))]
#[path = "linux_direct_io.rs"]
pub mod win_direct_io;

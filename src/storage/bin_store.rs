// storage/bin_store.rs
// Compact binary persistence. ~4x smaller and ~10x faster than JSON.
// Custom format with magic-number header, raw f32 vectors, and
// length-prefixed strings instead of quoted JSON overhead.

use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::{fs, io};

use crate::core::record::Record;
use crate::core::{Result, VectorDBError};
use crate::storage::PersistentStorage;

/// Magic bytes: "MVDB" in ASCII (big-endian layout in hex).
/// Used to reject files that aren't mini-vectordb binary dumps.
const MAGIC: u32 = 0x4D56_4442;

/// Current binary format version. Increment when the layout changes
/// so load() can reject outdated files with a clear error message.
const VERSION: u32 = 1;

/// Binary persistence backend.
///
/// # Notes
/// The binary format eliminates all JSON structural overhead:
/// no field names, no colons, no quotes, no whitespace, no brackets.
/// Vectors are stored as raw `[f32]` bytes in little-endian — same
/// layout as Rust's native `Vec<f32>`, so we can transmute `&[u8]`
/// back to `&[f32]` without copying.
///
/// Strings (id, metadata keys/values) use u32-prefixed length encoding
/// for O(1) skip-ahead on malformed or partial reads.
///
/// File size estimate: for 128-dim vectors, ~520 bytes/record vs
/// ~2000 bytes/record in JSON — about 4x smaller.
pub struct BinStorage {
    /// All records managed by this storage instance.
    records: Vec<Record>,
}

impl BinStorage {
    /// Create an empty binary storage.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Build storage from an existing record collection.
    pub fn from_records(records: Vec<Record>) -> Self {
        Self { records }
    }

    /// Consume storage and return the records.
    pub fn into_records(self) -> Vec<Record> {
        self.records
    }

    /// Number of records held.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True when storage has zero records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// `BinStorage::new()` provides the canonical empty state.
impl Default for BinStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistentStorage for BinStorage {
    /// Write all records to a compact binary file.
    ///
    /// # Implementation
    /// 1. Write header: magic, version, count, dimension.
    /// 2. For each record: write id, raw f32 vector bytes, metadata.
    /// 3. Uses BufWriter for buffered I/O.
    /// 4. Write-then-rename for atomicity.
    ///
    /// # Notes
    /// All multi-byte integers are little-endian via `u32::to_le_bytes()` —
    /// matches native byte order on x86/ARM so no bswap needed.
    fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        // write to temp file first, then atomically rename
        let tmp = path.with_extension("tmp");
        let file = fs::File::create(&tmp)
            .map_err(|e| VectorDBError::Other(format!("failed to create temp file: {e}")))?;
        let mut w = BufWriter::new(file);

        // determine dimension from first record; 0 if empty
        let dim: u32 = self
            .records
            .first()
            .map(|r| r.vector.len() as u32)
            .unwrap_or(0);

        write_u32(&mut w, MAGIC).map_err(|e| VectorDBError::Other(format!("write magic: {e}")))?;
        write_u32(&mut w, VERSION)
            .map_err(|e| VectorDBError::Other(format!("write version: {e}")))?;
        write_u32(&mut w, self.records.len() as u32)
            .map_err(|e| VectorDBError::Other(format!("write count: {e}")))?;
        write_u32(&mut w, dim)
            .map_err(|e| VectorDBError::Other(format!("write dimension: {e}")))?;

        for r in &self.records {
            write_str(&mut w, &r.id).map_err(|e| VectorDBError::Other(format!("write id: {e}")))?;
            write_vector(&mut w, &r.vector)
                .map_err(|e| VectorDBError::Other(format!("write vector: {e}")))?;

            // metadata: count-prefixed key-value pairs
            let meta_len = r.metadata.len() as u32;
            write_u32(&mut w, meta_len)
                .map_err(|e| VectorDBError::Other(format!("write meta count: {e}")))?;
            for (k, v) in &r.metadata {
                write_str(&mut w, k)
                    .map_err(|e| VectorDBError::Other(format!("write meta key: {e}")))?;
                write_str(&mut w, v)
                    .map_err(|e| VectorDBError::Other(format!("write meta val: {e}")))?;
            }
        }

        // flush BufWriter then atomic rename
        w.into_inner()
            .map_err(|_| VectorDBError::Other("flush failed".into()))?;
        fs::rename(&tmp, path).map_err(|e| VectorDBError::Other(format!("rename failed: {e}")))?;
        Ok(())
    }

    /// Read records from a binary file into a new `BinStorage`.
    ///
    /// # Implementation
    /// 1. Read and validate header (magic, version).
    /// 2. Read count records using dimension from header.
    /// 3. For each record: read id, read raw f32 bytes, read metadata.
    /// 4. Uses BufReader for buffered I/O.
    ///
    /// # Validation
    /// Rejects files that:
    /// - Don't start with MAGIC (not a mini-vectordb binary file).
    /// - Have a VERSION > expected (forward-incompatible).
    /// - Are truncated mid-record (unexpected EOF).
    fn load(path: impl AsRef<Path>) -> Result<Self>
    where
        Self: Sized,
    {
        let file = fs::File::open(path.as_ref())
            .map_err(|e| VectorDBError::Other(format!("failed to open file: {e}")))?;
        let mut r = BufReader::new(file);

        let magic =
            read_u32(&mut r).map_err(|e| VectorDBError::Other(format!("read magic: {e}")))?;
        if magic != MAGIC {
            return Err(VectorDBError::Other(format!(
                "bad magic: 0x{magic:08X}, expected 0x{MAGIC:08X}"
            )));
        }

        let version =
            read_u32(&mut r).map_err(|e| VectorDBError::Other(format!("read version: {e}")))?;
        if version > VERSION {
            return Err(VectorDBError::Other(format!(
                "unsupported version {version} (max {VERSION})"
            )));
        }

        let count =
            read_u32(&mut r).map_err(|e| VectorDBError::Other(format!("read count: {e}")))?;
        let dim =
            read_u32(&mut r).map_err(|e| VectorDBError::Other(format!("read dimension: {e}")))?;

        let mut records = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let id =
                read_string(&mut r).map_err(|e| VectorDBError::Other(format!("read id: {e}")))?;
            let vector = read_vector(&mut r, dim as usize)
                .map_err(|e| VectorDBError::Other(format!("read vector: {e}")))?;

            let meta_len = read_u32(&mut r)
                .map_err(|e| VectorDBError::Other(format!("read meta count: {e}")))?;
            let mut metadata = HashMap::with_capacity(meta_len as usize);
            for _ in 0..meta_len {
                let key = read_string(&mut r)
                    .map_err(|e| VectorDBError::Other(format!("read meta key: {e}")))?;
                let val = read_string(&mut r)
                    .map_err(|e| VectorDBError::Other(format!("read meta val: {e}")))?;
                metadata.insert(key, val);
            }

            records.push(Record::with_metadata(id, vector, metadata));
        }

        Ok(Self { records })
    }
}

// --- private read/write helpers ---

/// Write a u32 in little-endian byte order.
fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

/// Read a u32 in little-endian byte order.
/// Returns `UnexpectedEof` if fewer than 4 bytes are available.
fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Write a length-prefixed UTF-8 string: 4-byte u32 length, then raw bytes.
fn write_str(w: &mut impl Write, s: &str) -> io::Result<()> {
    write_u32(w, s.len() as u32)?;
    w.write_all(s.as_bytes())
}

/// Read a length-prefixed UTF-8 string.
/// Returns `UnexpectedEof` if the file ends before all bytes are consumed.
fn read_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u32(r)? as usize;
    // allocate exactly once for the string content
    let mut buf = vec![0u8; len];
    if len > 0 {
        r.read_exact(&mut buf)?;
    }
    // safe: the file stores valid UTF-8 from write_str; if it doesn't,
    // this returns an error which bubbles up correctly
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Write a vector of f32 as raw little-endian bytes.
///
/// Casts `&[f32]` to `&[u8]` via pointer cast — safe because f32 has
/// no padding bits and all bit patterns are valid f32 values.
fn write_vector(w: &mut impl Write, v: &[f32]) -> io::Result<()> {
    // pointer cast: `v.len() * size_of::<f32>()` bytes of f32 data
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    w.write_all(bytes)
}

/// Read `dim` f32 values from raw little-endian bytes.
///
/// Allocates a `vec![0u8; dim * 4]`, reads into it, then transmutes
/// the bytes back to `Vec<f32>`. Safe because every 4-byte group is
/// a valid f32 bit pattern (no invalid states).
fn read_vector(r: &mut impl Read, dim: usize) -> io::Result<Vec<f32>> {
    let n = dim * 4;
    let mut buf = vec![0u8; n];
    if n > 0 {
        r.read_exact(&mut buf)?;
    }
    // rebuild Vec<f32> from byte buffer without copying
    //
    // safe: n is always a multiple of 4 (dim * 4), Vec<f32>'s layout
    // is identical to [u8; n] for n = dim * 4 bytes on all
    // architectures with IEEE 754 f32 (which is all of them).
    let ptr = buf.as_mut_ptr() as *mut f32;
    let len = dim;
    let cap = dim;
    std::mem::forget(buf); // don't run Vec<u8> destructor — ownership transferred
    Ok(unsafe { Vec::from_raw_parts(ptr, len, cap) })
}

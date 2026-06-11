use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::{fs, io};

use crate::core::record::Record;
use crate::core::{Result, VectorDBError};
use crate::metadata::MetadataValue;
use crate::storage::PersistentStorage;

const MAGIC: u32 = 0x4D56_4442;
const VERSION: u32 = 1;

/// Binary persistence backend.
pub struct BinStorage {
    records: Vec<Record>,
}

impl BinStorage {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    pub fn from_records(records: Vec<Record>) -> Self {
        Self { records }
    }

    pub fn into_records(self) -> Vec<Record> {
        self.records
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

impl Default for BinStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistentStorage for BinStorage {
    fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        let file = fs::File::create(&tmp)
            .map_err(|e| VectorDBError::Other(format!("failed to create temp file: {e}")))?;
        let mut w = BufWriter::new(file);

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

            let meta_len = r.metadata.len() as u32;
            write_u32(&mut w, meta_len)
                .map_err(|e| VectorDBError::Other(format!("write meta count: {e}")))?;
            for (k, v) in &r.metadata {
                write_str(&mut w, k)
                    .map_err(|e| VectorDBError::Other(format!("write meta key: {e}")))?;
                let json = serde_json::to_string(v)
                    .map_err(|e| VectorDBError::Other(format!("meta serialize: {e}")))?;
                write_str(&mut w, &json)
                    .map_err(|e| VectorDBError::Other(format!("write meta val: {e}")))?;
            }
        }

        w.into_inner()
            .map_err(|_| VectorDBError::Other("flush failed".into()))?;
        fs::rename(&tmp, path).map_err(|e| VectorDBError::Other(format!("rename failed: {e}")))?;
        Ok(())
    }

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
                let json = read_string(&mut r)
                    .map_err(|e| VectorDBError::Other(format!("read meta val: {e}")))?;
                let val: MetadataValue = serde_json::from_str(&json)
                    .map_err(|e| VectorDBError::Other(format!("meta parse: {e}")))?;
                metadata.insert(key, val);
            }

            records.push(Record::with_metadata(id, vector, metadata));
        }

        Ok(Self { records })
    }
}

fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_str(w: &mut impl Write, s: &str) -> io::Result<()> {
    write_u32(w, s.len() as u32)?;
    w.write_all(s.as_bytes())
}

fn read_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u32(r)? as usize;
    let mut buf = vec![0u8; len];
    if len > 0 {
        r.read_exact(&mut buf)?;
    }
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Casts &[f32] to &[u8] — safe: f32 has no padding bits.
fn write_vector(w: &mut impl Write, v: &[f32]) -> io::Result<()> {
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    w.write_all(bytes)
}

/// Reads raw bytes and reconstructs a Vec<f32> without copying.
fn read_vector(r: &mut impl Read, dim: usize) -> io::Result<Vec<f32>> {
    let n = dim * 4;
    let mut buf = vec![0u8; n];
    if n > 0 {
        r.read_exact(&mut buf)?;
    }
    let ptr = buf.as_mut_ptr() as *mut f32;
    let len = dim;
    let cap = dim;
    std::mem::forget(buf);
    // safe: all f32 bit patterns are valid; n is always a multiple of 4
    Ok(unsafe { Vec::from_raw_parts(ptr, len, cap) })
}

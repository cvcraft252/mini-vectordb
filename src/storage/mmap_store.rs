//! Memory-mapped vector storage for GB-scale datasets.

use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::core::record::Record;
use crate::core::{Result, VectorDBError};
use crate::metadata::Metadata;

const MMAP_MAGIC: u32 = 0x4D56_4442;
const MMAP_VERSION: u32 = 1;

/// Memory-mapped vector storage. Vectors live on disk, IDs/metadata in memory.
pub struct MmapStore {
    ids: Vec<String>,
    metadata: Vec<Metadata>,
    id_to_idx: HashMap<String, usize>,
    mmap: Mmap,
    dimension: usize,
}

impl MmapStore {
    /// Memory-maps a binary vector file for zero-copy access.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = fs::File::open(path.as_ref())
            .map_err(|e| VectorDBError::Other(format!("open: {e}")))?;

        let file_len = file
            .metadata()
            .map_err(|e| VectorDBError::Other(format!("metadata: {e}")))?
            .len() as usize;
        if file_len < 16 {
            return Err(VectorDBError::Other("file too small".into()));
        }

        let mmap =
            unsafe { Mmap::map(&file).map_err(|e| VectorDBError::Other(format!("mmap: {e}")))? };

        let magic = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
        if magic != MMAP_MAGIC {
            return Err(VectorDBError::Other(format!("bad magic: 0x{magic:08X}")));
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version > MMAP_VERSION {
            return Err(VectorDBError::Other(format!(
                "unsupported version {version}"
            )));
        }
        let count = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
        let dimension = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;

        let mut ids = Vec::with_capacity(count);
        let metadata = vec![Metadata::new(); count];
        let mut id_to_idx = HashMap::with_capacity(count);

        for i in 0..count {
            let id = format!("{i}");
            id_to_idx.insert(id.clone(), i);
            ids.push(id);
        }

        Ok(Self {
            ids,
            metadata,
            id_to_idx,
            mmap,
            dimension,
        })
    }

    /// Writes records to a binary file suitable for mmap loading.
    pub fn write(path: impl AsRef<Path>, records: &[Record]) -> Result<()> {
        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        let file =
            fs::File::create(&tmp).map_err(|e| VectorDBError::Other(format!("create: {e}")))?;
        let mut w = BufWriter::new(file);

        let dim = records.first().map(|r| r.vector.len()).unwrap_or(0);
        w.write_all(&MMAP_MAGIC.to_le_bytes())
            .map_err(|e| VectorDBError::Other(format!("magic: {e}")))?;
        w.write_all(&MMAP_VERSION.to_le_bytes())
            .map_err(|e| VectorDBError::Other(format!("version: {e}")))?;
        w.write_all(&(records.len() as u32).to_le_bytes())
            .map_err(|e| VectorDBError::Other(format!("count: {e}")))?;
        w.write_all(&(dim as u32).to_le_bytes())
            .map_err(|e| VectorDBError::Other(format!("dim: {e}")))?;

        for r in records {
            // safe: f32 has no padding bits
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(r.vector.as_ptr() as *const u8, r.vector.len() * 4)
            };
            w.write_all(bytes)
                .map_err(|e| VectorDBError::Other(format!("vector: {e}")))?;
        }

        w.into_inner()
            .map_err(|_| VectorDBError::Other("flush failed".into()))?;
        fs::rename(&tmp, path).map_err(|e| VectorDBError::Other(format!("rename: {e}")))?;
        Ok(())
    }

    /// Looks up a record by ID. The vector is copied from the mmap'd region.
    pub fn get(&self, id: &str) -> Option<Record> {
        let idx = *self.id_to_idx.get(id)?;
        let offset = 16 + idx * self.dimension * 4;
        let end = offset + self.dimension * 4;
        if end > self.mmap.len() {
            return None;
        }
        let floats: &[f32] = unsafe {
            std::slice::from_raw_parts(
                self.mmap[offset..end].as_ptr() as *const f32,
                self.dimension,
            )
        };
        let mut record = Record::new(id, floats.to_vec());
        record.metadata = self.metadata[idx].clone();
        Some(record)
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn ids(&self) -> &[String] {
        &self.ids
    }
}

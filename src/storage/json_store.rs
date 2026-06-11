use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::core::record::Record;
use crate::core::{Result, VectorDBError};
use crate::storage::PersistentStorage;

/// JSON-based persistence backend.
pub struct JsonStorage {
    records: Vec<Record>,
}

impl JsonStorage {
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

impl Default for JsonStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistentStorage for JsonStorage {
    fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        let file = fs::File::create(&tmp)
            .map_err(|e| VectorDBError::Other(format!("failed to create temp file: {e}")))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.records)
            .map_err(|e| VectorDBError::Other(format!("serialization failed: {e}")))?;
        fs::rename(&tmp, path).map_err(|e| VectorDBError::Other(format!("rename failed: {e}")))?;
        Ok(())
    }

    fn load(path: impl AsRef<Path>) -> Result<Self>
    where
        Self: Sized,
    {
        let file = fs::File::open(path.as_ref())
            .map_err(|e| VectorDBError::Other(format!("failed to open file: {e}")))?;
        let reader = BufReader::new(file);
        let records: Vec<Record> = serde_json::from_reader(reader)
            .map_err(|e| VectorDBError::Other(format!("deserialization failed: {e}")))?;
        Ok(Self { records })
    }
}

use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::core::record::Record;
use crate::core::{Result, VectorDBError};
use crate::storage::PersistentStorage;

/// JSON-based persistence backend.
///
/// # Notes
/// The JSON format is a single array of Record objects:
/// ```json
/// [
///   {
///     "id": "doc1",
///     "vector": [0.1, 0.2, 0.3],
///     "metadata": {"category": "book"}
///   }
/// ]
/// ```
/// Pretty-printed with 2-space indent so diffs are meaningful.
/// File size is ~4x larger than binary format but human-auditable.
pub struct JsonStorage {
    /// All records managed by this storage instance.
    /// Public-only via `from_records` / `into_records` to keep
    /// the collection opaque to callers.
    records: Vec<Record>,
}

impl JsonStorage {
    /// Create an empty storage with zero records.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Build storage from an existing record collection.
    ///
    /// This is the canonical constructor for save-then-persist flows:
    /// 1. Extract records from index
    /// 2. `JsonStorage::from_records(records)`
    /// 3. `storage.save(path)`
    pub fn from_records(records: Vec<Record>) -> Self {
        Self { records }
    }

    /// Consume storage and return the records.
    ///
    /// Used after `load` to feed records into an index:
    /// ```ignore
    /// let storage = JsonStorage::load(path)?;
    /// for r in storage.into_records() {
    ///     index.insert(r)?;
    /// }
    /// ```
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

/// `JsonStorage::new()` provides the canonical empty state.
impl Default for JsonStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistentStorage for JsonStorage {
    /// Serialize all records to a pretty-printed JSON file.
    ///
    /// # Implementation
    /// Uses `serde_json::to_writer_pretty` for human-readable output.
    /// Record vectors are written with full float precision.
    /// The file is UTF-8 with 2-space indentation.
    ///
    /// # Atomicity
    /// Write-then-rename pattern: first write to `{path}.tmp`,
    /// then atomically rename. If the rename succeeds, the old
    /// file (if any) is replaced. If the process crashes mid-write,
    /// only the temp file is left behind — the original stays intact.
    fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        // write to a temp file first so a crash mid-write doesn't
        // corrupt the original — the atomic rename is the commit point
        let tmp = path.with_extension("tmp");
        let file = fs::File::create(&tmp)
            .map_err(|e| VectorDBError::Other(format!("failed to create temp file: {e}")))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.records)
            .map_err(|e| VectorDBError::Other(format!("serialization failed: {e}")))?;
        fs::rename(&tmp, path).map_err(|e| VectorDBError::Other(format!("rename failed: {e}")))?;
        Ok(())
    }

    /// Read records from a JSON file into a new `JsonStorage`.
    ///
    /// # Implementation
    /// Uses `serde_json::from_reader` with streaming deserialization
    /// via `BufReader` to avoid loading the entire file into a String
    /// before parsing. The file must be a single JSON array of Record
    /// objects. Missing metadata fields default to empty HashMap
    /// (via `#[serde(default)]` on Record).
    ///
    /// # Errors
    /// Returns `Other` on file-not-found (wraps io::Error),
    /// malformed JSON, or vector data that doesn't match Record schema.
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

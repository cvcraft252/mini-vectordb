/// REST API server.
pub mod api;
/// Core types: Record, DistanceMetric, errors.
pub mod core;
/// Index trait and implementations.
pub mod index;
/// Typed metadata schema.
pub mod metadata;
/// Query engine: filter parsing, inverted index, query planner.
pub mod query;
/// Persistent storage backends (JSON, binary, memory-mapped).
pub mod storage;

use std::path::PathBuf;
use std::sync::RwLock;

use crate::core::metric::DistanceMetric;
use crate::core::record::Record;
use crate::core::{Result, VectorDBError};
use crate::index::flat::FlatIndex;
use crate::index::hnsw::HnswIndex;
use crate::index::{Index, SearchResult};
use crate::metadata::index::MetadataIndex;
use crate::query::filter::parse_filter;
use crate::storage::PersistentStorage;
use crate::storage::bin_store::BinStorage;
use crate::storage::json_store::JsonStorage;

/// Which serialization format to use for auto-persistence.
#[derive(Debug, Clone, Copy)]
pub enum StorageFormat {
    /// Human-readable JSON with pretty-printing.
    Json,
    /// Compact binary with magic header and raw f32 encoding.
    Binary,
}

// Defaults to FlatIndex (brute-force exact search). The RwLock is
// never held across await points (blocking API, no async).
/// Thread-safe vector database backed by an in-memory index.
///
/// Wraps the underlying index in `RwLock<Box<dyn Index>>` so multiple
/// readers can search concurrently while writes (insert/delete/update)
/// are serialized. The `dyn Index` trait object lets us swap the
/// backend (e.g. FlatIndex → HnswIndex) without changing callers.
pub struct VectorDB {
    /// Protects all index access. Read lock for search/get/len,
    /// write lock for insert/delete/update/clear.
    ///
    /// Using `Box<dyn Index>` instead of a concrete type so the
    /// backing index can be replaced at runtime (e.g. `clear()` or
    /// future adaptive index selection).
    index: RwLock<Box<dyn Index>>,

    /// Auto-persistence configuration. `None` means the database is
    /// in-memory only and callers are responsible for explicit saves
    /// via the storage module. `Some` means every mutation triggers
    /// a full save to the configured path and format.
    persistence: Option<(StorageFormat, PathBuf)>,
}

/// Record count at which FlatIndex is automatically replaced by HnswIndex.
const UPGRADE_THRESHOLD: usize = 1000;

impl VectorDB {
    /// Create a database with an empty flat index.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::VectorDB;
    /// let db = VectorDB::new();
    /// assert_eq!(db.len(), 0);
    /// assert!(db.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            index: RwLock::new(Box::new(FlatIndex::new())),
            persistence: None,
        }
    }

    /// Create a database that auto-saves to disk after every mutation.
    ///
    /// For datasets under ~10k records this overhead is < 1ms (binary)
    /// to < 10ms (JSON). Larger datasets should use manual saves via
    /// the storage module instead.
    pub fn with_persistence(path: impl Into<PathBuf>, format: StorageFormat) -> Self {
        Self {
            index: RwLock::new(Box::new(FlatIndex::new())),
            persistence: Some((format, path.into())),
        }
    }

    /// True when auto-persistence is enabled.
    pub fn is_persistent(&self) -> bool {
        self.persistence.is_some()
    }

    /// Replace the current FlatIndex with HnswIndex, migrating all records.
    fn upgrade_to_hnsw(&self, index: &mut Box<dyn Index>) -> Result<()> {
        let records = index.records();
        let mut hnsw = HnswIndex::new();
        for r in records {
            hnsw.insert(r)?;
        }
        *index = Box::new(hnsw);
        Ok(())
    }

    /// Persist a snapshot of records via the configured format.
    ///
    /// Called by mutation methods while they hold the write lock.
    /// The caller extracts records from the index and passes them
    /// here — this avoids a second trait-object deref through the
    /// `RwLockWriteGuard<Box<dyn Index>>`.
    ///
    /// When persistence is disabled (the default), this is a no-op.
    fn save_all(&self, records: Vec<Record>) -> Result<()> {
        let Some((format, path)) = &self.persistence else {
            return Ok(());
        };
        match format {
            StorageFormat::Json => JsonStorage::from_records(records).save(path),
            StorageFormat::Binary => BinStorage::from_records(records).save(path),
        }
    }

    /// Insert a record. Acquires a write lock (blocks concurrent writes).
    ///
    /// # Errors
    /// `DimensionMismatch` if vector dimension differs from stored vectors.
    /// `EmptyVector` if the vector has zero elements.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::core::record::Record;
    /// # use mini_vectordb::VectorDB;
    /// # use mini_vectordb::core::VectorDBError;
    /// # fn main() -> Result<(), VectorDBError> {
    /// let db = VectorDB::new();
    /// let r = Record::new("doc1", vec![0.1, 0.2, 0.3]);
    /// db.insert(r)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert(&self, record: Record) -> Result<()> {
        let mut index = self
            .index
            .write()
            .expect("RwLock is never poisoned; no panics in write-locked sections");
        index.insert(record)?;
        // upgrade to HNSW exactly when crossing the threshold
        if index.len() == UPGRADE_THRESHOLD {
            self.upgrade_to_hnsw(&mut index)?;
        }
        self.save_all(index.records())
    }

    /// Search for the top_k most similar vectors. Acquires a read lock
    /// (multiple concurrent searches are allowed).
    ///
    /// # Errors
    /// `DimensionMismatch` if query dimension doesn't match stored vectors.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::core::record::Record;
    /// # use mini_vectordb::core::metric::DistanceMetric;
    /// # use mini_vectordb::VectorDB;
    /// # use mini_vectordb::core::VectorDBError;
    /// # fn main() -> Result<(), VectorDBError> {
    /// # let db = VectorDB::new();
    /// # db.insert(Record::new("a", vec![0.1, 0.2, 0.3]))?;
    /// let results = db.search(&[0.1, 0.2, 0.3], 5, DistanceMetric::Cosine)?;
    /// for r in &results {
    ///     println!("{} -> dist={:.4}", r.id, r.distance);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>> {
        self.index
            .read()
            .expect("RwLock is never poisoned; no panics in read-locked sections")
            .search(query, top_k, metric)
    }

    /// Batch search — parallel across queries under a single read lock.
    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<Vec<SearchResult>>> {
        self.index
            .read()
            .expect("RwLock is never poisoned; no panics in read-locked sections")
            .search_batch(queries, top_k, metric)
    }

    /// Search with a metadata filter applied before distance computation.
    ///
    /// The filter reduces the candidate set, then vector search runs only
    /// on the remaining records. Faster than search-then-filter when the
    /// filter is selective (matches < 50% of records).
    ///
    /// # Errors
    /// Returns `VectorDBError::Other` if the filter expression is malformed.
    ///
    /// ```
    /// # use mini_vectordb::VectorDB;
    /// # use mini_vectordb::core::record::Record;
    /// # use mini_vectordb::core::metric::DistanceMetric;
    /// # use mini_vectordb::metadata::MetadataValue;
    /// let db = VectorDB::new();
    /// let mut meta = mini_vectordb::metadata::Metadata::new();
    /// meta.insert("cat".into(), MetadataValue::String("book".into()));
    /// db.insert(Record::with_metadata("r1", vec![1.0], meta)).unwrap();
    /// let results = db.search_filtered(
    ///     &[1.0], 5, DistanceMetric::Euclidean, "cat = \"book\""
    /// ).unwrap();
    /// assert_eq!(results.len(), 1);
    /// ```
    pub fn search_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
        filter_expr: &str,
    ) -> Result<Vec<SearchResult>> {
        let filter = parse_filter(filter_expr)
            .map_err(|e| VectorDBError::Other(format!("filter parse error: {e}")))?;

        let index = self
            .index
            .read()
            .expect("RwLock is never poisoned; no panics in read-locked sections");

        // Build a temporary metadata index from the current record set.
        // Under the read lock the snapshot is consistent.
        let records = index.records();
        let mut meta_idx = MetadataIndex::new();
        for r in &records {
            meta_idx.index_record(&r.id, &r.metadata);
        }

        crate::query::planner::execute_filtered_search(
            query, top_k, metric, &filter, &meta_idx, &**index,
        )
    }

    /// Look up a record by ID. Acquires a read lock.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::core::record::Record;
    /// # use mini_vectordb::VectorDB;
    /// # use mini_vectordb::core::VectorDBError;
    /// # fn main() -> Result<(), VectorDBError> {
    /// # let db = VectorDB::new();
    /// # db.insert(Record::new("doc1", vec![0.1, 0.2, 0.3]))?;
    /// if let Some(r) = db.get("doc1")? {
    ///     assert_eq!(r.id, "doc1");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn get(&self, id: &str) -> Result<Option<Record>> {
        self.index
            .read()
            .expect("RwLock is never poisoned; no panics in read-locked sections")
            .get(id)
    }

    /// Remove a record by ID. Acquires a write lock.
    /// Silently succeeds if the ID doesn't exist.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::core::record::Record;
    /// # use mini_vectordb::VectorDB;
    /// # use mini_vectordb::core::VectorDBError;
    /// # fn main() -> Result<(), VectorDBError> {
    /// # let db = VectorDB::new();
    /// # db.insert(Record::new("doc1", vec![0.1, 0.2, 0.3]))?;
    /// db.delete("doc1")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn delete(&self, id: &str) -> Result<()> {
        let mut index = self
            .index
            .write()
            .expect("RwLock is never poisoned; no panics in write-locked sections");
        index.delete(id)?;
        self.save_all(index.records())
    }

    /// Replace a record's vector while keeping its ID and metadata intact.
    /// Acquires a write lock.
    ///
    /// # Errors
    /// `NotFound` if no record with this ID exists.
    /// `DimensionMismatch` if the new vector has wrong dimension.
    /// `EmptyVector` if the new vector is empty.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::core::record::Record;
    /// # use mini_vectordb::VectorDB;
    /// # use mini_vectordb::core::VectorDBError;
    /// # fn main() -> Result<(), VectorDBError> {
    /// # let db = VectorDB::new();
    /// # db.insert(Record::new("doc1", vec![0.1, 0.2, 0.3]))?;
    /// db.update("doc1", vec![0.5, 0.6, 0.7])?;
    /// # Ok(())
    /// # }
    /// ```
    // Delete-then-insert: reads old record for metadata, deletes it,
    // inserts updated version. O(N) with FlatIndex but avoids adding
    // a dedicated "replace" path to the Index trait.
    pub fn update(&self, id: &str, vector: Vec<f32>) -> Result<()> {
        let mut index = self
            .index
            .write()
            .expect("RwLock is never poisoned; no panics in write-locked sections");
        // fetch old record to preserve its metadata
        let old = index.get(id)?;
        if let Some(mut record) = old {
            // validate dimension before delete to avoid the edge case
            // where deleting the last record resets the dimension lock,
            // allowing a mismatched vector to slip through
            let dim = vector.len();
            if dim == 0 {
                return Err(VectorDBError::EmptyVector);
            }
            let stored_dim = record.vector.len();
            // if this is the only record and we delete it first, the
            // dimension resets to 0 — so we validate upfront
            if dim != stored_dim {
                return Err(VectorDBError::DimensionMismatch {
                    expected: stored_dim,
                    actual: dim,
                });
            }
            record.vector = vector;
            // delete-then-insert: simpler than adding a replace method
            // to the Index trait, and keeps insertion path consistent
            index.delete(id)?;
            index.insert(record)?;
            self.save_all(index.records())
        } else {
            // clone the id into the error — we can't move `id` since
            // it's a shared reference
            Err(VectorDBError::NotFound(id.to_string()))
        }
    }

    /// Number of records in the database. Acquires a read lock.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::core::record::Record;
    /// # use mini_vectordb::VectorDB;
    /// # use mini_vectordb::core::VectorDBError;
    /// # fn main() -> Result<(), VectorDBError> {
    /// # let db = VectorDB::new();
    /// # db.insert(Record::new("a", vec![1.0]))?;
    /// # db.insert(Record::new("b", vec![2.0]))?;
    /// assert_eq!(db.len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn len(&self) -> usize {
        self.index
            .read()
            .expect("RwLock is never poisoned; no panics in read-locked sections")
            .len()
    }

    /// True when the database has zero records. Acquires a read lock.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::VectorDB;
    /// assert!(VectorDB::new().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        // delegates to len() to avoid duplicating the read lock logic
        self.len() == 0
    }

    /// Remove all records and reset the dimension constraint.
    /// Acquires a write lock.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::core::record::Record;
    /// # use mini_vectordb::VectorDB;
    /// # use mini_vectordb::core::VectorDBError;
    /// # fn main() -> Result<(), VectorDBError> {
    /// # let db = VectorDB::new();
    /// # db.insert(Record::new("a", vec![1.0, 2.0, 3.0]))?;
    /// db.clear()?;
    /// assert!(db.is_empty());
    /// // dimension is reset — new vectors can have any shape
    /// db.insert(Record::new("fresh", vec![1.0, 2.0, 3.0]))?;
    /// # Ok(())
    /// # }
    /// ```
    // Replaces the entire index with a fresh FlatIndex — simpler than
    // draining records one-by-one and automatically resets dimension
    // tracking.
    pub fn clear(&self) -> Result<()> {
        let mut index = self
            .index
            .write()
            .expect("RwLock is never poisoned; no panics in write-locked sections");
        // replace the entire index with a fresh one — this is simpler
        // than draining individual records and automatically resets
        // the dimension constraint in FlatIndex
        *index = Box::new(FlatIndex::new());
        self.save_all(index.records())
    }
}

impl Default for VectorDB {
    fn default() -> Self {
        Self::new()
    }
}

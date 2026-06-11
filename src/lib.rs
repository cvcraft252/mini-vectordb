pub mod api;
pub mod core;
pub mod index;
pub mod metadata;
pub mod query;
pub mod rag;
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

/// Serialization format for auto-persistence.
#[derive(Debug, Clone, Copy)]
pub enum StorageFormat {
    Json,
    Binary,
}

/// Thread-safe vector database backed by an in-memory index with optional auto-persistence.
pub struct VectorDB {
    index: RwLock<Box<dyn Index>>,
    persistence: Option<(StorageFormat, PathBuf)>,
}

const UPGRADE_THRESHOLD: usize = 1000;

impl VectorDB {
    /// Creates an empty database with a flat index.
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

    /// Creates a database that auto-saves after every mutation.
    pub fn with_persistence(path: impl Into<PathBuf>, format: StorageFormat) -> Self {
        Self {
            index: RwLock::new(Box::new(FlatIndex::new())),
            persistence: Some((format, path.into())),
        }
    }

    pub fn is_persistent(&self) -> bool {
        self.persistence.is_some()
    }

    fn upgrade_to_hnsw(&self, index: &mut Box<dyn Index>) -> Result<()> {
        let records = index.records();
        let mut hnsw = HnswIndex::new();
        for r in records {
            hnsw.insert(r)?;
        }
        *index = Box::new(hnsw);
        Ok(())
    }

    fn save_all(&self, records: Vec<Record>) -> Result<()> {
        let Some((format, path)) = &self.persistence else {
            return Ok(());
        };
        match format {
            StorageFormat::Json => JsonStorage::from_records(records).save(path),
            StorageFormat::Binary => BinStorage::from_records(records).save(path),
        }
    }

    /// Inserts a record and upgrades to HNSW at the threshold.
    ///
    /// # Errors
    /// `DimensionMismatch` if dimensions differ, `EmptyVector` if vector is empty.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::core::record::Record;
    /// # use mini_vectordb::VectorDB;
    /// # use mini_vectordb::core::VectorDBError;
    /// # fn main() -> Result<(), VectorDBError> {
    /// let db = VectorDB::new();
    /// db.insert(Record::new("doc1", vec![0.1, 0.2, 0.3]))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert(&self, record: Record) -> Result<()> {
        let mut index = self
            .index
            .write()
            .expect("RwLock is never poisoned; no panics in write-locked sections");
        index.insert(record)?;
        if index.len() == UPGRADE_THRESHOLD {
            self.upgrade_to_hnsw(&mut index)?;
        }
        self.save_all(index.records())
    }

    /// Searches for the top_k most similar vectors.
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
    /// assert_eq!(results.len(), 1);
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

    /// Runs parallel batch search across multiple query vectors.
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

    /// Searches with a metadata filter applied before distance computation.
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

        let records = index.records();
        let mut meta_idx = MetadataIndex::new();
        for r in &records {
            meta_idx.index_record(&r.id, &r.metadata);
        }

        crate::query::planner::execute_filtered_search(
            query, top_k, metric, &filter, &meta_idx, &**index,
        )
    }

    /// Looks up a record by ID.
    pub fn get(&self, id: &str) -> Result<Option<Record>> {
        self.index
            .read()
            .expect("RwLock is never poisoned; no panics in read-locked sections")
            .get(id)
    }

    /// Removes a record by ID. Silently succeeds if it doesn't exist.
    pub fn delete(&self, id: &str) -> Result<()> {
        let mut index = self
            .index
            .write()
            .expect("RwLock is never poisoned; no panics in write-locked sections");
        index.delete(id)?;
        self.save_all(index.records())
    }

    /// Replaces a record's vector while keeping its ID and metadata.
    ///
    /// # Errors
    /// `NotFound`, `DimensionMismatch`, `EmptyVector`.
    pub fn update(&self, id: &str, vector: Vec<f32>) -> Result<()> {
        let mut index = self
            .index
            .write()
            .expect("RwLock is never poisoned; no panics in write-locked sections");
        let old = index.get(id)?;
        if let Some(mut record) = old {
            let dim = vector.len();
            if dim == 0 {
                return Err(VectorDBError::EmptyVector);
            }
            let stored_dim = record.vector.len();
            if dim != stored_dim {
                return Err(VectorDBError::DimensionMismatch {
                    expected: stored_dim,
                    actual: dim,
                });
            }
            record.vector = vector;
            index.delete(id)?;
            index.insert(record)?;
            self.save_all(index.records())
        } else {
            Err(VectorDBError::NotFound(id.to_string()))
        }
    }

    /// Returns the number of records.
    pub fn len(&self) -> usize {
        self.index
            .read()
            .expect("RwLock is never poisoned; no panics in read-locked sections")
            .len()
    }

    /// Returns true when the database has zero records.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Removes all records and resets the dimension constraint.
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
    /// db.insert(Record::new("fresh", vec![1.0, 2.0, 3.0]))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn clear(&self) -> Result<()> {
        let mut index = self
            .index
            .write()
            .expect("RwLock is never poisoned; no panics in write-locked sections");
        *index = Box::new(FlatIndex::new());
        self.save_all(index.records())
    }
}

impl Default for VectorDB {
    fn default() -> Self {
        Self::new()
    }
}

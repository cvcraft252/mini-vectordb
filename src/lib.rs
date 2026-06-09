// lib.rs
// mini-vectordb public API root. Each module tree is exposed
// individually so callers can import specific types without
// pulling in everything (e.g. `use mini_vectordb::core::metric`).
// 2026-06-09: core + index done. storage/query/metadata later.

/// Core types: Record, DistanceMetric, errors.
pub mod core;
/// Index trait and implementations.
pub mod index;

use std::sync::RwLock;

use crate::core::metric::DistanceMetric;
use crate::core::record::Record;
use crate::core::{Result, VectorDBError};
use crate::index::flat::FlatIndex;
use crate::index::{Index, SearchResult};

/// Thread-safe vector database backed by an in-memory index.
///
/// # Notes
/// Wraps the underlying index in `RwLock<Box<dyn Index>>` so multiple
/// readers can search concurrently while writes (insert/delete/update)
/// are serialized. The `dyn Index` trait object lets us swap the
/// backend (e.g. FlatIndex → HnswIndex) without changing callers.
///
/// Defaults to `FlatIndex` (brute-force exact search). For larger
/// datasets, a graph-based index would provide better throughput.
///
/// The `RwLock` is never held across await points (this is a blocking
/// API, no async), so deadlocks are impossible under normal use.
pub struct VectorDB {
    /// Protects all index access. Read lock for search/get/len,
    /// write lock for insert/delete/update/clear.
    ///
    /// Using `Box<dyn Index>` instead of a concrete type so the
    /// backing index can be replaced at runtime (e.g. `clear()` or
    /// future adaptive index selection).
    index: RwLock<Box<dyn Index>>,
}

impl VectorDB {
    /// Create a new database with an empty brute-force index.
    ///
    /// # Examples
    /// ```ignore
    /// let db = VectorDB::new();
    /// assert_eq!(db.len(), 0);
    /// assert!(db.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            index: RwLock::new(Box::new(FlatIndex::new())),
        }
    }

    /// Insert a record. Acquires a write lock (blocks concurrent writes).
    ///
    /// # Arguments
    /// * `record` - Must have a unique ID. Vector dimension must match
    ///   existing records, or any dimension if the database is empty.
    ///   ID uniqueness is not enforced — two records with the same ID
    ///   overwrite behavior depends on the index, but callers should
    ///   ensure uniqueness.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// `DimensionMismatch` if vector dimension differs from stored vectors.
    /// `EmptyVector` if the vector has zero elements.
    ///
    /// # Examples
    /// ```ignore
    /// let r = Record::new("doc1", vec![0.1, 0.2, 0.3]);
    /// db.insert(r)?;
    /// ```
    pub fn insert(&self, record: Record) -> Result<()> {
        self.index
            .write()
            .expect("RwLock is never poisoned; no panics in write-locked sections")
            .insert(record)
    }

    /// Search for the top_k most similar vectors. Acquires a read lock
    /// (multiple concurrent searches are allowed).
    ///
    /// # Arguments
    /// * `query` - The query vector. Dimension must match stored vectors.
    /// * `top_k` - Maximum results to return. If 0, returns empty.
    /// * `metric` - Distance function controlling similarity measurement.
    ///
    /// # Returns
    /// Up to `top_k` results sorted by distance ascending (closest first).
    /// Empty `Vec` if the database is empty or `top_k` is 0.
    ///
    /// # Errors
    /// `DimensionMismatch` if query dimension doesn't match stored vectors.
    ///
    /// # Examples
    /// ```ignore
    /// let results = db.search(&[0.1, 0.2, 0.3], 5, DistanceMetric::Cosine)?;
    /// for r in &results {
    ///     println!("{} -> dist={:.4}", r.id, r.distance);
    /// }
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
    ///
    /// # Arguments
    /// * `queries` — each query must match stored dimension.
    /// * `top_k` — results per query.
    /// * `metric` — distance function for all queries.
    ///
    /// # Returns
    /// One result set per query, in input order.
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

    /// Look up a record by ID. Acquires a read lock.
    ///
    /// # Arguments
    /// * `id` - Record identifier.
    ///
    /// # Returns
    /// `Ok(Some(record))` if found, `Ok(None)` if not. The returned
    /// Record is cloned from internal storage.
    ///
    /// # Examples
    /// ```ignore
    /// if let Some(r) = db.get("doc1")? {
    ///     assert_eq!(r.id, "doc1");
    /// }
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
    /// # Arguments
    /// * `id` - Record identifier to remove.
    ///
    /// # Returns
    /// `Ok(())` — deliberately infallible so callers can chain deletes
    /// without checking existence first.
    ///
    /// # Examples
    /// ```ignore
    /// db.delete("doc1")?;
    /// ```
    pub fn delete(&self, id: &str) -> Result<()> {
        self.index
            .write()
            .expect("RwLock is never poisoned; no panics in write-locked sections")
            .delete(id)
    }

    /// Replace a record's vector while keeping its ID and metadata intact.
    /// Acquires a write lock.
    ///
    /// # Arguments
    /// * `id` - Record to update. Must exist.
    /// * `vector` - New vector. Dimension must match the database.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// `NotFound` if no record with this ID exists.
    /// `DimensionMismatch` if the new vector has wrong dimension.
    /// `EmptyVector` if the new vector is empty.
    ///
    /// # Implementation
    /// Internally does delete-then-insert: reads the old record to
    /// preserve metadata, removes it, then inserts the updated version.
    /// This two-step approach is O(N) due to the internal linear scan
    /// in FlatIndex, but avoids adding a dedicated "replace" path to
    /// the Index trait.
    ///
    /// # Examples
    /// ```ignore
    /// db.update("doc1", vec![0.5, 0.6, 0.7])?;
    /// ```
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
            Ok(())
        } else {
            // clone the id into the error — we can't move `id` since
            // it's a shared reference
            Err(VectorDBError::NotFound(id.to_string()))
        }
    }

    /// Number of records in the database. Acquires a read lock.
    ///
    /// # Examples
    /// ```ignore
    /// assert_eq!(db.len(), 42);
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
    /// ```ignore
    /// assert!(VectorDB::new().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        // delegates to len() to avoid duplicating the read lock logic
        self.len() == 0
    }

    /// Remove all records and reset the dimension constraint.
    /// Acquires a write lock.
    ///
    /// # Implementation
    /// Replaces the internal index with a fresh `FlatIndex`. This is
    /// simpler than draining records one-by-one and automatically resets
    /// dimension tracking.
    ///
    /// # Examples
    /// ```ignore
    /// db.clear()?;
    /// assert!(db.is_empty());
    /// // dimension is reset — new vectors can have any shape
    /// db.insert(Record::new("fresh", vec![1.0, 2.0, 3.0]))?;
    /// ```
    pub fn clear(&self) -> Result<()> {
        let mut index = self
            .index
            .write()
            .expect("RwLock is never poisoned; no panics in write-locked sections");
        // replace the entire index with a fresh one — this is simpler
        // than draining individual records and automatically resets
        // the dimension constraint in FlatIndex
        *index = Box::new(FlatIndex::new());
        Ok(())
    }
}

/// `VectorDB::new()` provides a sensible default (flat index, empty).
impl Default for VectorDB {
    fn default() -> Self {
        Self::new()
    }
}

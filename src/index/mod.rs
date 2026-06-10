//! Index trait and implementations: brute-force flat and HNSW graph.
pub mod flat;
pub mod hnsw;

use crate::core::Result;
use crate::core::metric::DistanceMetric;
use crate::core::record::Record;

// We store the full Record instead of just id + distance so callers
// don't need a second lookup. The clone cost is acceptable because
// top_k is small (typically 5–100) and Record is under 1KB.
/// One match from a vector similarity search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Record ID that matched the query.
    pub id: String,
    /// Distance value. Smaller = more similar. Interpretation depends on
    /// the DistanceMetric chosen by the caller.
    pub distance: f32,
    /// The full record, cloned from index storage.
    pub record: Record,
}

// `Send + Sync` bound is required because the REST API layer wraps
// indices in `Arc<RwLock<dyn Index>>` for concurrent HTTP handlers.
// The dynamic dispatch overhead (~1ns per call) is negligible next to
// O(N·D) distance computation.
//
// Dimension is locked after the first insert — mixing vector dimensions
// in one index is almost always a caller bug, and we enforce it to
// catch errors early.
/// Core operations every index backend must implement.
pub trait Index: Send + Sync {
    /// Search for the top_k records most similar to `query`.
    ///
    /// # Arguments
    /// * `query` - The search vector. Dimension must match records already
    ///   in the index.
    /// * `top_k` - Max results. If index has fewer records, returns all.
    /// * `metric` - Distance function controlling similarity measurement.
    ///
    /// # Returns
    /// Up to `top_k` SearchResults sorted by distance ascending (closest
    /// first). Returns empty Vec if the index is empty.
    ///
    /// # Errors
    /// `DimensionMismatch` if `query.len()` != stored vector dimension.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::index::{Index, flat::FlatIndex};
    /// # use mini_vectordb::core::metric::DistanceMetric;
    /// # use mini_vectordb::core::record::Record;
    /// # let mut index = FlatIndex::new();
    /// # index.insert(Record::new("a", vec![0.1, 0.2, 0.3])).unwrap();
    /// let results = index.search(&[0.1, 0.2, 0.3], 5, DistanceMetric::Cosine).unwrap();
    /// assert!(results.len() <= 5);
    /// ```
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>>;

    /// Add a record to the index.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// `DimensionMismatch` if vector dimension doesn't match the index
    /// (applies to all inserts after the first).
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::index::{Index, flat::FlatIndex};
    /// # use mini_vectordb::core::record::Record;
    /// # let mut index = FlatIndex::new();
    /// index.insert(Record::new("doc1", vec![0.1, 0.2, 0.3])).unwrap();
    /// ```
    fn insert(&mut self, record: Record) -> Result<()>;

    /// Remove a record by ID. No error if the ID doesn't exist.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::index::{Index, flat::FlatIndex};
    /// # use mini_vectordb::core::record::Record;
    /// # let mut index = FlatIndex::new();
    /// # index.insert(Record::new("doc1", vec![0.1, 0.2])).unwrap();
    /// index.delete("doc1").unwrap();
    /// ```
    fn delete(&mut self, id: &str) -> Result<()>;

    /// Look up a record by ID without performing a vector search.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::index::{Index, flat::FlatIndex};
    /// # use mini_vectordb::core::record::Record;
    /// # let mut index = FlatIndex::new();
    /// # index.insert(Record::new("doc1", vec![0.1, 0.2])).unwrap();
    /// if let Some(r) = index.get("doc1").unwrap() {
    ///     println!("dim: {}", r.vector.len());
    /// }
    /// ```
    fn get(&self, id: &str) -> Result<Option<Record>>;

    /// Total number of records currently stored.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::index::{Index, flat::FlatIndex};
    /// # use mini_vectordb::core::record::Record;
    /// # let mut index = FlatIndex::new();
    /// # index.insert(Record::new("a", vec![0.1, 0.2])).unwrap();
    /// assert_eq!(index.len(), 1);
    /// ```
    fn len(&self) -> usize;

    /// True when the index has zero records.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::index::{Index, flat::FlatIndex};
    /// assert!(FlatIndex::new().is_empty());
    /// ```
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Run the same search against multiple query vectors in one call.
    ///
    /// # Returns
    /// One `Vec<SearchResult>` per input query, in the same order.
    /// Each inner Vec is sorted by distance ascending.
    fn search_batch(
        &self,
        queries: &[Vec<f32>],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<Vec<SearchResult>>>;

    /// Return a snapshot of all records currently in the index.
    ///
    /// Clones every record — the clone cost is acceptable because
    /// serialization already requires owned data, and this method
    /// is called at most once per mutation in auto-persistence mode.
    fn records(&self) -> Vec<Record>;
}

// index/mod.rs
// Index trait and associated types for vector search backends.
// 2026-06-09: single Index trait covers both search and CRUD.
//             If the query planner ever needs read-only search views,
//             we may split into Index + Searchable traits.

/// Brute-force flat index.
pub mod flat;

use crate::core::Result;
use crate::core::metric::DistanceMetric;
use crate::core::record::Record;

/// One match from a vector similarity search.
///
/// # Notes
/// We store the full Record instead of just id + distance so callers
/// don't need a second lookup. The clone cost is acceptable because
/// top_k is small (typically 5–100) and Record is under 1KB.
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

/// Core operations every index backend must implement.
///
/// # Notes
/// `Send + Sync` bound is required because the REST API layer wraps
/// indices in `Arc<RwLock<dyn Index>>` for concurrent HTTP handlers.
/// The dynamic dispatch overhead (~1ns per call) is negligible next to
/// O(N·D) distance computation.
///
/// Dimension is locked after the first insert — mixing vector dimensions
/// in one index is almost always a caller bug, and we enforce it to
/// catch errors early.
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
    /// ```ignore
    /// let results = index.search(&[0.1, 0.2, 0.3], 5, DistanceMetric::Cosine)?;
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
    /// # Arguments
    /// * `record` - Must have a unique ID. On first insert, this record's
    ///   dimension becomes the required dimension for the index lifetime.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// `DimensionMismatch` if vector dimension doesn't match the index
    /// (applies to all inserts after the first).
    ///
    /// # Examples
    /// ```ignore
    /// let r = Record::new("doc1", vec![0.1, 0.2, 0.3]);
    /// index.insert(r)?;
    /// ```
    fn insert(&mut self, record: Record) -> Result<()>;

    /// Remove a record by ID. No error if the ID doesn't exist.
    ///
    /// # Arguments
    /// * `id` - Record identifier to remove.
    ///
    /// # Returns
    /// `Ok(())` — this is deliberately infallible so callers can
    /// unconditionally chain deletes.
    ///
    /// # Examples
    /// ```ignore
    /// index.delete("doc1")?;
    /// ```
    fn delete(&mut self, id: &str) -> Result<()>;

    /// Look up a record by ID without performing a vector search.
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
    /// if let Some(r) = index.get("doc1")? {
    ///     println!("dim: {}", r.vector.len());
    /// }
    /// ```
    fn get(&self, id: &str) -> Result<Option<Record>>;

    /// Total number of records currently stored.
    ///
    /// # Examples
    /// ```ignore
    /// assert_eq!(index.len(), 42);
    /// ```
    fn len(&self) -> usize;

    /// True when the index has zero records.
    ///
    /// # Examples
    /// ```ignore
    /// assert!(FlatIndex::new().is_empty());
    /// ```
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Run the same search against multiple query vectors in one call.
    ///
    /// # Arguments
    /// * `queries` — each query must match the stored dimension.
    /// * `top_k` — results per query.
    /// * `metric` — distance function applied to all queries.
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
}

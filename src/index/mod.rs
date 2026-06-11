//! Index trait and implementations.
pub mod flat;
pub mod hnsw;

use crate::core::Result;
use crate::core::metric::DistanceMetric;
use crate::core::record::Record;

/// One match from a vector similarity search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub record: Record,
}

/// Core operations every index backend must implement.
pub trait Index: Send + Sync {
    /// Searches for the top_k records most similar to `query`.
    ///
    /// # Errors
    /// `DimensionMismatch` if `query.len()` != stored vector dimension.
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>>;

    /// Adds a record to the index.
    ///
    /// # Errors
    /// `DimensionMismatch` if vector dimension doesn't match.
    fn insert(&mut self, record: Record) -> Result<()>;

    /// Removes a record by ID. No error if the ID doesn't exist.
    fn delete(&mut self, id: &str) -> Result<()>;

    /// Looks up a record by ID.
    fn get(&self, id: &str) -> Result<Option<Record>>;

    /// Returns the number of records.
    fn len(&self) -> usize;

    /// Returns true when the index has zero records.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Runs the same search against multiple query vectors.
    fn search_batch(
        &self,
        queries: &[Vec<f32>],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<Vec<SearchResult>>>;

    /// Returns a snapshot of all records currently in the index.
    fn records(&self) -> Vec<Record>;
}

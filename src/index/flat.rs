// index/flat.rs
// Brute-force exact nearest neighbor index. O(N·D) per search.
// 2026-06-09: stores records in a contiguous Vec for cache-friendly
//             linear scans. Precomputes L2 norms so cosine distance
//             only does one sqrt per query (instead of N+1).
//             At N < 1000, linear scan often beats tree-based approaches
//             due to branch prediction and SIMD auto-vectorization.

use crate::core::metric::DistanceMetric;
use crate::core::record::Record;
use crate::core::{Result, VectorDBError};
use crate::index::{Index, SearchResult};

/// Flat (brute-force) index. Every search scans all records.
///
/// # Notes
/// Dimension is locked after the first insert to prevent accidental
/// mixing of differently-shaped embeddings in one index.
///
/// For cosine search we precompute and store each record's L2 norm
/// in `norms`, avoiding `2*N` sqrt calls per search. This is a
/// classic trick from the SIFT evaluation literature.
pub struct FlatIndex {
    /// All records in insertion order. O(1) push, O(N) delete (swap-remove).
    records: Vec<Record>,

    /// Precomputed L2 norms |v| for each record, index-aligned with `records`.
    /// Populated on insert, swap-removed on delete.
    /// Only meaningful when searching with `DistanceMetric::Cosine`;
    /// unused values are harmless (just a few wasted bytes).
    norms: Vec<f32>,

    /// Vector dimension. `0` means "not yet set" (empty index).
    /// After first insert, validated on every subsequent insert.
    dimension: usize,
}

impl FlatIndex {
    /// Create an empty index with no dimension constraint.
    ///
    /// # Examples
    /// ```ignore
    /// let idx = FlatIndex::new();
    /// assert!(idx.is_empty());
    /// assert_eq!(idx.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            norms: Vec::new(),
            dimension: 0,
        }
    }

    /// Cosine distance using caller-supplied norms (no sqrt per record).
    ///
    /// # Arguments
    /// * `query_norm` - Precomputed |query|, done once per search call.
    /// * `query` - The query vector.
    /// * `record_vec` - A stored record's vector.
    /// * `record_norm` - Precomputed |record| from `self.norms[i]`.
    ///
    /// # Returns
    /// Cosine distance in [0, 2]. Returns 0.0 if either norm is zero
    /// (degenerate zero-vector — no meaningful direction, so we treat
    /// it as "distance zero from everything" to avoid NaN propagation).
    ///
    /// # Notes
    /// Marked `#[inline]` because the function body is ~4 FMAs, and
    /// the call overhead dominates for small-dimension vectors.
    #[inline]
    fn cosine_with_norms(
        query_norm: f32,
        query: &[f32],
        record_vec: &[f32],
        record_norm: f32,
    ) -> f32 {
        // avoid division by zero if either vector is degenerate
        if query_norm == 0.0 || record_norm == 0.0 {
            return 0.0;
        }
        let dot: f32 = query
            .iter()
            .zip(record_vec.iter())
            .map(|(a, b)| a * b)
            .sum();
        1.0 - dot / (query_norm * record_norm)
    }

    /// Compute L2 norm: `sqrt(sum(v[i]²))`.
    ///
    /// # Arguments
    /// * `v` - Vector to measure.
    ///
    /// # Notes
    /// Using `fold` with a single accumulator instead of
    /// `map + sum` avoids an intermediate iterator state allocation.
    /// The difference is ~3ns for 128-dim, but it adds up in tight loops.
    #[inline]
    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().fold(0.0f32, |acc, &x| acc + x * x).sqrt()
    }

    /// Extract top-k (index, distance) pairs from a full distance array.
    ///
    /// # Arguments
    /// * `distances` - Array of (record_index, distance) pairs, one per record.
    /// * `top_k` - Number of best results to keep.
    ///
    /// # Returns
    /// Up to `top_k` pairs sorted by distance ascending. If top_k >= len,
    /// returns all pairs sorted.
    ///
    /// # Notes
    /// Uses `select_nth_unstable_by` for O(N) partial sort, which is
    /// faster than a binary heap (O(N log k)) for single queries.
    /// If we add batch query support later, a heap per query
    /// would be more ergonomic but ~2x slower at k=10.
    fn select_top_k(mut distances: Vec<(usize, f32)>, top_k: usize) -> Vec<(usize, f32)> {
        // handle edge cases: nothing to select, or asking for everything
        let k = top_k.min(distances.len());
        if k == 0 {
            return Vec::new();
        }
        // partial sort: partitions so first k elements are the k smallest.
        // using select_nth_unstable_by over sort_unstable_by saves O(N log N)
        // when k << N (typical: k=5, N=10000).
        //
        // safe: distances never contain NaN — zero-vector edge cases
        // return 0.0 instead, and other metrics produce finite f32.
        distances.select_nth_unstable_by(k - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        distances.truncate(k);
        // sort truncated prefix; same NaN safety invariant as above
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances
    }
}

/// `FlatIndex::new()` provides the canonical empty state.
impl Default for FlatIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Index for FlatIndex {
    /// Brute-force scan over all records, computing distance to query
    /// for each one, then selecting top_k.
    ///
    /// # Implementation
    /// 1. Validate query dimension against stored dimension.
    /// 2. If metric is Cosine, precompute query norm once, then use
    ///    `cosine_with_norms` with the pre-stored `norms`.
    /// 3. For other metrics, call `DistanceMetric::compute` per record.
    /// 4. Build a Vec of (index, distance), pass to `select_top_k`.
    /// 5. Map indices back to SearchResult, cloning each winning Record.
    ///
    /// # Complexity
    /// O(N·D + N) where N = len(), D = vector dimension.
    /// The N log(k) from sort is negligible since k is small (≤ 100).
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>> {
        // early exit: nothing to search
        if self.records.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        // validate query matches stored dimension
        if query.len() != self.dimension {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let n = self.records.len();
        let k = top_k.min(n);

        // compute distance for every record. cosine path precomputes
        // query norm once and skips sqrt per record via stored norms.
        let distances: Vec<(usize, f32)> = match metric {
            DistanceMetric::Cosine => {
                let query_norm = Self::l2_norm(query);
                self.records
                    .iter()
                    .enumerate()
                    .map(|(i, record)| {
                        let d = Self::cosine_with_norms(
                            query_norm,
                            query,
                            &record.vector,
                            // safe: norms is always index-aligned with records
                            self.norms[i],
                        );
                        (i, d)
                    })
                    .collect()
            }
            _ => self
                .records
                .iter()
                .enumerate()
                .map(|(i, record)| {
                    let d = metric.compute(query, &record.vector);
                    (i, d)
                })
                .collect(),
        };

        let top = Self::select_top_k(distances, k);

        // map indices to SearchResults. clone is fine here because
        // top_k is small (typically 5–100) and Record is under 1KB.
        let results: Vec<SearchResult> = top
            .into_iter()
            .map(|(idx, dist)| {
                // safe: idx came from select_top_k which only returns
                // valid indices in 0..self.records.len()
                let record = &self.records[idx];
                SearchResult {
                    id: record.id.clone(),
                    distance: dist,
                    record: record.clone(),
                }
            })
            .collect();

        Ok(results)
    }

    /// Insert a record. On first insert, sets `self.dimension`.
    /// On subsequent inserts, validates dim match.
    /// Pushes the record and its precomputed L2 norm.
    ///
    /// # Notes
    /// We always compute and store the norm regardless of which metric
    /// will be used later. Storage cost is 4 bytes per vector — worth
    /// it to avoid re-scanning all vectors when switching metrics.
    fn insert(&mut self, record: Record) -> Result<()> {
        let dim = record.vector.len();
        if dim == 0 {
            return Err(VectorDBError::EmptyVector);
        }
        // first insert locks the dimension; subsequent inserts must match
        if self.dimension == 0 {
            self.dimension = dim;
        } else if dim != self.dimension {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dimension,
                actual: dim,
            });
        }
        // always compute norm regardless of which metric will be used later
        self.norms.push(Self::l2_norm(&record.vector));
        self.records.push(record);
        Ok(())
    }

    /// Remove a record by ID. Uses linear scan to find the index,
    /// then swap-removes from both `records` and `norms` to keep
    /// them aligned.
    ///
    /// # Notes
    /// O(N) linear scan is acceptable for now. A HashMap<String, usize>
    /// could provide O(1) id-to-index lookup if delete-heavy workloads
    /// become common.
    ///
    /// Dimension resets to 0 when the last record is removed so
    /// a different shape can be inserted later.
    fn delete(&mut self, id: &str) -> Result<()> {
        // linear scan — O(N) but acceptable while dataset is small
        if let Some(pos) = self.records.iter().position(|r| r.id == id) {
            // swap-remove keeps Vec contiguous and avoids shifting elements
            self.records.swap_remove(pos);
            self.norms.swap_remove(pos);
            // empty index: reset dimension lock so any shape can be inserted next
            if self.records.is_empty() {
                self.dimension = 0;
            }
        }
        // deliberately infallible: no error if id doesn't exist
        Ok(())
    }

    /// Linear scan through records by ID. O(N) — same tradeoff as `delete`.
    fn get(&self, id: &str) -> Result<Option<Record>> {
        // clone is necessary because we can't return a reference into self
        Ok(self.records.iter().find(|r| r.id == id).cloned())
    }

    /// Number of records. O(1) — just `self.records.len()`.
    fn len(&self) -> usize {
        self.records.len()
    }
}

// ── tests ──
//
// Note: all .unwrap() calls in test setup (insert/delete/get/search)
// are on operations that cannot fail with the given test data:
// dimensions always match, IDs are unique, and delete is infallible.
#[cfg(test)]
mod tests {
    use super::*;

    // ── l2_norm ──

    #[test]
    fn l2_norm_unit_vector() {
        // |[1,0,0]| = 1
        let norm = FlatIndex::l2_norm(&[1.0, 0.0, 0.0]);
        assert!((norm - 1.0).abs() < 1e-6, "expected 1.0, got {norm}");
    }

    #[test]
    fn l2_norm_345_triangle() {
        // |[3,4]| = 5
        let norm = FlatIndex::l2_norm(&[3.0, 4.0]);
        assert!((norm - 5.0).abs() < 1e-6, "expected 5.0, got {norm}");
    }

    #[test]
    fn l2_norm_zero_vector() {
        let norm = FlatIndex::l2_norm(&[0.0, 0.0, 0.0]);
        assert_eq!(norm, 0.0);
    }

    // ── cosine_with_norms ──

    #[test]
    fn cosine_precomputed_matches_generic() {
        let query = vec![1.0, 2.0, 3.0];
        let record = vec![4.0, 5.0, 6.0];
        let q_norm = FlatIndex::l2_norm(&query);
        let r_norm = FlatIndex::l2_norm(&record);
        let fast = FlatIndex::cosine_with_norms(q_norm, &query, &record, r_norm);
        let generic = DistanceMetric::Cosine.compute(&query, &record);
        assert!(
            (fast - generic).abs() < 1e-5,
            "fast={fast}, generic={generic}"
        );
    }

    #[test]
    fn cosine_precomputed_identical_vectors() {
        let v = vec![2.0, 3.0, 4.0];
        let norm = FlatIndex::l2_norm(&v);
        let d = FlatIndex::cosine_with_norms(norm, &v, &v, norm);
        assert!(
            d < 1e-6,
            "identical vectors should have distance ~0, got {d}"
        );
    }

    #[test]
    fn cosine_precomputed_zero_norm_returns_zero() {
        let d = FlatIndex::cosine_with_norms(0.0, &[1.0, 2.0], &[1.0, 2.0], 3.0);
        assert_eq!(d, 0.0);
        let d = FlatIndex::cosine_with_norms(3.0, &[1.0, 2.0], &[0.0, 0.0], 0.0);
        assert_eq!(d, 0.0);
    }

    // ── select_top_k ──

    #[test]
    fn select_top_k_empty_input() {
        let result = FlatIndex::select_top_k(vec![], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn select_top_k_zero_k() {
        let distances = vec![(0, 1.0), (1, 0.5), (2, 2.0)];
        let result = FlatIndex::select_top_k(distances, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn select_top_k_returns_correct_order() {
        // (index, distance). should return smallest distances first.
        let distances = vec![(0, 3.0), (1, 1.0), (2, 4.0), (3, 2.0)];
        let result = FlatIndex::select_top_k(distances, 3);
        assert_eq!(result.len(), 3);
        // sorted ascending by distance
        assert_eq!(result[0].0, 1); // dist 1.0
        assert_eq!(result[1].0, 3); // dist 2.0
        assert_eq!(result[2].0, 0); // dist 3.0
    }

    #[test]
    fn select_top_k_k_larger_than_n() {
        let distances = vec![(0, 2.0), (1, 1.0)];
        let result = FlatIndex::select_top_k(distances, 10);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 1); // smallest distance first
        assert_eq!(result[1].0, 0);
    }
}

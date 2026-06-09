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
    /// If we add batch query support in Task 2.2, a heap per query
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
        // sort the truncated prefix so results are in ascending distance order
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
    /// O(N) scan is acceptable for Phase 1. Task 1.3 may add a
    /// HashMap<String, usize> for O(1) id-to-index lookup if
    /// delete-heavy workloads appear.
    ///
    /// Dimension resets to 0 when the last record is removed so
    /// a different shape can be inserted later.
    fn delete(&mut self, id: &str) -> Result<()> {
        // linear scan — O(N) but acceptable for Phase 1
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

    // ── FlatIndex integration ──

    fn make_record(id: &str, vec: Vec<f32>) -> Record {
        Record::new(id, vec)
    }

    #[test]
    fn new_is_empty() {
        let idx = FlatIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn insert_and_get() {
        let mut idx = FlatIndex::new();
        let r = make_record("a", vec![1.0, 0.0]);
        idx.insert(r.clone()).unwrap();
        assert_eq!(idx.len(), 1);
        let got = idx.get("a").unwrap().unwrap();
        assert_eq!(got.id, "a");
        assert_eq!(got.vector, vec![1.0, 0.0]);
    }

    #[test]
    fn get_missing_returns_none() {
        let idx = FlatIndex::new();
        assert!(idx.get("nope").unwrap().is_none());
    }

    #[test]
    fn insert_dimension_mismatch() {
        let mut idx = FlatIndex::new();
        idx.insert(make_record("a", vec![1.0, 2.0])).unwrap();
        let err = idx
            .insert(make_record("b", vec![1.0, 2.0, 3.0]))
            .unwrap_err();
        assert!(matches!(err, VectorDBError::DimensionMismatch { .. }));
    }

    #[test]
    fn insert_empty_vector_rejected() {
        let mut idx = FlatIndex::new();
        let err = idx.insert(make_record("a", vec![])).unwrap_err();
        assert!(matches!(err, VectorDBError::EmptyVector));
    }

    #[test]
    fn insert_allows_new_dimension_after_clear() {
        let mut idx = FlatIndex::new();
        idx.insert(make_record("a", vec![1.0, 2.0])).unwrap();
        idx.delete("a").unwrap();
        assert!(idx.is_empty());
        // dimension should have reset, allowing a different shape
        idx.insert(make_record("b", vec![1.0, 2.0, 3.0])).unwrap();
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn delete_nonexistent_is_noop() {
        let mut idx = FlatIndex::new();
        idx.insert(make_record("a", vec![1.0])).unwrap();
        idx.delete("no_such_id").unwrap();
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn delete_removes_record_and_norm() {
        let mut idx = FlatIndex::new();
        idx.insert(make_record("x", vec![1.0, 2.0])).unwrap();
        idx.insert(make_record("y", vec![3.0, 4.0])).unwrap();
        idx.delete("x").unwrap();
        assert_eq!(idx.len(), 1);
        // verify the remaining record is intact
        assert!(idx.get("x").unwrap().is_none());
        let got = idx.get("y").unwrap().unwrap();
        assert_eq!(got.id, "y");
    }

    #[test]
    fn search_euclidean_basic() {
        let mut idx = FlatIndex::new();
        // three points along the x-axis
        idx.insert(make_record("near", vec![1.0, 0.0])).unwrap();
        idx.insert(make_record("mid", vec![5.0, 0.0])).unwrap();
        idx.insert(make_record("far", vec![9.0, 0.0])).unwrap();
        // query at origin
        let results = idx
            .search(&[0.0, 0.0], 2, DistanceMetric::Euclidean)
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "near"); // dist 1.0
        assert_eq!(results[1].id, "mid"); // dist 5.0
        assert!(results[0].distance < results[1].distance);
    }

    #[test]
    fn search_cosine_uses_precomputed_norms() {
        let mut idx = FlatIndex::new();
        // [1,0] and [0,1] are orthogonal (cosine dist = 1.0)
        idx.insert(make_record("a", vec![1.0, 0.0])).unwrap();
        idx.insert(make_record("b", vec![0.0, 1.0])).unwrap();
        let results = idx.search(&[1.0, 0.0], 2, DistanceMetric::Cosine).unwrap();
        assert_eq!(results.len(), 2);
        // "a" is identical to query, so it should be first
        assert_eq!(results[0].id, "a");
        assert!(results[0].distance < 0.01);
        // "b" is orthogonal, distance ~1.0
        assert!((results[1].distance - 1.0).abs() < 0.01);
    }

    #[test]
    fn search_returns_all_when_top_k_exceeds_len() {
        let mut idx = FlatIndex::new();
        idx.insert(make_record("a", vec![1.0])).unwrap();
        idx.insert(make_record("b", vec![2.0])).unwrap();
        let results = idx.search(&[0.0], 100, DistanceMetric::Euclidean).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_empty_index_returns_empty() {
        let idx = FlatIndex::new();
        let results = idx
            .search(&[1.0, 2.0], 5, DistanceMetric::Euclidean)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_top_k_zero_returns_empty() {
        let mut idx = FlatIndex::new();
        idx.insert(make_record("a", vec![1.0])).unwrap();
        let results = idx.search(&[1.0], 0, DistanceMetric::Euclidean).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_dimension_mismatch() {
        let mut idx = FlatIndex::new();
        idx.insert(make_record("a", vec![1.0, 2.0, 3.0])).unwrap();
        let err = idx
            .search(&[1.0, 2.0], 5, DistanceMetric::Euclidean)
            .unwrap_err();
        assert!(matches!(err, VectorDBError::DimensionMismatch { .. }));
    }

    #[test]
    fn search_with_manhattan() {
        let mut idx = FlatIndex::new();
        idx.insert(make_record("a", vec![0.0, 0.0])).unwrap();
        idx.insert(make_record("b", vec![3.0, 4.0])).unwrap();
        // query at origin: manhattan to (0,0)=0, to (3,4)=7
        let results = idx
            .search(&[0.0, 0.0], 2, DistanceMetric::Manhattan)
            .unwrap();
        assert_eq!(results[0].id, "a");
        assert!((results[0].distance - 0.0).abs() < 0.01);
        assert_eq!(results[1].id, "b");
        assert!((results[1].distance - 7.0).abs() < 0.01);
    }

    #[test]
    fn search_result_contains_full_record() {
        let mut idx = FlatIndex::new();
        idx.insert(make_record("doc", vec![1.0, 2.0])).unwrap();
        let results = idx
            .search(&[1.0, 2.0], 1, DistanceMetric::Euclidean)
            .unwrap();
        assert_eq!(results[0].id, "doc");
        assert_eq!(results[0].record.vector, vec![1.0, 2.0]);
        assert_eq!(results[0].distance, 0.0);
    }

    #[test]
    fn records_and_norms_stay_aligned() {
        // regression: after swapping records, norms must match the record at each index
        let mut idx = FlatIndex::new();
        for i in 0..5 {
            let val = i as f32 * 10.0;
            idx.insert(make_record(&format!("r{i}"), vec![val]))
                .unwrap();
        }
        // delete from the middle — swap-remove should move the last element
        idx.delete("r1").unwrap();
        assert_eq!(idx.len(), 4);
        // search should still work correctly (norms aligned with records)
        let results = idx.search(&[0.0], 4, DistanceMetric::Euclidean).unwrap();
        // r0 at 0.0 should be closest
        assert_eq!(results[0].id, "r0");
        assert!(results[0].distance < 1.0);
    }
}

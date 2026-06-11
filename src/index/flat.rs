use crate::core::metric::DistanceMetric;
use crate::core::record::Record;
use crate::core::{Result, VectorDBError};
use crate::index::{Index, SearchResult};

/// Flat (brute-force) index. Every search scans all records.
pub struct FlatIndex {
    records: Vec<Record>,
    norms: Vec<f32>,
    dimension: usize,
}

impl FlatIndex {
    /// Creates an empty index.
    ///
    /// # Examples
    /// ```
    /// # use mini_vectordb::index::{Index, flat::FlatIndex};
    /// let idx = FlatIndex::new();
    /// assert!(idx.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            norms: Vec::new(),
            dimension: 0,
        }
    }

    #[inline]
    fn cosine_with_norms(
        query_norm: f32,
        query: &[f32],
        record_vec: &[f32],
        record_norm: f32,
    ) -> f32 {
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

    #[inline]
    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().fold(0.0f32, |acc, &x| acc + x * x).sqrt()
    }

    fn select_top_k(mut distances: Vec<(usize, f32)>, top_k: usize) -> Vec<(usize, f32)> {
        let k = top_k.min(distances.len());
        if k == 0 {
            return Vec::new();
        }
        // safe: partial_cmp on finite f32 values never fails
        distances.select_nth_unstable_by(k - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        distances.truncate(k);
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances
    }
}

impl Default for FlatIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl Index for FlatIndex {
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>> {
        if self.records.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        if query.len() != self.dimension {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let n = self.records.len();
        let k = top_k.min(n);

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

        let results: Vec<SearchResult> = top
            .into_iter()
            .map(|(idx, dist)| {
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

    fn search_batch(
        &self,
        queries: &[Vec<f32>],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<Vec<SearchResult>>> {
        for q in queries.iter() {
            if q.len() != self.dimension {
                return Err(VectorDBError::DimensionMismatch {
                    expected: self.dimension,
                    actual: q.len(),
                });
            }
        }
        use rayon::prelude::*;
        queries
            .par_iter()
            .map(|q| self.search(q, top_k, metric))
            .collect::<Result<Vec<_>>>()
    }

    fn insert(&mut self, record: Record) -> Result<()> {
        let dim = record.vector.len();
        if dim == 0 {
            return Err(VectorDBError::EmptyVector);
        }
        if self.dimension == 0 {
            self.dimension = dim;
        } else if dim != self.dimension {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dimension,
                actual: dim,
            });
        }
        self.norms.push(Self::l2_norm(&record.vector));
        self.records.push(record);
        Ok(())
    }

    fn delete(&mut self, id: &str) -> Result<()> {
        if let Some(pos) = self.records.iter().position(|r| r.id == id) {
            self.records.swap_remove(pos);
            self.norms.swap_remove(pos);
            if self.records.is_empty() {
                self.dimension = 0;
            }
        }
        Ok(())
    }

    fn get(&self, id: &str) -> Result<Option<Record>> {
        Ok(self.records.iter().find(|r| r.id == id).cloned())
    }

    fn len(&self) -> usize {
        self.records.len()
    }

    fn records(&self) -> Vec<Record> {
        self.records.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_norm_unit_vector() {
        let norm = FlatIndex::l2_norm(&[1.0, 0.0, 0.0]);
        assert!((norm - 1.0).abs() < 1e-6, "expected 1.0, got {norm}");
    }

    #[test]
    fn l2_norm_345_triangle() {
        let norm = FlatIndex::l2_norm(&[3.0, 4.0]);
        assert!((norm - 5.0).abs() < 1e-6, "expected 5.0, got {norm}");
    }

    #[test]
    fn l2_norm_zero_vector() {
        let norm = FlatIndex::l2_norm(&[0.0, 0.0, 0.0]);
        assert_eq!(norm, 0.0);
    }

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
        let distances = vec![(0, 3.0), (1, 1.0), (2, 4.0), (3, 2.0)];
        let result = FlatIndex::select_top_k(distances, 3);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].0, 1);
        assert_eq!(result[1].0, 3);
        assert_eq!(result[2].0, 0);
    }

    #[test]
    fn select_top_k_k_larger_than_n() {
        let distances = vec![(0, 2.0), (1, 1.0)];
        let result = FlatIndex::select_top_k(distances, 10);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 1);
        assert_eq!(result[1].0, 0);
    }
}

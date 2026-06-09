// core/metric.rs
// Distance metrics for vector similarity search.
// 2026-06-09: all four continuous metrics active. Hamming is the
//             first discrete metric — counts element-wise mismatches
//             and normalizes by vector length to [0, 1].
//             Cosine precomputes norms via single-pass fold to avoid
//             iterating the vector twice. Mixed results in benchmarks
//             but it's cleaner for small N.

use serde::{Deserialize, Serialize};

/// Common interface for every distance function in the library.
///
/// # Notes
/// The trait accepts `&self` so individual metric types can carry
/// parameters (e.g. a Hamming variant with configurable threshold).
/// The existing `DistanceMetric` enum implements this trait, keeping
/// the dispatch-based API while enabling generic code over `dyn Distance`.
pub trait Distance {
    /// Compute the distance between two equal-length vectors.
    ///
    /// # Arguments
    /// * `a`, `b` — vectors of f32. Caller must ensure equal length.
    ///
    /// # Returns
    /// f32 in a metric-specific range. Smaller = more similar.
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Which distance function to use. All return f32 where smaller = closer.
/// DotProduct is negated so `a·b = max` becomes `dist = min`,
/// keeping the "smaller is better" convention consistent across metrics.
///
/// Hamming is for binary-encoded vectors (each element expected to be
/// 0.0 or 1.0). Mismatch count divided by length gives distance in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
    /// Fraction of positions that differ. For binary feature vectors.
    Hamming,
}

impl DistanceMetric {
    /// Dispatch to the right distance function. Callers are responsible
    /// for checking dimension equality before calling this.
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::DotProduct => dot_product_distance(a, b),
            DistanceMetric::Manhattan => manhattan_distance(a, b),
            DistanceMetric::Hamming => hamming_distance(a, b),
        }
    }
}

impl Distance for DistanceMetric {
    /// Delegates to the inherent `compute()`. This trait impl exists
    /// so callers can write generic code over `dyn Distance` without
    /// depending on the concrete enum type.
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        self.compute(a, b)
    }
}

// --- Private impls ---

/// Cosine distance: 1 - (a·b)/(|a||b|).
/// Returns 0.0 if either vector is all-zeros — there's no meaningful
/// direction, so we treat it as "same as everything". Debatable, but
/// Faiss does the same and it avoids NaN in search results.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let (dot, norm_a, norm_b) = a
        .iter()
        .zip(b.iter())
        .fold((0.0f32, 0.0f32, 0.0f32), |(d, na, nb), (&x, &y)| {
            (d + x * y, na + x * x, nb + y * y)
        });
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    1.0 - dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Plain L2. We always sqrt because we need actual distances for
/// the flat index (callers compare distances across different
/// query vectors). If we ever do pure top-k with a single query
/// we can skip sqrt and compare squared distances.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Negated dot product so "higher similarity = lower distance".
/// Useful when vectors are already normalized (e.g. embedding outputs).
fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

/// Manhattan (L1) distance: sum of absolute element-wise differences.
/// Equivalent to the grid distance between two points in Rⁿ.
fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
}

/// Hamming distance for binary vectors. Counts element mismatches
/// and divides by vector length, yielding a fraction in [0, 1].
///
/// # Arguments
/// * `a`, `b` — vectors of 0.0 and 1.0 values. Non-binary values still
///   work (mismatch is `a[i] != b[i]`), but the metric is designed for
///   binary-encoded feature vectors.
///
/// # Returns
/// `mismatches / len` as f32. 0.0 when identical, 1.0 when every
/// position differs.
///
/// # Notes
/// Uses `f32::total_cmp` to check exact bitwise equality between
/// elements. For binary vectors this is equivalent to `==` and
/// avoids any floating-point tolerance debate, but it means 0.0
/// and -0.0 are treated as different — callers using binary vectors
/// should stick to 0.0 and 1.0 only.
fn hamming_distance(a: &[f32], b: &[f32]) -> f32 {
    // count positions where a[i] != b[i]; normalize by length
    let mismatches = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.total_cmp(y) != std::cmp::Ordering::Equal)
        .count();
    // safe: a.len() > 0 because dimension validation rejects empty vectors
    mismatches as f32 / a.len() as f32
}

use serde::{Deserialize, Serialize};

/// Common interface for every distance function in the library.
// The trait accepts `&self` so individual metric types can carry
// parameters; this keeps the dispatch-based API while enabling generic code.
pub trait Distance {
    /// Returns a distance value where smaller means more similar.
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
    // Delegates to the inherent `compute()` so callers can write
    // generic code over `dyn Distance`.
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        self.compute(a, b)
    }
}

// --- Private impls ---

// Cosine distance: 1 - (a·b)/(|a||b|). Returns 0.0 for zero-vectors.
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

// Plain L2 distance (sqrt of sum of squared differences).
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// Negated dot product so higher similarity = lower distance.
fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

// Manhattan (L1) distance: sum of absolute element-wise differences.
fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
}

// Hamming distance for binary vectors (0.0 / 1.0 elements).
// Returns mismatches / len as f32. 0.0 = identical, 1.0 = all differ.
// Uses f32::total_cmp for bit-exact equality; callers must use 0.0 and 1.0 only.
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

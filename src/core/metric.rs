// core/metric.rs
// Distance metrics for vector similarity search.
// 2026-06-09: only float32 vectors supported. Binary metrics (Hamming)
//             are not yet implemented.
//             Cosine precomputes norms via single-pass fold to avoid
//             iterating the vector twice. Mixed results in benchmarks
//             but it's cleaner for small N.

use serde::{Deserialize, Serialize};

/// Which distance function to use. All return f32 where smaller = closer.
/// DotProduct is negated so `a·b = max` becomes `dist = min`,
/// keeping the "smaller is better" convention consistent across metrics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
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
        }
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

fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
}

use serde::{Deserialize, Serialize};

/// Computes a distance between two vectors where smaller means more similar.
pub trait Distance {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Which distance function to use. DotProduct is negated so smaller = closer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
    Hamming,
}

impl DistanceMetric {
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
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        self.compute(a, b)
    }
}

#[inline]
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

#[inline]
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[inline]
fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

#[inline]
fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
}

fn hamming_distance(a: &[f32], b: &[f32]) -> f32 {
    let mismatches = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.total_cmp(y) != std::cmp::Ordering::Equal)
        .count();
    mismatches as f32 / a.len() as f32
}

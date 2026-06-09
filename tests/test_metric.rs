// Distance metric unit tests.

use mini_vectordb::core::metric::DistanceMetric;

macro_rules! assert_f32_eq {
    ($a:expr, $b:expr, $msg:expr) => {
        assert!(($a - $b).abs() < 1e-5, "{}: {} != {}", $msg, $a, $b);
    };
}

// ── Cosine ──

#[test]
fn cosine_identical_vectors() {
    let v = vec![1.0, 2.0, 3.0];
    assert_f32_eq!(
        DistanceMetric::Cosine.compute(&v, &v),
        0.0,
        "cosine of identical vectors should be 0"
    );
}

#[test]
fn cosine_orthogonal_vectors() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    assert_f32_eq!(
        DistanceMetric::Cosine.compute(&a, &b),
        1.0,
        "cosine of orthogonal vectors should be 1"
    );
}

#[test]
fn cosine_opposite_vectors() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    assert_f32_eq!(
        DistanceMetric::Cosine.compute(&a, &b),
        2.0,
        "cosine of opposite vectors should be 2"
    );
}

#[test]
fn cosine_zero_vector() {
    let a = vec![0.0, 0.0];
    let b = vec![1.0, 2.0];
    assert_f32_eq!(
        DistanceMetric::Cosine.compute(&a, &b),
        0.0,
        "cosine with zero vector should default to 0"
    );
}

#[test]
fn cosine_known_value() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    assert_f32_eq!(
        DistanceMetric::Cosine.compute(&a, &b),
        0.025368,
        "cosine distance for [1,2,3] vs [4,5,6]"
    );
}

// ── Euclidean ──

#[test]
fn euclidean_identical_vectors() {
    let v = vec![3.0, 4.0];
    assert_f32_eq!(
        DistanceMetric::Euclidean.compute(&v, &v),
        0.0,
        "euclidean of identical vectors should be 0"
    );
}

#[test]
fn euclidean_known_value() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];
    assert_f32_eq!(
        DistanceMetric::Euclidean.compute(&a, &b),
        5.0,
        "euclidean distance of (0,0)-(3,4) should be 5"
    );
}

#[test]
fn euclidean_negative_coordinates() {
    let a = vec![-1.0, -1.0];
    let b = vec![2.0, 3.0];
    assert_f32_eq!(
        DistanceMetric::Euclidean.compute(&a, &b),
        5.0,
        "euclidean distance of (-1,-1)-(2,3) should be 5"
    );
}

// ── DotProduct ──

#[test]
fn dot_product_identical_vectors() {
    let v = vec![2.0, 3.0];
    assert_f32_eq!(
        DistanceMetric::DotProduct.compute(&v, &v),
        -13.0,
        "dot product distance of [2,3] with itself should be -13"
    );
}

#[test]
fn dot_product_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    assert_f32_eq!(
        DistanceMetric::DotProduct.compute(&a, &b),
        0.0,
        "dot product distance of orthogonal vectors should be 0"
    );
}

#[test]
fn dot_product_positive_values() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    assert_f32_eq!(
        DistanceMetric::DotProduct.compute(&a, &b),
        -32.0,
        "dot product distance of [1,2,3]·[4,5,6] = -32"
    );
}

// ── Manhattan ──

#[test]
fn manhattan_identical_vectors() {
    let v = vec![1.0, 2.0, 3.0];
    assert_f32_eq!(
        DistanceMetric::Manhattan.compute(&v, &v),
        0.0,
        "manhattan of identical vectors should be 0"
    );
}

#[test]
fn manhattan_known_value() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];
    assert_f32_eq!(
        DistanceMetric::Manhattan.compute(&a, &b),
        7.0,
        "manhattan distance of (0,0)-(3,4) should be 7"
    );
}

#[test]
fn manhattan_negative_coordinates() {
    let a = vec![-1.0, -2.0];
    let b = vec![3.0, 4.0];
    assert_f32_eq!(
        DistanceMetric::Manhattan.compute(&a, &b),
        10.0,
        "manhattan distance of (-1,-2)-(3,4) should be 10"
    );
}

// ── Hamming ──

#[test]
fn hamming_identical_vectors_returns_zero() {
    let a = vec![0.0, 1.0, 0.0, 1.0];
    let b = vec![0.0, 1.0, 0.0, 1.0];
    assert_f32_eq!(
        DistanceMetric::Hamming.compute(&a, &b),
        0.0,
        "hamming of identical binary vectors should be 0"
    );
}

#[test]
fn hamming_known_mismatch_count() {
    let a = vec![0.0, 1.0, 0.0, 1.0];
    let b = vec![0.0, 0.0, 0.0, 1.0];
    assert_f32_eq!(
        DistanceMetric::Hamming.compute(&a, &b),
        0.25,
        "1 mismatch out of 4 positions = 0.25"
    );
}

#[test]
fn hamming_all_positions_differ_returns_one() {
    let a = vec![0.0, 0.0];
    let b = vec![1.0, 1.0];
    assert_f32_eq!(
        DistanceMetric::Hamming.compute(&a, &b),
        1.0,
        "all positions differ → distance 1.0"
    );
}

// ── DistanceMetric::compute dispatch ──

#[test]
fn compute_dispatches_correctly() {
    let a = vec![3.0, 4.0];
    let b = vec![0.0, 0.0];
    assert_f32_eq!(
        DistanceMetric::Euclidean.compute(&a, &b),
        5.0,
        "euclidean via compute"
    );
    assert_f32_eq!(
        DistanceMetric::Manhattan.compute(&a, &b),
        7.0,
        "manhattan via compute"
    );
}

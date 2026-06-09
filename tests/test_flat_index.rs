// tests/test_flat_index.rs
// Integration tests for FlatIndex public API.

use mini_vectordb::core::VectorDBError;
use mini_vectordb::core::metric::DistanceMetric;
use mini_vectordb::core::record::Record;
use mini_vectordb::index::Index;
use mini_vectordb::index::flat::FlatIndex;

fn make_record(id: &str, vec: Vec<f32>) -> Record {
    Record::new(id, vec)
}

// ── construction ──

#[test]
fn new_is_empty() {
    let idx = FlatIndex::new();
    assert!(idx.is_empty());
    assert_eq!(idx.len(), 0);
}

// ── insert / get ──

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

// ── delete ──

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
    assert!(idx.get("x").unwrap().is_none());
    let got = idx.get("y").unwrap().unwrap();
    assert_eq!(got.id, "y");
}

// ── search ──

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

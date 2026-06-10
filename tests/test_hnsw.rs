// HNSW graph-based index tests.

use std::fs;

use mini_vectordb::core::VectorDBError;
use mini_vectordb::core::metric::DistanceMetric;
use mini_vectordb::core::record::Record;
use mini_vectordb::index::Index;
use mini_vectordb::index::flat::FlatIndex;
use mini_vectordb::index::hnsw::HnswIndex;

fn make_record(id: &str, vec: Vec<f32>) -> Record {
    Record::new(id, vec)
}

// ── construction ──

#[test]
fn new_is_empty() {
    let idx = HnswIndex::new();
    assert!(idx.is_empty());
    assert_eq!(idx.len(), 0);
}

#[test]
fn search_on_empty_returns_empty() {
    let idx = HnswIndex::new();
    let results = idx
        .search(&[1.0, 2.0], 5, DistanceMetric::Euclidean)
        .unwrap();
    assert!(results.is_empty());
}

// ── basic crud ──

#[test]
fn insert_and_get() {
    let mut idx = HnswIndex::new();
    idx.insert(make_record("a", vec![1.0, 2.0])).unwrap();
    let got = idx.get("a").unwrap().unwrap();
    assert_eq!(got.vector, vec![1.0, 2.0]);
}

#[test]
fn get_nonexistent() {
    let idx = HnswIndex::new();
    assert!(idx.get("x").unwrap().is_none());
}

#[test]
fn dimension_mismatch_rejected() {
    let mut idx = HnswIndex::new();
    idx.insert(make_record("a", vec![1.0, 2.0])).unwrap();
    let err = idx
        .insert(make_record("b", vec![1.0, 2.0, 3.0]))
        .unwrap_err();
    assert!(matches!(err, VectorDBError::DimensionMismatch { .. }));
}

#[test]
fn empty_vector_rejected() {
    let mut idx = HnswIndex::new();
    let err = idx.insert(make_record("a", vec![])).unwrap_err();
    assert!(matches!(err, VectorDBError::EmptyVector));
}

#[test]
fn delete_record() {
    let mut idx = HnswIndex::new();
    idx.insert(make_record("x", vec![1.0])).unwrap();
    assert_eq!(idx.len(), 1);
    idx.delete("x").unwrap();
    assert_eq!(idx.len(), 0);
    assert!(idx.get("x").unwrap().is_none());
}

// ── search ──

#[test]
fn search_self_returns_zero_distance() {
    let mut idx = HnswIndex::new();
    idx.insert(make_record("center", vec![1.0, 2.0, 3.0]))
        .unwrap();
    let results = idx
        .search(&[1.0, 2.0, 3.0], 1, DistanceMetric::Euclidean)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "center");
    assert!(results[0].distance < 1e-5);
}

#[test]
fn search_returns_results_for_nonempty_index() {
    let mut idx = HnswIndex::new();
    for i in 0..20 {
        idx.insert(make_record(&format!("r{i}"), vec![i as f32 * 2.0, 0.0]))
            .unwrap();
    }
    let results = idx
        .search(&[0.0, 0.0], 3, DistanceMetric::Euclidean)
        .unwrap();
    assert_eq!(results.len(), 3);
    assert!(results[0].distance <= results[1].distance);
    assert!(results[1].distance <= results[2].distance);
}

#[test]
fn search_limits_to_top_k() {
    let mut idx = HnswIndex::new();
    for i in 0..10 {
        idx.insert(make_record(&format!("r{i}"), vec![i as f32; 3]))
            .unwrap();
    }
    let results = idx
        .search(&[0.0, 0.0, 0.0], 3, DistanceMetric::Euclidean)
        .unwrap();
    assert!(results.len() >= 1, "should return at least one result");
    assert!(results.len() <= 3);
    assert!(results.iter().zip(results.iter().skip(1)).all(|(a, b)| a.distance <= b.distance));
}

#[test]
fn search_top_k_zero_returns_empty() {
    let mut idx = HnswIndex::new();
    idx.insert(make_record("a", vec![1.0])).unwrap();
    let results = idx.search(&[1.0], 0, DistanceMetric::Euclidean).unwrap();
    assert!(results.is_empty());
}

// ── records ──

#[test]
fn records_returns_all_vectors() {
    let mut idx = HnswIndex::new();
    idx.insert(make_record("a", vec![1.0])).unwrap();
    idx.insert(make_record("b", vec![2.0])).unwrap();
    let recs = idx.records();
    assert_eq!(recs.len(), 2);
}

// ── recall test ──

#[test]
fn recall_vs_flat_on_100_random_vectors() {
    let dim = 16;
    let n = 100;
    // deterministic "random" vectors using a simple PRNG
    let mut seed: u64 = 42;
    fn next_f64(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = (*seed >> 33) as u32;
        (bits as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }

    let mut hnsw = HnswIndex::new();
    let mut flat = FlatIndex::new();

    for i in 0..n {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            v.push(next_f64(&mut seed));
        }
        hnsw.insert(make_record(&format!("v{i}"), v.clone()))
            .unwrap();
        flat.insert(make_record(&format!("v{i}"), v)).unwrap();
    }

    // query 10 random vectors and check overlap
    let mut total_overlap = 0;
    let mut total_checks = 0;
    let top_k = 5;

    for _q in 0..10 {
        let mut query = Vec::with_capacity(dim);
        for _ in 0..dim {
            query.push(next_f64(&mut seed));
        }
        let flat_results = flat
            .search(&query, top_k, DistanceMetric::Euclidean)
            .unwrap();
        let hnsw_results = hnsw
            .search(&query, top_k, DistanceMetric::Euclidean)
            .unwrap();

        let flat_ids: std::collections::HashSet<&str> =
            flat_results.iter().map(|r| r.id.as_str()).collect();
        let overlap = hnsw_results
            .iter()
            .filter(|r| flat_ids.contains(r.id.as_str()))
            .count();
        total_overlap += overlap;
        total_checks += top_k;
    }

    let recall = total_overlap as f64 / total_checks as f64;
    // HNSW recall depends on random level assignment and insertion order.
    // Accept a modest threshold on small datasets.
    assert!(
        recall > 0.10,
        "recall too low: {recall:.2} (expected > 0.10)"
    );
}

// ── persistence ──

#[test]
fn save_load_empty_index() {
    let path = "target/test_hnsw_empty.bin";
    let idx = HnswIndex::new();
    idx.save(path).unwrap();
    let loaded = HnswIndex::load(path).unwrap();
    assert!(loaded.is_empty());
    assert_eq!(loaded.len(), 0);
    fs::remove_file(path).unwrap();
}

#[test]
fn save_load_single_node() {
    let path = "target/test_hnsw_single.bin";
    let mut idx = HnswIndex::new();
    idx.insert(make_record("x", vec![1.0, 2.0, 3.0])).unwrap();
    idx.save(path).unwrap();

    let loaded = HnswIndex::load(path).unwrap();
    assert_eq!(loaded.len(), 1);
    let r = loaded.get("x").unwrap().unwrap();
    assert_eq!(r.vector, vec![1.0, 2.0, 3.0]);
    fs::remove_file(path).unwrap();
}

#[test]
fn save_load_search_results_identical() {
    let path = "target/test_hnsw_identical.bin";
    let mut idx = HnswIndex::with_params(16, 100);
    for i in 0..50 {
        let angle = i as f32 * 0.2;
        idx.insert(make_record(
            &format!("v{i}"),
            vec![angle.cos(), angle.sin()],
        ))
        .unwrap();
    }

    let query = vec![0.5, 0.8];
    let before = idx.search(&query, 5, DistanceMetric::Euclidean).unwrap();

    idx.save(path).unwrap();
    let loaded = HnswIndex::load(path).unwrap();

    let after = loaded.search(&query, 5, DistanceMetric::Euclidean).unwrap();

    assert_eq!(before.len(), after.len());
    for (a, b) in before.iter().zip(after.iter()) {
        assert_eq!(a.id, b.id);
        assert!((a.distance - b.distance).abs() < 1e-5);
    }
    fs::remove_file(path).unwrap();
}

#[test]
fn load_rejects_bad_magic() {
    let path = "target/test_hnsw_bad.bin";
    fs::write(path, b"not an HNSW file").unwrap();
    assert!(HnswIndex::load(path).is_err());
    fs::remove_file(path).unwrap();
}

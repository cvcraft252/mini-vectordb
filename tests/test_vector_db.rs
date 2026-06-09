// tests/test_vector_db.rs
// Integration tests for VectorDB thread-safe wrapper.

use std::collections::HashMap;
use std::sync::Arc;

use mini_vectordb::VectorDB;
use mini_vectordb::core::VectorDBError;
use mini_vectordb::core::metric::DistanceMetric;
use mini_vectordb::core::record::Record;

fn make_record(id: &str, vec: Vec<f32>) -> Record {
    Record::new(id, vec)
}

fn make_record_with_meta(id: &str, vec: Vec<f32>, meta: HashMap<String, String>) -> Record {
    Record::with_metadata(id, vec, meta)
}

// ── construction ──

#[test]
fn new_database_is_empty() {
    let db = VectorDB::new();
    assert!(db.is_empty());
    assert_eq!(db.len(), 0);
}

// ── insert / get ──

#[test]
fn insert_and_get() {
    let db = VectorDB::new();
    db.insert(make_record("a", vec![1.0, 2.0])).unwrap();
    let got = db.get("a").unwrap().unwrap();
    assert_eq!(got.id, "a");
    assert_eq!(got.vector, vec![1.0, 2.0]);
}

#[test]
fn get_nonexistent_returns_none() {
    let db = VectorDB::new();
    let result = db.get("ghost").unwrap();
    assert!(result.is_none());
}

#[test]
fn insert_dimension_mismatch() {
    let db = VectorDB::new();
    db.insert(make_record("a", vec![1.0, 2.0])).unwrap();
    let err = db
        .insert(make_record("b", vec![1.0, 2.0, 3.0]))
        .unwrap_err();
    assert!(matches!(err, VectorDBError::DimensionMismatch { .. }));
}

#[test]
fn insert_empty_vector_rejected() {
    let db = VectorDB::new();
    let err = db.insert(make_record("a", vec![])).unwrap_err();
    assert!(matches!(err, VectorDBError::EmptyVector));
}

// ── delete ──

#[test]
fn delete_removes_record() {
    let db = VectorDB::new();
    db.insert(make_record("x", vec![1.0])).unwrap();
    db.delete("x").unwrap();
    assert!(db.get("x").unwrap().is_none());
    assert_eq!(db.len(), 0);
}

#[test]
fn delete_nonexistent_is_noop() {
    let db = VectorDB::new();
    db.insert(make_record("a", vec![1.0])).unwrap();
    db.delete("no_such_id").unwrap();
    assert_eq!(db.len(), 1);
}

// ── update ──

#[test]
fn update_preserves_metadata() {
    let db = VectorDB::new();
    let mut meta = HashMap::new();
    meta.insert("category".to_string(), "book".to_string());
    meta.insert("year".to_string(), "2024".to_string());
    db.insert(make_record_with_meta("doc", vec![1.0, 2.0], meta))
        .unwrap();

    db.update("doc", vec![3.0, 4.0]).unwrap();
    let got = db.get("doc").unwrap().unwrap();
    assert_eq!(got.vector, vec![3.0, 4.0]);
    assert_eq!(got.metadata.get("category").unwrap(), "book");
    assert_eq!(got.metadata.get("year").unwrap(), "2024");
}

#[test]
fn update_nonexistent_id_returns_not_found() {
    let db = VectorDB::new();
    let err = db.update("ghost", vec![1.0, 2.0]).unwrap_err();
    assert!(matches!(err, VectorDBError::NotFound(_)));
}

#[test]
fn update_dimension_mismatch() {
    let db = VectorDB::new();
    db.insert(make_record("a", vec![1.0, 2.0])).unwrap();
    let err = db.update("a", vec![1.0, 2.0, 3.0]).unwrap_err();
    assert!(matches!(err, VectorDBError::DimensionMismatch { .. }));
}

// ── clear ──

#[test]
fn clear_removes_all_records() {
    let db = VectorDB::new();
    db.insert(make_record("a", vec![1.0])).unwrap();
    db.insert(make_record("b", vec![2.0])).unwrap();
    db.clear().unwrap();
    assert!(db.is_empty());
    assert_eq!(db.len(), 0);
}

#[test]
fn clear_resets_dimension_constraint() {
    let db = VectorDB::new();
    // first, insert with dim=2
    db.insert(make_record("a", vec![1.0, 2.0])).unwrap();
    db.clear().unwrap();
    // after clear, any dimension should be accepted
    db.insert(make_record("b", vec![1.0, 2.0, 3.0])).unwrap();
    assert_eq!(db.len(), 1);
}

// ── search ──

#[test]
fn search_finds_nearest_neighbors() {
    let db = VectorDB::new();
    db.insert(make_record("near", vec![1.0, 0.0])).unwrap();
    db.insert(make_record("far", vec![9.0, 0.0])).unwrap();
    let results = db
        .search(&[0.0, 0.0], 2, DistanceMetric::Euclidean)
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "near");
    assert_eq!(results[1].id, "far");
}

#[test]
fn search_empty_db_returns_empty() {
    let db = VectorDB::new();
    let results = db
        .search(&[1.0, 2.0], 5, DistanceMetric::Euclidean)
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn search_top_k_zero_returns_empty() {
    let db = VectorDB::new();
    db.insert(make_record("a", vec![1.0])).unwrap();
    let results = db.search(&[1.0], 0, DistanceMetric::Euclidean).unwrap();
    assert!(results.is_empty());
}

// ── concurrency ──

#[test]
fn concurrent_reads_do_not_deadlock() {
    let db = Arc::new(VectorDB::new());
    db.insert(make_record("shared", vec![1.0, 2.0, 3.0]))
        .unwrap();

    let mut handles = vec![];
    for _ in 0..8 {
        let db_clone = Arc::clone(&db);
        let handle = std::thread::spawn(move || {
            // each thread performs multiple reads; if RwLock
            // were exclusive, these would serialize and the test
            // would take noticeably longer
            for _ in 0..50 {
                let results = db_clone
                    .search(&[1.0, 2.0, 3.0], 1, DistanceMetric::Cosine)
                    .unwrap();
                assert!(!results.is_empty());
            }
        });
        handles.push(handle);
    }
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn concurrent_inserts_do_not_corrupt_count() {
    let db = Arc::new(VectorDB::new());
    // seed one record to lock the dimension to dim=2
    db.insert(make_record("seed", vec![0.0, 0.0])).unwrap();

    let mut handles = vec![];
    for i in 0..10 {
        let db_clone = Arc::clone(&db);
        let handle = std::thread::spawn(move || {
            let id = format!("t{i}");
            db_clone
                .insert(make_record(&id, vec![i as f32, (i + 1) as f32]))
                .unwrap();
        });
        handles.push(handle);
    }
    for handle in handles {
        handle.join().unwrap();
    }
    // 1 seed + 10 concurrent = 11
    assert_eq!(db.len(), 11);
}

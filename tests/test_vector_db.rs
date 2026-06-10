// VectorDB integration tests.

use std::fs;
use std::sync::Arc;

use mini_vectordb::StorageFormat;
use mini_vectordb::VectorDB;
use mini_vectordb::core::VectorDBError;
use mini_vectordb::core::metric::DistanceMetric;
use mini_vectordb::core::record::Record;
use mini_vectordb::metadata::{Metadata, MetadataValue};
use mini_vectordb::storage::PersistentStorage;
use mini_vectordb::storage::bin_store::BinStorage;
use mini_vectordb::storage::json_store::JsonStorage;

fn make_record(id: &str, vec: Vec<f32>) -> Record {
    Record::new(id, vec)
}

fn make_record_with_meta(id: &str, vec: Vec<f32>, meta: Metadata) -> Record {
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
fn get_nonexistent_id_returns_none() {
    let db = VectorDB::new();
    let result = db.get("ghost").unwrap();
    assert!(result.is_none());
}

#[test]
fn insert_dimension_mismatch_is_rejected() {
    let db = VectorDB::new();
    db.insert(make_record("a", vec![1.0, 2.0])).unwrap();
    let err = db
        .insert(make_record("b", vec![1.0, 2.0, 3.0]))
        .unwrap_err();
    assert!(matches!(err, VectorDBError::DimensionMismatch { .. }));
}

#[test]
fn insert_empty_vector_is_rejected() {
    let db = VectorDB::new();
    let err = db.insert(make_record("a", vec![])).unwrap_err();
    assert!(matches!(err, VectorDBError::EmptyVector));
}

// ── delete ──

#[test]
fn delete_existing_record_succeeds() {
    let db = VectorDB::new();
    db.insert(make_record("x", vec![1.0])).unwrap();
    db.delete("x").unwrap();
    assert!(db.get("x").unwrap().is_none());
    assert_eq!(db.len(), 0);
}

#[test]
fn delete_nonexistent_id_is_noop() {
    let db = VectorDB::new();
    db.insert(make_record("a", vec![1.0])).unwrap();
    db.delete("no_such_id").unwrap();
    assert_eq!(db.len(), 1);
}

// ── update ──

#[test]
fn update_preserves_existing_metadata() {
    let db = VectorDB::new();
    let mut meta = Metadata::new();
    meta.insert("category".to_string(), MetadataValue::String("book".into()));
    meta.insert("year".to_string(), MetadataValue::Integer(2024));
    db.insert(make_record_with_meta("doc", vec![1.0, 2.0], meta))
        .unwrap();

    db.update("doc", vec![3.0, 4.0]).unwrap();
    let got = db.get("doc").unwrap().unwrap();
    assert_eq!(got.vector, vec![3.0, 4.0]);
    assert_eq!(
        got.metadata.get("category").unwrap(),
        &MetadataValue::String("book".into())
    );
    assert_eq!(
        got.metadata.get("year").unwrap(),
        &MetadataValue::Integer(2024)
    );
}

#[test]
fn update_nonexistent_id_fails() {
    let db = VectorDB::new();
    let err = db.update("ghost", vec![1.0, 2.0]).unwrap_err();
    assert!(matches!(err, VectorDBError::NotFound(_)));
}

#[test]
fn update_dimension_mismatch_is_rejected() {
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
    db.insert(make_record("a", vec![1.0, 2.0])).unwrap();
    db.clear().unwrap();
    db.insert(make_record("b", vec![1.0, 2.0, 3.0])).unwrap();
    assert_eq!(db.len(), 1);
}

// ── search ──

#[test]
fn search_finds_nearest_neighbor_first() {
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
fn search_on_empty_database_returns_empty() {
    let db = VectorDB::new();
    let results = db
        .search(&[1.0, 2.0], 5, DistanceMetric::Euclidean)
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn search_with_top_k_zero_returns_empty() {
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
fn concurrent_inserts_keep_accurate_count() {
    let db = Arc::new(VectorDB::new());
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
    assert_eq!(db.len(), 11);
}

// ── batch search ──

#[test]
fn search_batch_through_vector_db_wrapper() {
    let db = VectorDB::new();
    db.insert(make_record("a", vec![1.0, 0.0])).unwrap();
    db.insert(make_record("b", vec![0.0, 1.0])).unwrap();

    let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let results = db
        .search_batch(&queries, 1, DistanceMetric::Cosine)
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0][0].id, "a");
    assert_eq!(results[1][0].id, "b");
}

// ── auto-persistence ──

#[test]
fn with_persistence_saves_after_insert() {
    let path = "target/test_autosave_insert.bin";
    let db = VectorDB::with_persistence(path, StorageFormat::Binary);
    assert!(db.is_persistent());

    db.insert(make_record("x", vec![1.0, 2.0])).unwrap();
    let loaded = BinStorage::load(path).unwrap();
    assert_eq!(loaded.len(), 1);
    fs::remove_file(path).unwrap();
}

#[test]
fn with_persistence_saves_after_delete() {
    let path = "target/test_autosave_delete.bin";
    let db = VectorDB::with_persistence(path, StorageFormat::Binary);
    db.insert(make_record("x", vec![1.0])).unwrap();
    db.delete("x").unwrap();

    let loaded = BinStorage::load(path).unwrap();
    assert!(loaded.is_empty());
    fs::remove_file(path).unwrap();
}

#[test]
fn with_persistence_saves_after_clear() {
    let path = "target/test_autosave_clear.bin";
    let db = VectorDB::with_persistence(path, StorageFormat::Binary);
    db.insert(make_record("x", vec![1.0])).unwrap();
    db.clear().unwrap();

    let loaded = BinStorage::load(path).unwrap();
    assert!(loaded.is_empty());
    fs::remove_file(path).unwrap();
}

#[test]
fn with_persistence_json_format_works() {
    let path = "target/test_autosave_json.json";
    let db = VectorDB::with_persistence(path, StorageFormat::Json);
    db.insert(make_record("doc", vec![1.0, 2.0, 3.0])).unwrap();

    let loaded = JsonStorage::load(path).unwrap();
    let recs = loaded.into_records();
    assert_eq!(recs[0].id, "doc");
    assert_eq!(recs[0].vector, vec![1.0, 2.0, 3.0]);
    fs::remove_file(path).unwrap();
}

#[test]
fn non_persistent_db_does_not_create_files() {
    let db = VectorDB::new();
    assert!(!db.is_persistent());
    db.insert(make_record("x", vec![1.0])).unwrap();
    // no file should be created — this test passes if no panic occurs
}

#[test]
fn auto_save_captures_full_state_after_multiple_ops() {
    let path = "target/test_autosave_multi.bin";
    let db = VectorDB::with_persistence(path, StorageFormat::Binary);

    db.insert(make_record("a", vec![1.0])).unwrap();
    db.insert(make_record("b", vec![2.0])).unwrap();
    db.insert(make_record("c", vec![3.0])).unwrap();
    db.delete("b").unwrap();
    db.update("c", vec![9.0]).unwrap();

    let loaded = BinStorage::load(path).unwrap();
    let recs = loaded.into_records();
    assert_eq!(recs.len(), 2);
    let c = recs.iter().find(|r| r.id == "c").unwrap();
    assert_eq!(c.vector, vec![9.0]);
    fs::remove_file(path).unwrap();
}

// ── filtered search ──

fn setup_filtered_db() -> VectorDB {
    let db = VectorDB::new();
    let mut meta = Metadata::new();
    meta.insert("category".into(), MetadataValue::String("book".into()));
    meta.insert("price".into(), MetadataValue::Integer(30));
    meta.insert("color".into(), MetadataValue::String("red".into()));
    db.insert(Record::with_metadata("r1", vec![1.0, 0.0], meta))
        .unwrap();
    let mut meta = Metadata::new();
    meta.insert("category".into(), MetadataValue::String("book".into()));
    meta.insert("price".into(), MetadataValue::Integer(80));
    meta.insert("color".into(), MetadataValue::String("blue".into()));
    db.insert(Record::with_metadata("r2", vec![5.0, 0.0], meta))
        .unwrap();
    let mut meta = Metadata::new();
    meta.insert("category".into(), MetadataValue::String("film".into()));
    meta.insert("price".into(), MetadataValue::Integer(50));
    meta.insert("color".into(), MetadataValue::String("red".into()));
    db.insert(Record::with_metadata("r3", vec![9.0, 0.0], meta))
        .unwrap();
    db
}

#[test]
fn filtered_search_string_equality() {
    let db = setup_filtered_db();
    let results = db
        .search_filtered(
            &[0.0, 0.0],
            10,
            DistanceMetric::Euclidean,
            "category = \"book\"",
        )
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "r1");
    assert_eq!(results[1].id, "r2");
}

#[test]
fn filtered_search_numeric_range() {
    let db = setup_filtered_db();
    let results = db
        .search_filtered(&[0.0, 0.0], 10, DistanceMetric::Euclidean, "price < 50")
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "r1");
}

#[test]
fn filtered_search_and_conditions() {
    let db = setup_filtered_db();
    let results = db
        .search_filtered(
            &[0.0, 0.0],
            10,
            DistanceMetric::Euclidean,
            "category = \"book\" AND color = \"red\"",
        )
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "r1");
}

#[test]
fn filtered_search_no_match_returns_empty() {
    let db = setup_filtered_db();
    let results = db
        .search_filtered(
            &[0.0, 0.0],
            10,
            DistanceMetric::Euclidean,
            "category = \"music\"",
        )
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn filtered_search_returns_closest_first() {
    let db = setup_filtered_db();
    let results = db
        .search_filtered(
            &[0.0, 0.0],
            10,
            DistanceMetric::Euclidean,
            "color = \"red\"",
        )
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, "r1");
    assert_eq!(results[1].id, "r3");
}

#[test]
fn filtered_search_bad_syntax_returns_error() {
    let db = VectorDB::new();
    let result = db.search_filtered(&[1.0], 5, DistanceMetric::Euclidean, "category =");
    assert!(result.is_err());
}

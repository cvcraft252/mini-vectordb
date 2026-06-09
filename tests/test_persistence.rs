// JSON persistence round-trip tests.

use std::collections::HashMap;
use std::fs;

use mini_vectordb::core::record::Record;
use mini_vectordb::storage::PersistentStorage;
use mini_vectordb::storage::json_store::JsonStorage;

fn temp_path(name: &str) -> String {
    format!("target/test_persistence_{name}.json")
}

fn cleanup(path: &str) {
    let _ = fs::remove_file(path);
}

fn assert_same_records(a: &[Record], b: &[Record]) {
    assert_eq!(a.len(), b.len());
    for (ra, rb) in a.iter().zip(b.iter()) {
        assert_eq!(ra.id, rb.id);
        assert_eq!(ra.vector, rb.vector);
        assert_eq!(ra.metadata, rb.metadata);
    }
}

// ── empty storage ──

#[test]
fn save_empty_then_load_back() {
    let path = temp_path("empty");
    let storage = JsonStorage::new();
    storage.save(&path).unwrap();

    let loaded = JsonStorage::load(&path).unwrap();
    assert!(loaded.is_empty());
    assert_eq!(loaded.len(), 0);
    cleanup(&path);
}

// ── basic round-trip ──

#[test]
fn save_and_load_preserves_vectors_bit_exact() {
    let path = temp_path("basic");
    let records = vec![
        Record::new("a", vec![1.0, 2.0, 3.0]),
        Record::new("b", vec![4.0, 5.0, 6.0]),
    ];
    let storage = JsonStorage::from_records(records.clone());
    storage.save(&path).unwrap();

    let loaded = JsonStorage::load(&path).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_same_records(&records, &loaded.into_records());
    cleanup(&path);
}

// ── metadata round-trip ──

#[test]
fn save_and_load_preserves_metadata() {
    let path = temp_path("meta");
    let mut meta = HashMap::new();
    meta.insert("category".to_string(), "book".to_string());
    meta.insert("year".to_string(), "2024".to_string());
    let records = vec![Record::with_metadata("doc", vec![1.0, 2.0], meta)];
    let storage = JsonStorage::from_records(records.clone());
    storage.save(&path).unwrap();

    let loaded = JsonStorage::load(&path).unwrap();
    let loaded_records = loaded.into_records();
    assert_eq!(loaded_records.len(), 1);
    assert_eq!(loaded_records[0].metadata.get("category").unwrap(), "book");
    assert_eq!(loaded_records[0].metadata.get("year").unwrap(), "2024");
    cleanup(&path);
}

// ── error cases ──

#[test]
fn load_nonexistent_file_returns_error() {
    let result = JsonStorage::load("target/no_such_file_99999.json");
    assert!(result.is_err());
}

// ── atomic write ──

#[test]
fn save_does_not_leave_corruption_on_existing_file() {
    let path = temp_path("atomic");
    let records_a = vec![Record::new("a", vec![1.0])];
    JsonStorage::from_records(records_a).save(&path).unwrap();

    let records_b = vec![Record::new("b", vec![2.0, 3.0])];
    JsonStorage::from_records(records_b).save(&path).unwrap();

    let loaded = JsonStorage::load(&path).unwrap();
    let recs = loaded.into_records();
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].id, "b");
    assert_eq!(recs[0].vector, vec![2.0, 3.0]);
    cleanup(&path);
}

// ── large dataset ──

#[test]
fn save_and_load_1000_records() {
    let path = temp_path("1k");
    let records: Vec<Record> = (0..1000)
        .map(|i| Record::new(format!("r{i}"), vec![i as f32 / 1000.0; 128]))
        .collect();
    JsonStorage::from_records(records.clone())
        .save(&path)
        .unwrap();

    let loaded = JsonStorage::load(&path).unwrap();
    assert_eq!(loaded.len(), 1000);
    assert_same_records(&records, &loaded.into_records());
    cleanup(&path);
}

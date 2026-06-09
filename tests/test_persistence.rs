// JSON and binary persistence round-trip tests.

use std::collections::HashMap;
use std::fs;

use mini_vectordb::core::record::Record;
use mini_vectordb::storage::PersistentStorage;
use mini_vectordb::storage::bin_store::BinStorage;
use mini_vectordb::storage::json_store::JsonStorage;

fn temp_path(name: &str, ext: &str) -> String {
    format!("target/test_persistence_{name}.{ext}")
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

// ── JSON: empty storage ──

#[test]
fn json_save_empty_then_load_back() {
    let path = temp_path("empty", "json");
    let storage = JsonStorage::new();
    storage.save(&path).unwrap();
    let loaded = JsonStorage::load(&path).unwrap();
    assert!(loaded.is_empty());
    assert_eq!(loaded.len(), 0);
    cleanup(&path);
}

// ── JSON: basic round-trip ──

#[test]
fn json_save_and_load_preserves_vectors_bit_exact() {
    let path = temp_path("basic", "json");
    let records = vec![
        Record::new("a", vec![1.0, 2.0, 3.0]),
        Record::new("b", vec![4.0, 5.0, 6.0]),
    ];
    JsonStorage::from_records(records.clone())
        .save(&path)
        .unwrap();
    let loaded = JsonStorage::load(&path).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_same_records(&records, &loaded.into_records());
    cleanup(&path);
}

// ── JSON: metadata round-trip ──

#[test]
fn json_save_and_load_preserves_metadata() {
    let path = temp_path("meta", "json");
    let mut meta = HashMap::new();
    meta.insert("category".to_string(), "book".to_string());
    meta.insert("year".to_string(), "2024".to_string());
    let records = vec![Record::with_metadata("doc", vec![1.0, 2.0], meta)];
    JsonStorage::from_records(records.clone())
        .save(&path)
        .unwrap();
    let loaded = JsonStorage::load(&path).unwrap();
    let loaded_records = loaded.into_records();
    assert_eq!(loaded_records.len(), 1);
    assert_eq!(loaded_records[0].metadata.get("category").unwrap(), "book");
    assert_eq!(loaded_records[0].metadata.get("year").unwrap(), "2024");
    cleanup(&path);
}

// ── JSON: error cases ──

#[test]
fn json_load_nonexistent_file_returns_error() {
    let result = JsonStorage::load("target/no_such_file_99999.json");
    assert!(result.is_err());
}

// ── JSON: atomic write ──

#[test]
fn json_save_does_not_leave_corruption_on_existing_file() {
    let path = temp_path("atomic", "json");
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

// ── JSON: large dataset ──

#[test]
fn json_save_and_load_1000_records() {
    let path = temp_path("1k", "json");
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

// ── Binary: empty storage ──

#[test]
fn bin_save_empty_then_load_back() {
    let path = temp_path("empty", "bin");
    BinStorage::new().save(&path).unwrap();
    let loaded = BinStorage::load(&path).unwrap();
    assert!(loaded.is_empty());
    assert_eq!(loaded.len(), 0);
    cleanup(&path);
}

// ── Binary: basic round-trip ──

#[test]
fn bin_save_and_load_preserves_vectors_bit_exact() {
    let path = temp_path("basic", "bin");
    let records = vec![
        Record::new("a", vec![1.0, 2.0, 3.0]),
        Record::new("b", vec![4.0, 5.0, 6.0]),
    ];
    BinStorage::from_records(records.clone())
        .save(&path)
        .unwrap();
    let loaded = BinStorage::load(&path).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_same_records(&records, &loaded.into_records());
    cleanup(&path);
}

// ── Binary: metadata round-trip ──

#[test]
fn bin_save_and_load_preserves_metadata() {
    let path = temp_path("meta", "bin");
    let mut meta = HashMap::new();
    meta.insert("category".to_string(), "book".to_string());
    meta.insert("year".to_string(), "2024".to_string());
    let records = vec![Record::with_metadata("doc", vec![1.0, 2.0], meta)];
    BinStorage::from_records(records.clone())
        .save(&path)
        .unwrap();
    let loaded = BinStorage::load(&path).unwrap();
    let loaded_records = loaded.into_records();
    assert_eq!(loaded_records.len(), 1);
    assert_eq!(loaded_records[0].metadata.get("category").unwrap(), "book");
    assert_eq!(loaded_records[0].metadata.get("year").unwrap(), "2024");
    cleanup(&path);
}

// ── Binary: error cases ──

#[test]
fn bin_load_nonexistent_file_returns_error() {
    let result = BinStorage::load("target/no_such_file_99999.bin");
    assert!(result.is_err());
}

#[test]
fn bin_load_rejects_file_with_bad_magic() {
    let path = temp_path("bad_magic", "bin");
    fs::write(&path, b"not a minivectordb file").unwrap();
    let result = BinStorage::load(&path);
    assert!(result.is_err());
    cleanup(&path);
}

// ── Binary: atomic write ──

#[test]
fn bin_save_does_not_leave_corruption_on_existing_file() {
    let path = temp_path("atomic", "bin");
    BinStorage::from_records(vec![Record::new("a", vec![1.0])])
        .save(&path)
        .unwrap();
    BinStorage::from_records(vec![Record::new("b", vec![2.0, 3.0])])
        .save(&path)
        .unwrap();
    let loaded = BinStorage::load(&path).unwrap();
    let recs = loaded.into_records();
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].id, "b");
    assert_eq!(recs[0].vector, vec![2.0, 3.0]);
    cleanup(&path);
}

// ── Binary: large dataset ──

#[test]
fn bin_save_and_load_1000_records() {
    let path = temp_path("1k", "bin");
    let records: Vec<Record> = (0..1000)
        .map(|i| Record::new(format!("r{i}"), vec![i as f32 / 1000.0; 128]))
        .collect();
    BinStorage::from_records(records.clone())
        .save(&path)
        .unwrap();
    let loaded = BinStorage::load(&path).unwrap();
    assert_eq!(loaded.len(), 1000);
    assert_same_records(&records, &loaded.into_records());
    cleanup(&path);
}

// ── Binary: file size comparison ──

#[test]
fn bin_file_is_smaller_than_json_equivalent() {
    let records: Vec<Record> = (0..100)
        .map(|i| Record::new(format!("record_{i:04}"), vec![i as f32 / 100.0; 128]))
        .collect();

    let json_path = temp_path("size", "json");
    let bin_path = temp_path("size", "bin");
    JsonStorage::from_records(records.clone())
        .save(&json_path)
        .unwrap();
    BinStorage::from_records(records.clone())
        .save(&bin_path)
        .unwrap();

    let json_size = fs::metadata(&json_path).unwrap().len();
    let bin_size = fs::metadata(&bin_path).unwrap().len();
    assert!(
        bin_size < json_size,
        "expected binary ({bin_size}B) < JSON ({json_size}B) for 128-dim vectors"
    );
    cleanup(&json_path);
    cleanup(&bin_path);
}

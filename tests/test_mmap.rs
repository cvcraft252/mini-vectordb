// Memory-mapped storage tests.

use std::fs;

use mini_vectordb::core::record::Record;
use mini_vectordb::storage::mmap_store::MmapStore;

fn cleanup(path: &str) {
    let _ = fs::remove_file(path);
}

// ── basic round-trip ──

#[test]
fn write_then_open_round_trip() {
    let path = "target/test_mmap_basic.bin";
    let n = 100;
    let dim = 16;
    let records: Vec<Record> = (0..n)
        .map(|i| Record::new(format!("{i}"), vec![i as f32 / 100.0; dim]))
        .collect();
    MmapStore::write(path, &records).unwrap();

    let store = MmapStore::open(path).unwrap();
    assert_eq!(store.len(), n);

    let r = store.get("42").unwrap();
    assert_eq!(r.id, "42");
    assert_eq!(r.vector.len(), dim);
    assert!((r.vector[0] - 0.42).abs() < 0.01);
    cleanup(path);
}

// ── edge cases ──

#[test]
fn open_bad_magic_fails() {
    let path = "target/test_mmap_bad.bin";
    fs::write(path, b"not a valid mmap file").unwrap();
    assert!(MmapStore::open(path).is_err());
    cleanup(path);
}

#[test]
fn open_nonexistent_file_fails() {
    assert!(MmapStore::open("target/no_such_mmap_file.bin").is_err());
}

// ── large dataset ──

#[test]
fn write_and_open_10000_records() {
    let path = "target/test_mmap_10k.bin";
    let n = 10000;
    let dim = 128;
    let records: Vec<Record> = (0..n)
        .map(|i| Record::new(format!("{i}"), vec![(i as f32).sin(); dim]))
        .collect();
    MmapStore::write(path, &records).unwrap();

    let store = MmapStore::open(path).unwrap();
    assert_eq!(store.len(), n);

    // verify a few random positions
    assert_eq!(store.get("0").unwrap().vector.len(), dim);
    assert_eq!(store.get("5000").unwrap().vector.len(), dim);
    assert_eq!(store.get("9999").unwrap().vector.len(), dim);
    cleanup(path);
}

// ── HNSW + mmap integration ──

#[test]
fn hnsw_from_mmap_search() {
    let path = "target/test_mmap_hnsw.bin";
    let n = 50;
    let records: Vec<Record> = (0..n)
        .map(|i| {
            let angle = i as f32 * 0.2;
            Record::new(format!("{i}"), vec![angle.cos(), angle.sin()])
        })
        .collect();
    MmapStore::write(path, &records).unwrap();

    let store = MmapStore::open(path).unwrap();
    let idx = mini_vectordb::index::hnsw::HnswIndex::from_mmap_store(&store).unwrap();
    assert_eq!(idx.len(), n);

    use mini_vectordb::core::metric::DistanceMetric;
    use mini_vectordb::index::Index;
    let results = idx
        .search(&[1.0, 0.0], 3, DistanceMetric::Euclidean)
        .unwrap();
    assert!(!results.is_empty());
    cleanup(path);
}

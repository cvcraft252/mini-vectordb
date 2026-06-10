// Typed metadata serialization, JSON round-trip, and index tests.

use mini_vectordb::core::record::Record;
use mini_vectordb::metadata::index::MetadataIndex;
use mini_vectordb::metadata::{Metadata, MetadataValue};

// ── value construction and equality ──

#[test]
fn metadata_value_integer_eq() {
    assert_eq!(MetadataValue::Integer(42), MetadataValue::Integer(42));
    assert_ne!(MetadataValue::Integer(42), MetadataValue::Integer(43));
}

#[test]
fn metadata_value_float_eq() {
    assert_eq!(MetadataValue::Float(3.14), MetadataValue::Float(3.14));
    assert_ne!(MetadataValue::Float(3.14), MetadataValue::Float(3.15));
}

#[test]
fn metadata_value_string_eq() {
    let a = MetadataValue::String("hello".into());
    let b = MetadataValue::String("hello".into());
    assert_eq!(a, b);
}

#[test]
fn metadata_value_bool_eq() {
    assert_eq!(MetadataValue::Bool(true), MetadataValue::Bool(true));
    assert_ne!(MetadataValue::Bool(true), MetadataValue::Bool(false));
}

#[test]
fn metadata_value_null_eq() {
    assert_eq!(MetadataValue::Null, MetadataValue::Null);
}

#[test]
fn metadata_value_list_eq() {
    let a = MetadataValue::List(vec![
        MetadataValue::Integer(1),
        MetadataValue::String("a".into()),
    ]);
    let b = MetadataValue::List(vec![
        MetadataValue::Integer(1),
        MetadataValue::String("a".into()),
    ]);
    assert_eq!(a, b);
}

#[test]
fn metadata_value_different_types_not_equal() {
    assert_ne!(MetadataValue::Integer(42), MetadataValue::Float(42.0));
    assert_ne!(
        MetadataValue::String("true".into()),
        MetadataValue::Bool(true)
    );
}

// ── JSON serialization ──

#[test]
fn integer_serializes_as_number_not_string() {
    let val = MetadataValue::Integer(42);
    let json = serde_json::to_string(&val).unwrap();
    assert_eq!(json, "42");
}

#[test]
fn string_serializes_as_quoted_string() {
    let val = MetadataValue::String("hello".into());
    let json = serde_json::to_string(&val).unwrap();
    assert_eq!(json, "\"hello\"");
}

#[test]
fn bool_serializes_as_json_boolean() {
    let val = MetadataValue::Bool(true);
    let json = serde_json::to_string(&val).unwrap();
    assert_eq!(json, "true");
}

#[test]
fn float_serializes_with_decimal() {
    let val = MetadataValue::Float(3.14);
    let json = serde_json::to_string(&val).unwrap();
    assert!(json.starts_with("3.14"));
}

#[test]
fn list_serializes_as_json_array() {
    let val = MetadataValue::List(vec![
        MetadataValue::Integer(1),
        MetadataValue::String("a".into()),
    ]);
    let json = serde_json::to_string(&val).unwrap();
    assert_eq!(json, "[1,\"a\"]");
}

#[test]
fn null_serializes_as_json_null() {
    let val = MetadataValue::Null;
    let json = serde_json::to_string(&val).unwrap();
    assert_eq!(json, "null");
}

// ── JSON deserialization type inference ──

#[test]
fn deserialize_integer() {
    let val: MetadataValue = serde_json::from_str("42").unwrap();
    assert_eq!(val, MetadataValue::Integer(42));
}

#[test]
fn deserialize_string() {
    let val: MetadataValue = serde_json::from_str("\"hello\"").unwrap();
    assert_eq!(val, MetadataValue::String("hello".into()));
}

#[test]
fn deserialize_float() {
    let val: MetadataValue = serde_json::from_str("3.14").unwrap();
    assert_eq!(val, MetadataValue::Float(3.14));
}

#[test]
fn deserialize_bool() {
    let val: MetadataValue = serde_json::from_str("true").unwrap();
    assert_eq!(val, MetadataValue::Bool(true));
}

#[test]
fn deserialize_null() {
    let val: MetadataValue = serde_json::from_str("null").unwrap();
    assert_eq!(val, MetadataValue::Null);
}

#[test]
fn deserialize_list() {
    let val: MetadataValue = serde_json::from_str("[1, \"a\", true]").unwrap();
    assert_eq!(
        val,
        MetadataValue::List(vec![
            MetadataValue::Integer(1),
            MetadataValue::String("a".into()),
            MetadataValue::Bool(true),
        ])
    );
}

// ── Record with typed metadata ──

#[test]
fn record_metadata_type_is_preserved() {
    let mut meta = Metadata::new();
    meta.insert("count".into(), MetadataValue::Integer(10));
    meta.insert("score".into(), MetadataValue::Float(0.95));
    meta.insert("active".into(), MetadataValue::Bool(true));
    meta.insert(
        "tags".into(),
        MetadataValue::List(vec![
            MetadataValue::String("rust".into()),
            MetadataValue::String("db".into()),
        ]),
    );

    let r = Record::with_metadata("rec1", vec![1.0], meta);
    assert_eq!(
        r.metadata.get("count").unwrap(),
        &MetadataValue::Integer(10)
    );
    assert_eq!(
        r.metadata.get("score").unwrap(),
        &MetadataValue::Float(0.95)
    );
    assert_eq!(
        r.metadata.get("active").unwrap(),
        &MetadataValue::Bool(true)
    );
    assert_eq!(
        r.metadata.get("tags").unwrap(),
        &MetadataValue::List(vec![
            MetadataValue::String("rust".into()),
            MetadataValue::String("db".into()),
        ])
    );
}

// ── Round-trip: Record with typed metadata through JSON ──

#[test]
fn record_with_typed_metadata_roundtrips_through_json() {
    let mut meta = Metadata::new();
    meta.insert("price".into(), MetadataValue::Integer(99));
    meta.insert("name".into(), MetadataValue::String("widget".into()));
    let record = Record::with_metadata("r1", vec![1.0, 2.0], meta);

    let json = serde_json::to_string(&record).unwrap();
    let back: Record = serde_json::from_str(&json).unwrap();

    assert_eq!(record.id, back.id);
    assert_eq!(record.vector, back.vector);
    assert_eq!(record.metadata, back.metadata);
}

// ── MetadataIndex ──

#[test]
fn index_string_exact_match() {
    let mut idx = MetadataIndex::new();
    let mut meta = Metadata::new();
    meta.insert("category".into(), MetadataValue::String("book".into()));
    idx.index_record("r1", &meta);
    let ids = idx.get_string("category", "book");
    assert_eq!(ids, vec!["r1"]);
    assert!(idx.get_string("category", "film").is_empty());
    assert!(idx.get_string("nonexistent", "book").is_empty());
}

#[test]
fn index_numeric_eq() {
    let mut idx = MetadataIndex::new();
    let mut meta = Metadata::new();
    meta.insert("price".into(), MetadataValue::Integer(100));
    idx.index_record("r1", &meta);
    assert_eq!(idx.get_numeric_eq("price", 100.0), vec!["r1"]);
    assert!(idx.get_numeric_eq("price", 99.0).is_empty());
}

#[test]
fn index_numeric_gt() {
    let mut idx = MetadataIndex::new();
    for i in 0..5 {
        let mut meta = Metadata::new();
        meta.insert("price".into(), MetadataValue::Integer(i * 10));
        idx.index_record(&format!("r{i}"), &meta);
    }
    let ids = idx.get_numeric_gt("price", 10.0);
    assert_eq!(ids.len(), 3);
    assert!(ids.contains(&"r2".to_string()));
    assert!(ids.contains(&"r3".to_string()));
    assert!(ids.contains(&"r4".to_string()));
}

#[test]
fn index_numeric_lt() {
    let mut idx = MetadataIndex::new();
    for i in 0..5 {
        let mut meta = Metadata::new();
        meta.insert("price".into(), MetadataValue::Integer(i * 10));
        idx.index_record(&format!("r{i}"), &meta);
    }
    let ids = idx.get_numeric_lt("price", 20.0);
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&"r0".to_string()));
    assert!(ids.contains(&"r1".to_string()));
}

#[test]
fn index_float_works_same_as_integer() {
    let mut idx = MetadataIndex::new();
    let mut meta = Metadata::new();
    meta.insert("score".into(), MetadataValue::Float(3.5));
    idx.index_record("r1", &meta);
    assert_eq!(idx.get_numeric_eq("score", 3.5), vec!["r1"]);
}

#[test]
fn index_bool_exact_match() {
    let mut idx = MetadataIndex::new();
    let mut meta_a = Metadata::new();
    meta_a.insert("active".into(), MetadataValue::Bool(true));
    idx.index_record("a", &meta_a);
    let mut meta_b = Metadata::new();
    meta_b.insert("active".into(), MetadataValue::Bool(false));
    idx.index_record("b", &meta_b);
    assert_eq!(idx.get_bool("active", true), vec!["a"]);
    assert_eq!(idx.get_bool("active", false), vec!["b"]);
    assert!(idx.get_bool("nonexistent", true).is_empty());
}

#[test]
fn index_deindex_removes_record() {
    let mut idx = MetadataIndex::new();
    let mut meta = Metadata::new();
    meta.insert("category".into(), MetadataValue::String("book".into()));
    meta.insert("price".into(), MetadataValue::Integer(50));
    idx.index_record("r1", &meta);
    idx.deindex_record("r1", &meta);
    assert!(idx.get_string("category", "book").is_empty());
    assert!(idx.get_numeric_eq("price", 50.0).is_empty());
}

#[test]
fn index_clear_field_removes_all_entries() {
    let mut idx = MetadataIndex::new();
    let mut meta = Metadata::new();
    meta.insert("category".into(), MetadataValue::String("book".into()));
    idx.index_record("r1", &meta);
    assert_eq!(idx.string_field_count(), 1);
    idx.clear_field("category");
    assert_eq!(idx.string_field_count(), 0);
    assert!(idx.get_string("category", "book").is_empty());
}

#[test]
fn index_multiple_records_same_value() {
    let mut idx = MetadataIndex::new();
    for id in ["a", "b", "c"] {
        let mut meta = Metadata::new();
        meta.insert("tag".into(), MetadataValue::String("rust".into()));
        idx.index_record(id, &meta);
    }
    let ids = idx.get_string("tag", "rust");
    assert_eq!(ids.len(), 3);
}

#[test]
fn index_deindex_one_of_three_shared_value() {
    let mut idx = MetadataIndex::new();
    for id in ["a", "b", "c"] {
        let mut meta = Metadata::new();
        meta.insert("tag".into(), MetadataValue::String("rust".into()));
        idx.index_record(id, &meta);
    }
    let mut meta = Metadata::new();
    meta.insert("tag".into(), MetadataValue::String("rust".into()));
    idx.deindex_record("b", &meta);
    let ids = idx.get_string("tag", "rust");
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&"a".to_string()));
    assert!(ids.contains(&"c".to_string()));
}

#[test]
fn index_null_and_list_are_skipped() {
    let mut idx = MetadataIndex::new();
    let mut meta = Metadata::new();
    meta.insert("null_field".into(), MetadataValue::Null);
    meta.insert(
        "list_field".into(),
        MetadataValue::List(vec![MetadataValue::Integer(1)]),
    );
    idx.index_record("r1", &meta);
    assert_eq!(idx.string_field_count(), 0);
    assert_eq!(idx.numeric_field_count(), 0);
}

#[test]
fn index_mixed_field_types() {
    let mut idx = MetadataIndex::new();
    let mut meta = Metadata::new();
    meta.insert("name".into(), MetadataValue::String("alice".into()));
    meta.insert("age".into(), MetadataValue::Integer(30));
    meta.insert("score".into(), MetadataValue::Float(9.5));
    meta.insert("active".into(), MetadataValue::Bool(true));
    idx.index_record("r1", &meta);

    assert_eq!(idx.get_string("name", "alice"), vec!["r1"]);
    assert_eq!(idx.get_numeric_eq("age", 30.0), vec!["r1"]);
    assert_eq!(idx.get_numeric_eq("score", 9.5), vec!["r1"]);
    assert_eq!(idx.get_bool("active", true), vec!["r1"]);
}

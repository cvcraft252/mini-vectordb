// Typed metadata serialization and JSON round-trip tests.

use mini_vectordb::core::record::Record;
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

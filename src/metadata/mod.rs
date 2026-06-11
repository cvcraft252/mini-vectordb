use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A typed metadata value compatible with JSON round-trips.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MetadataValue {
    String(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    List(Vec<MetadataValue>),
    Null,
}

/// Typed metadata map used by Record.
///
/// ```
/// # use std::collections::HashMap;
/// # use mini_vectordb::metadata::{Metadata, MetadataValue};
/// let mut meta = Metadata::new();
/// meta.insert("price".into(), MetadataValue::Integer(42));
/// ```
pub type Metadata = HashMap<String, MetadataValue>;
pub mod index;

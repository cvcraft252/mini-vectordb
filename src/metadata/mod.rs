use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// `#[serde(untagged)]` tries variants in declaration order.
// String listed first for backward compat; Integer before Float
// to avoid silently converting `42` → `42.0`.
/// A typed metadata value compatible with JSON round-trips.
///
/// Every variant maps to a distinct JSON type, making the in-memory
/// representation match the on-disk format exactly. A number is stored
/// as a number, not as a quoted string — this enables range queries
/// and type-safe comparisons in downstream filter operations.
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

/// Alias for the typed metadata map used by Record.
///
/// ```
/// # use std::collections::HashMap;
/// # use mini_vectordb::metadata::{Metadata, MetadataValue};
/// let mut meta = Metadata::new();
/// meta.insert("price".into(), MetadataValue::Integer(42));
/// ```
pub type Metadata = HashMap<String, MetadataValue>;
/// Metadata index: string hashmap, numeric btreemap, bool pairs.
pub mod index;

// metadata/mod.rs
// Typed metadata values attached to vector records.
// Replaces flat HashMap<String, String> with a discriminated union
// so callers can query by type without parsing strings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A typed metadata value compatible with JSON round-trips.
///
/// Every variant maps to a distinct JSON type, making the in-memory
/// representation match the on-disk format exactly. A number is stored
/// as a number, not as a quoted string — this enables range queries
/// and type-safe comparisons in downstream filter operations.
///
/// # Notes
/// `#[serde(untagged)]` tells serde to try variants in declaration
/// order. `String` is listed first so existing data (all-strings flat
/// metadata) deserializes correctly. Integer comes before Float to
/// avoid converting `42` to `42.0`.
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
/// ```ignore
/// let mut meta = Metadata::new();
/// meta.insert("price".into(), MetadataValue::Integer(42));
/// ```
pub type Metadata = HashMap<String, MetadataValue>;
/// Metadata index: string hashmap, numeric btreemap, bool pairs.
pub mod index;

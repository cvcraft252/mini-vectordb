// core/record.rs
// The fundamental data unit: a float32 vector with string metadata.
// 2026-06-09: metadata is flat HashMap<String,String> for now.
//             A typed MetadataValue enum would enable range queries
//             on numeric fields (e.g. price > 100).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// One vector + its identity + optional key-value tags.
/// Callers own ID generation — we never auto-assign.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl Record {
    /// Create a record with no metadata.
    ///
    /// # Examples
    /// ```ignore
    /// let r = Record::new("doc-1", vec![1.0, 2.0, 3.0]);
    /// assert_eq!(r.id, "doc-1");
    /// assert!(r.metadata.is_empty());
    /// ```
    pub fn new(id: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            vector,
            metadata: HashMap::new(),
        }
    }

    /// Create a record with key-value metadata attached.
    ///
    /// Metadata is stored as `HashMap<String, String>` — flat text tags
    /// that survive JSON round-trips. For typed fields (int, float, bool),
    /// serialize to string on insert and parse on read.
    ///
    /// # Examples
    /// ```ignore
    /// let meta = HashMap::from([("category".into(), "book".into())]);
    /// let r = Record::with_metadata("doc-2", vec![1.0], meta);
    /// assert_eq!(r.metadata.get("category").unwrap(), "book");
    /// ```
    pub fn with_metadata(
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            id: id.into(),
            vector,
            metadata,
        }
    }
}

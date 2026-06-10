// core/record.rs
// The fundamental data unit: a float32 vector with typed metadata.

use serde::{Deserialize, Serialize};

use crate::metadata::Metadata;

/// One vector + its identity + optional key-value tags.
/// Callers own ID generation — we never auto-assign.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: Metadata,
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
            metadata: Metadata::new(),
        }
    }

    /// Create a record with typed metadata attached.
    ///
    /// # Examples
    /// ```ignore
    /// use mini_vectordb::metadata::MetadataValue;
    ///
    /// let mut meta = Metadata::new();
    /// meta.insert("price".into(), MetadataValue::Integer(42));
    /// meta.insert("label".into(), MetadataValue::String("book".into()));
    /// let r = Record::with_metadata("doc-2", vec![1.0], meta);
    /// ```
    pub fn with_metadata(id: impl Into<String>, vector: Vec<f32>, metadata: Metadata) -> Self {
        Self {
            id: id.into(),
            vector,
            metadata,
        }
    }
}

use serde::{Deserialize, Serialize};

use crate::metadata::Metadata;

/// One vector with an ID and optional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: Metadata,
}

impl Record {
    /// Creates a record with no metadata.
    ///
    /// # Examples
    /// ```rust
    /// # use mini_vectordb::core::record::Record;
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

    /// Creates a record with typed metadata.
    pub fn with_metadata(id: impl Into<String>, vector: Vec<f32>, metadata: Metadata) -> Self {
        Self {
            id: id.into(),
            vector,
            metadata,
        }
    }
}

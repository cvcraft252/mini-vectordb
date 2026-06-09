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
    pub fn new(id: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            vector,
            metadata: HashMap::new(),
        }
    }

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

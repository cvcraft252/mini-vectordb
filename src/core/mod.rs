// core/mod.rs
// Central error type and module root for the vector database.
// 2026-06-09: using thiserror instead of hand-rolled Display impls —
//             cuts boilerplate and gives us #[from] for free later.

pub mod metric;
pub mod record;

use thiserror::Error;

/// Every fallible operation in the library returns this.
/// We keep it flat (no nesting) because callers rarely need
/// to distinguish error categories beyond what the message says.
#[derive(Error, Debug)]
pub enum VectorDBError {
    #[error("Record not found: {0}")]
    NotFound(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Empty vector")]
    EmptyVector,

    /// Escape hatch for errors that don't deserve their own variant (yet).
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, VectorDBError>;

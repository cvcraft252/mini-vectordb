//! Core types: error handling, records, and distance metrics.
pub mod metric;
pub mod record;

use thiserror::Error;

/// Every fallible operation in the library returns this.
#[derive(Error, Debug)]
pub enum VectorDBError {
    #[error("Record not found: {0}")]
    NotFound(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Empty vector")]
    EmptyVector,

    #[error("{0}")]
    Other(String),
}

/// Shorthand for `std::result::Result<T, VectorDBError>`.
pub type Result<T> = std::result::Result<T, VectorDBError>;

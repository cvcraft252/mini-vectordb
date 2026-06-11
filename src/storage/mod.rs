pub mod bin_store;
pub mod json_store;
pub mod mmap_store;

use crate::core::Result;
use std::path::Path;

/// Writes a full dataset to durable storage and reads it back.
pub trait PersistentStorage {
    /// Serializes all records to the given file path.
    ///
    /// # Errors
    /// Returns `VectorDBError::Other` on I/O or serialization failure.
    fn save(&self, path: impl AsRef<Path>) -> Result<()>;

    /// Deserializes records from the given file path.
    ///
    /// # Errors
    /// Returns `VectorDBError::Other` on I/O or deserialization failure.
    fn load(path: impl AsRef<Path>) -> Result<Self>
    where
        Self: Sized;
}

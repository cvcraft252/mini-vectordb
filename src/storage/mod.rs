// storage/mod.rs
// Persistence backends for vector database durability.
// Two backends planned: JSON (human-readable, debug-friendly)
// and binary (compact, fast). Both implement PersistentStorage.

/// Binary file persistence.
pub mod bin_store;
/// JSON file persistence.
pub mod json_store;

use crate::core::Result;
use std::path::Path;

/// Write the full dataset to durable storage and read it back.
///
/// # Notes
/// Implementors are responsible for atomicity guarantees.
/// `JsonStorage` delegates to serde_json which provides best-effort
/// write safety — the file is truncated on write, but power loss
/// mid-write can leave a partially-written file.
pub trait PersistentStorage {
    /// Serialize all records to the given file path.
    ///
    /// # Arguments
    /// * `path` — File to write. Will be created or truncated.
    ///
    /// # Errors
    /// Returns `VectorDBError::Other` on I/O or serialization failure.
    fn save(&self, path: impl AsRef<Path>) -> Result<()>;

    /// Deserialize records from the given file path.
    ///
    /// # Arguments
    /// * `path` — File to read. Must exist and contain valid data.
    ///
    /// # Returns
    /// A new storage instance populated with the loaded records.
    ///
    /// # Errors
    /// Returns `VectorDBError::Other` on I/O or deserialization failure.
    fn load(path: impl AsRef<Path>) -> Result<Self>
    where
        Self: Sized;
}

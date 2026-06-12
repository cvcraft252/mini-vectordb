use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::core::Result;
use crate::core::VectorDBError;
use crate::core::record::Record;

#[derive(Serialize, Deserialize)]
struct RecordEntry {
    record: Record,
}

fn project_dir(name: &str) -> Result<PathBuf> {
    let base = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("mini-vectordb")
        .join("projects")
        .join(name);
    fs::create_dir_all(&base).map_err(|e| VectorDBError::Other(e.to_string()))?;
    Ok(base)
}

pub fn save_records(name: &str, records: &[Record]) -> Result<()> {
    let entries: Vec<RecordEntry> = records
        .iter()
        .map(|r| RecordEntry { record: r.clone() })
        .collect();
    let data = bincode::serialize(&entries).map_err(|e| VectorDBError::Other(e.to_string()))?;
    let path = project_dir(name)?.join("records.bincode");
    fs::write(path, data).map_err(|e| VectorDBError::Other(e.to_string()))?;
    Ok(())
}

pub fn load_records(name: &str) -> Result<Vec<Record>> {
    let path = project_dir(name)?.join("records.bincode");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let data = fs::read(&path).map_err(|e| VectorDBError::Other(e.to_string()))?;
    let entries: Vec<RecordEntry> =
        bincode::deserialize(&data).map_err(|e| VectorDBError::Other(e.to_string()))?;
    Ok(entries.into_iter().map(|e| e.record).collect())
}

pub fn project_exists(name: &str) -> bool {
    project_dir(name)
        .map(|p| p.join("records.bincode").exists())
        .unwrap_or(false)
}

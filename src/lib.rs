//! Vector database with auto-upgrade to HNSW and metadata filtering.

pub mod core;
pub mod embed;
pub mod index;
pub mod metadata;
pub mod query;
pub mod storage;

use std::sync::RwLock;

use crate::core::metric::DistanceMetric;
use crate::core::record::Record;
use crate::core::{Result, VectorDBError};
use crate::index::flat::FlatIndex;
use crate::index::hnsw::HnswIndex;
use crate::index::{Index, SearchResult};
use crate::metadata::index::MetadataIndex;
use crate::query::filter::parse_filter;

const UPGRADE_THRESHOLD: usize = 1000;

/// Thread-safe vector database. Uses FlatIndex below `UPGRADE_THRESHOLD`,
/// auto-upgrades to HNSW above it. Build metric matches `with_metric()`.
pub struct VectorDB {
    index: RwLock<Box<dyn Index>>,
    metadata_index: RwLock<MetadataIndex>,
    metric: DistanceMetric,
}

impl VectorDB {
    /// Creates an empty database with Cosine metric.
    pub fn new() -> Self {
        Self::with_metric(DistanceMetric::Cosine)
    }

    /// Creates an empty database with the given distance metric.
    pub fn with_metric(metric: DistanceMetric) -> Self {
        Self {
            index: RwLock::new(Box::new(FlatIndex::new())),
            metadata_index: RwLock::new(MetadataIndex::new()),
            metric,
        }
    }

    /// Best-effort upgrade to HNSW. On failure, keeps FlatIndex — the
    /// database is still functional, just slower.
    fn upgrade_to_hnsw(&self, index: &mut Box<dyn Index>) {
        let records = index.records();
        let mut hnsw = HnswIndex::with_params_and_metric(16, 200, self.metric);
        for r in records {
            if hnsw.insert(r).is_err() {
                // Upgrade failed: keep FlatIndex, it still works correctly.
                return;
            }
        }
        *index = Box::new(hnsw);
    }

    /// Inserts a record and syncs the metadata index. Triggers HNSW upgrade at threshold.
    pub fn insert(&self, record: Record) -> Result<()> {
        let id = record.id.clone();
        let meta = record.metadata.clone();

        {
            let mut index = self.index.write().expect("RwLock is never poisoned");
            index.insert(record)?;
            if index.len() == UPGRADE_THRESHOLD {
                self.upgrade_to_hnsw(&mut index);
            }
        }

        self.metadata_index
            .write()
            .expect("RwLock is never poisoned")
            .index_record(&id, &meta);

        Ok(())
    }

    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>> {
        self.index
            .read()
            .expect("RwLock is never poisoned")
            .search(query, top_k, metric)
    }

    pub fn search_batch(
        &self,
        queries: &[Vec<f32>],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<Vec<SearchResult>>> {
        self.index
            .read()
            .expect("RwLock is never poisoned")
            .search_batch(queries, top_k, metric)
    }

    pub fn search_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
        filter_expr: &str,
    ) -> Result<Vec<SearchResult>> {
        let filter = parse_filter(filter_expr)
            .map_err(|e| VectorDBError::Other(format!("filter parse error: {e}")))?;
        let index = self.index.read().expect("RwLock is never poisoned");
        let meta_idx = self
            .metadata_index
            .read()
            .expect("RwLock is never poisoned");
        crate::query::planner::execute_filtered_search(
            query, top_k, metric, &filter, &meta_idx, &**index,
        )
    }

    pub fn get(&self, id: &str) -> Result<Option<Record>> {
        self.index.read().expect("RwLock is never poisoned").get(id)
    }

    /// Deletes a record. Syncs the metadata index.
    pub fn delete(&self, id: &str) -> Result<()> {
        let meta = {
            let index = self.index.read().expect("RwLock is never poisoned");
            index.get(id)?.map(|r| r.metadata)
        };

        {
            let mut index = self.index.write().expect("RwLock is never poisoned");
            index.delete(id)?;
        }

        if let Some(meta) = meta {
            self.metadata_index
                .write()
                .expect("RwLock is never poisoned")
                .deindex_record(id, &meta);
        }

        Ok(())
    }

    /// Updates the vector of an existing record. Metadata is preserved.
    pub fn update(&self, id: &str, vector: Vec<f32>) -> Result<()> {
        let dim = vector.len();
        if dim == 0 {
            return Err(VectorDBError::EmptyVector);
        }

        let old_meta = {
            let index = self.index.read().expect("RwLock is never poisoned");
            let record = index.get(id)?;
            let record = record.ok_or_else(|| VectorDBError::NotFound(id.to_string()))?;
            if dim != record.vector.len() {
                return Err(VectorDBError::DimensionMismatch {
                    expected: record.vector.len(),
                    actual: dim,
                });
            }
            record.metadata.clone()
        };

        // Delete+re-insert under write lock so the index stays consistent.
        {
            let mut index = self.index.write().expect("RwLock is never poisoned");
            index.delete(id)?;
            index.insert(Record::with_metadata(
                id.to_string(),
                vector,
                old_meta.clone(),
            ))?;
        }

        // Re-sync metadata index even though values haven't changed — keeps
        // things correct if MetadataIndex ever adds ref-counting semantics.
        let mut meta_idx = self
            .metadata_index
            .write()
            .expect("RwLock is never poisoned");
        meta_idx.deindex_record(id, &old_meta);
        meta_idx.index_record(id, &old_meta);

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.index.read().expect("RwLock is never poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for VectorDB {
    fn default() -> Self {
        Self::new()
    }
}

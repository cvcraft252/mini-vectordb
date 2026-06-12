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

pub struct VectorDB {
    index: RwLock<Box<dyn Index>>,
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            index: RwLock::new(Box::new(FlatIndex::new())),
        }
    }

    fn upgrade_to_hnsw(&self, index: &mut Box<dyn Index>) -> Result<()> {
        let records = index.records();
        let mut hnsw = HnswIndex::new();
        for r in records {
            hnsw.insert(r)?;
        }
        *index = Box::new(hnsw);
        Ok(())
    }

    pub fn insert(&self, record: Record) -> Result<()> {
        let mut index = self.index.write().expect("RwLock is never poisoned");
        index.insert(record)?;
        if index.len() == UPGRADE_THRESHOLD {
            self.upgrade_to_hnsw(&mut index)?;
        }
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
        let records = index.records();
        let mut meta_idx = MetadataIndex::new();
        for r in &records {
            meta_idx.index_record(&r.id, &r.metadata);
        }
        crate::query::planner::execute_filtered_search(
            query, top_k, metric, &filter, &meta_idx, &**index,
        )
    }

    pub fn get(&self, id: &str) -> Result<Option<Record>> {
        self.index.read().expect("RwLock is never poisoned").get(id)
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        let mut index = self.index.write().expect("RwLock is never poisoned");
        index.delete(id)
    }

    pub fn update(&self, id: &str, vector: Vec<f32>) -> Result<()> {
        let mut index = self.index.write().expect("RwLock is never poisoned");
        let old = index.get(id)?;
        if let Some(mut record) = old {
            let dim = vector.len();
            if dim == 0 {
                return Err(VectorDBError::EmptyVector);
            }
            let stored_dim = record.vector.len();
            if dim != stored_dim {
                return Err(VectorDBError::DimensionMismatch {
                    expected: stored_dim,
                    actual: dim,
                });
            }
            record.vector = vector;
            index.delete(id)?;
            index.insert(record)
        } else {
            Err(VectorDBError::NotFound(id.to_string()))
        }
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

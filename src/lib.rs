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
use crate::embed::EmbedEngine;
use crate::index::flat::FlatIndex;
use crate::index::hnsw::HnswIndex;
use crate::index::{Index, SearchResult};
use crate::metadata::index::MetadataIndex;
use crate::metadata::{Metadata, MetadataValue};
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

    pub fn insert_batch(&self, records: &[Record]) -> Result<()> {
        for r in records {
            self.insert(r.clone())?;
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

    pub fn records(&self) -> Vec<Record> {
        self.index
            .read()
            .expect("RwLock is never poisoned")
            .records()
    }
}

impl Default for VectorDB {
    fn default() -> Self {
        Self::new()
    }
}

pub fn chunk_text(text: &str, chunk_size: usize) -> Vec<String> {
    text_splitter::TextSplitter::new(chunk_size)
        .chunks(text)
        .map(|c| c.to_string())
        .collect()
}

pub struct Engine {
    db: VectorDB,
    embedder: Box<dyn EmbedEngine>,
}

impl Engine {
    pub fn new(embedder: Box<dyn EmbedEngine>) -> Self {
        Self {
            db: VectorDB::new(),
            embedder,
        }
    }

    pub fn init(name: &str) -> Result<()> {
        if crate::storage::project_store::project_exists(name) {
            return Err(VectorDBError::Other(format!(
                "project '{name}' already exists"
            )));
        }
        crate::storage::project_store::save_records(name, &[])
    }

    pub fn load(name: &str) -> Result<Self> {
        let records = crate::storage::project_store::load_records(name)?;
        if records.is_empty() {
            return Err(VectorDBError::Other(format!(
                "project '{name}' is empty. Run 'add' first."
            )));
        }
        let db = VectorDB::new();
        db.insert_batch(&records)?;
        Ok(Self {
            db,
            embedder: Box::new(
                crate::embed::FastEmbedEngine::try_new()
                    .map_err(|e| VectorDBError::Other(e.to_string()))?,
            ),
        })
    }

    pub fn ingest(&self, path: &str, chunk_size: usize) -> Result<usize> {
        let text =
            std::fs::read_to_string(path).map_err(|e| VectorDBError::Other(e.to_string()))?;
        let chunks = chunk_text(&text, chunk_size);
        let embeddings = self
            .embedder
            .embed(&chunks)
            .map_err(|e| VectorDBError::Other(e.to_string()))?;

        let mut records = Vec::with_capacity(chunks.len());
        for (i, (chunk, vec)) in chunks.iter().zip(embeddings).enumerate() {
            let mut meta = Metadata::new();
            meta.insert("source".into(), MetadataValue::String(path.into()));
            meta.insert("text".into(), MetadataValue::String(chunk.clone()));
            records.push(Record::with_metadata(format!("{path}:{i}"), vec, meta));
        }
        self.db.insert_batch(&records)?;
        Ok(chunks.len())
    }

    pub fn query(&self, text: &str, top_k: usize) -> Result<Vec<String>> {
        let q_vec = self
            .embedder
            .embed(&[text.into()])
            .map_err(|e| VectorDBError::Other(e.to_string()))?;
        let query_vec = &q_vec[0];
        let results = self.db.search(query_vec, top_k, DistanceMetric::Cosine)?;
        let mut chunks = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for r in results {
            if seen.contains(&r.id) {
                continue;
            }
            seen.insert(r.id.clone());
            if let Ok(Some(rec)) = self.db.get(&r.id)
                && let Some(MetadataValue::String(t)) = rec.metadata.get("text")
            {
                chunks.push(t.clone());
            }
        }
        Ok(chunks)
    }

    pub fn query_filtered(&self, text: &str, filter: &str, top_k: usize) -> Result<Vec<String>> {
        let q_vec = self
            .embedder
            .embed(&[text.into()])
            .map_err(|e| VectorDBError::Other(e.to_string()))?;
        let results = self
            .db
            .search_filtered(&q_vec[0], top_k, DistanceMetric::Cosine, filter)?;
        let mut chunks = Vec::new();
        for r in results {
            if let Ok(Some(rec)) = self.db.get(&r.id)
                && let Some(MetadataValue::String(t)) = rec.metadata.get("text")
            {
                chunks.push(t.clone());
            }
        }
        Ok(chunks)
    }

    pub fn save(&self, name: &str) -> Result<()> {
        let records = self.db.records();
        crate::storage::project_store::save_records(name, &records)
    }

    pub fn len(&self) -> usize {
        self.db.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

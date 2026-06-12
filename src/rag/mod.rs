use std::collections::HashSet;
use std::fs;

use crate::VectorDB;
use crate::core::metric::DistanceMetric;
use crate::core::record::Record;
use crate::metadata::{Metadata, MetadataValue};

pub fn chunk_text(text: &str) -> Vec<String> {
    text_splitter::TextSplitter::new(1000)
        .chunks(text)
        .map(|c| c.to_string())
        .collect()
}

/// Abstract embedding backend. Implement for fastembed, ONNX, or API-based models.
pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
}

/// Local embedding via fastembed.
pub struct FastEmbedBackend {
    model: fastembed::TextEmbedding,
}

impl FastEmbedBackend {
    pub fn try_new() -> Result<Self, String> {
        let model =
            fastembed::TextEmbedding::try_new(Default::default()).map_err(|e| e.to_string())?;
        Ok(Self { model })
    }
}

impl Embedder for FastEmbedBackend {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let texts: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self.model.embed(texts, None).map_err(|e| e.to_string())?;
        Ok(embeddings.into_iter().collect())
    }
}

pub fn ingest_file(db: &VectorDB, path: &str, embedder: &dyn Embedder) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let chunks = chunk_text(&text);
    if chunks.is_empty() {
        return Ok(());
    }
    let embeddings = embedder.embed(&chunks)?;
    for (i, (chunk, vec)) in chunks.iter().zip(embeddings).enumerate() {
        let mut meta = Metadata::new();
        meta.insert("source".into(), MetadataValue::String(path.into()));
        meta.insert("text".into(), MetadataValue::String(chunk.clone()));
        db.insert(Record::with_metadata(format!("{path}:{i}"), vec, meta))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn query_semantic(
    db: &VectorDB,
    question: &str,
    embedder: &dyn Embedder,
) -> Result<Vec<String>, String> {
    let q_vec = embedder.embed(&[question.into()])?;
    let query_vec = q_vec.into_iter().next().ok_or("no embedding")?;
    let results = db
        .search(&query_vec, 3, DistanceMetric::Cosine)
        .map_err(|e| e.to_string())?;
    let mut seen = HashSet::new();
    let mut chunks = Vec::new();
    for r in results {
        if seen.contains(&r.id) {
            continue;
        }
        seen.insert(r.id.clone());
        if let Ok(Some(rec)) = db.get(&r.id)
            && let Some(MetadataValue::String(t)) = rec.metadata.get("text")
        {
            chunks.push(t.clone());
        }
    }
    Ok(chunks)
}

pub fn query_keywords(db: &VectorDB, question: &str, top_k: usize) -> Vec<String> {
    let words: Vec<&str> = question.split_whitespace().collect();
    let mut results = Vec::new();
    let mut seen = HashSet::new();
    for word in &words {
        let pattern = if word.len() > 2 { word } else { continue };
        if let Ok(rs) = db.search_filtered(
            &[0.0; 4],
            top_k,
            DistanceMetric::Euclidean,
            &format!("text LIKE \"%{pattern}%\""),
        ) {
            for r in rs {
                if seen.contains(&r.id) {
                    continue;
                }
                seen.insert(r.id.clone());
                if let Ok(Some(rec)) = db.get(&r.id)
                    && let Some(MetadataValue::String(t)) = rec.metadata.get("text")
                {
                    results.push(t.clone());
                }
            }
        }
    }
    results.truncate(top_k);
    results
}

pub fn demo(path: &str, questions: &[&str]) {
    let db = VectorDB::new();
    println!("Ingesting {path}...");
    match ingest_file(&db, path, &FastEmbedBackend::try_new().unwrap()) {
        Ok(()) => println!("  {} chunks indexed\n", db.len()),
        Err(e) => {
            eprintln!("Failed: {e}");
            return;
        }
    }
    for q in questions {
        println!("Q: {q}");
        match query_semantic(&db, q, &FastEmbedBackend::try_new().unwrap()) {
            Ok(results) => {
                for (i, r) in results.iter().enumerate() {
                    println!("  {}. {}", i + 1, r);
                }
            }
            Err(e) => eprintln!("Error: {e}"),
        }
        println!();
    }
}

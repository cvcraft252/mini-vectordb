use std::fs;

use crate::VectorDB;
use crate::core::metric::DistanceMetric;
use crate::core::record::Record;
use crate::metadata::{Metadata, MetadataValue};

/// Split text into overlapping chunks by character count.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk);
        if end >= chars.len() {
            break;
        }
        start += chunk_size - overlap;
    }
    chunks
}

/// Ingest a text file: chunk it and store in the database.
pub fn ingest_file(
    db: &VectorDB,
    path: &str,
    chunk_size: usize,
    overlap: usize,
) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let chunks = chunk_text(&text, chunk_size, overlap);
    for (i, chunk) in chunks.iter().enumerate() {
        let mut meta = Metadata::new();
        meta.insert("source".into(), MetadataValue::String(path.into()));
        meta.insert("chunk".into(), MetadataValue::Integer(i as i64));
        meta.insert("text".into(), MetadataValue::String(chunk.clone()));
        db.insert(Record::with_metadata(
            format!("{path}:{i}"),
            vec![i as f32; 4],
            meta,
        ))
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn query_keywords(db: &VectorDB, question: &str, top_k: usize) -> Vec<String> {
    let words: Vec<&str> = question.split_whitespace().collect();
    let mut results = Vec::new();
    let mut seen = std::collections::HashSet::new();
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

/// Interactive RAG demo: ingest a file and answer questions.
pub fn demo(path: &str, questions: &[&str]) {
    let db = VectorDB::new();
    println!("Ingesting {path}...");
    match ingest_file(&db, path, 500, 50) {
        Ok(()) => println!("  {} chunks indexed\n", db.len()),
        Err(e) => {
            eprintln!("Failed: {e}");
            return;
        }
    }
    for q in questions {
        println!("Q: {q}");
        let results = query_keywords(&db, q, 3);
        for (i, r) in results.iter().enumerate() {
            let preview: String = r.chars().take(80).collect();
            println!("  {}. {}", i + 1, preview);
        }
        println!();
    }
}

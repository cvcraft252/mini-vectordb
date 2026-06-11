use std::collections::HashSet;
use std::fs;

use crate::VectorDB;
use crate::core::metric::DistanceMetric;
use crate::core::record::Record;
use crate::metadata::{Metadata, MetadataValue};

/// Split text into chunks by sentence, grouping ~3 sentences per chunk.
pub fn chunk_text(text: &str) -> Vec<String> {
    text_splitter::TextSplitter::new(1000)
        .chunks(text)
        .map(|c| c.to_string())
        .collect()
}

pub fn ingest_file(db: &VectorDB, path: &str) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let chunks = chunk_text(&text);
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
    match ingest_file(&db, path) {
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

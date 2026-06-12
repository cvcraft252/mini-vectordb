use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use text_splitter::TextSplitter;

use mini_vectordb::VectorDB;
use mini_vectordb::core::metric::DistanceMetric;
use mini_vectordb::core::record::Record;
use mini_vectordb::embed::{EmbedEngine, FastEmbedEngine};
use mini_vectordb::metadata::{Metadata, MetadataValue};
use mini_vectordb::storage::vectra_store;

#[derive(Parser)]
#[command(name = "mini-vectordb")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Init {
        name: String,
    },
    Add {
        path: String,
        #[arg(long, default_value = "default")]
        name: String,
    },
    Query {
        text: String,
        #[arg(long, default_value = "default")]
        name: String,
        #[arg(long)]
        json: bool,
        #[arg(long, default_value = "3")]
        top_k: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Init { name } => {
            vectra_store::save_records(&name, &[]).map_err(|e| e.to_string())?;
            println!("Project '{name}' created.");
        }
        Command::Add { path, name } => {
            if !Path::new(&path).exists() {
                return Err(format!("file not found: {path}"));
            }
            let embedder = FastEmbedEngine::try_new()?;
            let text = fs::read_to_string(&path).map_err(|e| e.to_string())?;
            let chunks = chunk_text(&text);

            let mut existing = vectra_store::load_records(&name).map_err(|e| e.to_string())?;
            let db = VectorDB::new();
            for (r, _) in &existing {
                db.insert(r.clone()).map_err(|e| e.to_string())?;
            }

            let embeddings = embedder.embed(&chunks)?;
            for (i, (chunk, vec)) in chunks.iter().zip(embeddings).enumerate() {
                let mut meta = Metadata::new();
                meta.insert("source".into(), MetadataValue::String(path.clone()));
                meta.insert("text".into(), MetadataValue::String(chunk.clone()));
                let record = Record::with_metadata(format!("{path}:{i}"), vec, meta);
                db.insert(record.clone()).map_err(|e| e.to_string())?;
                existing.push((record, vec![]));
            }
            vectra_store::save_records(&name, &existing).map_err(|e| e.to_string())?;
            println!("Indexed {} chunks from {path}", chunks.len());
        }
        Command::Query {
            text,
            name,
            json,
            top_k,
        } => {
            let embedder = FastEmbedEngine::try_new()?;
            let records = vectra_store::load_records(&name).map_err(|e| e.to_string())?;
            if records.is_empty() {
                return Err(format!("no data in project '{name}'. Run 'add' first."));
            }
            let db = VectorDB::new();
            for (r, _) in &records {
                db.insert(r.clone()).map_err(|e| e.to_string())?;
            }

            let q_vec = embedder.embed(&[text])?;
            let query_vec = &q_vec[0];
            let results = db
                .search(query_vec, top_k, DistanceMetric::Cosine)
                .map_err(|e| e.to_string())?;

            if json {
                let out: Vec<_> = results
                    .iter()
                    .filter_map(|r| {
                        db.get(&r.id).ok()?.and_then(|rec| {
                            rec.metadata.get("text").and_then(|v| {
                                if let MetadataValue::String(s) = v {
                                    Some(s.clone())
                                } else {
                                    None
                                }
                            })
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&out).unwrap());
            } else {
                for (i, r) in results.iter().enumerate() {
                    if let Ok(Some(rec)) = db.get(&r.id)
                        && let Some(MetadataValue::String(t)) = rec.metadata.get("text")
                    {
                        println!("{}. {}", i + 1, t);
                    }
                }
            }
        }
    }
    Ok(())
}

fn chunk_text(text: &str) -> Vec<String> {
    TextSplitter::new(1000)
        .chunks(text)
        .map(|c| c.to_string())
        .collect()
}

use clap::{Parser, Subcommand};

use mini_vectordb::VectraEngine;
use mini_vectordb::embed::FastEmbedEngine;

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
        #[arg(long, default_value = "1000")]
        chunk_size: usize,
    },
    Query {
        text: String,
        #[arg(long, default_value = "default")]
        name: String,
        #[arg(long)]
        json: bool,
        #[arg(long, default_value = "3")]
        top_k: usize,
        #[arg(long)]
        filter: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Init { name } => {
            VectraEngine::init(&name).map_err(|e| e.to_string())?;
            println!("Project '{name}' created.");
        }
        Command::Add {
            path,
            name,
            chunk_size,
        } => {
            if !std::path::Path::new(&path).exists() {
                return Err(format!("file not found: {path}"));
            }
            let embedder = Box::new(FastEmbedEngine::try_new()?);
            let engine = VectraEngine::new(embedder);
            engine
                .ingest(&path, chunk_size)
                .map_err(|e| e.to_string())?;
            engine.save(&name).map_err(|e| e.to_string())?;
            println!("Indexed {} chunks from {path}", engine.len());
        }
        Command::Query {
            text,
            name,
            json,
            top_k,
            filter,
        } => {
            let engine = VectraEngine::load(&name).map_err(|e| e.to_string())?;
            let results = if let Some(f) = &filter {
                engine.query_filtered(&text, f, top_k)
            } else {
                engine.query(&text, top_k)
            }
            .map_err(|e| e.to_string())?;

            if json {
                println!("{}", serde_json::to_string_pretty(&results).unwrap());
            } else {
                for (i, r) in results.iter().enumerate() {
                    println!("{}. {}", i + 1, r);
                }
            }
        }
    }
    Ok(())
}

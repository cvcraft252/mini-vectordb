use std::env;

use mini_vectordb::{Engine, embed::try_new_auto};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: cargo run -- <file> [question]");
        return;
    }
    let path = &args[1];
    let question = if args.len() > 2 {
        args[2..].join(" ")
    } else {
        "what is this document about".into()
    };

    println!(
        "embed backend: {}",
        if env::var("EMBED_API_URL").is_ok() {
            "api"
        } else {
            "local"
        }
    );
    println!(
        "llm backend:   {}",
        if env::var("LLM_API_URL").is_ok() {
            "api"
        } else {
            "not set"
        }
    );

    let embedder = try_new_auto().expect("failed to create embedder");
    let mut engine = Engine::new(embedder);

    println!("ingesting {path}...");
    let n = engine.ingest(path, 1000).expect("ingest failed");
    println!("{n} chunks indexed\n");

    // show top 3 retrieved chunks before LLM
    let chunks = engine.query(&question, 3).expect("query failed");
    println!("--- retrieved chunks ---");
    for (i, c) in chunks.iter().enumerate() {
        let preview: String = c.chars().take(120).collect();
        println!("[{i}] {preview}...");
    }
    println!("--- llm answer ---\n");

    match engine.generate(&question) {
        Ok(answer) => println!("{answer}"),
        Err(e) => eprintln!("llm error: {e}"),
    }
}

use std::env;

use mini_vectordb::{Engine, embed::try_new_auto};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: cargo run -- <file> [question] [top_k]");
        return;
    }
    let path = &args[1];
    let question = if args.len() > 2 {
        args[2..].join(" ")
    } else {
        "what is this document about".into()
    };
    let top_k: usize = env::var("TOP_K")
        .unwrap_or_else(|_| "3".into())
        .parse()
        .unwrap_or(3);

    println!(
        "embed: {} | llm: {} | top_k: {top_k}",
        if env::var("EMBED_API_URL").is_ok() {
            "api"
        } else {
            "local"
        },
        if env::var("LLM_API_URL").is_ok() {
            "api"
        } else {
            "not set"
        },
    );

    let embedder = try_new_auto().expect("failed to create embedder");
    let mut engine = Engine::new(embedder);

    println!("ingesting {path}...");
    let n = engine.ingest(path, 1000).expect("ingest failed");
    println!("{n} chunks indexed\n");

    let chunks = engine.query(&question, top_k).expect("query failed");
    println!("--- top {top_k} chunks ---");
    for (i, c) in chunks.iter().enumerate() {
        let preview: String = c.chars().take(150).collect();
        println!("[{i}] {preview}...");
    }
    println!("--- answer ---\n");

    match engine.generate(&question) {
        Ok(answer) => println!("{answer}"),
        Err(e) => eprintln!("llm error: {e}"),
    }
}

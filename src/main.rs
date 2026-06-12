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

    let embedder = try_new_auto().expect("failed to create embedder");
    let engine = Engine::new(embedder);

    println!("ingesting {path}...");
    let n = engine.ingest(path, 1000).expect("ingest failed");
    println!("  {n} chunks indexed\n");

    println!("Q: {question}");
    let answer = engine.generate(&question).expect("generate failed");
    println!("\n{answer}");
}

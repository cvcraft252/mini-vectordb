use mini_vectordb::StorageFormat;
use mini_vectordb::rag::FastEmbedBackend;
use mini_vectordb::storage::PersistentStorage;
use mini_vectordb::storage::bin_store::BinStorage;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run -- <ingest|search> [args...]");
        return;
    }
    let embedder = FastEmbedBackend::try_new().expect("Failed to load embedding model");
    let db_path = "rag_db.bin";
    let db = if std::path::Path::new(db_path).exists() {
        let storage = BinStorage::load(db_path).unwrap();
        let db = mini_vectordb::VectorDB::new();
        for r in storage.into_records() {
            db.insert(r).unwrap();
        }
        db
    } else {
        mini_vectordb::VectorDB::with_persistence(db_path, StorageFormat::Binary)
    };

    match args[1].as_str() {
        "ingest" => {
            let path = args.get(2).expect("usage: cargo run -- ingest <file>");
            if let Err(e) = mini_vectordb::rag::ingest_file(&db, path, &embedder) {
                eprintln!("{e}");
                return;
            }
            println!("Indexed {} chunks from {path}", db.len());
        }
        "search" => {
            let question = &args[2..].join(" ");
            match mini_vectordb::rag::query_semantic(&db, question, &embedder) {
                Ok(results) => {
                    for (i, r) in results.iter().enumerate() {
                        println!("{}. {}", i + 1, r);
                    }
                }
                Err(e) => eprintln!("{e}"),
            }
        }
        _ => eprintln!("unknown command: {}", args[1]),
    }
}

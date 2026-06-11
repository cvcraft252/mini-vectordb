use mini_vectordb::storage::bin_store::BinStorage;
use mini_vectordb::storage::PersistentStorage;
use mini_vectordb::StorageFormat;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run -- <ingest|search> [args...]");
        return;
    }
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
            let text = std::fs::read_to_string(path).unwrap();
            let chunks = mini_vectordb::rag::chunk_text(&text);
            for (i, chunk) in chunks.iter().enumerate() {
                let mut meta = mini_vectordb::metadata::Metadata::new();
                meta.insert(
                    "source".into(),
                    mini_vectordb::metadata::MetadataValue::String(path.into()),
                );
                meta.insert(
                    "text".into(),
                    mini_vectordb::metadata::MetadataValue::String(chunk.clone()),
                );
                db.insert(mini_vectordb::core::record::Record::with_metadata(
                    format!("{path}:{i}"),
                    vec![i as f32; 4],
                    meta,
                ))
                .unwrap();
            }
            println!("Indexed {} chunks from {path}", chunks.len());
        }
        "search" => {
            let question = &args[2..].join(" ");
            let results = mini_vectordb::rag::query_keywords(&db, question, 3);
            for (i, r) in results.iter().enumerate() {
                println!("{}. {}", i + 1, r);
            }
        }
        _ => eprintln!("unknown command: {}", args[1]),
    }
}

# mini-vectordb

[![Rust](https://img.shields.io/badge/Rust-1.95%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A minimal vector database in Rust. Flat and HNSW indexing, metadata filtering,
JSON/binary/mmap persistence, and keyword retrieval.

## Usage

```rust
use mini_vectordb::core::metric::DistanceMetric;
use mini_vectordb::core::record::Record;
use mini_vectordb::metadata::{Metadata, MetadataValue};
use mini_vectordb::VectorDB;

let db = VectorDB::new();

// insert with typed metadata
let mut meta = Metadata::new();
meta.insert("cat".into(), MetadataValue::String("book".into()));
db.insert(Record::with_metadata("doc1", vec![0.1, 0.2, 0.3], meta))?;

// vector search (switches to HNSW at 1000 records)
let results = db.search(&[0.1, 0.2, 0.3], 5, DistanceMetric::Cosine)?;

// filtered search
let results = db.search_filtered(&[0.1, 0.2], 5, DistanceMetric::Euclidean, "cat = \"book\"")?;
```

## Usage

```rust
use mini_vectordb::{Engine, embed::FastEmbedEngine};

let embedder = Box::new(FastEmbedEngine::try_new().unwrap());
let engine = Engine::new(embedder);

// ingest documents
engine.ingest("./docs/business_faq.txt", 1000).unwrap();
engine.save("my_project").unwrap();

// semantic search
let chunks = engine.query("what is the return policy", 3).unwrap();
```

## Architecture

```
Engine ──→ VectorDB ──→ Index (FlatIndex | HnswIndex)
              │         │
              │         └──→ Storage (Json | Binary | Mmap)
              │
              └──→ Query Engine ──→ MetadataIndex
                       │
                       └──→ Filter parser → evaluate → set ops
```

## Development

```bash
cargo build
cargo test
cargo clippy
cargo fmt
```

## Acknowledgements

- [HNSW paper](https://arxiv.org/abs/1603.09320) — Malkov & Yashunin
- [Faiss](https://github.com/facebookresearch/faiss) — Meta's vector search library
- [Qdrant](https://github.com/qdrant/qdrant) — Rust vector database inspiration
- [pgvector](https://github.com/pgvector/pgvector) — Postgres vector extension

## License

[MIT](https://opensource.org/licenses/MIT)

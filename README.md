# mini-vectordb

[![Rust](https://img.shields.io/badge/Rust-1.95%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Star History Chart](https://api.star-history.com/svg?repos=cvcraft252/mini-vectordb&type=Date)](https://star-history.com/#cvcraft252/mini-vectordb&Date)

A minimal progressive vector database in Rust — from brute-force exact search to HNSW,
with metadata filtering, persistence, and a REST API. Each milestone is standalone.

## Features

### Vector Search
- [x] Cosine, Euclidean, DotProduct, Manhattan, Hamming distance metrics
- [x] `Distance` trait for generic metric dispatch
- [x] Brute-force flat index with exact nearest neighbor search
- [x] Precomputed L2 norms for fast cosine distance
- [x] Rayon-parallel batch search across multiple queries
- [x] HNSW approximate nearest neighbor index
- [x] HNSW graph serialization for fast restart
- [x] Adaptive index selection (flat → HNSW at 1000 records)

### CRUD Operations
- [x] Insert, get, delete, update, clear, len, is_empty
- [x] Thread-safe `RwLock<Box<dyn Index>>` wrapper
- [x] Concurrent reads with serialized writes
- [x] Dimension locking (rejects mixed-shape vectors)

### Persistence
- [x] `PersistentStorage` trait with save/load API
- [x] JSON persistence with pretty-printing and atomic write-then-rename
- [x] Binary persistence (MVDB magic header, raw f32 encoding)
- [x] Auto-persistence — configurable auto-save on every mutation
- [x] Memory-mapped vector storage for GB-scale datasets
- [x] Mmap + HNSW integration for million-scale on consumer hardware

### Metadata
- [x] Typed metadata schema (String, Integer, Float, Bool, List, Null)
- [x] BTreeMap indexes for range queries on numeric fields
- [x] HashMap indexes for exact string match

### Query Engine
- [x] Filter AST with expression parser (And, Or, Not, Eq, Gt, Lt, In, Like)
- [x] Inverted index with posting list intersection/union
- [x] Query planner with cost-based optimization (filter-first vs search-first)

### REST API
- [x] Axum-based HTTP server
- [x] CRUD endpoints (POST /insert, GET /get/:id, POST /search, DELETE /delete)
- [ ] Batch endpoints (POST /insert_batch, POST /search_batch)
- [ ] Health check and database statistics endpoints

### RAG Demo
- [ ] Document chunking and embedding pipeline
- [ ] Semantic retrieval with context assembly
- [ ] Interactive CLI or web demo with source attribution

## Quick Start

```rust
use mini_vectordb::core::metric::DistanceMetric;
use mini_vectordb::core::record::Record;
use mini_vectordb::VectorDB;

let db = VectorDB::new();

db.insert(Record::new("red",  vec![1.0, 0.0, 0.0])).unwrap();
db.insert(Record::new("green", vec![0.0, 1.0, 0.0])).unwrap();
db.insert(Record::new("blue",  vec![0.0, 0.0, 1.0])).unwrap();

let results = db.search(&[0.9, 0.1, 0.0], 2, DistanceMetric::Euclidean).unwrap();
for sr in &results {
    println!("{}  dist={:.4}", sr.id, sr.distance);
}
// red  dist=0.1414
// green  dist=1.2728
```

## CRUD Operations

```rust
db.insert(Record::new("doc1", vec![0.1, 0.2, 0.3]))?;
if let Some(record) = db.get("doc1")? {
    println!("{:?}", record.vector);
}
db.update("doc1", vec![0.5, 0.6, 0.7])?;
db.delete("doc1")?;
assert_eq!(db.len(), 0);
```

## Batch Search

```rust
let queries = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
let batch = db.search_batch(&queries, 1, DistanceMetric::Cosine)?;
```

## Persistence

```rust
use mini_vectordb::storage::json_store::JsonStorage;
use mini_vectordb::storage::bin_store::BinStorage;
use mini_vectordb::storage::PersistentStorage;

// JSON — human-readable, diffable
JsonStorage::from_records(records).save("db.json")?;
let loaded = JsonStorage::load("db.json")?;

// Binary — compact, fast
BinStorage::from_records(records).save("db.bin")?;
let loaded = BinStorage::load("db.bin")?;
```

### Auto-Persistence

```rust
use mini_vectordb::StorageFormat;

let db = VectorDB::with_persistence("db.bin", StorageFormat::Binary);
db.insert(Record::new("x", vec![1.0, 2.0])).unwrap();
// auto-saved — survives process restart
```

## HNSW Approximate Search

```rust
use mini_vectordb::index::hnsw::HnswIndex;
use mini_vectordb::index::Index;

let mut idx = HnswIndex::with_params(16, 200);
idx.insert(Record::new("a", vec![1.0, 2.0, 3.0])).unwrap();
let results = idx.search(&[1.0, 2.0, 3.0], 5, DistanceMetric::Cosine).unwrap();
```

## Filtered Search

```rust
use mini_vectordb::metadata::{Metadata, MetadataValue};

let db = VectorDB::new();
let mut meta = Metadata::new();
meta.insert("cat".into(), MetadataValue::String("book".into()));
db.insert(Record::with_metadata("r1", vec![1.0], meta)).unwrap();

let results = db.search_filtered(
    &[1.0], 5, DistanceMetric::Euclidean,
    "cat = \"book\"",
).unwrap();
```

## Architecture

```
src/
├── core/           Record, DistanceMetric, Distance trait, errors
├── index/          Index trait, FlatIndex, HnswIndex
├── storage/        PersistentStorage, JsonStorage, BinStorage, MmapStore
├── metadata/       MetadataValue enum, MetadataIndex
├── query/          Filter parser, evaluate_filter, query planner
└── lib.rs          VectorDB, StorageFormat, adaptive index selection
```

Trait-based design: `Index`, `Distance`, `PersistentStorage` — swap backends without
changing callers.

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

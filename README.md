# mini-vectordb

[![Rust](https://img.shields.io/badge/Rust-1.95%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A minimal vector database in Rust with HNSW search, metadata filtering, persistence, and a REST API.

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
- [x] Batch endpoints (POST /insert_batch, POST /search_batch)
- [x] Health check and database statistics endpoints

### RAG Demo
- [x] Document chunking and embedding pipeline
- [x] Semantic retrieval with context assembly
- [x] Interactive CLI or web demo with source attribution

## Usage

```rust
use mini_vectordb::core::metric::DistanceMetric;
use mini_vectordb::core::record::Record;
use mini_vectordb::VectorDB;

let db = VectorDB::new();
db.insert(Record::new("a", vec![1.0, 2.0, 3.0]))?;
let results = db.search(&[1.0, 2.0, 3.0], 5, DistanceMetric::Cosine)?;
```

## Architecture

```
src/
├── core/           Record, DistanceMetric, Distance trait, errors
├── index/          Index trait, FlatIndex, HnswIndex
├── storage/        PersistentStorage, JsonStorage, BinStorage, MmapStore
├── metadata/       MetadataValue enum, MetadataIndex
├── query/          Filter parser, evaluate_filter, query planner
├── api/            REST API server (insert, search, delete, batch, health)
├── rag/            Document chunking and keyword retrieval
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

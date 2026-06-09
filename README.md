# mini-vectordb

[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Cargo](https://img.shields.io/badge/Cargo-1.75%2B-green.svg)](https://doc.rust-lang.org/cargo/)

> A progressive vector database built from scratch in Rust. From brute-force search to HNSW, metadata filtering, mmap persistence, REST API, and RAG.

This project demonstrates how to build a modern vector database incrementally. Each phase adds a real-world feature, with clean trait-based architecture and comprehensive tests.

## Features by Phase

| Phase | Feature | Status | Description |
|-------|---------|--------|-------------|
| 1 | **Brute-Force Search** | Planned | Exact nearest neighbor with flat index |
| 2 | **Similarity Metrics** | Planned | Cosine, Euclidean, DotProduct, Manhattan + SIMD |
| 3 | **Persistence** | Planned | JSON and binary serialization |
| 4 | **Metadata** | Planned | Typed metadata with BTree/Hash indexes |
| 5 | **Inverted Filter** | Planned | Metadata pre-filtering before vector search |
| 6 | **HNSW** | Planned | Approximate nearest neighbor with graph index |
| 7 | **Mmap Storage** | Planned | GB-scale datasets without loading into RAM |
| 8 | **REST API** | Planned | Axum-based HTTP interface |
| 9 | **RAG Demo** | Planned | End-to-end document Q&A pipeline |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   CLI       │  │  REST API   │  │   RAG Demo          │  │
│  │  (main.rs)  │  │  (Axum)     │  │  (Pipeline)         │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼────────────────────┼─────────────┘
          │                │                    │
          └────────────────┴────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                      Query Engine                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Filter    │──▶│   Planner   │──▶│   Index (Flat/    │  │
│  │  (Inverted) │  │  (Cost-based)│  │   HNSW)             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                      Storage Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Memory    │  │   JSON      │  │   Mmap              │  │
│  │  (HashMap)  │  │  (File)     │  │  (Zero-copy)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- Linux / macOS / Windows (WSL2 recommended)

### Build

```bash
git clone https://github.com/cvcraftz252/mini-vectordb.git
cd mini-vectordb
cargo build --release
```

### CLI Usage

```bash
# Insert vectors
cargo run -- insert --id doc1 --vector "0.1,0.2,0.3,..."

# Search top-5 similar vectors
cargo run -- search --query "0.1,0.2,0.3,..." --top-k 5

# Search with metadata filter
cargo run -- search --query "0.1,0.2,..." --filter "category='book' AND price<50"

# Persist database
cargo run -- save --path ./mydb.json

# Load and serve REST API
cargo run -- serve --port 3000
```

### REST API (Phase 8+)

```bash
# Start server
cargo run --bin mini-vectordb-server -- --port 3000

# Insert
curl -X POST http://localhost:3000/insert \
  -H "Content-Type: application/json" \
  -d '{"id":"doc1","vector":[0.1,0.2,0.3],"metadata":{"category":"book"}}'

# Search
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{"query":[0.1,0.2,0.3],"top_k":5,"filter":"category=\"book\""}'
```

## Project Structure

```
.
├── Cargo.toml
├── README.md
├── benches/
│   └── search_benchmark.rs
├── src/
│   ├── lib.rs              # Public API
│   ├── main.rs             # CLI entry
│   ├── core/
│   │   ├── mod.rs          # Core types and error definitions
│   │   ├── record.rs       # Vector record with metadata
│   │   └── metric.rs       # Distance computation traits
│   ├── index/
│   │   ├── mod.rs          # Index trait definitions
│   │   ├── flat.rs         # Brute-force exact search
│   │   └── hnsw.rs         # HNSW approximate search
│   ├── storage/
│   │   ├── mod.rs          # Storage trait definitions
│   │   ├── memory.rs       # In-memory backend
│   │   ├── json_store.rs   # JSON persistence
│   │   └── mmap_store.rs   # Memory-mapped storage
│   ├── query/
│   │   ├── mod.rs          # Query engine
│   │   ├── filter.rs       # Metadata filtering (inverted index)
│   │   └── planner.rs      # Query planning and optimization
│   ├── metadata/
│   │   └── mod.rs          # Metadata schema and indexes
│   ├── api/
│   │   ├── mod.rs          # API module
│   │   └── rest.rs         # Axum HTTP handlers
│   └── rag/
│       └── mod.rs          # RAG pipeline demo
└── tests/
    ├── test_flat_index.rs
    ├── test_metric.rs
    ├── test_persistence.rs
    ├── test_metadata.rs
    ├── test_filter.rs
    ├── test_hnsw.rs
    ├── test_mmap.rs
    └── integration_tests.rs
```

## Performance Roadmap

| Scale | Index | Search Latency | Memory | Phase |
|-------|-------|----------------|--------|-------|
| 1K | Flat | < 1ms | ~2MB | 1 |
| 10K | Flat + Rayon | < 5ms | ~20MB | 2 |
| 100K | HNSW | < 10ms | ~200MB | 6 |
| 1M | HNSW + Mmap | < 20ms | ~500MB | 7 |

*Benchmarks on Apple M3 / AMD Ryzen 7, 128-dim vectors, top-10 search.*

## Development Phases

- [ ] Phase 1: Brute-force flat index with CRUD
- [ ] Phase 2: Multiple distance metrics + batch search
- [ ] Phase 3: JSON and binary persistence
- [ ] Phase 4: Typed metadata with field indexes
- [ ] Phase 5: Inverted index for metadata filtering
- [ ] Phase 6: HNSW approximate nearest neighbor
- [ ] Phase 7: Memory-mapped storage for large datasets
- [ ] Phase 8: REST API with Axum
- [ ] Phase 9: RAG document Q&A demo

## Testing

```bash
# Run all tests
cargo test

# Run benchmarks
cargo bench

# Run specific phase tests
cargo test --test test_hnsw
cargo test --test test_filter
```

## Contributing

This is an educational project. While the scope is intentionally progressive, suggestions for architecture improvements are welcome. Please open an issue before major changes.

## Acknowledgements

- [HNSW paper](https://arxiv.org/abs/1603.09320) — Malkov & Yashunin
- [Faiss](https://github.com/facebookresearch/faiss) — Meta's vector search library
- [Qdrant](https://github.com/qdrant/qdrant) — Rust vector database inspiration
- [pgvector](https://github.com/pgvector/pgvector) — Postgres vector extension

## License

[MIT](LICENSE)

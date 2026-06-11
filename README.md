# mini-vectordb

[![Rust](https://img.shields.io/badge/Rust-1.95%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A minimal vector database in Rust. Flat and HNSW indexing, metadata filtering,
JSON/binary/mmap persistence, and a REST API.

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

## REST API

Start the server:

```bash
cargo run
```

| Method | Path | Body | Response |
|--------|------|------|----------|
| `POST` | `/insert` | `{"id":"...","vector":[...]}` | `ok` |
| `GET` | `/get/:id` | — | `{"id":"...","vector":[...]}` |
| `POST` | `/search` | `{"vector":[...],"top_k":N,"metric":"euclidean\|cosine\|dotproduct\|manhattan\|hamming"}` | `[{"id":"...","distance":...}]` |
| `DELETE` | `/delete/:id` | — | `ok` |
| `POST` | `/update` | `{"id":"...","vector":[...]}` | `ok` |
| `POST` | `/insert_batch` | `{"records":[{...},...]}` | `inserted N` |
| `POST` | `/search_batch` | `{"queries":[{...},...]}` | `[[{...}],...]` |
| `GET` | `/health` | — | `ok` |
| `GET` | `/stats` | — | `{"vector_count":N}` |

Example:

```bash
$ curl -X POST localhost:3000/insert \
    -H 'Content-Type: application/json' \
    -d '{"id":"doc1","vector":[1,2,3]}'
ok

$ curl -X POST localhost:3000/search \
    -H 'Content-Type: application/json' \
    -d '{"vector":[1,2,3],"top_k":3,"metric":"cosine"}'
[{"id":"doc1","distance":0.0}]

$ curl localhost:3000/health
ok
$ curl localhost:3000/stats
{"vector_count":1}
```

## Architecture

```
HTTP ──→ VectorDB ──→ Index (FlatIndex | HnswIndex)
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

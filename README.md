# mini-vectordb

[![Rust](https://img.shields.io/badge/Rust-1.95%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A minimal vector database in Rust. Flat and HNSW indexing, metadata filtering,
semantic search with local embeddings.

## Usage

```toml
[dependencies]
mini-vectordb = { git = "https://github.com/cvcraft252/mini-vectordb" }
```

Local model (fastembed):

```rust
use mini_vectordb::{Engine, embed::FastEmbedEngine};

let mut engine = Engine::new(Box::new(FastEmbedEngine::try_new().unwrap()));
engine.ingest("./doc.pdf", 1000).unwrap();
let answer = engine.generate("what is this document about").unwrap();
```

Ollama or any OpenAI-compatible API:

```rust
use mini_vectordb::{Engine, embed::try_new_auto};

// set env: EMBED_API_URL, EMBED_API_KEY, EMBED_MODEL_NAME,
//          LLM_API_URL, LLM_API_KEY, LLM_MODEL_NAME
let mut engine = Engine::new(try_new_auto().unwrap());
engine.ingest("./doc.pdf", 1000).unwrap();
let answer = engine.generate("what is multi-head attention").unwrap();
```

## Architecture

```
Engine ──→ VectorDB ──→ Index (FlatIndex | HnswIndex)
              │         │
              │         └──→ Embed (FastEmbedEngine)
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

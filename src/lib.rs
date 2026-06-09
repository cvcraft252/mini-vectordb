// lib.rs
// mini-vectordb public API root. Each module tree is exposed
// individually so callers can import specific types without
// pulling in everything (e.g. `use mini_vectordb::core::metric`).
// 2026-06-09: core + index done. storage/query/metadata later.

pub mod core;
pub mod index;

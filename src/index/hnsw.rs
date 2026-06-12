//! HNSW graph-based index for approximate nearest neighbor search.
//! Reference: Malkov & Yashunin (2016), arXiv:1603.09320.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::{fs, io};

use crate::core::metric::DistanceMetric;
use crate::core::record::Record;
use crate::core::{Result, VectorDBError};
use crate::index::{Index, SearchResult};
use crate::metadata::Metadata;

const HNSW_MAGIC: u32 = 0x484E5357;
const HNSW_VERSION: u32 = 2;

#[derive(PartialEq)]
struct Candidate {
    distance: f32,
    index: usize,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.index.cmp(&other.index))
    }
}

fn get_random_level(m: usize) -> usize {
    let ml = 1.0 / (m as f64).ln();
    let r: f64 = rand::random();
    (-r.ln() * ml).floor() as usize
}

struct Node {
    id: String,
    vector: Vec<f32>,
    metadata: Metadata,
    layers: Vec<Vec<usize>>,
}

/// HNSW graph index. Higher layers are sparser highways for fast traversal,
/// layer 0 is exhaustive. Build metric determines edge connectivity.
pub struct HnswIndex {
    nodes: Vec<Option<Node>>,
    entry_point: Option<usize>,
    m: usize,
    m0: usize,
    ef: usize,
    max_level: usize,
    dimension: usize,
    count: usize,
    id_to_idx: HashMap<String, usize>,
    /// Distance metric used for graph construction and pruning.
    metric: DistanceMetric,
}

impl HnswIndex {
    /// Creates an empty HNSW index with default parameters and Euclidean metric.
    pub fn new() -> Self {
        Self::with_params_and_metric(DEFAULT_M, DEFAULT_EF, DistanceMetric::Euclidean)
    }

    /// Creates an HNSW index with custom M and ef, defaulting to Euclidean metric.
    pub fn with_params(m: usize, ef: usize) -> Self {
        Self::with_params_and_metric(m, ef, DistanceMetric::Euclidean)
    }

    /// Creates an HNSW index with custom M, ef, and distance metric.
    pub fn with_params_and_metric(m: usize, ef: usize, metric: DistanceMetric) -> Self {
        Self {
            nodes: vec![None],
            entry_point: None,
            m,
            m0: m.saturating_mul(2),
            ef,
            max_level: 0,
            dimension: 0,
            count: 0,
            id_to_idx: HashMap::new(),
            metric,
        }
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }
}

/// Default parameters: M=16, ef=200.
const DEFAULT_M: usize = 16;
const DEFAULT_EF: usize = 200;

impl Default for HnswIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl HnswIndex {
    fn search_layer(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        layer: usize,
        metric: DistanceMetric,
    ) -> Vec<(usize, f32)> {
        let mut visited = vec![false; self.nodes.len()];
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        let node = self.nodes[entry].as_ref().expect("entry node exists");
        let dist = metric.compute(query, &node.vector);

        candidates.push(Reverse(Candidate {
            distance: dist,
            index: entry,
        }));
        results.push(Candidate {
            distance: dist,
            index: entry,
        });
        visited[entry] = true;

        while let Some(Reverse(current)) = candidates.pop() {
            if let Some(worst) = results.peek()
                && current.distance > worst.distance
                && results.len() >= ef
            {
                break;
            }

            let current_node = self.nodes[current.index].as_ref().expect("node exists");
            let neighbors = match current_node.layers.get(layer) {
                Some(n) => n.as_slice(),
                None => continue,
            };

            for &neighbor in neighbors {
                if visited[neighbor] {
                    continue;
                }
                visited[neighbor] = true;

                let neighbor_node = self.nodes[neighbor].as_ref().expect("neighbor exists");
                let d = metric.compute(query, &neighbor_node.vector);

                if match results.peek() {
                    Some(w) if results.len() >= ef => d < w.distance,
                    _ => true,
                } {
                    candidates.push(Reverse(Candidate {
                        distance: d,
                        index: neighbor,
                    }));
                    results.push(Candidate {
                        distance: d,
                        index: neighbor,
                    });
                    while results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .map(|c| (c.index, c.distance))
            .collect()
    }

    fn select_neighbors(&self, candidates: &[(usize, f32)], m: usize) -> Vec<usize> {
        candidates.iter().take(m).map(|(idx, _)| *idx).collect()
    }

    /// Saves the graph to a binary file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let tmp = path.with_extension("tmp");
        let file = fs::File::create(&tmp)
            .map_err(|e| VectorDBError::Other(format!("create temp: {e}")))?;
        let mut w = BufWriter::new(file);

        write_u32(&mut w, HNSW_MAGIC).map_err(|e| VectorDBError::Other(format!("magic: {e}")))?;
        write_u32(&mut w, HNSW_VERSION)
            .map_err(|e| VectorDBError::Other(format!("version: {e}")))?;
        write_u32(&mut w, self.m as u32).map_err(|e| VectorDBError::Other(format!("m: {e}")))?;
        write_u32(&mut w, self.m0 as u32).map_err(|e| VectorDBError::Other(format!("m0: {e}")))?;
        write_u32(&mut w, self.ef as u32).map_err(|e| VectorDBError::Other(format!("ef: {e}")))?;
        write_u32(&mut w, self.max_level as u32)
            .map_err(|e| VectorDBError::Other(format!("max_level: {e}")))?;
        write_u32(&mut w, self.dimension as u32)
            .map_err(|e| VectorDBError::Other(format!("dim: {e}")))?;
        write_u32(&mut w, self.count as u32)
            .map_err(|e| VectorDBError::Other(format!("count: {e}")))?;
        let ep = self
            .entry_point
            .map(|i| i as u64)
            .unwrap_or(u64::from(u32::MAX));
        write_u64(&mut w, ep).map_err(|e| VectorDBError::Other(format!("entry: {e}")))?;
        write_u8(&mut w, metric_to_byte(self.metric))
            .map_err(|e| VectorDBError::Other(format!("metric: {e}")))?;

        for node in &self.nodes {
            match node {
                Some(n) => {
                    w.write_all(&[1u8])
                        .map_err(|e| VectorDBError::Other(format!("present: {e}")))?;
                    write_str(&mut w, &n.id)
                        .map_err(|e| VectorDBError::Other(format!("id: {e}")))?;
                    write_vector(&mut w, &n.vector)
                        .map_err(|e| VectorDBError::Other(format!("vector: {e}")))?;
                    write_u32(&mut w, n.layers.len() as u32)
                        .map_err(|e| VectorDBError::Other(format!("layers: {e}")))?;
                    for layer in &n.layers {
                        write_u32(&mut w, layer.len() as u32)
                            .map_err(|e| VectorDBError::Other(format!("neighbors: {e}")))?;
                        for &n_idx in layer {
                            write_u32(&mut w, n_idx as u32)
                                .map_err(|e| VectorDBError::Other(format!("neighbor: {e}")))?;
                        }
                    }
                    write_metadata(&mut w, &n.metadata)
                        .map_err(|e| VectorDBError::Other(format!("metadata: {e}")))?;
                }
                None => {
                    w.write_all(&[0u8])
                        .map_err(|e| VectorDBError::Other(format!("absent: {e}")))?;
                }
            }
        }

        w.into_inner()
            .map_err(|_| VectorDBError::Other("flush failed".into()))?;
        fs::rename(&tmp, path).map_err(|e| VectorDBError::Other(format!("rename: {e}")))?;
        Ok(())
    }

    /// Loads a saved HNSW graph. Only v2 format (with metric + metadata) is supported.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = fs::File::open(path.as_ref())
            .map_err(|e| VectorDBError::Other(format!("open: {e}")))?;
        let mut r = BufReader::new(file);

        let magic = read_u32(&mut r).map_err(|e| VectorDBError::Other(format!("magic: {e}")))?;
        if magic != HNSW_MAGIC {
            return Err(VectorDBError::Other(format!("bad magic: 0x{magic:08X}")));
        }
        let version =
            read_u32(&mut r).map_err(|e| VectorDBError::Other(format!("version: {e}")))?;
        if version != HNSW_VERSION {
            return Err(VectorDBError::Other(format!(
                "unsupported version {version}, expected {HNSW_VERSION}"
            )));
        }

        let m = read_u32(&mut r).map_err(|e| VectorDBError::Other(format!("m: {e}")))? as usize;
        let m0 = read_u32(&mut r).map_err(|e| VectorDBError::Other(format!("m0: {e}")))? as usize;
        let ef = read_u32(&mut r).map_err(|e| VectorDBError::Other(format!("ef: {e}")))? as usize;
        let max_level =
            read_u32(&mut r).map_err(|e| VectorDBError::Other(format!("max_level: {e}")))? as usize;
        let dimension =
            read_u32(&mut r).map_err(|e| VectorDBError::Other(format!("dim: {e}")))? as usize;
        let count =
            read_u32(&mut r).map_err(|e| VectorDBError::Other(format!("count: {e}")))? as usize;
        let ep_raw = read_u64(&mut r).map_err(|e| VectorDBError::Other(format!("entry: {e}")))?;
        let entry_point = if ep_raw == u64::from(u32::MAX) {
            None
        } else {
            Some(ep_raw as usize)
        };
        let metric_byte =
            read_u8(&mut r).map_err(|e| VectorDBError::Other(format!("metric: {e}")))?;
        let metric = byte_to_metric(metric_byte)
            .ok_or_else(|| VectorDBError::Other(format!("unknown metric byte: {metric_byte}")))?;

        let mut nodes = Vec::new();
        let mut id_to_idx = HashMap::new();

        loop {
            let mut buf = [0u8; 1];
            match r.read_exact(&mut buf) {
                Ok(_) => {}
                Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    return Err(VectorDBError::Other(format!("read present: {e}")));
                }
            }
            if buf[0] == 0 {
                nodes.push(None);
                continue;
            }

            let id = read_string(&mut r).map_err(|e| VectorDBError::Other(format!("id: {e}")))?;
            let vector = read_vector(&mut r, dimension)
                .map_err(|e| VectorDBError::Other(format!("vector: {e}")))?;
            let num_layers = read_u32(&mut r)
                .map_err(|e| VectorDBError::Other(format!("layers: {e}")))?
                as usize;

            let mut layers = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                let num_neighbors = read_u32(&mut r)
                    .map_err(|e| VectorDBError::Other(format!("neighbors: {e}")))?
                    as usize;
                let mut neighbors = Vec::with_capacity(num_neighbors);
                for _ in 0..num_neighbors {
                    let n_idx = read_u32(&mut r)
                        .map_err(|e| VectorDBError::Other(format!("neighbor: {e}")))?
                        as usize;
                    neighbors.push(n_idx);
                }
                layers.push(neighbors);
            }

            let metadata = read_metadata(&mut r)
                .map_err(|e| VectorDBError::Other(format!("metadata: {e}")))?;

            let node_idx = nodes.len();
            id_to_idx.insert(id.clone(), node_idx);
            nodes.push(Some(Node {
                id,
                vector,
                metadata,
                layers,
            }));
        }

        Ok(Self {
            nodes,
            entry_point,
            m,
            m0,
            ef,
            max_level,
            dimension,
            count,
            id_to_idx,
            metric,
        })
    }
}

impl Index for HnswIndex {
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>> {
        let Some(entry) = self.entry_point else {
            return Ok(Vec::new());
        };
        if top_k == 0 {
            return Ok(Vec::new());
        }

        if query.len() != self.dimension {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let mut current = entry;
        for layer in (1..=self.max_level).rev() {
            let layer_results = self.search_layer(query, current, 1, layer, metric);
            current = layer_results[0].0;
        }

        let ef = self.ef.max(top_k);
        let candidates = self.search_layer(query, current, ef, 0, metric);
        let k = top_k.min(candidates.len());

        let results: Vec<SearchResult> = candidates[..k]
            .iter()
            .map(|(idx, dist)| {
                let node = self.nodes[*idx].as_ref().expect("node exists");
                SearchResult {
                    id: node.id.clone(),
                    distance: *dist,
                    record: Record::with_metadata(
                        &node.id,
                        node.vector.clone(),
                        node.metadata.clone(),
                    ),
                }
            })
            .collect();

        Ok(results)
    }

    fn search_batch(
        &self,
        queries: &[Vec<f32>],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<Vec<SearchResult>>> {
        for q in queries.iter() {
            if q.len() != self.dimension {
                return Err(VectorDBError::DimensionMismatch {
                    expected: self.dimension,
                    actual: q.len(),
                });
            }
        }
        use rayon::prelude::*;
        queries
            .par_iter()
            .map(|q| self.search(q, top_k, metric))
            .collect::<Result<Vec<_>>>()
    }

    fn insert(&mut self, record: Record) -> Result<()> {
        let dim = record.vector.len();
        if dim == 0 {
            return Err(VectorDBError::EmptyVector);
        }
        if self.dimension == 0 {
            self.dimension = dim;
        } else if dim != self.dimension {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dimension,
                actual: dim,
            });
        }

        let node_idx = self.nodes.len();
        let level = get_random_level(self.m);

        if self.entry_point.is_none() {
            self.entry_point = Some(node_idx);
            self.max_level = level;
        } else if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(node_idx);
        }

        let mut layers = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            layers.push(Vec::new());
        }

        let node = Node {
            id: record.id.clone(),
            vector: record.vector.clone(),
            metadata: record.metadata.clone(),
            layers,
        };
        self.nodes.push(Some(node));
        self.id_to_idx.insert(record.id.clone(), node_idx);
        self.count += 1;

        let Some(entry) = self.entry_point else {
            return Ok(());
        };

        // Use the configured metric, not a hardcoded default.
        let metric = self.metric;
        let mut current = entry;
        for l in ((level + 1)..=self.max_level).rev() {
            let layer_results = self.search_layer(&record.vector, current, 1, l, metric);
            current = layer_results[0].0;
        }

        for l in (0..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(&record.vector, current, self.ef, l, metric);
            let m_max = if l == 0 { self.m0 } else { self.m };
            let neighbors = self.select_neighbors(&candidates, m_max);

            let node_ref = self.nodes[node_idx].as_mut().expect("new node exists");
            node_ref.layers[l] = neighbors.clone();

            let mut prunes: Vec<(usize, Vec<usize>)> = Vec::new();
            for &n_idx in &neighbors {
                let neighbor = self.nodes[n_idx].as_mut().expect("neighbor exists");
                if !neighbor.layers[l].contains(&node_idx) {
                    neighbor.layers[l].push(node_idx);
                }
                if neighbor.layers[l].len() > m_max {
                    prunes.push((n_idx, neighbor.layers[l].clone()));
                }
            }

            for (n_idx, conns) in prunes {
                let n_vec = self.nodes[n_idx]
                    .as_ref()
                    .expect("neighbor exists")
                    .vector
                    .clone();
                let mut scored: Vec<(usize, f32)> = conns
                    .iter()
                    .map(|&idx| {
                        let v = &self.nodes[idx].as_ref().expect("conn exists").vector;
                        let d = metric.compute(&n_vec, v);
                        (idx, d)
                    })
                    .collect();
                scored.sort_by(|a, b| a.1.total_cmp(&b.1));
                self.nodes[n_idx].as_mut().expect("neighbor exists").layers[l] =
                    scored.iter().take(m_max).map(|(i, _)| *i).collect();
            }

            if !candidates.is_empty() {
                current = candidates[0].0;
            }
        }

        Ok(())
    }

    fn delete(&mut self, id: &str) -> Result<()> {
        if let Some(&idx) = self.id_to_idx.get(id) {
            // keep entry point as tombstone to avoid re-wiring the graph
            self.nodes[idx] = None;
            self.count = self.count.saturating_sub(1);
            self.id_to_idx.remove(id);
            if self.count == 0 {
                self.entry_point = None;
                self.dimension = 0;
            }
        }
        Ok(())
    }

    fn get(&self, id: &str) -> Result<Option<Record>> {
        Ok(self
            .id_to_idx
            .get(id)
            .and_then(|&idx| self.nodes[idx].as_ref())
            .map(|n| Record::with_metadata(&n.id, n.vector.clone(), n.metadata.clone())))
    }

    fn len(&self) -> usize {
        self.count
    }

    fn records(&self) -> Vec<Record> {
        self.nodes
            .iter()
            .filter_map(|n| n.as_ref())
            .map(|n| Record::with_metadata(&n.id, n.vector.clone(), n.metadata.clone()))
            .collect()
    }
}

// ── Metric byte encoding ──

fn metric_to_byte(m: DistanceMetric) -> u8 {
    match m {
        DistanceMetric::Cosine => 0,
        DistanceMetric::Euclidean => 1,
        DistanceMetric::DotProduct => 2,
        DistanceMetric::Manhattan => 3,
        DistanceMetric::Hamming => 4,
    }
}

fn byte_to_metric(v: u8) -> Option<DistanceMetric> {
    match v {
        0 => Some(DistanceMetric::Cosine),
        1 => Some(DistanceMetric::Euclidean),
        2 => Some(DistanceMetric::DotProduct),
        3 => Some(DistanceMetric::Manhattan),
        4 => Some(DistanceMetric::Hamming),
        _ => None,
    }
}

// ── Binary serialization helpers ──

fn write_u8(w: &mut impl Write, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_u64(w: &mut impl Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn write_str(w: &mut impl Write, s: &str) -> io::Result<()> {
    write_u32(w, s.len() as u32)?;
    w.write_all(s.as_bytes())
}

fn read_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u32(r)? as usize;
    let mut buf = vec![0u8; len];
    if len > 0 {
        r.read_exact(&mut buf)?;
    }
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn write_vector(w: &mut impl Write, v: &[f32]) -> io::Result<()> {
    // cast &[f32] to &[u8] — safe: f32 has no padding bits
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
    w.write_all(bytes)
}

fn read_vector(r: &mut impl Read, dim: usize) -> io::Result<Vec<f32>> {
    let n = dim * 4;
    let mut buf = vec![0u8; n];
    if n > 0 {
        r.read_exact(&mut buf)?;
    }
    let ptr = buf.as_mut_ptr() as *mut f32;
    let len = dim;
    let cap = dim;
    std::mem::forget(buf);
    // safe: n = dim * 4, all f32 bit patterns are valid
    Ok(unsafe { Vec::from_raw_parts(ptr, len, cap) })
}

/// Serializes metadata as length-prefixed bincode bytes.
fn write_metadata(w: &mut impl Write, meta: &Metadata) -> io::Result<()> {
    let bytes =
        bincode::serialize(meta).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    write_u32(w, bytes.len() as u32)?;
    w.write_all(&bytes)
}

/// Deserializes length-prefixed bincode metadata bytes.
fn read_metadata(r: &mut impl Read) -> io::Result<Metadata> {
    let len = read_u32(r)? as usize;
    let mut buf = vec![0u8; len];
    if len > 0 {
        r.read_exact(&mut buf)?;
    }
    bincode::deserialize(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

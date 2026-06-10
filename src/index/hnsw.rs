//! HNSW graph-based index for approximate nearest neighbor search.
//! Implemented from Malkov & Yashunin (2016): "Efficient and robust
//! approximate nearest neighbor search using Hierarchical Navigable
//! Small World graphs" (arXiv:1603.09320).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::{fs, io};

use crate::core::metric::DistanceMetric;
use crate::core::record::Record;
use crate::core::{Result, VectorDBError};
use crate::index::{Index, SearchResult};

/// HNSW binary format magic number: "HNSW" in ASCII.
const HNSW_MAGIC: u32 = 0x484E5357;
/// Current binary format version.
const HNSW_VERSION: u32 = 1;

// Candidate wrapper that orders by distance (smallest first) for BinaryHeap (max-heap).
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

/// Returns a random level via exponential decay.
/// `ml = 1/ln(M)` gives roughly `1/M` of nodes reaching level ≥ 1.
fn get_random_level(m: usize) -> usize {
    let ml = 1.0 / (m as f64).ln();
    let r: f64 = rand::random();
    (-r.ln() * ml).floor() as usize
}

/// One node in the HNSW graph.
struct Node {
    id: String,
    vector: Vec<f32>,
    /// Adjacency lists indexed by layer: layers[0] is the base layer.
    layers: Vec<Vec<usize>>,
}

/// HNSW graph-based index for approximate nearest neighbor search.
///
/// Multi-layer structure: higher layers are sparser highways enabling
/// logarithmic search time. All nodes exist in layer 0.
///
/// # Parameters
/// - `m` — max connections per node per layer (default 16). Layer 0 uses 2*m.
/// - `ef` — candidate list size during construction and search (default 200).
///
/// Higher `ef` values improve recall at the cost of speed.
pub struct HnswIndex {
    /// All nodes indexed by internal usize ID.
    nodes: Vec<Option<Node>>,
    /// Node index of the top-layer entry point.
    entry_point: Option<usize>,
    /// Max connections per node for layers > 0.
    m: usize,
    /// Max connections per node for layer 0 (2*m).
    m0: usize,
    /// Candidate list size during both construction and search.
    ef: usize,
    /// Current maximum level in the graph.
    max_level: usize,
    /// Vector dimension, locked on first insert.
    dimension: usize,
    /// Number of live (non-deleted) nodes.
    count: usize,
    /// Maps record ID to node index for O(1) get/delete.
    id_to_idx: HashMap<String, usize>,
}

impl HnswIndex {
    /// Create an empty HNSW index with default parameters.
    ///
    /// ```
    /// # use mini_vectordb::index::hnsw::HnswIndex;
    /// # use mini_vectordb::index::Index;
    /// let idx = HnswIndex::new();
    /// assert!(idx.is_empty());
    /// ```
    pub fn new() -> Self {
        Self::with_params(DEFAULT_M, DEFAULT_EF)
    }

    /// Create an HNSW index with custom M and ef parameters.
    pub fn with_params(m: usize, ef: usize) -> Self {
        Self {
            // reserve index 0 as sentinel
            nodes: vec![None],
            entry_point: None,
            m,
            m0: m.saturating_mul(2),
            ef,
            max_level: 0,
            dimension: 0,
            count: 0,
            id_to_idx: HashMap::new(),
        }
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
    /// Greedy search at a single layer, returning up to `ef` nearest neighbors.
    ///
    /// Maintains a candidate set (closest unevaluated) and a result set
    /// (best so far). Stops when the closest unevaluated candidate is
    /// farther than the farthest result.
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

        // Candidates is a min-heap via Reverse: smallest distance first.
        candidates.push(Reverse(Candidate {
            distance: dist,
            index: entry,
        }));
        // Results is a max-heap: largest distance first (for easy pruning).
        results.push(Candidate {
            distance: dist,
            index: entry,
        });
        visited[entry] = true;

        while let Some(Reverse(current)) = candidates.pop() {
            // stop: current candidate is farther than the worst result
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

                let add = match results.peek() {
                    Some(w) if results.len() >= ef => d < w.distance,
                    _ => true,
                };

                if add {
                    let cand = Candidate {
                        distance: d,
                        index: neighbor,
                    };
                    candidates.push(Reverse(cand));
                    results.push(Candidate {
                        distance: d,
                        index: neighbor,
                    });
                    // keep results size bounded to ef
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

    /// Select up to `m` nearest indices from a distance-sorted candidate list.
    fn select_neighbors(&self, candidates: &[(usize, f32)], m: usize) -> Vec<usize> {
        candidates.iter().take(m).map(|(idx, _)| *idx).collect()
    }

    /// Save the graph structure to a binary file for fast restart.
    ///
    /// Writes the header (magic, version, parameters), then each node's
    /// id, vector, and adjacency lists. Tombstones (deleted nodes) are
    /// skipped.
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

    /// Load a previously saved HNSW graph from a binary file.
    ///
    /// Restores the exact same graph structure — search results
    /// are identical before and after save/load.
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
        if version > HNSW_VERSION {
            return Err(VectorDBError::Other(format!(
                "unsupported version {version}"
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

            let node_idx = nodes.len();
            id_to_idx.insert(id.clone(), node_idx);
            nodes.push(Some(Node { id, vector, layers }));
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
        // handle empty index or zero top_k
        let Some(entry) = self.entry_point else {
            return Ok(Vec::new());
        };
        if top_k == 0 {
            return Ok(Vec::new());
        }

        // validate dimension
        if query.len() != self.dimension {
            return Err(VectorDBError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        // descend from top layer to layer 1, greedy single-path
        let mut current = entry;
        for layer in (1..=self.max_level).rev() {
            let layer_results = self.search_layer(query, current, 1, layer, metric);
            current = layer_results[0].0;
        }

        // exhaustive search at layer 0
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
                    record: Record::new(&node.id, node.vector.clone()),
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

        // update max_level and entry_point if this node goes higher
        if self.entry_point.is_none() {
            self.entry_point = Some(node_idx);
            self.max_level = level;
        } else if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(node_idx);
        }

        // build node layers
        let mut layers = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            layers.push(Vec::new());
        }

        let node = Node {
            id: record.id.clone(),
            vector: record.vector.clone(),
            layers,
        };
        self.nodes.push(Some(node));
        self.id_to_idx.insert(record.id.clone(), node_idx);
        self.count += 1;

        // connect the new node into the graph
        let Some(entry) = self.entry_point else {
            return Ok(());
        };

        let mut current = entry;
        // descend from top layer down to level+1
        for l in ((level + 1)..=self.max_level).rev() {
            let layer_results =
                self.search_layer(&record.vector, current, 1, l, metric_for_insert());
            current = layer_results[0].0;
        }

        // connect at each layer from level down to 0
        for l in (0..=level.min(self.max_level)).rev() {
            let candidates =
                self.search_layer(&record.vector, current, self.ef, l, metric_for_insert());
            let m_max = if l == 0 { self.m0 } else { self.m };
            let neighbors = self.select_neighbors(&candidates, m_max);

            // add bidirectional connections
            let node_ref = self.nodes[node_idx].as_mut().expect("new node exists");
            node_ref.layers[l] = neighbors.clone();

            // collect neighbors that need pruning before mutating them
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

            // score and prune outside the mutable borrow
            for (n_idx, conns) in prunes {
                // score neighbor's connections by distance from the neighbor's
                // own vector (not the new node's vector)
                let n_vec = self.nodes[n_idx]
                    .as_ref()
                    .expect("neighbor exists")
                    .vector
                    .clone();
                let mut scored: Vec<(usize, f32)> = conns
                    .iter()
                    .map(|&idx| {
                        let v = &self.nodes[idx].as_ref().expect("conn exists").vector;
                        let d = metric_for_insert().compute(&n_vec, v);
                        (idx, d)
                    })
                    .collect();
                scored.sort_by(|a, b| a.1.total_cmp(&b.1));
                self.nodes[n_idx].as_mut().expect("neighbor exists").layers[l] =
                    scored.iter().take(m_max).map(|(i, _)| *i).collect();
            }

            // update entry for next layer down
            if !candidates.is_empty() {
                current = candidates[0].0;
            }
        }

        Ok(())
    }

    fn delete(&mut self, id: &str) -> Result<()> {
        if let Some(&idx) = self.id_to_idx.get(id) {
            if idx == self.entry_point.unwrap_or(0) {
                // to simplify, mark the node as deleted in-place
                // and keep it in the graph as a tombstone
                self.nodes[idx] = None;
                self.count = self.count.saturating_sub(1);
                self.id_to_idx.remove(id);
                if self.count == 0 {
                    self.entry_point = None;
                    self.dimension = 0;
                }
            } else {
                self.nodes[idx] = None;
                self.count = self.count.saturating_sub(1);
                self.id_to_idx.remove(id);
            }
        }
        Ok(())
    }

    fn get(&self, id: &str) -> Result<Option<Record>> {
        Ok(self
            .id_to_idx
            .get(id)
            .and_then(|&idx| self.nodes[idx].as_ref())
            .map(|n| Record::new(&n.id, n.vector.clone())))
    }

    fn len(&self) -> usize {
        self.count
    }

    fn records(&self) -> Vec<Record> {
        self.nodes
            .iter()
            .filter_map(|n| n.as_ref())
            .map(|n| Record::new(&n.id, n.vector.clone()))
            .collect()
    }
}

/// Default metric used during construction: Euclidean provides stable
/// spacial layout. Query-time metric is caller-specified.
fn metric_for_insert() -> DistanceMetric {
    DistanceMetric::Euclidean
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

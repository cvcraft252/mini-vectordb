//! HNSW graph-based index for approximate nearest neighbor search.
//! Implemented from Malkov & Yashunin (2016): "Efficient and robust
//! approximate nearest neighbor search using Hierarchical Navigable
//! Small World graphs" (arXiv:1603.09320).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::core::metric::DistanceMetric;
use crate::core::record::Record;
use crate::core::{Result, VectorDBError};
use crate::index::{Index, SearchResult};

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

use std::collections::{BTreeMap, HashMap};
use std::ops::Bound;

use crate::metadata::MetadataValue;

/// Strongly-ordered f64 wrapper. Uses `total_cmp` so every bit pattern
/// (including NaN, subnormals) has a defined position. The ordering is
/// deterministic but not numerically meaningful for NaN — callers should
/// avoid storing NaN in numeric metadata fields.
#[derive(Debug, Clone, Copy)]
struct OrderedF64(f64);

impl PartialEq for OrderedF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == std::cmp::Ordering::Equal
    }
}

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// String and numeric indexes are independent — a single record's
// metadata fields appear in both if they contain both types.
// Insert is O(log N) for numeric fields, O(1) amortized for strings.
// The index stores only record IDs, not the records themselves.
/// In-memory index for metadata-driven record filtering.
pub struct MetadataIndex {
    /// String equality index: field_name -> (value -> [ids]).
    string_index: HashMap<String, HashMap<String, Vec<String>>>,
    /// Numeric range index: field_name -> BTreeMap<value -> [ids]>.
    /// Multiple records can share the same numeric value.
    numeric_index: HashMap<String, BTreeMap<OrderedF64, Vec<String>>>,
    /// Bool index: field_name -> (true_ids, false_ids).
    bool_index: HashMap<String, (Vec<String>, Vec<String>)>,
}

impl MetadataIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            string_index: HashMap::new(),
            numeric_index: HashMap::new(),
            bool_index: HashMap::new(),
        }
    }

    /// Index all indexable metadata fields from a record.
    ///
    /// Called after insert. Skips Null and List values (lists are not
    /// indexable at the top level; nested indexable values could be
    /// an optimization later).
    pub fn index_record(&mut self, id: &str, metadata: &HashMap<String, MetadataValue>) {
        for (field, value) in metadata {
            match value {
                MetadataValue::String(s) => {
                    self.string_index
                        .entry(field.clone())
                        .or_default()
                        .entry(s.clone())
                        .or_default()
                        .push(id.to_string());
                }
                MetadataValue::Integer(i) => {
                    self.numeric_index
                        .entry(field.clone())
                        .or_default()
                        .entry(OrderedF64(*i as f64))
                        .or_default()
                        .push(id.to_string());
                }
                MetadataValue::Float(f) => {
                    self.numeric_index
                        .entry(field.clone())
                        .or_default()
                        .entry(OrderedF64(*f))
                        .or_default()
                        .push(id.to_string());
                }
                MetadataValue::Bool(b) => {
                    let entry = self.bool_index.entry(field.clone()).or_default();
                    if *b {
                        entry.0.push(id.to_string());
                    } else {
                        entry.1.push(id.to_string());
                    }
                }
                _ => {}
            }
        }
    }

    /// Remove a record's metadata entries from all indexes.
    ///
    /// Called before delete or during update. Linear scan through
    /// indexed values to find and remove the ID — acceptable while
    /// per-field value cardinality is moderate.
    pub fn deindex_record(&mut self, id: &str, metadata: &HashMap<String, MetadataValue>) {
        for (field, value) in metadata {
            match value {
                MetadataValue::String(s) => {
                    if let Some(values) = self.string_index.get_mut(field)
                        && let Some(ids) = values.get_mut(s.as_str())
                    {
                        ids.retain(|i| i != id);
                        if ids.is_empty() {
                            values.remove(s.as_str());
                        }
                    }
                }
                MetadataValue::Integer(i) => {
                    if let Some(tree) = self.numeric_index.get_mut(field) {
                        let key = OrderedF64(*i as f64);
                        if let Some(ids) = tree.get_mut(&key) {
                            ids.retain(|i| i != id);
                            if ids.is_empty() {
                                tree.remove(&key);
                            }
                        }
                    }
                }
                MetadataValue::Float(f) => {
                    if let Some(tree) = self.numeric_index.get_mut(field) {
                        let key = OrderedF64(*f);
                        if let Some(ids) = tree.get_mut(&key) {
                            ids.retain(|i| i != id);
                            if ids.is_empty() {
                                tree.remove(&key);
                            }
                        }
                    }
                }
                MetadataValue::Bool(b) => {
                    if let Some((true_ids, false_ids)) = self.bool_index.get_mut(field) {
                        let target = if *b { true_ids } else { false_ids };
                        target.retain(|i| i != id);
                    }
                }
                _ => {}
            }
        }
    }

    /// Exact-match lookup for a string field. O(1).
    pub fn get_string(&self, field: &str, value: &str) -> Vec<String> {
        self.string_index
            .get(field)
            .and_then(|m| m.get(value))
            .cloned()
            .unwrap_or_default()
    }

    /// Get all IDs where numeric field == value. O(log N).
    pub fn get_numeric_eq(&self, field: &str, value: f64) -> Vec<String> {
        self.numeric_index
            .get(field)
            .and_then(|tree| tree.get(&OrderedF64(value)))
            .cloned()
            .unwrap_or_default()
    }

    /// Get all IDs where numeric field > value. O(log N + K).
    ///
    /// Uses `Bound::Excluded` on the lower bound for strict greater-than.
    /// The upper bound is unbounded.
    pub fn get_numeric_gt(&self, field: &str, value: f64) -> Vec<String> {
        let Some(tree) = self.numeric_index.get(field) else {
            return Vec::new();
        };
        tree.range((Bound::Excluded(OrderedF64(value)), Bound::Unbounded))
            .flat_map(|(_, ids)| ids.iter().cloned())
            .collect()
    }

    /// Get all IDs where numeric field < value. O(log N + K).
    ///
    /// Uses `Bound::Excluded` on the upper bound for strict less-than.
    /// The lower bound is unbounded.
    pub fn get_numeric_lt(&self, field: &str, value: f64) -> Vec<String> {
        let Some(tree) = self.numeric_index.get(field) else {
            return Vec::new();
        };
        tree.range((Bound::Unbounded, Bound::Excluded(OrderedF64(value))))
            .flat_map(|(_, ids)| ids.iter().cloned())
            .collect()
    }

    /// Get all IDs where bool field matches the given value. O(1).
    pub fn get_bool(&self, field: &str, value: bool) -> Vec<String> {
        self.bool_index
            .get(field)
            .map(|(t, f)| if value { t.clone() } else { f.clone() })
            .unwrap_or_default()
    }

    /// Remove all entries for a given field from every index. O(1).
    pub fn clear_field(&mut self, field: &str) {
        self.string_index.remove(field);
        self.numeric_index.remove(field);
        self.bool_index.remove(field);
    }

    /// Number of indexed string fields.
    pub fn string_field_count(&self) -> usize {
        self.string_index.len()
    }

    /// Number of indexed numeric fields.
    pub fn numeric_field_count(&self) -> usize {
        self.numeric_index.len()
    }
}

impl Default for MetadataIndex {
    fn default() -> Self {
        Self::new()
    }
}

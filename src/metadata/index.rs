use std::collections::{BTreeMap, HashMap};
use std::ops::Bound;

use crate::metadata::MetadataValue;

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

/// In-memory index for metadata-driven record filtering.
pub struct MetadataIndex {
    string_index: HashMap<String, HashMap<String, Vec<String>>>,
    numeric_index: HashMap<String, BTreeMap<OrderedF64, Vec<String>>>,
    bool_index: HashMap<String, (Vec<String>, Vec<String>)>,
}

impl MetadataIndex {
    pub fn new() -> Self {
        Self {
            string_index: HashMap::new(),
            numeric_index: HashMap::new(),
            bool_index: HashMap::new(),
        }
    }

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

    pub fn get_string(&self, field: &str, value: &str) -> Vec<String> {
        self.string_index
            .get(field)
            .and_then(|m| m.get(value))
            .cloned()
            .unwrap_or_default()
    }

    pub fn get_numeric_eq(&self, field: &str, value: f64) -> Vec<String> {
        self.numeric_index
            .get(field)
            .and_then(|tree| tree.get(&OrderedF64(value)))
            .cloned()
            .unwrap_or_default()
    }

    pub fn get_numeric_gt(&self, field: &str, value: f64) -> Vec<String> {
        let Some(tree) = self.numeric_index.get(field) else {
            return Vec::new();
        };
        tree.range((Bound::Excluded(OrderedF64(value)), Bound::Unbounded))
            .flat_map(|(_, ids)| ids.iter().cloned())
            .collect()
    }

    pub fn get_numeric_lt(&self, field: &str, value: f64) -> Vec<String> {
        let Some(tree) = self.numeric_index.get(field) else {
            return Vec::new();
        };
        tree.range((Bound::Unbounded, Bound::Excluded(OrderedF64(value))))
            .flat_map(|(_, ids)| ids.iter().cloned())
            .collect()
    }

    pub fn get_bool(&self, field: &str, value: bool) -> Vec<String> {
        self.bool_index
            .get(field)
            .map(|(t, f)| if value { t.clone() } else { f.clone() })
            .unwrap_or_default()
    }

    pub fn get_string_prefix(&self, field: &str, prefix: &str) -> Vec<String> {
        let Some(map) = self.string_index.get(field) else {
            return Vec::new();
        };
        map.iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .flat_map(|(_, ids)| ids.iter().cloned())
            .collect()
    }

    pub(crate) fn get_field_ids(&self, field: &str) -> Vec<String> {
        let mut ids: Vec<String> = Vec::new();
        if let Some(map) = self.string_index.get(field) {
            ids.extend(map.values().flat_map(|v| v.iter().cloned()));
        }
        if let Some(tree) = self.numeric_index.get(field) {
            ids.extend(tree.values().flat_map(|v| v.iter().cloned()));
        }
        if let Some((t, f)) = self.bool_index.get(field) {
            ids.extend(t.iter().cloned());
            ids.extend(f.iter().cloned());
        }
        ids
    }

    pub fn clear_field(&mut self, field: &str) {
        self.string_index.remove(field);
        self.numeric_index.remove(field);
        self.bool_index.remove(field);
    }

    pub fn string_field_count(&self) -> usize {
        self.string_index.len()
    }

    pub fn get_string_contains(&self, field: &str, substr: &str) -> Vec<String> {
        let Some(map) = self.string_index.get(field) else {
            return Vec::new();
        };
        map.iter()
            .filter(|(k, _)| k.contains(substr))
            .flat_map(|(_, ids)| ids.iter().cloned())
            .collect()
    }

    pub fn get_string_suffix(&self, field: &str, suffix: &str) -> Vec<String> {
        let Some(map) = self.string_index.get(field) else {
            return Vec::new();
        };
        map.iter()
            .filter(|(k, _)| k.ends_with(suffix))
            .flat_map(|(_, ids)| ids.iter().cloned())
            .collect()
    }

    pub fn numeric_field_count(&self) -> usize {
        self.numeric_index.len()
    }
}

impl Default for MetadataIndex {
    fn default() -> Self {
        Self::new()
    }
}

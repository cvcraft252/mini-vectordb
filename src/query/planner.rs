//! Combines metadata filtering with vector search for end-to-end queries.

use std::collections::HashSet;

use crate::core::Result;
use crate::core::metric::DistanceMetric;
use crate::index::{Index, SearchResult};
use crate::metadata::index::MetadataIndex;
use crate::query::filter::{Filter, evaluate_filter};

/// Search with pre-filtering: evaluate the filter, then compute distances
/// only for matching records.
///
/// When the filter eliminates most records, this is faster than a full
/// scan. When the filter matches nearly everything, the overhead of
/// building a temporary MetadataIndex makes it slightly slower than
/// calling [`Index::search`] directly.
pub fn execute_filtered_search(
    query: &[f32],
    top_k: usize,
    metric: DistanceMetric,
    filter: &Filter,
    metadata_index: &MetadataIndex,
    vector_index: &dyn Index,
) -> Result<Vec<SearchResult>> {
    let candidate_ids: HashSet<String> = evaluate_filter(filter, metadata_index)
        .into_iter()
        .collect();

    if candidate_ids.is_empty() {
        return Ok(Vec::new());
    }

    let mut scored: Vec<SearchResult> = Vec::with_capacity(candidate_ids.len());
    for id in &candidate_ids {
        if let Some(record) = vector_index.get(id)? {
            let dist = metric.compute(query, &record.vector);
            scored.push(SearchResult {
                id: id.clone(),
                distance: dist,
                record,
            });
        }
    }

    scored.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(top_k.min(scored.len()));
    Ok(scored)
}

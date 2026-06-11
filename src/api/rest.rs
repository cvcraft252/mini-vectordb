//! Axum-based HTTP handlers for vector CRUD and search.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{Json, Router, routing};
use serde::{Deserialize, Serialize};

use crate::VectorDB;
use crate::core::metric::DistanceMetric;
use crate::core::record::Record;

#[derive(Clone)]
struct AppState {
    db: Arc<VectorDB>,
}

#[derive(Deserialize)]
struct InsertRequest {
    id: String,
    vector: Vec<f32>,
}

#[derive(Deserialize)]
struct SearchRequest {
    vector: Vec<f32>,
    top_k: usize,
    #[serde(default = "default_metric")]
    metric: String,
}

#[derive(Deserialize)]
struct UpdateRequest {
    id: String,
    vector: Vec<f32>,
}

fn default_metric() -> String {
    "euclidean".into()
}

#[derive(Serialize)]
struct SearchResultResponse {
    id: String,
    distance: f32,
}

#[derive(Serialize)]
struct RecordResponse {
    id: String,
    vector: Vec<f32>,
}

fn parse_metric(s: &str) -> Result<DistanceMetric, String> {
    match s.to_lowercase().as_str() {
        "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
        "cosine" => Ok(DistanceMetric::Cosine),
        "dotproduct" | "dot" => Ok(DistanceMetric::DotProduct),
        "manhattan" | "l1" => Ok(DistanceMetric::Manhattan),
        "hamming" => Ok(DistanceMetric::Hamming),
        _ => Err(format!("unknown metric: {s}")),
    }
}

async fn insert(
    State(state): State<AppState>,
    Json(req): Json<InsertRequest>,
) -> impl IntoResponse {
    match state.db.insert(Record::new(&req.id, req.vector)) {
        Ok(()) => (StatusCode::OK, "ok").into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

async fn get(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    match state.db.get(&id) {
        Ok(Some(r)) => Json(RecordResponse {
            id: r.id,
            vector: r.vector,
        })
        .into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, "not found").into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
    let metric = match parse_metric(&req.metric) {
        Ok(m) => m,
        Err(e) => return (StatusCode::BAD_REQUEST, e).into_response(),
    };
    match state.db.search(&req.vector, req.top_k, metric) {
        Ok(results) => {
            let out: Vec<SearchResultResponse> = results
                .iter()
                .map(|r| SearchResultResponse {
                    id: r.id.clone(),
                    distance: r.distance,
                })
                .collect();
            Json(out).into_response()
        }
        Err(e) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

async fn delete(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    match state.db.delete(&id) {
        Ok(()) => (StatusCode::OK, "ok").into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn update(
    State(state): State<AppState>,
    Json(req): Json<UpdateRequest>,
) -> impl IntoResponse {
    match state.db.update(&req.id, req.vector) {
        Ok(()) => (StatusCode::OK, "ok").into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

/// Starts the HTTP server on the given port.
pub async fn serve(db: VectorDB, port: u16) {
    let state = AppState { db: Arc::new(db) };
    let app = Router::new()
        .route("/insert", routing::post(insert))
        .route("/get/:id", routing::get(get))
        .route("/search", routing::post(search))
        .route("/delete/:id", routing::delete(delete))
        .route("/update", routing::post(update))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    println!("Listening on http://{addr}");
    axum::serve(listener, app).await.unwrap();
}

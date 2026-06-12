use std::path::PathBuf;

use serde::Deserialize;

use crate::core::Result;

pub trait EmbedEngine: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub fn try_new_auto() -> std::result::Result<Box<dyn EmbedEngine>, String> {
    if std::env::var("EMBED_API_URL").is_ok() {
        Ok(Box::new(ApiEmbedEngine::try_new()?))
    } else {
        Ok(Box::new(FastEmbedEngine::try_new()?))
    }
}

#[derive(Deserialize)]
struct ApiEmbedResponse {
    data: Vec<ApiEmbedData>,
}

#[derive(Deserialize)]
struct ApiEmbedData {
    embedding: Vec<f32>,
}

pub struct ApiEmbedEngine {
    url: String,
    key: String,
    model: String,
}

impl ApiEmbedEngine {
    fn try_new() -> std::result::Result<Self, String> {
        let url =
            std::env::var("EMBED_API_URL").map_err(|_| "EMBED_API_URL not set".to_string())?;
        let key =
            std::env::var("EMBED_API_KEY").map_err(|_| "EMBED_API_KEY not set".to_string())?;
        let model = std::env::var("EMBED_MODEL_NAME")
            .map_err(|_| "EMBED_MODEL_NAME not set".to_string())?;
        Ok(Self { url, key, model })
    }
}

impl EmbedEngine for ApiEmbedEngine {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let client = reqwest::blocking::Client::new();
        let body = serde_json::json!({
            "model": self.model,
            "input": texts,
        });
        let resp: ApiEmbedResponse = client
            .post(&self.url)
            .header("Authorization", format!("Bearer {}", self.key))
            .json(&body)
            .send()
            .map_err(|e| crate::core::VectorDBError::Other(e.to_string()))?
            .json()
            .map_err(|e| crate::core::VectorDBError::Other(e.to_string()))?;
        Ok(resp.data.into_iter().map(|d| d.embedding).collect())
    }
}

pub struct FastEmbedEngine {
    model: fastembed::TextEmbedding,
}

impl FastEmbedEngine {
    pub fn try_new() -> std::result::Result<Self, String> {
        let model = match std::env::var("MINI_VECTORDB_MODEL_PATH") {
            Ok(path) => {
                let dir = PathBuf::from(&path);
                if !dir.exists() {
                    return Err(format!("model path not found: {path}"));
                }
                fastembed::TextEmbedding::try_new(
                    fastembed::InitOptions::new(fastembed::EmbeddingModel::AllMiniLML6V2)
                        .with_cache_dir(dir),
                )
                .map_err(|e| e.to_string())?
            }
            Err(_) => {
                fastembed::TextEmbedding::try_new(Default::default()).map_err(|e| e.to_string())?
            }
        };
        Ok(Self { model })
    }
}

impl EmbedEngine for FastEmbedEngine {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self
            .model
            .embed(texts, None)
            .map_err(|e| crate::core::VectorDBError::Other(e.to_string()))?;
        Ok(embeddings.into_iter().collect())
    }
}

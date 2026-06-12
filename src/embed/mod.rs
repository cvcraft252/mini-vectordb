use std::path::PathBuf;

use crate::core::Result;

pub trait EmbedEngine: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub struct FastEmbedEngine {
    model: fastembed::TextEmbedding,
}

impl FastEmbedEngine {
    pub fn try_new() -> std::result::Result<Self, String> {
        let model = match std::env::var("VECTRA_MODEL_PATH") {
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

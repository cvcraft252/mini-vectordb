pub trait EmbedEngine: Send + Sync {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String>;
}

pub struct FastEmbedEngine {
    model: fastembed::TextEmbedding,
}

impl FastEmbedEngine {
    pub fn try_new() -> Result<Self, String> {
        let model =
            fastembed::TextEmbedding::try_new(Default::default()).map_err(|e| e.to_string())?;
        Ok(Self { model })
    }
}

impl EmbedEngine for FastEmbedEngine {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let texts: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self.model.embed(texts, None).map_err(|e| e.to_string())?;
        Ok(embeddings.into_iter().collect())
    }
}

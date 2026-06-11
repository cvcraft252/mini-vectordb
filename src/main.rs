#[tokio::main]
async fn main() {
    let db = mini_vectordb::VectorDB::new();
    mini_vectordb::api::rest::serve(db, 3000).await;
}

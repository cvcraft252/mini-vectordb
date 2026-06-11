use mini_vectordb::StorageFormat;

#[tokio::main]
async fn main() {
    let db = mini_vectordb::VectorDB::with_persistence("data.bin", StorageFormat::Binary);
    mini_vectordb::api::rest::serve(db, 3000).await;
}

from rag_chroma import load_collection

collection, patterns = load_collection()

# 直接用 where filter 查
results = collection.query(
    query_embeddings=[[0.0] * 1536],  # 随便一个向量
    n_results=5,
    where={"categories": {"$contains": "Cowl"}},
    include=["metadatas"],
)
print(f"Results: {len(results['ids'][0])}")
for m in results["metadatas"][0]:
    print(f"  {m['categories']}")
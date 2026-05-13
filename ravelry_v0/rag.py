import json
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 你来实现这四个函数

def load_patterns(path: str = "data/patterns.json") -> list[dict]:
    """加载 patterns 数据"""
    patterns = json.loads(Path(path).read_text(encoding="utf-8"))
    return patterns

def get_or_create_embeddings(
    patterns: list[dict], 
    client: OpenAI,
    cache_path: str = "data/embeddings.npy"
) -> np.ndarray:
    """
    生成或加载缓存的 embeddings。
    如果 cache_path 存在，直接加载。
    如果不存在，调用 OpenAI API 生成，然后保存。
    注意：1000 条不要一条一条调，用批量请求。
    """
    if Path(cache_path).exists():
        print("加载缓存的 embeddings...")
        return np.load(cache_path)
    
    print("生成新的 embeddings...")
    texts = [p["text_for_embedding"] for p in patterns]

    BATCH_SIZE = 100
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        all_embeddings.extend([e.embedding for e in response.data])
        print(f"  {min(i + BATCH_SIZE, len(texts))}/{len(texts)} embeddings 生成完成")

    embeddings = np.array(all_embeddings)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的 cosine similarity"""
    # 公式：dot(a,b) / (norm(a) * norm(b))
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(
    query: str,
    patterns: list[dict],
    embeddings: np.ndarray,
    client: OpenAI,
    top_k: int = 5
) -> list[dict]:
    """
    用 cosine similarity 检索最相关的 top_k 个 pattern。
    返回完整的 pattern dict（不只是文字）。
    """
    query_embedding = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    ).data[0].embedding
    query_vec = np.array(query_embedding)

    similarities = [cosine_similarity(query_vec, e) for e in embeddings]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [patterns[i] for i in top_indices]

def main():
    client = OpenAI()
    patterns = load_patterns()
    embeddings = get_or_create_embeddings(patterns, client)
    
    query = input("搜索编织图解：")
    results = search(query, patterns, embeddings, client)
    
    for i, p in enumerate(results):
        print(f"\n{i+1}. {p['name']}")
        print(f"   {p.get('craft', {}).get('name', '')} · {p.get('yarn_weight_description', '')}")
        print(f"   评分：{p.get('rating_average', 0):.1f} ({p.get('rating_count') or 0} 评分)")
        print(f"   链接：https://www.ravelry.com/patterns/library/{p['permalink']}")

if __name__ == "__main__":
    main()
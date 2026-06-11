import json
import time
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 你来实现这四个函数

def load_patterns(path: str = "data/patterns.json") -> list[dict]:
    """加载 patterns 数据"""
    patterns = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"  ✅ 加载了 {len(patterns)} 个 pattern")
    return patterns

def _embed_batch(client: OpenAI, batch: list[str], out: list) -> None:
    """Embed a batch, retrying on 429 and halving the batch on 431."""
    for attempt in range(6):
        try:
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            out.extend([e.embedding for e in response.data])
            return
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                wait = 2 ** attempt
                print(f"  Rate limited, waiting {wait}s... (attempt {attempt + 1}/6)")
                time.sleep(wait)
            elif "431" in msg or "headers too large" in msg.lower():
                if len(batch) == 1:
                    raise  # single text still too large — give up
                mid = len(batch) // 2
                print(f"  431 on batch of {len(batch)}, splitting into {mid} + {len(batch) - mid}")
                _embed_batch(client, batch[:mid], out)
                _embed_batch(client, batch[mid:], out)
                return
            else:
                raise
    raise RuntimeError(f"Failed after 6 retries on batch of {len(batch)}")


def get_or_create_embeddings(
    patterns: list[dict],
    client: OpenAI,
    cache_path: str = "data/embeddings.npy"
) -> np.ndarray:
    """
    生成或加载缓存的 embeddings。
    如果 cache_path 存在，直接加载。
    如果不存在，调用 OpenAI API 生成，然后保存。
    支持断点续传（checkpoint）和 429 指数退避重试。
    """
    if Path(cache_path).exists():
        print("加载缓存的 embeddings...")
        return np.load(cache_path)

    checkpoint_path = Path(cache_path).with_suffix(".checkpoint.npy")
    # text-embedding-3-small max is 8,191 tokens (~32K chars). Truncate conservatively.
    MAX_CHARS = 30_000
    texts = [p["text_for_embedding"][:MAX_CHARS] for p in patterns]
    BATCH_SIZE = 50
    all_embeddings = []
    start_idx = 0

    if checkpoint_path.exists():
        saved = np.load(checkpoint_path)
        all_embeddings = list(saved)
        start_idx = len(all_embeddings)
        print(f"恢复进度：已有 {start_idx} 条，继续从第 {start_idx} 条开始...")
    else:
        print("生成新的 embeddings...")

    for i in range(start_idx, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        _embed_batch(client, batch, all_embeddings)

        print(f"  {min(i + BATCH_SIZE, len(texts))}/{len(texts)} embeddings 生成完成")

        # Save checkpoint every 10 batches (~1000 patterns)
        if ((i - start_idx) // BATCH_SIZE) % 10 == 9:
            np.save(checkpoint_path, np.array(all_embeddings))

    embeddings = np.array(all_embeddings)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

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
# Improvement Backlog

记录已发现但暂不处理的问题，供后续 sprint 或面试故事素材使用。

## 1. 高人气图解未出现在语义搜索结果中（"summer top" 案例）

**现象**：搜索 "summer top" / "summer knitting top" 等查询，
没有返回一些知名度很高的图解（评分 4.7-4.8，favorites 数千），
比如 Lela Top、Sabai Top 这类经典夏季上衣图解。

**已确认**：直接按图解名称搜索可以找到这些图解，说明它们确实在
data/patterns.json 数据集里，不是数据缺失问题。

**可能原因（待排查）**：
- embedding/semantic_query 提取的 "summer top" 语义查询，可能和这些
  图解 text_for_embedding 里的描述对不上（如果它们的 notes 字段
  没有明确出现 "summer" 这个词）
- 可能是 retrieval 阶段（BM25 / vector）的 recall 问题——这些图解
  可能根本没进入 top-20 候选集，reranker 也无能为力
- 也可能是 Cohere rerank 阶段的排序问题——query 措辞和图解描述的
  语义距离较大，导致 rerank score 偏低
- 当前架构完全基于语义相关性排序，没有考虑"人气信号"
  （rating + favorites）作为排序因子之一——这可能是核心问题：
  纯语义搜索 vs 人气加权排序的 trade-off

**排查步骤**：
1. 把 "summer top" 加入 golden_set.json 作为新的测试 case，
   预期结果里包含 Lela Top / Sabai Top
2. 单独跑 hybrid_search() 看这些图解是否进入 top-20 候选
   （定位问题出在 retrieval 还是 reranking 阶段）
3. 检查这些图解的 text_for_embedding 内容，看是否缺少
   "summer"、"top" 等关键语义信号

**面试价值**：这是一个很好的 "semantic relevance vs popularity bias" 
案例——纯语义搜索可能会让冷门但语义匹配度高的内容，排在热门但措辞
不完全匹配的内容前面。可以讨论如何在 reranking 之后加入一个
"popularity boost" 信号（比如 final_score = cohere_score * 0.7 + 
normalized_popularity * 0.3）。

---

# Future optimization: ijson streaming parse (内存优化)

**问题**：load_collection() 峰值内存 3804MB，远超最终稳定值 2330MB
（瞬时膨胀 +1474MB）。

**根因**：patterns.json（170MB）先读成原始字符串（~2.5GB在内存中），
json.loads 解析期间原始字符串和解析后对象同时驻留内存。

**方案**：用 ijson 做流式解析，避免"整个文件+整个解析结果"同时驻留。

**已完成的相关优化**：
- batched tolist()（消除了另一个 +1769MB 的瞬时膨胀）
- 去重 np.load（-351MB）

**当前临时方案**：Cloud Run 配置为 cpu=2, memory=8Gi

**面试价值**：完整的 profile → diagnose → incremental fix 故事，
有真实的 debug_memory.py 数据支撑（峰值 4137MB → 3804MB，
目标 ~2400MB）
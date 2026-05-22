from rag_chroma import PatternSearchIntent, _build_where

intent = PatternSearchIntent(
    semantic_query="cardigan",
    yarn_weight="DK",
    min_rating=4.5,
)
where = _build_where(intent)
print(where)
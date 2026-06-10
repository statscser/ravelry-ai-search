"""
比较 gpt-4o-mini vs claude-haiku 在 parse_query 上的准确率
用 golden_set.json 的 20 条 query 作为测试集
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # 加入 ravelry_search/ 到路径

import json
from openai import OpenAI
import anthropic
import instructor
from dotenv import load_dotenv
from rag_chroma import PatternSearchIntent, SYSTEM_PROMPT

load_dotenv()

GOLDEN_SET = json.loads(
    (Path(__file__).parent.parent / "data" / "golden_set.json")
    .read_text(encoding="utf-8")
)

def parse_with_openai(query: str) -> PatternSearchIntent:
    client = instructor.from_openai(OpenAI())
    return client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=500,
        response_model=PatternSearchIntent,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )

def parse_with_haiku(query: str) -> PatternSearchIntent:
    client = instructor.from_anthropic(anthropic.Anthropic())
    return client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        response_model=PatternSearchIntent,
        messages=[{"role": "user", "content": query}],
        system=SYSTEM_PROMPT,
    )

def score_intent(intent: PatternSearchIntent, query: str) -> dict:
    """判断 parsing 结果是否合理——检查几个关键字段"""
    checks = {}
    q = query.lower()
    
    # craft 识别
    if "crochet" in q or "钩针" in q:
        checks["craft"] = intent.craft == "Crochet"
    elif "knit" in q or "棒针" in q:
        checks["craft"] = intent.craft == "Knitting"
    
    # free_only 识别
    if "free" in q or "免费" in q:
        checks["free_only"] = intent.free_only == True
    
    # exclude_fibers 识别
    if "mohair" in q or "马海毛" in q:
        if "no " in q or "不用" in q or "without" in q:
            checks["exclude_fibers"] = len(intent.exclude_fibers) > 0
    
    # semantic_query 非空
    checks["has_semantic_query"] = len(intent.semantic_query) > 0
    
    return checks

def main():
    queries = [item["query"] for item in GOLDEN_SET]

    EXTRA_QUERIES = [
        "粗线粗针但轻盈不用马海毛的毛衣",      # 中文负向条件
        "cozy sweater no mohair",               # 英文负向条件
        "soft cardigan without wool or mohair", # 多个 exclude
        "bulky sweater avoid alpaca",           # 动词形式的排除
        "free hat no acrylic yarn",             # 免费 + 负向
    ]
    queries.extend(EXTRA_QUERIES)
    
    print(f"{'Query':<45} {'GPT':>5} {'Haiku':>7}")
    print("-" * 60)
    
    gpt_scores, haiku_scores = [], []
    
    for query in queries:
        gpt_intent   = parse_with_openai(query)
        haiku_intent = parse_with_haiku(query)
        
        gpt_checks   = score_intent(gpt_intent, query)
        haiku_checks = score_intent(haiku_intent, query)
        
        gpt_score   = sum(gpt_checks.values()) / len(gpt_checks) if gpt_checks else 1.0
        haiku_score = sum(haiku_checks.values()) / len(haiku_checks) if haiku_checks else 1.0
        
        gpt_scores.append(gpt_score)
        haiku_scores.append(haiku_score)
        
        status = "✅" if haiku_score >= gpt_score else "⚠️"
        print(f"{query:<45} {gpt_score:>5.2f} {haiku_score:>7.2f} {status}")

        # print(f"  GPT   exclude_fibers: {gpt_intent.exclude_fibers}")
        # print(f"  Haiku exclude_fibers: {haiku_intent.exclude_fibers}")

    print("-" * 60)
    print(f"{'AVERAGE':<45} {sum(gpt_scores)/len(gpt_scores):>5.2f} {sum(haiku_scores)/len(haiku_scores):>7.2f}")
    print(f"\nCost comparison:")
    print(f"  gpt-4o-mini:  ~$0.0001/query")
    print(f"  claude-haiku: ~$0.00003/query  (约便宜 3x)")

if __name__ == "__main__":
    main()
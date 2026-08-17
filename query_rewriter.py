"""
Step 2.3: 查询改写（Query Rewriting）

学习目标：
- 理解 HyDE（Hypothetical Document Embedding）原理
- 理解 Multi-Query 的作用
- 体验查询改写对检索效果的影响

关键概念：
- HyDE: 让 LLM 先假想一个答案，用假想答案去检索（答案比问题更像文档）
- Multi-Query: 把一个问题拆成多个角度，分别检索再合并
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class QueryRewriter:
    """查询改写器"""

    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    def hyde(self, query: str) -> str:
        """
        HyDE（Hypothetical Document Embedding）。

        原理：
        1. 让 LLM 假想一个答案（可能是错的）
        2. 用这个假想答案去检索
        3. 假想答案比原始问题更像真实文档 → 检索效果更好

        为什么有效？
        用户的问题: "巴菲特怎么看通胀"
          ↓ 在向量空间中可能距离真实文档较远

        假想答案: "巴菲特认为通胀是股票投资者的最大敌人..."
          ↓ 在向量空间中距离真实文档更近
          因为假想答案的"写法"跟文档更相似

        风险：如果 LLM 假想的方向完全错了，可能检索到错误的内容
        """
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个巴菲特投资知识专家。请根据问题，写一段假设性的回答，"
                               "就像你是从巴菲特致股东信中摘录的一样。不需要正确，只需要风格和用词相似。",
                },
                {"role": "user", "content": query},
            ],
            temperature=0.7,  # 高一点温度，让假想更发散
            max_tokens=300,
        )
        return response.choices[0].message.content

    def multi_query(self, query: str, n: int = 3) -> list[str]:
        """
        Multi-Query：把一个问题改写成多个不同角度的查询。

        原理：
        单个查询可能只从一个角度检索，遗漏其他相关内容。
        改写成多个角度 → 分别检索 → 合并去重 → 更全面。

        例子：
        原始: "巴菲特怎么看护城河"
          ↓
        Q1: "巴菲特护城河的定义和判断标准"
        Q2: "伯克希尔旗下公司有哪些护城河案例"
        Q3: "护城河概念在巴菲特投资决策中的演变"
        """
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"你是一个查询改写专家。请把用户的问题改写成 {n} 个不同角度的搜索查询，"
                               f"用于在巴菲特致股东信中搜索相关信息。"
                               f"每个查询一行，不要编号，不要解释。",
                },
                {"role": "user", "content": query},
            ],
            temperature=0.5,
            max_tokens=200,
        )
        queries = [q.strip() for q in response.choices[0].message.content.strip().split("\n") if q.strip()]
        return queries[:n]


# ============================================================
# 对比实验
# ============================================================
def compare_query_strategies(rewriter, hybrid_search):
    """对比不同查询策略的检索效果"""
    queries = [
        "巴菲特怎么看通胀",
        "喜诗糖果的投资故事",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"原始查询: {query}")
        print(f"{'='*60}")

        # 1. 原始查询
        print("\n  [原始查询]:")
        results = hybrid_search.search(query, top_k=3, mode="hybrid")
        for i, r in enumerate(results, 1):
            preview = r["text"][:60].replace("\n", " ")
            print(f"    {i}. {r['source']} {preview}...")

        # 2. HyDE
        print("\n  [HyDE]:")
        hypothetical_answer = rewriter.hyde(query)
        print(f"    假想答案: {hypothetical_answer[:100]}...")
        results = hybrid_search.search(hypothetical_answer, top_k=3, mode="hybrid")
        for i, r in enumerate(results, 1):
            preview = r["text"][:60].replace("\n", " ")
            print(f"    {i}. {r['source']} {preview}...")

        # 3. Multi-Query
        print("\n  [Multi-Query]:")
        sub_queries = rewriter.multi_query(query)
        all_results = {}  # id -> result, 去重
        for sq in sub_queries:
            print(f"    子查询: {sq}")
            results = hybrid_search.search(sq, top_k=3, mode="hybrid")
            for r in results:
                if r["id"] not in all_results:
                    all_results[r["id"]] = r

        # 合并后按 rrf_score 排序
        merged = sorted(all_results.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)
        print(f"    合并去重后共 {len(merged)} 个结果，Top-3:")
        for i, r in enumerate(merged[:3], 1):
            preview = r["text"][:60].replace("\n", " ")
            print(f"    {i}. {r['source']} {preview}...")


if __name__ == "__main__":
    from hybrid_search import HybridSearch

    rewriter = QueryRewriter()
    hs = HybridSearch()
    compare_query_strategies(rewriter, hs)

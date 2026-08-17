"""
Step 2.2: Reranking（重排序）

学习目标：
- 理解 Bi-Encoder vs Cross-Encoder 的区别
- 理解为什么 Reranking 能提升检索质量
- 实际体验 Reranking 前后的效果差异

关键概念：
- Bi-Encoder（Embedding 检索）：query 和 doc 分别编码 → 快但粗
- Cross-Encoder（Reranker）：query + doc 拼在一起编码 → 慢但准
- 两阶段检索：先用 Bi-Encoder 粗筛 Top-50，再用 Cross-Encoder 精排 Top-5
"""

from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-Encoder Reranker。

    使用 Qwen3-Reranker-0.6B 模型（阿里 Qwen 系列，中英双语）。

    原理：
    输入: [query, doc1], [query, doc2], [query, doc3] ...
    输出: 0.85, 0.32, 0.71  ← 每对的匹配分数

    为什么比 Embedding 更准：
    Embedding: query→向量, doc→向量, 比较两个向量（不交互）
    CrossEncoder: query+doc → 一个分数（深度交互，能看到两者关系）

    代价：慢。所以不能用来做全量检索，只能对少量候选精排。
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B"):
        print(f"加载 Reranker 模型: {model_name}")
        self.model = CrossEncoder(model_name)
        print("Reranker 加载完成")

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """
        对候选文档重排序。

        参数：
            query: 用户查询
            documents: 初步检索结果列表，每个需有 "text" 字段
            top_k: 返回前 K 个

        返回：按相关性重排序后的文档列表（带 rerank_score）
        """
        if not documents:
            return []

        # 构造 Cross-Encoder 输入: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc["text"]] for doc in documents]

        # 批量打分
        scores = self.model.predict(pairs)

        # 把分数附加到文档上
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # 按 rerank_score 降序排序
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


# ============================================================
# 对比实验
# ============================================================
def compare_with_reranking(hybrid_search, reranker):
    """对比：混合检索 vs 混合检索 + Reranking"""
    from dotenv import load_dotenv
    load_dotenv()

    queries = [
        "伯克希尔1985年的浮存金",
        "巴菲特为什么关闭纺织业务",
        "什么是安全边际？巴菲特如何应用？",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        # Step 1: 混合检索 Top-20（粗筛）
        candidates = hybrid_search.search(query, top_k=20, mode="hybrid")

        # Step 2: Rerank 精排 Top-5
        reranked = reranker.rerank(query, candidates, top_k=5)

        print("\n  [HYBRID Top-5（重排前）]:")
        for i, doc in enumerate(candidates[:5], 1):
            preview = doc["text"][:60].replace("\n", " ")
            print(f"    {i}. {doc['source']} (rrf={doc.get('rrf_score', 0):.4f}) {preview}...")

        print("\n  [HYBRID + RERANK Top-5（重排后）]:")
        for i, doc in enumerate(reranked, 1):
            preview = doc["text"][:60].replace("\n", " ")
            print(f"    {i}. {doc['source']} (rerank={doc['rerank_score']:.4f}) {preview}...")

        # 检查排名变化
        before_ids = [doc["id"] for doc in candidates[:5]]
        after_ids = [doc["id"] for doc in reranked[:5]]
        if before_ids != after_ids:
            print("\n  ⬆️ 排名发生了变化！Reranking 调整了顺序")
        else:
            print("\n  ➡️ 排名未变化")


if __name__ == "__main__":
    from hybrid_search import HybridSearch

    hs = HybridSearch()
    reranker = Reranker()
    compare_with_reranking(hs, reranker)

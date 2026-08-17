"""
Level 2 完整 RAG：混合检索 + Reranking + 查询改写

这是 Level 1 的升级版，整合了三个高级检索技术：
1. 混合检索（向量 + BM25 + RRF）
2. Reranking（Cross-Encoder 精排）
3. 查询改写（HyDE / Multi-Query 可选）

对比 Level 1：
  Level 1: query → 向量检索 Top-5 → LLM
  Level 2: query → (改写) → 混合检索 Top-20 → Rerank Top-5 → LLM
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from hybrid_search import HybridSearch
from query_rewriter import QueryRewriter

load_dotenv()


class BuffettRAGv2:
    """巴菲特信件 RAG v2 — 高级检索版"""

    def __init__(self, use_reranker: bool = True, use_query_rewrite: str = "none"):
        """
        参数：
            use_reranker: 是否启用 Reranking（需要额外下载 560MB 模型）
            use_query_rewrite: "none" / "hyde" / "multi"
        """
        print("=" * 50)
        print("  巴菲特 RAG v2 — 高级检索版")
        print("=" * 50)

        # 混合检索
        self.search = HybridSearch()
        self.use_reranker = use_reranker
        self.use_query_rewrite = use_query_rewrite

        # Reranker（可选）
        self.reranker = None
        if use_reranker:
            from reranker import Reranker
            self.reranker = Reranker()

        # 查询改写器
        self.rewriter = QueryRewriter() if use_query_rewrite != "none" else None

        # LLM
        self.llm = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.llm_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        print(f"\n配置: reranker={'ON' if use_reranker else 'OFF'} | query_rewrite={use_query_rewrite}")
        print("✅ RAG v2 就绪！")

    def _retrieve(self, query: str, top_k: int = 5):
        """完整检索流程：改写 → 混合检索 → Rerank"""

        # Step 1: 查询改写
        search_query = query
        extra_context = []

        if self.rewriter and self.use_query_rewrite == "hyde":
            search_query = self.rewriter.hyde(query)
            print(f"   [HyDE] 用假想答案检索")
        elif self.rewriter and self.use_query_rewrite == "multi":
            sub_queries = self.rewriter.multi_query(query)
            print(f"   [Multi-Query] 拆分为 {len(sub_queries)} 个子查询")
            # 多路检索合并
            all_results = {}
            for sq in sub_queries:
                results = self.search.search(sq, top_k=10, mode="hybrid")
                for r in results:
                    if r["id"] not in all_results:
                        all_results[r["id"]] = r
            candidates = sorted(all_results.values(), key=lambda x: x.get("rrf_score", 0), reverse=True)
            # 跳到 rerank 步骤
            if self.reranker:
                print(f"   [Rerank] 精排 {len(candidates)} → {top_k}")
                return self.reranker.rerank(query, candidates[:30], top_k=top_k)
            return candidates[:top_k]

        # Step 2: 混合检索 Top-20
        candidates = self.search.search(search_query, top_k=20, mode="hybrid")

        # Step 3: Rerank 精排
        if self.reranker:
            results = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            results = candidates[:top_k]

        return results

    def query(self, question: str, verbose: bool = True) -> str:
        """完整 RAG v2 流程"""
        if verbose:
            print(f"\n🔍 问题: {question}")

        # 检索
        chunks = self._retrieve(question, top_k=5)

        if verbose:
            print(f"\n📚 检索到 {len(chunks)} 个相关分块:")
            for i, c in enumerate(chunks, 1):
                score = c.get("rerank_score", c.get("rrf_score", 0))
                preview = c["text"][:60].replace("\n", " ")
                print(f"   [{i}] {c['source']} (score={score:.4f}) {preview}...")

        # 生成
        if verbose:
            print(f"\n🤖 生成回答...")

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[来源 {i}: {chunk['source']} ({chunk['year']}年)]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "你是巴菲特投资知识助手。只根据提供的上下文回答，不要编造。引用来源年份。",
                },
                {
                    "role": "user",
                    "content": f"参考上下文：\n{context}\n\n---\n\n用户问题：{question}\n\n请基于以上上下文回答。",
                },
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content

        if verbose:
            print(f"\n💡 回答:\n{'-'*50}")
            print(answer)
            print(f"{'-'*50}")

        return answer


def interactive_mode():
    print("""
可选模式：
  1. 基础混合检索（无 Reranker，无改写）— 最快
  2. 混合检索 + Reranking — 更准
  3. 混合检索 + HyDE — 更全面
  4. 混合检索 + Multi-Query — 最全面
""")
    choice = input("选择模式 [1-4]: ").strip()

    configs = {
        "1": (False, "none"),
        "2": (True, "none"),
        "3": (False, "hyde"),
        "4": (False, "multi"),
    }
    use_reranker, use_rewrite = configs.get(choice, (False, "none"))

    rag = BuffettRAGv2(use_reranker=use_reranker, use_query_rewrite=use_rewrite)

    while True:
        try:
            question = input("\n❓ 请输入问题: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            rag.query(question)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    interactive_mode()

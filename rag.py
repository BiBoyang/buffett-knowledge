"""
Step 1.5: 完整 RAG 问答系统

学习目标：
- 理解 RAG 的完整流程：Query → Retrieve → Augment → Generate
- 理解 prompt engineering 在 RAG 中的作用
- 理解为什么 RAG 能减少 LLM 幻觉

RAG 全流程：
  用户提问 "巴菲特怎么看护城河？"
       │
       ▼
  [1] Query Embedding: 把问题变成向量
       │
       ▼
  [2] Vector Search: 在 Chroma 中找最相似的 Top-K 个分块
       │
       ▼
  [3] Prompt 组装: 把检索到的分块作为上下文拼进 prompt
       │
       ▼
  [4] LLM 生成: DeepSeek 根据上下文生成回答
"""

import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from embedder import load_vector_store, get_embedding_model, build_vector_store


# ============================================================
# 配置
# ============================================================
DATA_DIR = "/Users/boyang/Downloads/巴菲特致股东信"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "buffett_letters"

# DeepSeek 配置（OpenAI 兼容接口）
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


# ============================================================
# RAG 核心组件
# ============================================================

class BuffettRAG:
    """巴菲特信件 RAG 问答系统"""

    def __init__(self, rebuild=False):
        """
        初始化 RAG 系统。

        rebuild=True: 重新构建向量库（首次运行或数据更新后用）
        rebuild=False: 加载已有向量库（秒级启动）
        """
        # 加载 Embedding 模型（用于把 query 转向量）
        self.model, self.tokenizer = get_embedding_model()

        # 加载或构建向量库
        if rebuild:
            self.collection, _ = build_vector_store(DATA_DIR, DB_PATH, COLLECTION_NAME)
        else:
            self.collection = load_vector_store(DB_PATH, COLLECTION_NAME)

        # 初始化 DeepSeek LLM
        self.llm = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        print("✅ RAG 系统就绪！输入问题开始提问，输入 'quit' 退出")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        [检索阶段] 根据查询找到最相关的文本分块。

        流程：
        1. 把 query 编码为向量
        2. 在 Chroma 中做余弦相似度搜索
        3. 返回 Top-K 个最相似的分块

        参数：
            query: 用户问题
            top_k: 返回最相关的 K 个分块（默认5）

        关键概念：
        - top_k 越大，上下文越丰富，但也可能引入噪声
        - 一般 3-5 是比较好的平衡点
        """
        # 把 query 转为向量
        import mlx.core as mx
        ids = self.tokenizer.encode(query)
        mlx_ids = mx.array(np.array(ids, dtype=np.int32))[None, :]
        outputs = self.model(mlx_ids)
        mx.eval(outputs.text_embeds)
        query_embedding = np.array(outputs.text_embeds.astype(mx.float32).tolist(), dtype=np.float32).tolist()

        # 在 Chroma 中搜索
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # 整理结果
        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "year": results["metadatas"][0][i]["year"],
                "distance": results["distances"][0][i],  # 越小越相似（余弦距离）
            })

        return retrieved

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """
        [生成阶段] 根据检索到的上下文，让 LLM 生成回答。

        这里是 RAG 的 "Augment" 和 "Generate" 步骤：
        - Augment: 把检索到的分块拼接到 prompt 中
        - Generate: LLM 基于上下文生成回答

        Prompt 设计原则：
        1. 明确告诉 LLM 只基于提供的上下文回答（减少幻觉）
        2. 如果上下文中没有相关信息，让 LLM 承认不知道
        3. 要求引用来源（增加可信度）
        """
        # 组装上下文
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[来源 {i}: {chunk['source']} ({chunk['year']}年)]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # 构建 Prompt
        system_prompt = """你是一个巴菲特投资知识助手。请根据提供的上下文内容回答用户问题。

规则：
1. 只根据提供的上下文内容回答，不要编造信息
2. 如果上下文中没有足够信息，请明确说明
3. 回答时尽量引用来源（信件年份）
4. 用清晰简洁的中文回答"""

        user_prompt = f"""参考上下文：
{context}

---

用户问题：{query}

请基于以上上下文回答问题。"""

        # 调用 DeepSeek API
        response = self.llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # 低温度 = 更确定性的回答（RAG 场景推荐）
            max_tokens=1000,
        )

        return response.choices[0].message.content

    def query(self, question: str, top_k: int = 5, verbose: bool = True) -> str:
        """
        完整 RAG 流程：Query → Retrieve → Generate

        这就是 RAG 的全貌：
        1. 把问题变成向量
        2. 在向量库中找相似的分块
        3. 把分块作为上下文 + 问题一起交给 LLM
        4. LLM 基于真实上下文生成回答
        """
        if verbose:
            print(f"\n🔍 问题: {question}")
            print(f"   检索 Top-{top_k} 个最相关分块...")

        # Step 1: 检索
        chunks = self.retrieve(question, top_k=top_k)

        if verbose:
            print(f"\n📚 检索到 {len(chunks)} 个相关分块:")
            for i, c in enumerate(chunks, 1):
                dist = c['distance']
                sim = 1 - dist  # 余弦距离 → 余弦相似度
                preview = c['text'][:60].replace('\n', ' ')
                print(f"   [{i}] {c['source']} (相似度={sim:.3f}) {preview}...")

        # Step 2: 生成
        if verbose:
            print(f"\n🤖 DeepSeek 生成回答...")
        answer = self.generate(question, chunks)

        if verbose:
            print(f"\n💡 回答:\n{'-'*50}")
            print(answer)
            print(f"{'-'*50}")

        return answer


# ============================================================
# 交互式问答
# ============================================================
def interactive_mode():
    """交互式问答循环"""
    print("=" * 50)
    print("  巴菲特致股东信 RAG 问答系统")
    print("=" * 50)

    rag = BuffettRAG(rebuild=False)

    print("\n试试这些问题：")
    print("  - 巴菲特怎么看护城河？")
    print("  - 伯克希尔为什么关闭纺织业务？")
    print("  - 巴菲特对可口可乐的投资逻辑是什么？")
    print("  - 什么是内在价值？巴菲特如何定义？")
    print("  - 浮存金在保险业务中扮演什么角色？")
    print()

    while True:
        try:
            question = input("\n❓ 请输入问题: ").strip()
            if question.lower() in ("quit", "exit", "q", "退出"):
                print("再见！")
                break
            if not question:
                continue
            rag.query(question)
        except KeyboardInterrupt:
            print("\n再见！")
            break


if __name__ == "__main__":
    interactive_mode()

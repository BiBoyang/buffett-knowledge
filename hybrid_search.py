"""
Step 2.1: 混合检索（Hybrid Search）= 向量检索 + BM25 + RRF 融合

学习目标：
- 理解为什么需要混合检索（向量擅长语义，BM25擅长关键词）
- 理解 RRF（Reciprocal Rank Fusion）融合算法
- 对比纯向量检索 vs 混合检索的效果差异

关键概念：
- Bi-Encoder（双编码器）：query 和 doc 分别编码，快但粗
- BM25：基于词频的经典信息检索算法，理解"这个词出现了几次"
- RRF：根据排名而非分数来融合两路结果，简单且有效
"""

import hashlib
import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from chunker import Chunk, chunk_recursive, load_all_documents
from embedder import load_vector_store, get_embedding_model


def chinese_tokenizer(text: str) -> list[str]:
    """
    中文分词器。

    BM25 需要分词，但中文没有天然空格分隔。
    简单方案：按字符切分 + 过滤标点。
    生产环境应该用 jieba 或 HanLP 做真正的中文分词。

    这里用字符级分词是因为：
    1. 不需要额外依赖
    2. 对"1985"、"浮存金"这种精确匹配够用
    3. 简单易懂
    """
    # 移除 Markdown 链接标记，只保留文字
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # 移除标点符号，保留中文字符、英文字母、数字
    tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+|\d+', text)
    return tokens


def _cache_fingerprint(data_dir: str, chunk_size: int, overlap: int) -> str:
    """
    计算 BM25 缓存的指纹。

    缓存是否有效取决于两件事：
    1. 源文档有没有变（文件清单：相对路径 + 大小 + 修改时间）
    2. 分块参数有没有变（chunk_size / overlap 变了，分块结果就全变了）

    用内容 hash 最严谨，但要读全部文件、太慢；
    "大小 + mtime" 足够可靠且几乎零成本。
    """
    base = Path(data_dir)
    entries = []
    if base.exists():
        for f in sorted(base.rglob("*.md")):
            st = f.stat()
            entries.append(f"{f.relative_to(base)}:{st.st_size}:{st.st_mtime}")
    payload = json.dumps(
        {"files": entries, "chunk_size": chunk_size, "overlap": overlap},
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class HybridSearch:
    """
    混合检索系统：向量检索 + BM25 关键词检索 + RRF 融合

    使用方法：
        hs = HybridSearch()
        results = hs.search("1985年浮存金", top_k=5)

    三步走：
    1. 向量检索（Chroma）→ 找语义相关的
    2. BM25 检索（rank_bm25）→ 找关键词匹配的
    3. RRF 融合 → 合并两路结果，取长补短
    """

    def __init__(
        self,
        data_dir: str = "/Users/boyang/Downloads/巴菲特致股东信",
        db_path: str = "./chroma_db",
        collection_name: str = "buffett_letters",
        bm25_cache_path: str = "./bm25_cache.json",
    ):
        # 1. 加载向量检索组件
        print("加载向量检索组件...")
        self.model, self.tokenizer = get_embedding_model()
        self.collection = load_vector_store(db_path, collection_name)

        # 2. 构建 BM25 索引（优先用缓存，指纹校验通过才敢用）
        print("构建 BM25 索引...")
        chunk_size, overlap = 500, 50
        cache_path = Path(bm25_cache_path)
        fingerprint = _cache_fingerprint(data_dir, chunk_size, overlap)

        # 为什么必须校验指纹：
        # 缓存里只有 tokenized/texts，没有"它对应哪份数据、哪组分块参数"的信息。
        # 源文档或 chunk_size/overlap 一变，旧缓存的 corpus 就和新分块错位，
        # _bm25_search 里 self.chunks[idx] 会张冠李戴（静默返回错误结果）。
        cached = None
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
            except (json.JSONDecodeError, OSError):
                cached = None  # 缓存损坏，当作不存在

        if cached and cached.get("fingerprint") == fingerprint:
            # 指纹一致：数据和分块参数都没变，缓存 corpus 与分块一一对应。
            # 缓存里已存 texts + 每个分块的元信息，直接重建 chunks，
            # 跳过"重新加载全部文档 + 重新分块 + 重新分词"这三步重活。
            print("  加载 BM25 缓存（指纹匹配）...")
            self.docs = None  # 命中时无需加载原文档
            self.chunk_texts = cached["texts"]
            self.tokenized_corpus = cached["tokenized"]
            self.chunks = [
                Chunk(
                    text=text,
                    source=meta["source"],
                    year=meta["year"],
                    category=meta["category"],
                    chunk_index=meta["chunk_index"],
                    char_count=len(text),
                )
                for text, meta in zip(self.chunk_texts, cached["meta"])
            ]
        else:
            # 无缓存或指纹失配：重新加载文档、分块、分词，并覆盖旧缓存
            if cached:
                print("  缓存指纹不匹配（源文档或分块参数已变化），自动重建...")
            self.docs = load_all_documents(data_dir)
            self.chunks = chunk_recursive(self.docs, chunk_size=chunk_size, overlap=overlap)
            print(f"  分词处理 {len(self.chunks)} 个分块...")
            self.chunk_texts = [c.text for c in self.chunks]
            self.tokenized_corpus = [chinese_tokenizer(t) for t in self.chunk_texts]
            # 缓存到文件（带上指纹和分块元信息，供下次校验和重建）
            with open(cache_path, "w") as f:
                json.dump({
                    "fingerprint": fingerprint,
                    "tokenized": self.tokenized_corpus,
                    "texts": self.chunk_texts,
                    "meta": [
                        {
                            "source": c.source,
                            "year": c.year,
                            "category": c.category,
                            "chunk_index": c.chunk_index,
                        }
                        for c in self.chunks
                    ],
                }, f)
            print(f"  BM25 缓存已保存到 {bm25_cache_path}")

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"✅ 混合检索系统就绪（向量 + BM25）")

    def _vector_search(self, query: str, top_k: int = 20) -> list[dict]:
        """向量检索：找语义相关的分块"""
        import mlx.core as mx
        import numpy as np
        ids = self.tokenizer.encode(query)
        mlx_ids = mx.array(np.array(ids, dtype=np.int32))[None, :]
        outputs = self.model(mlx_ids)
        mx.eval(outputs.text_embeds)
        query_embedding = np.array(outputs.text_embeds.astype(mx.float32).tolist(), dtype=np.float32).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        items = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "year": results["metadatas"][0][i]["year"],
                "score": 1 - results["distances"][0][i],  # 距离 → 相似度
            })
        return items

    def _bm25_search(self, query: str, top_k: int = 20) -> list[dict]:
        """BM25 检索：找关键词精确匹配的分块"""
        tokenized_query = chinese_tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 按分数排序，取 top_k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        items = []
        for idx in top_indices:
            if scores[idx] > 0:  # 过滤零分
                chunk = self.chunks[idx]
                items.append({
                    "id": f"{chunk.source}_{chunk.chunk_index}",
                    "text": chunk.text,
                    "source": chunk.source,
                    "year": chunk.year,
                    "score": float(scores[idx]),
                })
        return items

    def _rrf_merge(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        RRF（Reciprocal Rank Fusion）融合算法。

        原理：
        每个文档的最终分数 = 在各路检索中的排名倒数之和
        score(d) = Σ 1/(k + rank_i(d))

        为什么用 RRF 而不是加权平均：
        1. 不依赖原始分数（向量相似度和BM25分数尺度不同）
        2. 只依赖排名 → 对分数尺度不敏感
        3. k=60 是经验最佳值（原论文推荐）

        例子：
        向量排名: A=1, B=2, C=3
        BM25排名: C=1, A=2, D=3

        A: 1/(60+1) + 1/(60+2) = 0.0325  ← 两路都靠前
        C: 1/(60+3) + 1/(60+1) = 0.0323
        B: 1/(60+2) + 0         = 0.0161  ← 只有一路
        D: 0         + 1/(60+3) = 0.0159  ← 只有一路

        排序: A > C > B > D
        """
        # 用 id 作为文档唯一标识
        rrf_scores = {}

        # 向量检索的排名贡献
        for rank, item in enumerate(vector_results, 1):
            doc_id = item["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {**item, "rrf_score": 0}
            rrf_scores[doc_id]["rrf_score"] += 1.0 / (k + rank)

        # BM25 检索的排名贡献
        for rank, item in enumerate(bm25_results, 1):
            doc_id = item["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {**item, "rrf_score": 0}
            rrf_scores[doc_id]["rrf_score"] += 1.0 / (k + rank)

        # 按 RRF 分数排序
        merged = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        return merged

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        vector_weight_top_k: int = 20,
        bm25_top_k: int = 20,
    ) -> list[dict]:
        """
        统一搜索接口。

        mode:
          - "vector":  只用向量检索（Level 1 的方式）
          - "bm25":    只用 BM25 检索
          - "hybrid":  混合检索（向量 + BM25 + RRF）← 推荐
        """
        if mode == "vector":
            return self._vector_search(query, top_k=top_k)
        elif mode == "bm25":
            return self._bm25_search(query, top_k=top_k)
        elif mode == "hybrid":
            # 两路各检索 top_k*4 个候选，RRF 融合后取 top_k
            vector_results = self._vector_search(query, top_k=vector_weight_top_k)
            bm25_results = self._bm25_search(query, top_k=bm25_top_k)
            merged = self._rrf_merge(vector_results, bm25_results)
            return merged[:top_k]
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# 对比实验
# ============================================================
def compare_search_modes(hs: HybridSearch, queries: list[str]):
    """对比三种检索模式的效果"""
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        for mode in ["vector", "bm25", "hybrid"]:
            results = hs.search(query, top_k=3, mode=mode)
            print(f"\n  [{mode.upper()}] Top-3:")
            for i, r in enumerate(results, 1):
                preview = r["text"][:80].replace("\n", " ")
                score_key = "rrf_score" if mode == "hybrid" else "score"
                print(f"    {i}. {r['source']} ({score_key}={r[score_key]:.4f}) {preview}...")


if __name__ == "__main__":
    hs = HybridSearch()

    test_queries = [
        "伯克希尔1985年的浮存金",        # 精确年份 + 关键词 → BM25 应该更好
        "巴菲特怎么看护城河？",            # 语义问题 → 向量应该更好
        "可口可乐的投资逻辑",              # 语义 + 关键词混合 → 混合应该最好
    ]

    compare_search_modes(hs, test_queries)

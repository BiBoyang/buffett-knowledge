"""
Step 1.3 & 1.4: Embedding 向量化 + 存入 Chroma 向量库

学习目标：
- 理解什么是 embedding（把文本变成高维向量）
- 理解向量相似度 = 语义相似度
- 学会使用 Chroma 向量数据库的增删查

关键概念：
- Embedding 模型：把文本 → 固定维度的浮点数向量
- 余弦相似度：衡量两个向量方向的接近程度（-1 到 1，越大越相似）
- 向量库的作用：预先计算并存储所有分块的向量，查询时只需算 query 的向量然后做相似度搜索
"""

import chromadb
import mlx.core as mx
import numpy as np
from chunker import Chunk, chunk_recursive, load_all_documents


def get_embedding_model():
    """
    加载 Qwen3-Embedding-0.6B 模型（MLX 加速）。

    为什么选这个模型：
    - 2026年中文 MTEB 基准最强
    - 0.6B 参数，Mac 本地可跑
    - 32K 上下文（远超 bge-m3 的 8K）
    - 输出维度可调（默认1024维）
    - 使用 MLX 框架，Apple Silicon 原生加速

    首次运行会下载约1.2GB模型文件。
    """
    from mlx_embeddings.utils import load as mlx_load
    import mlx.core as mx

    print("正在加载 Qwen3-Embedding-0.6B (MLX)...")
    print("首次运行需下载约1.2GB模型文件，请耐心等待")
    model, tokenizer = mlx_load("Qwen/Qwen3-Embedding-0.6B")
    print(f"模型加载完成！（MLX 加速）")
    return model, tokenizer


def build_vector_store(
    data_dir: str,
    db_path: str = "./chroma_db",
    collection_name: str = "buffett_letters",
    chunk_size: int = 500,
    overlap: int = 50,
):
    """
    完整的索引构建流程：加载 → 分块 → Embedding → 存入 Chroma

    参数：
        data_dir:       巴菲特信件目录
        db_path:        Chroma 数据持久化路径
        collection_name: 集合名称（类似数据库的表名）
        chunk_size:     分块大小
        overlap:        分块重叠

    返回：(chroma_collection, embedding_model)
    """
    # 1. 加载文档
    print("=" * 50)
    print("Step 1: 加载文档")
    print("=" * 50)
    docs = load_all_documents(data_dir)
    print(f"  加载了 {len(docs)} 篇文档")

    # 2. 分块（使用递归分块策略）
    print("\nStep 2: 递归分块")
    chunks = chunk_recursive(docs, chunk_size=chunk_size, overlap=overlap)
    print(f"  生成 {len(chunks)} 个分块")

    # 3. 加载 Embedding 模型
    print("\nStep 3: 加载 Embedding 模型")
    model, tokenizer = get_embedding_model()

    # 4. 生成向量并存入 Chroma
    print("\nStep 4: 生成向量并存入 Chroma")
    client = chromadb.PersistentClient(path=db_path)

    # 如果已存在同名集合，先删除（避免重复）
    try:
        client.delete_collection(collection_name)
        print(f"  删除旧集合: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
    )

    # 分批处理（每批100个，避免内存溢出）
    batch_size = 100
    texts = [c.text for c in chunks]
    metadatas = [
        {"source": c.source, "year": c.year, "category": c.category, "chunk_index": c.chunk_index}
        for c in chunks
    ]
    ids = [f"{c.source}_{c.chunk_index}" for c in chunks]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        # 生成向量
        all_embs = []
        for text in batch_texts:
            token_ids = tokenizer.encode(text)
            mlx_ids = mx.array(np.array(token_ids, dtype=np.int32))[None, :]
            outputs = model(mlx_ids)
            all_embs.append(outputs.text_embeds)
        combined = mx.concatenate(all_embs, axis=0).astype(mx.float32)
        mx.eval(combined)
        embeddings = combined.tolist()

        # 存入 Chroma
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        print(f"  已处理 {min(i + batch_size, len(texts))}/{len(texts)} 个分块")

    print(f"\n✅ 向量库构建完成！")
    print(f"   路径: {db_path}")
    print(f"   集合: {collection_name}")
    print(f"   向量数: {collection.count()}")
    print(f"   维度: 1024")

    return collection, (model, tokenizer)


def load_vector_store(
    db_path: str = "./chroma_db",
    collection_name: str = "buffett_letters",
):
    """加载已构建好的向量库（不需要重新 embedding）"""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)
    print(f"已加载向量库: {collection.count()} 个向量")
    return collection


if __name__ == "__main__":
    data_dir = "/Users/boyang/Downloads/巴菲特致股东信"
    build_vector_store(data_dir)

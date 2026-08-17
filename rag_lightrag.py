"""
Level 3: LightRAG — 知识图谱 + 向量检索双层 RAG

LightRAG 核心思路：
  文档 → LLM 自动抽取实体和关系 → 构建知识图谱 + 向量索引
  查询时同时走图谱层和向量层，合并结果

使用方式：
  1. 初始化 + 插入数据：python rag_lightrag.py --init
  2. 查询：python rag_lightrag.py --query "巴菲特怎么看护城河？"
"""

import argparse
import asyncio
import re
from pathlib import Path

from lightrag import QueryParam

# 配置、LLM / Embedding 函数、实例创建等公共部分已抽到 rag_common.py
from rag_common import (
    DATA_DIR,
    WORK_DIR,
    EMBEDDING_DIM,
    DEEPSEEK_MODEL,
    create_rag_instance,
)


# ============================================================
# 数据加载
# ============================================================
def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """加载所有 Markdown 文档"""
    docs = []
    base = Path(data_dir)

    category_map = {
        "伯克希尔股东信": "berkshire",
        "合伙人信": "partnership",
        "特别信件": "special",
    }

    for folder, category in category_map.items():
        folder_path = base / folder
        if not folder_path.exists():
            print(f"⚠️  目录不存在: {folder_path}")
            continue

        for md_file in sorted(folder_path.glob("*.md")):
            name = md_file.stem
            year_match = re.match(r'(\d{4})', name)
            year = int(year_match.group(1)) if year_match else 0

            text = md_file.read_text(encoding="utf-8")
            docs.append({
                "content": text,
                "meta": {
                    "filename": name,
                    "year": year,
                    "category": category,
                },
            })

    return docs


# ============================================================
# 初始化 LightRAG（create_rag_instance 来自 rag_common）
# ============================================================
async def insert_all(rag, docs):
    """初始化存储 + 逐篇插入文档"""
    await rag.initialize_storages()

    for i, doc in enumerate(docs):
        print(f"  [{i+1}/{len(docs)}] 插入: {doc['meta']['filename']}")
        try:
            await rag.ainsert(doc["content"])
        except Exception as e:
            print(f"  ⚠️ 失败: {e}")
            continue


def init_and_insert(work_dir: str = WORK_DIR):
    """初始化 LightRAG 并插入所有文档"""
    print("=" * 50)
    print("  LightRAG 初始化 + 数据插入")
    print("=" * 50)

    docs = load_documents()
    print(f"共 {len(docs)} 篇文档")

    rag = create_rag_instance(work_dir)
    print(f"工作目录: {work_dir}")
    print(f"Embedding: 本地 Qwen3-Embedding-0.6B ({EMBEDDING_DIM}维)")
    print(f"LLM: DeepSeek ({DEEPSEEK_MODEL})")

    asyncio.run(insert_all(rag, docs))
    print(f"\n✅ 所有文档已插入！索引保存在: {work_dir}")


def query(question: str, mode: str = "hybrid", work_dir: str = WORK_DIR):
    """查询 LightRAG"""
    rag = create_rag_instance(work_dir)

    async def do_query():
        await rag.initialize_storages()
        result = await rag.aquery(question, param=QueryParam(mode=mode, hl_keywords=[], ll_keywords=[]))
        return result

    answer = asyncio.run(do_query())
    print(f"\n{'='*50}")
    print(f"  问题: {question}")
    print(f"  模式: {mode}")
    print(f"{'='*50}")
    print(answer)
    return answer


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightRAG for Buffett Letters")
    parser.add_argument("--init", action="store_true", help="初始化并插入数据")
    parser.add_argument("--query", type=str, help="查询问题")
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["naive", "local", "global", "hybrid"],
                        help="查询模式")
    args = parser.parse_args()

    if args.init:
        init_and_insert()
    elif args.query:
        query(args.query, mode=args.mode)
    else:
        parser.print_help()

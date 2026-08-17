# 巴菲特致股东信 知识图谱 + RAG 学习项目

以巴菲特致股东信（伯克希尔股东信、合伙人信、特别信件）为语料，循序实现三代 RAG 并做对比评测的学习项目：从基础向量检索，到混合检索 + 重排 + 查询改写，再到 LightRAG 知识图谱。检索结果还可导出为静态网站做图谱可视化。

## 环境配置

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 然后填入 DEEPSEEK_API_KEY
```

`.env` 需要 `DEEPSEEK_API_KEY`（DeepSeek 兼容 OpenAI API），字段参考 `.env.example`。

## 数据依赖

源代码中硬编码了数据目录 `/Users/boyang/Downloads/巴菲特致股东信`（含 `伯克希尔股东信/`、`合伙人信/`、`特别信件/` 三个子目录的 Markdown 文件）。换机器或换路径时需自行修改各脚本顶部的 `DATA_DIR`。

## 三代 RAG

| 代际 | 入口 | 说明 |
|---|---|---|
| L1 基础向量检索 | `python rag.py` | query → Chroma 向量检索 Top-5 → LLM。组件：`chunker.py`（分块）、`embedder.py`（本地 Qwen3-Embedding + Chroma） |
| L2 混合检索 | `python rag_v2.py` | query →（可选查询改写）→ 向量 + BM25 混合检索 → Reranker 精排 → LLM。组件：`hybrid_search.py`、`query_rewriter.py`、`reranker.py` |
| L3 知识图谱 | `python rag_lightrag.py --init` 建索引；`python rag_lightrag.py --query "问题"` 查询 | LightRAG：LLM 自动抽取实体关系建图谱，查询走图谱 + 向量双层。`insert_one.py` 用于补插漏掉的单篇文档 |

三代是递进关系：L1 打底（分块 + 向量库），L2 在 L1 的向量库上加 BM25/重排/改写提升检索质量，L3 换用 LightRAG 框架引入知识图谱以回答全局性、关系型问题。L3 的 LLM/Embedding 公共函数在 `rag_common.py`。

## 目录结构

```
├── rag.py / chunker.py / embedder.py   # L1：基础向量检索
├── rag_v2.py / hybrid_search.py / query_rewriter.py / reranker.py
│                                       # L2：混合检索 + 重排 + 查询改写
├── rag_lightrag.py / rag_common.py / insert_one.py
│                                       # L3：LightRAG 知识图谱
├── ragas_eval.py                       # Ragas 评测（L2/L3 对比，结果存 evaluation/）
├── export_graph.py / export_site.py    # 从 lightrag_workspace/ 导出图谱数据与静态网站（website/）
├── translations.py                     # 网站展示用的英文实体名 → 中文映射
├── chroma_db/                          # L1/L2 的向量库
├── bm25_cache.json                     # L2 的 BM25 分词缓存（带数据指纹，源文档或分块参数变化会自动重建）
├── lightrag_workspace/                 # L3 的索引（图谱 + 向量）
├── evaluation/                         # Ragas 评测结果
├── archive/                            # 已废弃的旧管线（自写实体抽取），仅存档
└── LEARNING_LOG_*.md / ARTICLE_*.md    # 学习笔记与文章
```

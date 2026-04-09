# buffett-knowledge

一个关于巴菲特致股东信的知识图谱项目。用一周时间断断续续、借助纯 AI 开发完成。

在线访问：[biboyang.github.io/buffett-knowledge](https://biboyang.github.io/buffett-knowledge/)

## 背景

灵感来自和朋友做财报 RAG 时的讨论——巴菲特致股东信适不适合做 RAG？调研后发现，这种跨年份、多实体的内容更适合做知识图谱，而非单纯的向量检索。但我之前从未接触过知识图谱开发，好在现在有 AI 的加持。

## 功能

- **信件浏览**：98 篇信件全文阅读，按年份和类别组织
- **实体索引**：概念（717 个）、公司（642 个）、人物（342 个），按出现频率排序
- **交叉链接**：点击实体可查看它在哪些信件中被提及，点击信件可查看涉及哪些实体
- **搜索**：支持按实体名和信件标题搜索

数据规模：98 篇信件 · 1,972 个实体 · 3,000 条关系 · 9,593 条交叉链接

## 技术栈

### 当前方案

| 组件 | 技术 | 说明 |
|------|------|------|
| 知识图谱框架 | LightRAG v1.4.13 | 实体抽取 + 图构建 + 查询 |
| LLM | DeepSeek | 实体抽取、问答、评测 |
| Embedding | Qwen3-Embedding-0.6B (1024维) | 向量化（本地运行） |
| Reranker | Qwen3-Reranker-0.6B | 重排序（本地运行） |
| 前端 | HTML + CSS + JS | 纯静态，无后端 |
| 图存储 | NetworkX (GraphML) | 轻量级图数据库 |
| 评测 | Ragas v0.4 | RAG 质量评测 |

### 模型演进

项目过程中经历了两次模型切换：

**Embedding & Reranker：从混合栈到统一 Qwen 栈**

最初使用了三个不同厂家的模型：all-MiniLM-L6-v2（Embedding，384维）、bge-reranker-v2-m3（Reranker）、DeepSeek（LLM）。后发现混用不同厂家模型风格不统一，中文表达能力也不够好。统一切换到 Qwen 系列后（Qwen3-Embedding-0.6B，1024维 + Qwen3-Reranker-0.6B），中文语义理解更强，实体数量从 10,836 增长到 11,064（+2.1%），关系数量从 14,711 增长到 15,134（+2.9%）。得益于 LightRAG 的 LLM 缓存机制，切换 Embedding 模型只需重新计算向量，不需要重新调用 LLM API，重建耗时从约 2 小时缩短到 40 分钟。

**知识图谱框架：为什么选 LightRAG 而不是 Microsoft GraphRAG**

调研阶段对比了 Microsoft GraphRAG 和 LightRAG，最终选择 LightRAG 的原因：

- 更轻量，小规模数据集（<10 万实体）完全够用
- Microsoft GraphRAG 依赖 Azure 服务，部署成本高
- LightRAG 内置 LLM 缓存，换模型可以避免重新调 API，节省最大头的费用
- 学习项目，先用简单的入门

## 数据集

| 类别 | 数量 | 时间范围 |
|------|------|----------|
| 伯克希尔股东信 | 60 篇 | 1965–2024 |
| 合伙人信 | 35 篇 | 1956–1970 |
| 特别信件 | 3 篇 | 2014 回顾、芒格思考、2025 感恩节致辞 |

原始数据为中文译本，约 160 万字。

## 项目结构

```
├── index.html              # 静态网站主页面
├── site_data.json          # 网站数据（实体、信件、关系）
├── graph_data.json         # 完整图谱数据（用于可视化）
├── export_site.py          # 数据导出脚本
├── rag_lightrag.py         # LightRAG 核心脚本
├── export_graph.py         # 图谱数据导出
├── ragas_eval.py           # Ragas 评测脚本
├── reranker.py             # Reranker 模块
├── hybrid_search.py        # 混合检索模块
├── translations.py         # 实体名中英文映射
├── lightrag_workspace/     # LightRAG 索引数据
├── evaluation/             # 评测结果
└── ARTICLE_KNOWLEDGE_GRAPH_JOURNEY.md  # 踩坑记录文章
```

## 学习记录

开发过程中记录了详细的学习日志：

- [Level 2 学习日志：高级检索](LEARNING_LOG_LEVEL2.md) — 混合检索、Reranking、查询改写
- [Level 3 学习日志：GraphRAG](LEARNING_LOG_LEVEL3.md) — 知识图谱构建、LightRAG 使用
- [从零开始搞知识图谱：踩坑实录](ARTICLE_KNOWLEDGE_GRAPH_JOURNEY.md) — 完整的踩坑记录文章

## 致谢

- 数据来源：巴菲特致股东信中文译本
- 开发工具：Claude Code + GLM-5.1

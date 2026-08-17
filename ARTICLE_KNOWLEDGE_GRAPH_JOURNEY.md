# 从零开始搞知识图谱：一个 RAG 初学者的踩坑实录

> 背景：我在做一个"巴菲特致股东信"的 RAG 项目，Level 1 做了基础向量检索，Level 2 做了混合检索 + Reranking，到了 Level 3 想试试知识图谱。整个过程从"自己手写实体抽取"到"用 LightRAG 一键搞定"，中间踩了无数的坑。这篇文章记录下真实的过程，希望能帮到同样从零开始的人。

---

## 1. 起点：为什么需要知识图谱？

我之前的 RAG 系统有一个明显短板：

```
问："巴菲特怎么看护城河？"
→ 向量检索能找到相关段落 ✅

问："巴菲特的投资理念经历了哪些变化？"
→ 需要跨 60 年的信件关联，向量检索做不到 ❌

问："护城河、安全边际、内在价值这三个概念之间是什么关系？"
→ 需要概念网络，向量检索只知道"相似"，不知道"关联" ❌
```

知识图谱的思路很直觉：把文档里的实体（人、公司、概念）和它们之间的关系（投资、收购、定义）抽出来，构建一张网，查询时沿着关系走，而不是靠语义相似度。

---

## 2. 第一次尝试：手写实体抽取（失败）

我的第一反应是："这不就是 NER + 关系抽取吗？我自己写一个。"

### 设计方案

我设计了一套 Schema：

```python
entity_types = ["person", "company", "concept"]
relation_types = ["invests_in", "acquired", "defines", "mentions", "related_to", "evolves_to"]
```

然后用 DeepSeek API 对每个文本分块做结构化抽取：

```python
prompt = """
从以下文本中抽取实体和关系...
实体类型：person, company, concept
关系类型：invests_in, acquired, defines, mentions, related_to
输出 JSON 格式...
"""
```

### 结果

跑到第 8 篇（1972 年）就卡住了。

```
99 篇文档 → 4310 个 chunks → 每个都要调一次 LLM API
8 篇文档就花了将近 1 小时
只抽出了 154 个实体、192 条关系
```

**问题：**
1. **太慢**：逐 chunk 调 API，速度完全不可接受
2. **格式不稳定**：LLM 有时输出合法 JSON，有时输出一坨带注释的"准 JSON"
3. **关系质量差**：大量 `mentions` 关系，几乎没有深层的 `defines`、`evolves_to`
4. **去重噩梦**：同一个实体"巴菲特"在不同 chunk 里被抽成 "Warren Buffett"、"巴菲特"、"沃伦·巴菲特"，合并逻辑越写越复杂

**教训：不要自己写实体抽取 pipeline，除非你有特殊需求。**

---

## 3. 第二次尝试：LightRAG（成功但有坑）

### 什么是 LightRAG

[LightRAG](https://github.com/HKUDS/LightRAG) 是一个开源的 GraphRAG 框架，核心功能：
- 自动对文档分块、调用 LLM 抽取实体和关系
- 自动构建知识图谱（用 NetworkX）
- 同时维护向量索引和图索引
- 查询时支持 4 种模式：naive（纯向量）、local（图局部）、global（图全局）、hybrid（混合）

### 配置

```python
rag = LightRAG(
    working_dir="./lightrag_workspace",
    llm_model_func=llm_model_func,       # DeepSeek API
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,                # Qwen3-Embedding-0.6B
        func=embedding_func,
    ),
)
```

### 踩坑 1："43 篇已处理，卡住了"

第一次跑 `--init`，过了一段时间来看，发现状态是：

```
processed: 53, failed: 50, TODO: 1, processing: 1
```

50 个 "failed"！乍一看以为 API 出了问题。

仔细一看 error message：

```
"Content already exists. Original doc_id: doc-79d01e..., Status: processed"
```

**原因：之前跑过一次，这次是重复插入。LightRAG 的去重机制把重复文档标记为 failed。** 实际上 53 篇都已经成功处理了。

**教训：LightRAG 的 `failed` 状态不一定代表真正的错误，要先看 `error_msg` 再判断。**

### 踩坑 2：GraphML 文件损坏

某天重启后，突然报错：

```
xml.etree.ElementTree.ParseError: not well-formed (invalid token): line 204949, column 17
```

打开文件一看：

```xml
    </edge>
  </graph>
</graphml>
">1.0</data>      ← 这里开始是重复数据
      <data key="d8">...
```

**原因：LightRAG 在写入 GraphML 文件时，没有先清空再写，而是追加了。导致文件包含了两个完整的 `</graphml>` 结构。**

修复方法：找到第一个 `</graphml>` 并截断：

```python
with open(graphml_file) as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if '</graphml>' in line:
        clean = lines[:i+1]
        break
with open(graphml_file, 'w') as f:
    f.writelines(clean)
```

**教训：如果 LightRAG 进程被强制中断，GraphML 文件可能损坏。建议定期备份。**

### 踩坑 3：空文档导致 "Set of Tasks/Futures is empty"

处理过程中反复出现这个错误：

```
ValueError: Set of Tasks/Futures is empty.
```

**原因：源数据目录里有一个空的 `.md` 文件（0 字节），LightRAG 对它分块后得到 0 个 chunks，创建了一个空的任务集合，`asyncio.wait([])` 抛出 ValueError。**

这个错误不会影响其他文档的处理，但会在日志里刷屏。解决方法是在数据加载时过滤掉空文件：

```python
if len(text.strip()) < 10:
    continue  # 跳过空文件
```

**教训：数据清洗永远是第一步。不要假设数据是干净的。**

### 踩坑 4：后台命令用 `| head` 导致 Broken Pipe

我在跑索引时用了 `| head -20` 想看看开头几行输出：

```bash
./venv/bin/python rag_lightrag.py --init 2>&1 | head -20
```

结果所有文档都报了 `[Errno 32] Broken pipe` 错误。

**原因：`head -20` 读够 20 行就退出了，管道断裂。Python 进程虽然继续运行，但 LLM 的输出管道已经断了，所有写操作都失败。**

进程看起来还活着（占内存、有 PID），但实际已经是个僵尸。

**教训：后台运行长任务时，不要用 `| head` 截断输出。直接重定向到文件：**

```bash
./venv/bin/python rag_lightrag.py --init > init.log 2>&1 &
```

---

## 4. 模型切换：从 bge 到 Qwen

### 为什么要换

项目初期用了混合模型栈：

| 组件 | 模型 | 来源 |
|------|------|------|
| Embedding | all-MiniLM-L6-v2 | sentence-transformers |
| Reranker | bge-reranker-v2-m3 | BAAI |
| LLM | DeepSeek | API |

三个不同厂家的模型，风格和能力不统一。之前和 AI 助手讨论时就决定了要统一到 Qwen 系列，但日志里一直写着 bge，代码也用的 bge，直到做完索引才发现。

### 换模型需要推倒重来吗？

**Embedding 模型（要重来）**：换 embedding 意味着所有向量都要重新生成，因为不同模型的向量维度和语义空间完全不同。

**Reranker（不用重来）**：Reranker 是无状态组件，只在查询时对候选结果重新打分，不参与索引构建。

| 组件 | 换模型的影响 | 需要重建？ |
|------|------------|-----------|
| Embedding | 所有向量失效 | ✅ 必须 |
| Reranker | 只影响查询时排序 | ❌ 不用 |
| LLM | 影响实体抽取质量 | ✅ 必须（但可缓存） |

### 重建过程

好消息是 LightRAG 有 LLM 缓存机制：

```
lightrag_workspace/
├── kv_store_llm_response_cache.json  ← 59MB, 3262 条 LLM 调用缓存
├── vdb_chunks.json                   ← 向量数据库（需要重建）
├── vdb_entities.json                 ← 向量数据库（需要重建）
├── vdb_relationships.json            ← 向量数据库（需要重建）
├── graph_chunk_entity_relation.graphml ← 知识图谱（需要重建）
└── ...其他 KV 存储（文本数据，不受影响）
```

**最贵的部分（3000+ 次 DeepSeek API 调用做实体抽取）被缓存了！**

重建步骤：
1. 改代码：embedding 维度 384 → 1024，模型名换成 Qwen3-Embedding-0.6B
2. 备份 `kv_store_llm_response_cache.json`
3. 清空 workspace 其他文件
4. 恢复 LLM 缓存
5. 重新跑 `--init` → LLM 抽取命中缓存（秒回），只需重新做 embedding + 图合并

### 重建结果

| 指标 | 旧 (all-MiniLM-L6-v2) | 新 (Qwen3-Embedding-0.6B) |
|------|---|---|
| 文档 | 98 篇 | 98 篇 |
| 实体 | 10,836 | **11,064** (+2.1%) |
| 关系 | 14,711 | **15,134** (+2.9%) |
| 向量维度 | 384 | 1024 |
| 重建耗时 | ~2 小时（含 API） | **~40 分钟**（缓存命中） |

实体和关系的增加可能是因为 Qwen3-Embedding 在中文文本上的表达能力更强，导致图合并时能识别出更多的等价实体。

---

## 5. 数据集概况

最终的数据集：

```
源数据：
  伯克希尔股东信：60 篇（1965-2024）
  合伙人信：35 篇（1956-1970）
  特别信件：3 篇（2014 伯克希尔回顾、芒格思考、2025 感恩节致辞）
  合计：98 篇有效文档，约 160 万字

知识图谱：
  11,064 个实体节点
  15,134 条关系边
  向量维度：1024（Qwen3-Embedding-0.6B）

技术栈：
  LLM：DeepSeek（实体抽取 + 问答）
  Embedding：Qwen3-Embedding-0.6B（本地，MLX 加速）
  Reranker：Qwen3-Reranker-0.6B（本地）
  GraphRAG 框架：LightRAG v1.4.13
  图存储：NetworkX（GraphML 格式）
  向量存储：nano-vectordb（LightRAG 内置）
```

---

## 6. 总结：给后来者的建议

### 技术选型

| 决策 | 建议 | 原因 |
|------|------|------|
| 实体抽取 | **不要自己写，用 LightRAG** | 自己写的 pipeline 慢、格式不稳定、去重复杂 |
| Embedding 模型 | **统一到一个系列** | 混用不同厂家的模型会导致风格不一致 |
| 图存储 | **先用 NetworkX** | 小规模数据（<10 万实体）完全够用，没必要上来就 Neo4j |
| 数据清洗 | **第一步就做** | 空文件、重复文档、编码问题都会在后面变成莫名其妙的 bug |

### 常见坑速查

| 错误 | 原因 | 解决 |
|------|------|------|
| `Content already exists` | 重复插入文档 | 正常行为，不是真的错误 |
| `ParseError: not well-formed` | GraphML 被重复写入 | 找到第一个 `</graphml>` 截断 |
| `Set of Tasks/Futures is empty` | 空文档导致 0 个 chunks | 过滤掉空文件 |
| `[Errno 32] Broken pipe` | 命令用了 `| head` 截断 | 不要截断长时任务的输出 |
| processed 没增长但进程还在 | 进程可能已变成僵尸 | 检查 workspace 文件的修改时间 |

### 心态

1. **索引构建是离线的，不要怕出错。** 大不了清掉重来，LLM 缓存能省掉最贵的部分。
2. **不要追求一步到位。** 先用默认配置跑通，再根据效果调优。
3. **数据质量 > 模型能力 > 算法复杂度。** 99 篇精心翻译的巴菲特信件，比 999 篇垃圾数据有价值得多。
4. **踩坑本身就是学习。** 如果一切顺利，你什么也学不到。

---

*本文基于真实项目 [buffett-rag] 的开发过程写成。项目使用巴菲特致股东信中文译本作为数据集，目的是学习 RAG 和知识图谱技术。*

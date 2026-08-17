# 从零开始构建知识图谱：用 99 篇巴菲特股东信做一次完整实践

我拿到了巴菲特从 1956 年到 2024 年写给股东和合伙人的信，想用知识图谱把这些信里的人物、公司、投资概念以及它们之间的关系结构化地整理出来。这篇文章记录整个构建过程，包括数据准备、技术选型、实体抽取、图谱构建，以及中间遇到的坑。

---

## 数据准备：从 PDF 到可用的文本

原始数据是英文 PDF。我用 LLM 辅助翻译整理成了中文 Markdown，一共 99 篇，约 160 万字：

- 伯克希尔股东信：60 篇（1965-2024）
- 合伙人信：35 篇（1956-1970）
- 特别信件：3 篇

翻译质量直接影响后续所有环节。如果翻译丢掉了关键实体名称（比如把 "National Indemnity Company" 统一翻译成"国民灾害保险公司"，而不是有时翻成"国民保险"有时翻成"国家赔偿"），实体抽取和去重会省很多事。这个坑后面会提到。

数据准备阶段的几件事值得注意：

- 统一文件命名（`1965-巴菲特致股东信.md`、`1957-巴菲特致合伙人信.md`），方便后续按年份和类型筛选
- 过滤掉空文件和内容过短的文件（我有一个空 `.md` 文件，后面引发了莫名其妙的报错）
- 表格数据尽量保留 Markdown 格式，不要丢失财务数字

---

## 为什么不用向量检索，要建知识图谱

向量检索能解决大部分问题。"巴菲特怎么看护城河？"这类问题，直接找语义相似的段落就能答得不错。

但有两类问题向量检索搞不定：

```
"巴菲特的投资理念经历了哪些变化？"
→ 跨 60 年的关联，向量检索只认识"相似"，不懂时序

"护城河、安全边际、内在价值之间是什么关系？"
→ 需要概念之间的关联网络，向量检索只知道"像不像"
```

知识图谱的思路：把文档里的实体（人、公司、概念）和它们之间的关系（投资、收购、定义）抽出来，建成一张网。查询的时候沿着关系走，而不是靠语义相似度。

需要说明的是，知识图谱和 RAG 不是一回事。知识图谱是一种数据结构，RAG 是一种架构模式。知识图谱可以作为 RAG 的一个检索源，但它本身是独立的。

---

## 技术选型

构建一个知识图谱，要做的核心决策有三个：用什么模型做 embedding、用什么调 LLM 做实体抽取、用什么框架把整个流程串起来。

### Embedding 模型

候选方案：

| 模型 | 参数量 | 上下文长度 | 中文 MTEB | 本地可跑 |
|------|--------|------------|-----------|----------|
| bge-m3 | 568M | 8K | 中上 | 可以 |
| Qwen3-Embedding-0.6B | 0.6B | 32K | 强 | 可以 |
| OpenAI text-embedding-3 | API | 8K | 强 | 付费 |

我选了 Qwen3-Embedding-0.6B。理由：

- 32K 上下文是关键优势——巴菲特的很多信件超过 8K token，用 bge-m3 会被截断
- 中文能力在 MTEB 基准上明显领先
- 0.6B 参数量，Mac 上跑得动

### LLM（实体抽取）

选了 DeepSeek API。便宜（大约 1 元/百万 token），中文好，接口兼容 OpenAI 格式。实体抽取需要大量调用 LLM，成本是个现实考量。

### 图谱构建框架

一开始我打算自己写实体抽取 pipeline，后来放弃了。原因下面单独说。最终选了 [LightRAG](https://github.com/HKUDS/LightRAG)，它把分块、实体抽取、图构建、向量索引全包了，查询时还支持四种模式（纯向量、图局部、图全局、混合）。

### 推理引擎

Embedding 推理从 `sentence-transformers`（PyTorch）换成了 `mlx-embeddings`（MLX）。模型没换，还是 Qwen3-Embedding-0.6B，只是跑推理的框架变了。MLX 是 Apple 为自家芯片写的框架，利用 Metal GPU 和统一内存架构，在 Apple Silicon 上比 PyTorch 快大约 3 倍。

---

## 构建流程

整个流程分四步：分块 → 实体抽取 → 图谱构建 → 向量索引。

### 分块

99 篇文档用递归分块策略，按标题→段落→句子→字符的优先级切分，尽量不破坏语义。最终切成 1018 个 chunks。

### 实体抽取：自己写的尝试和失败

我的第一版方案是自己写实体抽取。定义了实体类型和关系类型：

```python
entity_types = ["person", "company", "concept"]
relation_types = ["invests_in", "acquired", "defines", "mentions", "related_to", "evolves_to"]
```

给每个 chunk 拼 prompt，调 DeepSeek API 做结构化抽取。4310 个 chunks（早期分块策略），每个调一次 API。

跑到第 8 篇就放弃了，有几个根本性的问题：

**速度。** 逐 chunk 调 API，每条要等 LLM 返回，全部跑完预估 12 小时以上。

**格式不稳定。** LLM 有时输出合法 JSON，有时输出带注释的"准 JSON"，需要 json_repair 兜底。这个看起来是小问题，但当你有几千条调用的时候，异常处理逻辑写不完。

**实体去重。** 这是最麻烦的。同一个"巴菲特"，不同 chunk 里被抽成 "Warren Buffett"、"巴菲特"、"沃伦·巴菲特"、"Warren E. Buffett"。要做实体消歧，简单的字符串匹配不够用，但引入复杂的相似度计算又增加系统复杂度。

**关系质量。** 抽出来的大多是 `mentions` 关系（"某封信提到了某个概念"），真正有价值的 `defines`、`evolves_to` 很少。prompt 工程能改善一些，但根本问题在于单 chunk 的上下文太窄，LLM 很难判断两个实体的深层关系。

### 实体抽取：用 LightRAG

换成 LightRAG 之后这些问题基本解决了。它内部做了几件关键的事：

- 分块后对每个 chunk 调 LLM 抽取实体和关系，但有内置的 prompt 模板和格式兜底
- 跨 chunk 做实体合并（包括 LLM 辅助的语义消歧）
- 同时构建图结构和向量索引

配置代码：

```python
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from mlx_embeddings.utils import load as mlx_load

model, tokenizer = mlx_load("Qwen/Qwen3-Embedding-0.6B")

async def embedding_func(texts):
    all_embs = []
    for text in texts:
        ids = tokenizer.encode(text)
        mlx_ids = mx.array(np.array(ids, dtype=np.int32))[None, :]
        outputs = model(mlx_ids)
        all_embs.append(outputs.text_embeds)
    combined = mx.concatenate(all_embs, axis=0).astype(mx.float32)
    mx.eval(combined)
    return np.array(combined.tolist(), dtype=np.float32)

rag = LightRAG(
    working_dir="./lightrag_workspace",
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        func=embedding_func,
    ),
)
```

跑 `--init`，99 篇文档处理完毕：11,064 个实体，15,134 条关系。

### 图谱构建和向量索引

LightRAG 在抽取实体的同时完成两件事：

- 把实体和关系写入 GraphML 格式的知识图谱（基于 NetworkX）
- 把实体、关系、文本块的 embedding 存到 nano-vectordb（轻量向量库）

最终产出的文件：

```
lightrag_workspace/
├── graph_chunk_entity_relation.graphml    ← 知识图谱
├── vdb_chunks.json                        ← 文本块向量
├── vdb_entities.json                      ← 实体向量
├── vdb_relationships.json                 ← 关系向量
├── kv_store_llm_response_cache.json      ← LLM 调用缓存（61MB）
└── kv_store_full_docs.json                ← 原始文档
```

其中 `kv_store_llm_response_cache.json` 很有价值——它缓存了所有 LLM 调用的结果。如果后续需要重建（比如换 embedding 模型），实体抽取这一步不需要重新调 API，直接走缓存。

---

## 技术难点和踩坑

### LightRAG 重复插入的误报

跑完 `--init`，看到 `processed: 53, failed: 50`，吓一跳。50 个 failed 看着像出大事了。翻错误信息才发现全是 "Content already exists"——之前跑过一次，重复插入被去重机制拦了。实际上 53 篇都成功了。LightRAG 的 `failed` 不一定是真失败，先看 error message。

### GraphML 文件损坏

有次重启后报 `ParseError: not well-formed (invalid token): line 204949`。打开文件看，里面有两个完整的 `</graphml>` 结构。LightRAG 写入 GraphML 时没清空旧文件，直接追上了。

修复方法：找到第一个 `</graphml>` 截断后面的内容。如果进程被强制中断过，这个文件可能会坏，建议定期备份。

### 空文档引发的幽灵错误

日志里反复刷 `ValueError: Set of Tasks/Futures is empty`。查了半天，是数据目录里有个空 `.md` 文件。LightRAG 分块后得到 0 个 chunks，创建空任务集合，`asyncio.wait([])` 就抛 ValueError。不影响其他文档，但日志刷屏很烦。解决方法就是数据加载时过滤掉空文件：

```python
if len(text.strip()) < 10:
    continue
```

### 后台命令的 Broken Pipe

跑 `python rag_lightrag.py --init 2>&1 | head -20` 想看开头几行，`head` 读够 20 行就退出了，管道断裂，后面所有写入全报 `Broken pipe`。进程还在跑但已经是僵尸。长任务不要用管道截断，直接重定向到文件。

### MLX 的 bfloat16 兼容性

从 `sentence-transformers` 切换到 `mlx-embeddings` 时遇到了一个坑。MLX 输出的 dtype 是 bfloat16，numpy 不支持直接转换：

```python
np.array(embeddings, dtype=np.float32)
# RuntimeError: Item size 2 for PEP 3118 buffer format string B ...
```

解决方法是在 MLX 侧先转 float32，再通过 `tolist()` 过渡到 numpy：

```python
combined = mx.concatenate(all_embs, axis=0).astype(mx.float32)
mx.eval(combined)
return np.array(combined.tolist(), dtype=np.float32)
```

### 重建知识图谱

换 embedding 推理引擎后所有向量要重新生成。但实体抽取不需要重来——LightRAG 的 LLM 缓存（61MB，3500+ 条 API 结果）能直接复用。实际重建步骤：备份缓存文件，清空 workspace 其他文件，恢复缓存，重新跑 `--init`。LLM 抽取走缓存秒回，只重新做 embedding 和图合并，总耗时 5-10 分钟。

---

## 踩坑速查

| 报错 | 原因 | 怎么办 |
|------|------|--------|
| `Content already exists` | 重复插入文档 | 不用管，不是真报错 |
| `ParseError: not well-formed` | GraphML 被追加写入 | 找第一个 `</graphml>` 截断 |
| `Set of Tasks/Futures is empty` | 空文件导致 0 chunks | 数据加载时过滤空文件 |
| `[Errno 32] Broken pipe` | 管道命令截断 | 长任务重定向到文件 |
| `bfloat16` numpy 转换失败 | MLX 输出格式问题 | `astype(mx.float32).tolist()` |

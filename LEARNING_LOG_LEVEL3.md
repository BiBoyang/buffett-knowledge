# Level 3 学习日志：GraphRAG（知识图谱 + RAG）

> 日期：2026-04-09
> 前置：Level 1 基础 RAG + Level 2 高级检索已完成

---

## Level 3 要解决什么问题？

Level 1 和 2 的检索都是"局部"的 —— 找到语义相似的段落片段。
但有些问题是"全局"的：

```
Level 1/2 擅长的（局部问题）：
  "巴菲特怎么看护城河？" → 找到相关段落 → 回答

Level 2 做不到的（全局问题）：
  "巴菲特的投资理念经历了哪些变化？" → 需要跨年份关联
  "伯克希尔的保险业务是怎么发展起来的？" → 需要理清事件链
  "护城河、安全边际、内在价值这三个概念之间是什么关系？" → 需要概念网络

为什么向量检索做不到？
因为向量检索是"找相似的段落"，而不是"理解实体间的关系"。
```

GraphRAG 的思路：
```
文档 → LLM 抽取实体和关系 → 构建知识图谱 → 图遍历检索 → LLM 回答

相比向量检索：
  向量检索: "找到跟问题最像的段落"（基于语义相似度）
  图检索:   "沿着实体关系走，找到相关联的所有信息"（基于结构关系）
```

---

## 技术选型记录

### 3.1 实体和关系抽取

**方案**：用 DeepSeek LLM 对每个分块做结构化抽取

**Schema 设计**（预定义实体类型和关系类型）：

实体类型：
| 类型 | 说明 | 示例 |
|------|------|------|
| person | 人物 | 巴菲特、芒格、格雷厄姆 |
| company | 公司 | 伯克希尔、可口可乐、盖可保险 |
| concept | 投资概念 | 护城河、安全边际、内在价值 |

关系类型：
| 关系 | 说明 | 示例 |
|------|------|------|
| invests_in | 投资 | (巴菲特, invests_in, 可口可乐) |
| acquired | 收购 | (伯克希尔, acquired, 喜诗糖果) |
| defines | 定义 | (巴菲特, defines, 护城河) |
| mentions | 提到 | (1985年信, mentions, 浮存金) |
| related_to | 相关 | (护城河, related_to, 竞争优势) |
| evolves_to | 演变为 | (格雷厄姆式投资, evolves_to, 巴菲特式投资) |

**为什么用 LLM 做抽取而不是传统 NER**：
- 传统 NER 只能抽取通用实体（人名、地名）
- LLM 能理解金融/投资领域的语义关系
- LLM 能处理中英文混合文本
- 对于这种固定数据集，离线跑一次就行，成本可控

### 3.2 知识图谱存储

**方案对比**：

| 方案 | 特点 | 适合 |
|------|------|------|
| **NetworkX** | Python 库，最简单，纯内存 | 学习、小规模 |
| Neo4j | 专业图数据库，Cypher 查询 | 生产环境 |
| NebulaGraph | 大规模分布式图 | 超大规模 |

**决定**：NetworkX
- 99篇文档的实体量不大（预计几百个实体、几千条关系）
- 学习项目，不需要持久化图数据库
- NetworkX 的图算法 API 很直观

### 3.3 图检索 vs 向量检索

```
向量检索的局限：
  Query: "护城河和安全边际是什么关系？"
  → 可能只找到讲护城河的段落，或只找到讲安全边际的段落
  → 找不到两者之间的联系

图检索的优势：
  → 从"护城河"节点出发
  → 沿 related_to 边找到"竞争优势"、"特许经营权"
  → 沿 defines 边找到"巴菲特"
  → 沿 mentions 边找到讲这些概念的信件
  → 整个关联子图就是答案的上下文
```

---

## 实验记录

### Step 1: 安装 LightRAG + 数据准备 ✅

**决策：放弃自写 entity_extractor，改用 LightRAG**

原因：
- 自写方案太慢（4310 chunks 逐个调 LLM）
- 只跑到 1965-1972 年就卡住了（154 实体、192 关系）
- LightRAG 内置实体抽取 + 图构建，一步到位

**已完成：**
1. `pip install lightrag-hku` v1.4.13
2. 数据加载验证通过：99 篇文档，160 万字（伯克希尔股东信 60 + 合伙人信 35 + 特别信件 3）
3. 核心脚本 `rag_lightrag.py` 已写好，包含：
   - DeepSeek API 配置（llm_model_func + embedding_func）
   - 数据加载（load_documents，从原始 md 目录读取）
   - LightRAG 实例创建（create_rag_instance）
   - CLI 支持 `--init`（插入数据）和 `--query`（查询）

### Step 2: 索引构建（第一轮） ✅

**过程：**
- 首次跑 `--init`，99 篇文档通过 DeepSeek API 逐篇抽取实体和关系
- 逐篇插入模式（`insert_one.py`），每篇完成后通知用户
- 中途遇到 GraphML 文件损坏（重复写入导致 XML 格式错误），通过截断修复

**第一轮结果（all-MiniLM-L6-v2, 384维）：**
- 文档：98 篇（1 个空文件跳过）
- 实体：10,836 个
- 关系：14,711 条
- 耗时：约 2 小时（含 API 调用 + 停顿确认）

### Step 2b: 模型切换 + 索引重建 ✅

**背景：**
- Level 2 的 reranker 误用了 bge-reranker-v2-m3，实际讨论时已决定统一用 Qwen 系列
- 同时将 embedding 从 all-MiniLM-L6-v2 换成 Qwen3-Embedding-0.6B

**模型切换决策：**

| 组件 | 旧模型 | 新模型 | 原因 |
|------|--------|--------|------|
| Embedding | all-MiniLM-L6-v2 (384维) | **Qwen3-Embedding-0.6B** (1024维) | 统一 Qwen 栈，中文能力更强 |
| Reranker | bge-reranker-v2-m3 | **Qwen3-Reranker-0.6B** | 同上，已删除 bge |

**重建过程：**
1. 代码更新：`rag_lightrag.py`、`insert_one.py`、`reranker.py` 三个文件
2. 备份 LLM 缓存（`kv_store_llm_response_cache.json`, 59MB, 3262 条）→ 最贵的部分不需要重来
3. 清空 workspace 其他文件 → 重新跑 `--init`
4. LLM 抽取命中缓存（跳过 API 调用），只需重新做 embedding + 图合并

**重建结果（Qwen3-Embedding-0.6B, 1024维）：**
- 文档：98 篇，0 失败
- 实体：11,064 个（比旧模型多 ~230 个）
- 关系：15,134 条（比旧模型多 ~420 条）
- 耗时：约 40 分钟（LLM 缓存命中，无 API 费用）

**经验教训：**
1. LightRAG 的 LLM 缓存（`kv_store_llm_response_cache.json`）是重建时最宝贵的资产，换 embedding 模型不需要重新调 LLM
2. 切换 embedding 维度（384→1024）必须清空向量数据库，但图结构和文本数据不受影响
3. 后台命令不要用 `| head -20` 截断输出，会导致 Broken pipe 错误

### Step 2c: Embedding 推理引擎切换为 MLX ✅

**背景**：项目在 Apple Silicon Mac 上运行，将 embedding 推理引擎从 `sentence-transformers`（PyTorch）切换到 `mlx-embeddings`（MLX），利用 Apple 原生 Metal GPU 加速。

**改动文件**：`rag_lightrag.py`、`insert_one.py`、`embedder.py`、`rag.py`、`hybrid_search.py`

**性能对比**：

| | PyTorch (sentence-transformers) | MLX (mlx-embeddings) |
|---|---|---|
| 吞吐量 | ~3 texts/s | ~9 texts/s |
| 加速比 | - | **~3x** |
| 数值差异 | - | < 0.003（可忽略） |

**注意事项**：
- `mlx-embeddings` 的文本模型使用 `mlx_embeddings.utils.load`（不是 `mlx_embeddings.load`）
- 模型输出为 bfloat16，numpy 不直接支持，需用 `.astype(mx.float32).tolist()` 转换
- 模型权重不变（仍然是 `Qwen/Qwen3-Embedding-0.6B`），只是推理引擎不同
- 重建知识图谱时需清空 vdb 文件和 doc 状态，但 LLM 缓存可复用

**下一步（Step 3）：查询接口**
- 4 种模式：naive（纯向量）、local（图谱局部）、global（图谱全局）、hybrid（混合）
- 用 Level 2 擅长和不擅长的问题分别测试

**下一步（Step 4）：对比实验**
- 同样问题分别跑 Level 2 和 LightRAG
- 重点对比全局问题的回答质量

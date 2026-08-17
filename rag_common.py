"""
LightRAG 公共组件：配置 + LLM / Embedding 函数 + 实例创建

rag_lightrag.py（初始化 + 查询）和 insert_one.py（补插单篇）原来各自
重复定义了这套东西，抽到这里统一维护。

DeepSeek 的两个关键约定：
- 兼容 OpenAI API，环境变量 OPENAI_API_KEY 直接复用同一个 key
- 不支持 response_format，调用时要移除 keyword_extraction 标记
"""

import os
from pathlib import Path

import numpy as np
import mlx.core as mx
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from mlx_embeddings.utils import load as mlx_load

load_dotenv(Path(__file__).parent / ".env")

# ============================================================
# 配置
# ============================================================
DATA_DIR = "/Users/boyang/Downloads/巴菲特致股东信"
WORK_DIR = "./lightrag_workspace"
EMBEDDING_DIM = 1024  # Qwen3-Embedding-0.6B 的维度

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# LightRAG 内部部分代码检查 OPENAI_API_KEY 环境变量
# DeepSeek 兼容 OpenAI API，所以用同一个 key
if DEEPSEEK_API_KEY:
    os.environ.setdefault("OPENAI_API_KEY", DEEPSEEK_API_KEY)

# 本地 embedding 模型（MLX 加速，Apple Silicon 优化）
_embedding_model = None
_embedding_tokenizer = None

def _get_embedding_model():
    global _embedding_model, _embedding_tokenizer
    if _embedding_model is None:
        _embedding_model, _embedding_tokenizer = mlx_load("Qwen/Qwen3-Embedding-0.6B")
    return _embedding_model, _embedding_tokenizer


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    # DeepSeek 不支持 response_format，移除关键字抽取标记
    # LLM 仍会按 prompt 要求输出 JSON，json_repair 能解析
    kwargs.pop("keyword_extraction", None)
    return await openai_complete_if_cache(
        DEEPSEEK_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        **kwargs,
    )


async def embedding_func(texts):
    model, tokenizer = _get_embedding_model()
    all_embs = []
    for text in texts:
        ids = tokenizer.encode(text)
        mlx_ids = mx.array(np.array(ids, dtype=np.int32))[None, :]
        outputs = model(mlx_ids)
        all_embs.append(outputs.text_embeds)
    # 合并并转为 float32 numpy
    combined = mx.concatenate(all_embs, axis=0).astype(mx.float32)
    mx.eval(combined)
    return np.array(combined.tolist(), dtype=np.float32)


def create_rag_instance(work_dir: str = WORK_DIR) -> LightRAG:
    """创建 LightRAG 实例"""
    rag = LightRAG(
        working_dir=work_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    return rag

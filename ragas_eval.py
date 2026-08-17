"""
Level 4: Ragas 评测
用同一组测试问题，对比 Level 2 / Level 3 的回答质量。
"""

import os
import json
import asyncio
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

EVAL_DIR = Path(__file__).parent / "evaluation"

# ============================================================
# 测试问题集（覆盖不同类型）
# ============================================================
TEST_QUESTIONS = [
    # 局部问题（Level 1/2 擅长）
    {
        "question": "巴菲特怎么看护城河？",
        "ground_truth": "巴菲特认为护城河是企业能够长期维持竞争优势的保护机制，包括品牌、成本优势、网络效应、转换成本等。他在多个年份的股东信中都强调了护城河的重要性，特别是可口可乐、GEICO等公司拥有的护城河。",
    },
    {
        "question": "伯克希尔1985年的浮存金是多少？",
        "ground_truth": "1985年伯克希尔的保险浮存金约为4.5亿美元左右，保险业务为伯克希尔提供了大量可投资的低成本资金。",
    },
    {
        "question": "巴菲特为什么关闭纺织业务？",
        "ground_truth": "巴菲特关闭纺织业务是因为该业务长期资本回报率低下，虽然有情感上的不舍（这是伯克希尔的起源业务），但理性分析后认为继续投入资本是不明智的。纺织业面临海外低成本竞争，无论投入多少资本都难以获得合理回报。",
    },
    # 全局问题（Level 3 擅长）
    {
        "question": "巴菲特的投资理念经历了哪些变化？",
        "ground_truth": "巴菲特的投资理念经历了从格雷厄姆式的'捡烟蒂'投资（寻找被严重低估的股票）到费雪/芒格式的'以合理价格买入优秀企业'的转变。早期（1950s-1960s）专注于定量分析寻找便宜股票，后期（1970s以后）更注重企业质量和护城河，典型案例包括从纺织业务转向收购See's Candies等优质企业。",
    },
    {
        "question": "伯克希尔的保险业务是怎么发展起来的？",
        "ground_truth": "伯克希尔保险业务始于1967年收购National Indemnity Company（杰克·林沃尔特的公司），之后不断扩张，包括收购GEICO并最终全资拥有。保险浮存金（float）成为伯克希尔最重要的投资资金来源，Ajit Jain在再保险领域做出了巨大贡献。",
    },
    {
        "question": "护城河、安全边际、内在价值这三个概念之间是什么关系？",
        "ground_truth": "护城河保证企业能长期创造现金流（决定内在价值的可持续性），安全边际是以低于内在价值的价格买入（保护投资者免受判断错误的损失）。三者关系：护城河→可持续现金流→可计算内在价值→在安全边际内买入。这是巴菲特在格雷厄姆的基础上融入芒格的质量投资理念。",
    },
    # 关系型问题
    {
        "question": "芒格对巴菲特的投资风格有什么影响？",
        "ground_truth": "芒格促使巴菲特从格雷厄姆式的'捡烟蒂'投资转向'以合理价格买入伟大企业'。芒格认为以便宜价格买入平庸企业不如以合理价格买入优秀企业，这一思想深刻改变了伯克希尔的投资方向，如See's Candies的收购就是这一转变的标志性案例。",
    },
    {
        "question": "巴菲特和格雷厄姆的投资方法有什么不同？",
        "ground_truth": "格雷厄姆注重定量分析和安全边际，寻找价格远低于账面价值的股票（捡烟蒂策略）。巴菲特在格雷厄姆的基础上，融入了费雪的定性分析和芒格的企业质量理念，更关注企业的护城河、管理质量和长期竞争力，愿意为优秀企业支付合理价格。",
    },
    {
        "question": "伯克希尔收购了哪些重要公司？",
        "ground_truth": "伯克希尔重要收购包括：National Indemnity（1967）、See's Candies（1972）、GEICO（逐步收购至1996全资）、BNSF铁路（2010）、Precision Castparts（2016）等。这些收购横跨保险、食品、铁路、工业等多个行业，体现了巴菲特多元化的投资策略。",
    },
    {
        "question": "巴菲特怎么看股票回购？",
        "ground_truth": "巴菲特支持在股价低于内在价值时进行股票回购，认为这是回馈股东的有效方式。他多次在股东信中批评那些为了支撑股价或掩饰每股收益增长而进行的回购。2018年后伯克希尔也进行了大规模回购。",
    },
    {
        "question": "浮存金在伯克希尔的商业模式中扮演什么角色？",
        "ground_truth": "浮存金是保险业务收取的保费在赔付之前可以用于投资的资金。伯克希尔的保险浮存金从最初的几千万美元增长到超过1500亿美元，这些资金可以免费或极低成本使用，是伯克希尔投资帝国最重要的资金来源。Ajit Jain管理的再保险业务是浮存金增长的关键。",
    },
    {
        "question": "巴菲特如何看待通胀对投资的影响？",
        "ground_truth": "巴菲特认为通胀是投资者最大的敌人，因为它侵蚀购买力。他在1981年股东信等多次讨论通胀，认为有定价权的企业（如See's Candies）能抵御通胀，而需要大量资本支出的企业（如纺织业）在通胀中会受损。他倾向于投资轻资产、有定价权的企业。",
    },
]


# ============================================================
# Ragas 评测核心
# ============================================================
async def run_ragas_evaluation(questions_data, level_name, answer_func, context_func):
    """对某个 Level 的 RAG 运行 Ragas 评测"""
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    # 用 DeepSeek 作为 Ragas 的评判 LLM
    judge_llm = LangchainLLMWrapper(ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    ))

    samples = []
    for item in questions_data:
        q = item["question"]
        gt = item["ground_truth"]

        # 获取回答和上下文
        answer = await answer_func(q)
        contexts = await context_func(q)

        samples.append(SingleTurnSample(
            user_input=q,
            response=answer,
            reference=gt,
            retrieved_contexts=contexts,
        ))

    dataset = EvaluationDataset(samples=samples)

    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        ContextPrecision,
        ContextRecall,
    )

    metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    print(f"\n{'='*60}")
    print(f"  评测: {level_name}")
    print(f"{'='*60}")

    result = evaluate(dataset=dataset, metrics=metrics)

    # 转为可序列化的格式（Ragas v0.4+ 返回 EvaluationResult 对象）
    # result.scores 是 list[dict]，每个 sample 一个 dict
    raw_scores = result.scores
    scores = {}
    if raw_scores and isinstance(raw_scores, list):
        # 收集每个指标的所有值，取平均
        metric_values = {}
        for sample_scores in raw_scores:
            for metric_name, value in sample_scores.items():
                if metric_name == "user_input":
                    continue
                if value is None:
                    continue
                if hasattr(value, 'item'):
                    value = value.item()
                try:
                    metric_values.setdefault(metric_name, []).append(float(value))
                except (TypeError, ValueError):
                    pass
        for metric_name, values in metric_values.items():
            scores[metric_name] = round(sum(values) / len(values), 4)

    # 同时保存每个问题的详细分数
    per_question = []
    for i, sample_scores in enumerate(raw_scores):
        qs = {}
        for k, v in sample_scores.items():
            if hasattr(v, 'item'):
                v = v.item()
            try:
                qs[k] = round(float(v), 4) if v is not None else None
            except (TypeError, ValueError):
                qs[k] = str(v)
        per_question.append(qs)

    print(f"  结果: {json.dumps(scores, ensure_ascii=False, indent=2)}")
    return scores, per_question


# ============================================================
# Level 回答函数
# ============================================================
# 共享实例：HybridSearch / Reranker / LightRAG 初始化都要加载约 1.2GB 模型，
# 每个问题新建一次实例会把整轮评测拖慢几十倍，这里全局只创建一次、复用。
_shared_instances = {}

def _get_hybrid_search():
    if "hybrid_search" not in _shared_instances:
        from hybrid_search import HybridSearch
        _shared_instances["hybrid_search"] = HybridSearch()
    return _shared_instances["hybrid_search"]

def _get_reranker():
    if "reranker" not in _shared_instances:
        from reranker import Reranker
        _shared_instances["reranker"] = Reranker()
    return _shared_instances["reranker"]

async def _get_lightrag():
    if "lightrag" not in _shared_instances:
        from rag_lightrag import create_rag_instance
        rag = create_rag_instance()
        await rag.initialize_storages()
        _shared_instances["lightrag"] = rag
    return _shared_instances["lightrag"]


async def level2_answer(question):
    """Level 2: 混合检索 + Reranking"""
    hs = _get_hybrid_search()
    reranker = _get_reranker()
    candidates = hs.search(question, top_k=20, mode="hybrid")
    reranked = reranker.rerank(question, candidates, top_k=5)
    contexts = [doc["text"] for doc in reranked]

    # 用 DeepSeek 生成回答
    from lightrag.llm.openai import openai_complete_if_cache
    context_text = "\n\n".join(contexts)
    prompt = f"基于以下上下文回答问题。如果上下文中没有答案，请说明。\n\n上下文：\n{context_text}\n\n问题：{question}"
    answer = await openai_complete_if_cache(
        DEEPSEEK_MODEL, prompt,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
    return answer

async def level2_context(question):
    hs = _get_hybrid_search()
    reranker = _get_reranker()
    candidates = hs.search(question, top_k=20, mode="hybrid")
    reranked = reranker.rerank(question, candidates, top_k=5)
    return [doc["text"] for doc in reranked]

async def level3_answer(question):
    """Level 3: LightRAG"""
    from lightrag import QueryParam
    rag = await _get_lightrag()
    # 预设空关键字列表，跳过 LLM 关键字抽取（DeepSeek 不支持 response_format）
    result = await rag.aquery(question, param=QueryParam(mode="hybrid", hl_keywords=[], ll_keywords=[]))
    return result

async def level3_context(question):
    from lightrag import QueryParam
    rag = await _get_lightrag()
    # naive 模式也需要跳过关键字抽取
    result = await rag.aquery(question, param=QueryParam(mode="naive", hl_keywords=[], ll_keywords=[]))
    return [str(result)]


# ============================================================
# 主流程
# ============================================================
async def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "test_questions": len(TEST_QUESTIONS),
        "levels": {},
    }

    # Level 1 已移除：rag.py 没有 create_rag / get_contexts 接口，无法评测
    levels_to_eval = [
        ("Level 2: Hybrid + Rerank", level2_answer, level2_context),
        ("Level 3: LightRAG (hybrid)", level3_answer, level3_context),
    ]

    for level_name, answer_fn, context_fn in levels_to_eval:
        try:
            scores, per_question = await run_ragas_evaluation(
                TEST_QUESTIONS, level_name, answer_fn, context_fn
            )
            results["levels"][level_name] = {
                "avg_scores": scores,
                "per_question": per_question,
            }
        except Exception as e:
            print(f"  ❌ {level_name} 评测失败: {e}")
            results["levels"][level_name] = {"error": str(e)}

    # 保存结果
    result_file = EVAL_DIR / f"ragas_eval_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  评测结果已保存: {result_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())

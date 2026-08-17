"""
Step 3.1 优化版：智能抽样抽取实体和关系

策略：从每个年份挑1-2篇关键信件，按段落级抽取。
覆盖全部年份但减少 API 调用次数。

预计 API 调用：约 100-150 次（而非 4310 次）
"""

import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from chunker import load_all_documents

load_dotenv()

ENTITY_TYPES = {
    "person": "人物",
    "company": "公司/组织",
    "concept": "投资概念/理念",
}

RELATION_TYPES = {
    "invests_in": "投资",
    "acquired": "收购",
    "defines": "定义/解释",
    "related_to": "相关/相似",
    "contrasts_with": "对比/相反",
    "leads_to": "导致/推动",
    "evolves_to": "演变为",
    "mentions": "提到",
}

EXTRACTION_PROMPT = """从以下巴菲特致股东信段落中抽取实体和关系。

实体类型: person(人物), company(公司), concept(概念)
关系类型: invests_in(投资), acquired(收购), defines(定义), related_to(相关), contrasts_with(对比), leads_to(导致), evolves_to(演变), mentions(提到)

规则：
1. 实体名称标准化用最常用名称（如"伯克希尔"而非"伯克希尔·哈撒韦公司"）
2. 尽量用具体语义关系，少用 mentions
3. 只输出 JSON

来源：{source} ({year}年)

{text}

输出JSON：
```json
{{"entities": [{{"name": "名", "type": "类型"}}], "relations": [{{"head": "A", "relation": "关系", "tail": "B"}}]}}
```"""


def smart_sample(docs: list[dict], max_paragraphs: int = 600) -> list[dict]:
    """
    智能抽样：从每年选关键段落，而非逐块处理全部4310个分块。

    策略：
    1. 每年选1封最重要的信（年度信，非半年度/季度信）
    2. 每封信按段落拆分，过滤掉太短的（<100字）和纯表格数据
    3. 保留最多 max_paragraphs 个段落
    """
    # 按年份分组，每年选主信
    by_year = {}
    for doc in docs:
        y = doc["year"]
        name = doc["filename"]
        # 优先选不带"年中"/"11月"/"12月"等修饰的年度信
        if y not in by_year:
            by_year[y] = doc
        else:
            # 年度信优先于半年度/季度信
            current = by_year[y]["filename"]
            is_annual = not any(s in current for s in ["年中", "11月", "12月", "5月", "10月"])
            new_is_annual = not any(s in name for s in ["年中", "11月", "12月", "5月", "10月"])
            if new_is_annual and not is_annual:
                by_year[y] = doc

    print(f"  选出 {len(by_year)} 个年份的主信")

    # 按段落拆分，过滤
    paragraphs = []
    for year, doc in sorted(by_year.items()):
        paras = re.split(r'\n\n+', doc["text"])
        for p in paras:
            p = p.strip()
            # 过滤：太短、纯表格行、纯链接
            if len(p) < 100:
                continue
            if p.startswith('|') and p.endswith('|'):
                continue
            if len(p) > 2000:
                p = p[:2000]  # 截断
            paragraphs.append({
                "text": p,
                "source": doc["filename"],
                "year": year,
            })

    # 如果还是太多，均匀采样
    if len(paragraphs) > max_paragraphs:
        step = len(paragraphs) / max_paragraphs
        paragraphs = [paragraphs[int(i * step)] for i in range(max_paragraphs)]

    print(f"  抽样出 {len(paragraphs)} 个段落")
    return paragraphs


def extract_batch(extractor, paragraphs: list[dict], batch_size: int = 3) -> tuple:
    """
    批量抽取：把多个段落合并成一个 API 调用。

    batch_size=3: 每次把3个段落拼在一起送给 LLM
    → API 调用次数 = 段落数 / 3
    """
    all_entities = {}
    all_relations = []

    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i + batch_size]
        if (i + 1) % 30 == 0 or i == 0:
            print(f"  处理 {i+1}/{len(paragraphs)}...")

        # 拼接多个段落
        combined_text = ""
        for p in batch:
            combined_text += f"\n---\n来源: {p['source']} ({p['year']}年)\n{p['text']}\n"

        prompt = EXTRACTION_PROMPT.format(
            source=", ".join(p["source"] for p in batch),
            year=batch[0]["year"],
            text=combined_text,
        )

        try:
            response = extractor.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是精确的JSON输出器。只输出JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            for e in result.get("entities", []):
                name = e.get("name", "")
                etype = e.get("type", "")
                if name and etype in ENTITY_TYPES:
                    if name not in all_entities:
                        all_entities[name] = {"name": name, "type": etype, "count": 0, "years": set()}
                    all_entities[name]["count"] += 1
                    for p in batch:
                        all_entities[name]["years"].add(p["year"])

            for r in result.get("relations", []):
                if r.get("head") and r.get("relation") in RELATION_TYPES and r.get("tail"):
                    all_relations.append({
                        "head": r["head"],
                        "relation": r["relation"],
                        "tail": r["tail"],
                        "year": batch[0]["year"],
                        "source": batch[0]["source"],
                    })

        except Exception as e:
            print(f"  ⚠️ 失败: {e}")
            continue

    return list(all_entities.values()), all_relations


def build_knowledge_base(
    data_dir: str = "/Users/boyang/Downloads/巴菲特致股东信",
    output_path: str = "./knowledge_base.json",
):
    """构建知识库（优化版）"""
    print("=" * 50)
    print("  Step 3.1: 实体和关系抽取（智能抽样版）")
    print("=" * 50)

    docs = load_all_documents(data_dir)
    print(f"共 {len(docs)} 篇文档")

    paragraphs = smart_sample(docs)
    print(f"API 调用次数预估: ~{len(paragraphs) // 3 + 1}")

    extractor = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    entities, relations = extract_batch(extractor, paragraphs, batch_size=3)

    # 序列化
    for e in entities:
        if isinstance(e.get("years"), set):
            e["years"] = sorted(e["years"])

    kb = {
        "entities": entities,
        "relations": relations,
        "stats": {
            "total_entities": len(entities),
            "total_relations": len(relations),
            "entity_types": {
                t: sum(1 for e in entities if e["type"] == t)
                for t in ENTITY_TYPES
            },
            "relation_types": {
                t: sum(1 for r in relations if r["relation"] == t)
                for t in RELATION_TYPES
            },
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 抽取完成！")
    print(f"   实体: {kb['stats']['total_entities']} ({kb['stats']['entity_types']})")
    print(f"   关系: {kb['stats']['total_relations']} ({kb['stats']['relation_types']})")
    print(f"   保存到: {output_path}")
    return kb


if __name__ == "__main__":
    kb = build_knowledge_base()

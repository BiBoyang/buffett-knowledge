"""
导出静态网站所需的全部数据。
从 LightRAG 的 KV store 提取实体-信件映射，生成 JSON 给前端使用。
"""

import json
import os
import re
import time
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
from openai import OpenAI

from translations import ENTITY_ZH

WS = Path("./lightrag_workspace")
OUT = Path(".")  # 输出到仓库根目录（GitHub Pages 直接部署根目录）
DESC_CACHE = Path("./desc_translations.json")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def translate_descriptions(entities):
    """用 DeepSeek API 批量翻译实体描述，结果缓存到本地文件"""
    # 加载缓存
    cache = {}
    if DESC_CACHE.exists():
        with open(DESC_CACHE, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  已加载翻译缓存: {len(cache)} 条")

    # 找出需要翻译的
    to_translate = []
    for e in entities:
        if not e.get("description"):
            continue
        if e["name"] in cache:
            e["description"] = cache[e["name"]]
        else:
            to_translate.append(e)

    if not to_translate:
        print("  所有描述已翻译（缓存命中）")
        return

    print(f"  需要翻译: {len(to_translate)} 条描述")

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    batch_size = 20
    translated = 0
    for i in range(0, len(to_translate), batch_size):
        batch = to_translate[i:i + batch_size]
        # 构建批量翻译 prompt
        items = []
        for j, e in enumerate(batch):
            items.append(f"[{j}] {e['name']}: {e['description']}")
        prompt = (
            "将以下实体描述翻译成中文。保留实体名称不翻译，只翻译描述部分。"
            "输出格式为 JSON 数组，每个元素是翻译后的描述字符串。"
            "不要输出任何其他内容。\n\n"
            + "\n".join(items)
        )

        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            text = resp.choices[0].message.content.strip()
            # 提取 JSON
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            results = json.loads(text)

            for j, e in enumerate(batch):
                if j < len(results):
                    e["description"] = results[j]
                    cache[e["name"]] = results[j]
                    translated += 1
        except Exception as ex:
            print(f"  翻译批次 {i//batch_size} 失败: {ex}")
            # 单条翻译兜底
            for e in batch:
                try:
                    resp = client.chat.completions.create(
                        model=DEEPSEEK_MODEL,
                        messages=[{"role": "user", "content": f"翻译成中文，只输出翻译结果:\n{e['name']}: {e['description']}"}],
                        temperature=0.1,
                    )
                    translated_text = resp.choices[0].message.content.strip()
                    # 去掉可能的 "实体名: " 前缀
                    prefix = e["name"] + ": "
                    if translated_text.startswith(prefix):
                        translated_text = translated_text[len(prefix):]
                    elif translated_text.startswith(e["name"] + "："):
                        translated_text = translated_text[len(e["name"]) + 1:]
                    e["description"] = translated_text
                    cache[e["name"]] = translated_text
                    translated += 1
                except Exception:
                    pass

        # 每批保存缓存
        with open(DESC_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

        print(f"  翻译进度: {translated}/{len(to_translate)}")
        time.sleep(0.5)  # 避免 rate limit

    print(f"  翻译完成: {translated} 条")


def load_kv(name):
    with open(WS / f"kv_store_{name}.json") as f:
        return json.load(f)


def parse_letter_title(summary):
    """从 content_summary 提取年份和标题"""
    if not summary:
        return None, None
    summary = summary.strip()
    # 第一行通常是 "# 1965 巴菲特致股东信" 之类
    first_line = summary.split("\n")[0].strip().lstrip("#").strip()
    year_match = re.search(r"(\d{4})", first_line)
    year = int(year_match.group(1)) if year_match else 0
    return year, first_line


def categorize_letter(title, file_path=""):
    """根据标题或路径分类信件"""
    if "合伙" in title or "partnership" in file_path.lower():
        return "partnership"
    elif "特别" in title or "感恩节" in title or "回顾" in title or "思考" in title:
        return "special"
    else:
        return "berkshire"


def export_site():
    print("加载 LightRAG 数据...")

    doc_status = load_kv("doc_status")
    full_docs = load_kv("full_docs")
    full_entities = load_kv("full_entities")
    entity_chunks = load_kv("entity_chunks")

    # 加载图谱获取实体类型、描述和关系
    G = nx.read_graphml(WS / "graph_chunk_entity_relation.graphml")
    entity_types = {nid: d.get("entity_type", "unknown") for nid, d in G.nodes(data=True)}
    entity_descriptions = {}
    for nid, d in G.nodes(data=True):
        desc = d.get("description", "")
        if desc:
            # 取第一段描述（多段用 <SEP> 分隔）
            entity_descriptions[nid] = desc.split("<SEP>")[0].strip()

    print(f"  文档: {len(doc_status)}")
    print(f"  实体: {len(entity_chunks)}")

    # ==========================================
    # 1. 构建信件数据
    # ==========================================
    letters = []
    doc_id_map = {}  # doc_id -> letter index

    for doc_id, info in doc_status.items():
        if info.get("status") != "processed":
            continue
        summary = info.get("content_summary", "")
        year, title = parse_letter_title(summary)
        if not title:
            continue

        category = categorize_letter(title, info.get("file_path", ""))

        # 该信件包含的实体
        entity_names = []
        if doc_id in full_entities:
            entity_names = full_entities[doc_id].get("entity_names", [])

        letter = {
            "id": len(letters),
            "doc_id": doc_id,
            "year": year,
            "title": title,
            "category": category,
            "length": info.get("content_length", 0),
            "entity_count": len(entity_names),
            "content": full_docs.get(doc_id, {}).get("content", ""),
        }
        doc_id_map[doc_id] = letter["id"]
        letters.append(letter)

    letters.sort(key=lambda l: l["year"])

    # 重建 id（排序后）
    for i, l in enumerate(letters):
        doc_id_map[l["doc_id"]] = i
        l["id"] = i

    print(f"  有效信件: {len(letters)}")

    # ==========================================
    # 2. 构建实体数据（按类型分组）
    # ==========================================
    # 统计每个实体出现在多少封信里
    entity_letter_map = defaultdict(set)  # entity_name -> set of letter ids

    for doc_id, info in full_entities.items():
        if doc_id not in doc_id_map:
            continue
        letter_id = doc_id_map[doc_id]
        for name in info.get("entity_names", []):
            entity_letter_map[name].add(letter_id)

    # 按类型分组，只保留有信件关联的实体
    type_names = {
        "person": "人物",
        "organization": "公司",
        "concept": "概念",
        "event": "事件",
        "location": "地点",
    }

    # 筛选有意义的实体（出现在 >= 2 封信里）
    # Step 1: 收集所有符合条件的原始实体，建立 raw_name -> display_name 映射
    raw_to_display = {}  # raw_name -> 翻译后的中文名
    raw_to_letter_ids = {}  # raw_name -> set of letter ids
    raw_to_type = {}  # raw_name -> entity type
    raw_to_desc = {}  # raw_name -> description

    for name, letter_ids in entity_letter_map.items():
        if len(letter_ids) < 2:
            continue
        etype = entity_types.get(name, "unknown")
        if etype not in ("person", "organization", "concept", "event", "location"):
            continue
        raw_to_display[name] = ENTITY_ZH.get(name, name)
        raw_to_letter_ids[name] = letter_ids
        raw_to_type[name] = etype
        raw_to_desc[name] = entity_descriptions.get(name, "")

    # Step 2: 按 display_name 分组，合并同名实体
    display_groups = defaultdict(list)  # display_name -> [raw_names]
    for raw_name, display in raw_to_display.items():
        display_groups[display].append(raw_name)

    entities = []
    for display_name, raw_names in display_groups.items():
        all_letter_ids = set()
        for rn in raw_names:
            all_letter_ids.update(raw_to_letter_ids[rn])
        etype = raw_to_type[raw_names[0]]
        # 取最长的描述（同名实体可能有不同描述）
        desc = ""
        for rn in raw_names:
            d = raw_to_desc.get(rn, "")
            if len(d) > len(desc):
                desc = d
        entities.append({
            "id": len(entities),
            "name": display_name,
            "type": etype,
            "type_zh": type_names.get(etype, etype),
            "letter_count": len(all_letter_ids),
            "letter_ids": sorted(all_letter_ids),
            "description": desc,
        })

    # Step 3: 按 letter_count 降序排序，分配最终 id
    entities.sort(key=lambda e: e["letter_count"], reverse=True)
    display_to_final_id = {}  # display_name -> final id
    for i, e in enumerate(entities):
        e["id"] = i
        display_to_final_id[e["name"]] = i

    # Step 4: 建立 raw_name -> final_id 映射（用于关系查询）
    entity_name_to_id = {}
    for raw_name, display in raw_to_display.items():
        fid = display_to_final_id.get(display)
        if fid is not None:
            entity_name_to_id[raw_name] = fid

    print(f"  有效实体（>=2封信）: {len(entities)}")
    type_dist = Counter(e["type"] for e in entities)
    for t, c in type_dist.most_common():
        print(f"    {type_names.get(t, t)}: {c}")

    # ==========================================
    # 3. 构建信件→实体关联
    # ==========================================
    for letter in letters:
        letter["entity_ids"] = []

    for entity in entities:
        for lid in entity["letter_ids"]:
            letters[lid]["entity_ids"].append(entity["id"])

    # ==========================================
    # 4. 构建实体间关系（取重要关系）
    # ==========================================
    relationships = []
    for u, v, data in G.edges(data=True):
        uid = entity_name_to_id.get(u)
        vid = entity_name_to_id.get(v)
        if uid is None or vid is None:
            continue
        keywords = data.get("keywords", "")
        main_kw = keywords.split(",")[0].strip() if keywords else ""
        relationships.append({
            "source": uid,
            "target": vid,
            "type": main_kw,
            "weight": float(data.get("weight", 1)),
        })

    # 按权重降序，取 top 关系
    relationships.sort(key=lambda r: r["weight"], reverse=True)

    print(f"  实体间关系: {len(relationships)}")

    # ==========================================
    # 5. 汇总统计
    # ==========================================
    category_counts = Counter(l["category"] for l in letters)
    category_names = {"berkshire": "伯克希尔股东信", "partnership": "合伙人信", "special": "特别信件"}

    meta = {
        "total_letters": len(letters),
        "total_entities": len(entities),
        "total_relationships": len(relationships),
        "total_links": sum(e["letter_count"] for e in entities),
        "letter_categories": {category_names.get(k, k): v for k, v in category_counts.items()},
        "entity_types": {type_names.get(t, t): c for t, c in type_dist.most_common()},
        "year_range": f"{min(l['year'] for l in letters)}–{max(l['year'] for l in letters)}",
    }

    # ==========================================
    # 6. 翻译描述
    # ==========================================
    if DEEPSEEK_API_KEY:
        print("\n翻译实体描述...")
        translate_descriptions(entities)
    else:
        print("\n未设置 DEEPSEEK_API_KEY，跳过描述翻译")

    # ==========================================
    # 7. 输出
    # ==========================================
    output = {
        "meta": meta,
        "letters": letters,
        "entities": entities,
        "relationships": relationships[:3000],  # 限制关系数量
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "site_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n导出完成: {out_path} ({size_mb:.1f} MB)")
    print(f"  信件: {meta['total_letters']}")
    print(f"  实体: {meta['total_entities']}")
    print(f"  关系: {len(output['relationships'])}")
    print(f"  交叉链接: {meta['total_links']}")


if __name__ == "__main__":
    export_site()

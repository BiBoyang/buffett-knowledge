"""
导出知识图谱数据为静态网站用的 JSON
"""

import json
import networkx as nx
from pathlib import Path
from collections import Counter


def export_graph(graphml_path: str, output_dir: str):
    G = nx.read_graphml(graphml_path)

    print(f"加载图谱: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    # 统计实体类型分布
    type_counts = Counter(d.get("entity_type", "unknown") for _, d in G.nodes(data=True))
    print(f"实体类型: {dict(type_counts.most_common(10))}")

    # 导出实体
    entities = []
    entity_id_map = {}  # 原始 node_id -> 索引
    for i, (node_id, data) in enumerate(G.nodes(data=True)):
        entity_type = data.get("entity_type", "unknown")
        description = data.get("description", "")

        # 截断过长描述（保留前 200 字符）
        short_desc = description.replace("<SEP>", " | ").strip()
        if len(short_desc) > 200:
            short_desc = short_desc[:200] + "..."

        entity = {
            "id": i,
            "name": node_id,
            "type": entity_type,
            "desc": short_desc,
        }
        entities.append(entity)
        entity_id_map[node_id] = i

    # 导出关系
    relationships = []
    for u, v, data in G.edges(data=True):
        src_id = entity_id_map.get(u)
        tgt_id = entity_id_map.get(v)
        if src_id is None or tgt_id is None:
            continue

        keywords = data.get("keywords", "")
        # 取第一个关键词作为主要关系类型
        main_keyword = keywords.split(",")[0].strip() if keywords else "related_to"

        description = data.get("description", "")
        if len(description) > 150:
            description = description[:150] + "..."

        relationships.append({
            "source": src_id,
            "target": tgt_id,
            "type": main_keyword,
            "keywords": keywords,
            "desc": description,
            "weight": float(data.get("weight", 1)),
        })

    # 按连接度排序实体（degree 高的排前面）
    degree = dict(G.degree())
    entities.sort(key=lambda e: degree.get(e["name"], 0), reverse=True)
    # 重建 id map（排序后 id 变了）
    entity_id_map = {}
    for i, e in enumerate(entities):
        e["id"] = i
        entity_id_map[e["name"]] = i
    # 更新关系中的 source/target
    for r in relationships:
        r["source"] = entity_id_map.get(list(G.nodes())[r["source"]], r["source"])
        r["target"] = entity_id_map.get(list(G.nodes())[r["target"]], r["target"])

    # 用 node name 查找新 id
    name_to_new_id = {e["name"]: e["id"] for e in entities}
    for r in relationships:
        src_name = None
        tgt_name = None
        for u, v, _ in G.edges(data=True):
            if entity_id_map.get(u) == r["source"]:
                src_name = u
            if entity_id_map.get(v) == r["target"]:
                tgt_name = v
            if src_name and tgt_name:
                break

    # 简单方法：重新遍历所有边，用 name 直接映射
    relationships = []
    for u, v, data in G.edges(data=True):
        src_id = name_to_new_id.get(u)
        tgt_id = name_to_new_id.get(v)
        if src_id is None or tgt_id is None:
            continue

        keywords = data.get("keywords", "")
        main_keyword = keywords.split(",")[0].strip() if keywords else "related_to"

        description = data.get("description", "")
        if len(description) > 150:
            description = description[:150] + "..."

        relationships.append({
            "source": src_id,
            "target": tgt_id,
            "type": main_keyword,
            "keywords": keywords,
            "desc": description,
            "weight": float(data.get("weight", 1)),
        })

    # 为每个实体预计算关系数
    entity_rel_count = Counter()
    for r in relationships:
        entity_rel_count[r["source"]] += 1
        entity_rel_count[r["target"]] += 1
    for e in entities:
        e["rel_count"] = entity_rel_count.get(e["id"], 0)

    output_data = {
        "meta": {
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "entity_types": dict(type_counts.most_common()),
        },
        "entities": entities,
        "relationships": relationships,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 写 JSON
    json_path = out_path / "graph_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = json_path.stat().st_size / 1024 / 1024
    print(f"\n导出完成:")
    print(f"  实体: {len(entities)}")
    print(f"  关系: {len(relationships)}")
    print(f"  文件: {json_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    export_graph(
        "./lightrag_workspace/graph_chunk_entity_relation.graphml",
        "./website",
    )

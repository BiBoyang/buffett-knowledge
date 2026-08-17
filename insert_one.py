"""
逐篇插入文档到 LightRAG
用法: python insert_one.py <索引号>
  索引号对应 missing_docs.txt 中的序号
"""
import sys
import json
import asyncio
from pathlib import Path

# 配置、LLM / Embedding 函数、实例创建已抽到 rag_common.py
from rag_common import DATA_DIR, WORK_DIR, create_rag_instance

def get_missing_docs():
    """找出所有未索引的文档"""
    base = Path(DATA_DIR)

    # Get stored doc lengths
    stored_path = Path(WORK_DIR) / "kv_store_full_docs.json"
    if stored_path.exists():
        with open(stored_path) as f:
            stored = json.load(f)
        stored_lens = set()
        for v in stored.values():
            if isinstance(v, dict):
                stored_lens.add(len(v.get('content', '').strip()))
    else:
        stored_lens = set()

    missing = []
    for folder in ['伯克希尔股东信', '合伙人信', '特别信件']:
        fp = base / folder
        if not fp.exists():
            continue
        for f in sorted(fp.glob('*.md')):
            text = f.read_text(encoding='utf-8').strip()
            if len(text) < 10:  # skip empty/tiny files
                continue
            if len(text) not in stored_lens:
                missing.append({'file': f, 'name': f.stem, 'folder': folder, 'len': len(text)})
            else:
                stored_lens.discard(len(text))
    return missing

async def insert_one(rag, doc_info):
    await rag.initialize_storages()
    text = doc_info['file'].read_text(encoding='utf-8')
    await rag.ainsert(text)

if __name__ == "__main__":
    missing = get_missing_docs()
    print(f"待处理文档: {len(missing)} 篇\n")

    if len(sys.argv) < 2:
        for i, d in enumerate(missing):
            print(f"  [{i+1:2d}] {d['name']} ({d['folder']}, {d['len']}字)")
        print(f"\n用法: python {sys.argv[0]} <序号>")
        sys.exit(0)

    idx = int(sys.argv[1]) - 1
    if idx < 0 or idx >= len(missing):
        print(f"序号超出范围 (1-{len(missing)})")
        sys.exit(1)

    doc = missing[idx]
    print(f"正在处理 [{idx+1}/{len(missing)}]: {doc['name']} ({doc['folder']}, {doc['len']}字)")

    rag = create_rag_instance(WORK_DIR)
    try:
        asyncio.run(insert_one(rag, doc))
        print(f"\n✅ 完成: {doc['name']}")
    except Exception as e:
        print(f"\n❌ 失败: {doc['name']} — {e}")
        sys.exit(1)

"""
Step 1.2: 文本分块 (Chunking)

学习目标：
- 理解为什么需要分块（LLM 和 embedding 都有上下文长度限制）
- 体验不同分块策略对检索效果的影响
- 理解 chunk_size 和 chunk_overlap 的作用

关键概念：
- chunk_size: 每个分块的最大字符数
- chunk_overlap: 相邻分块的重叠字符数（防止语义被切断）
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    """一个文本分块"""
    text: str           # 分块文本
    source: str         # 来源文件名（如 "1985-巴菲特致股东信"）
    year: int           # 年份
    category: str       # 分类（berkshire / partnership / special）
    chunk_index: int    # 在原文中的第几个分块
    char_count: int     # 字符数

    def __str__(self):
        preview = self.text[:80].replace('\n', ' ')
        return f"[{self.source}#{self.chunk_index}] ({self.char_count}字) {preview}..."


def load_all_documents(data_dir: str) -> list[dict]:
    """加载所有 Markdown 文档"""
    docs = []
    base = Path(data_dir)

    category_map = {
        "伯克希尔股东信": "berkshire",
        "合伙人信": "partnership",
        "特别信件": "special",
    }

    for folder, category in category_map.items():
        folder_path = base / folder
        if not folder_path.exists():
            print(f"⚠️  目录不存在: {folder_path}")
            continue

        for md_file in sorted(folder_path.glob("*.md")):
            # 从文件名提取年份（如 "1985-巴菲特致股东信.md" → 1985）
            name = md_file.stem
            year_match = re.match(r'(\d{4})', name)
            year = int(year_match.group(1)) if year_match else 0

            text = md_file.read_text(encoding="utf-8")
            docs.append({
                "filename": name,
                "year": year,
                "category": category,
                "text": text,
                "char_count": len(text),
            })

    return docs


# ============================================================
# 分块策略 1：固定大小分块（最简单，但可能切断语义）
# ============================================================
def chunk_fixed(docs: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """
    按固定字符数切分，带重叠。

    优点：实现简单，分块大小均匀
    缺点：可能在句子中间切断，破坏语义完整性

    参数说明：
    - chunk_size=500: 每块约500字符（中文约250-300字）
    - overlap=50: 相邻块重叠50字符，减少信息丢失
    """
    chunks = []
    for doc in docs:
        text = doc["text"]
        start = 0
        idx = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append(Chunk(
                text=chunk_text,
                source=doc["filename"],
                year=doc["year"],
                category=doc["category"],
                chunk_index=idx,
                char_count=len(chunk_text),
            ))
            start += chunk_size - overlap
            idx += 1
    return chunks


# ============================================================
# 分块策略 2：按段落分块（保留语义完整性）
# ============================================================
def chunk_by_paragraph(docs: list[dict], max_chunk_size: int = 800) -> list[Chunk]:
    """
    按段落（双换行符）切分。如果单段落超过 max_chunk_size，则回退到固定大小切分。
    相邻短段落会合并，直到达到 max_chunk_size。

    优点：保留段落语义完整性
    缺点：分块大小不均匀，某些段落可能很长
    """
    chunks = []
    for doc in docs:
        paragraphs = re.split(r'\n\n+', doc["text"])
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # 合并短段落
        merged = []
        buffer = ""
        for para in paragraphs:
            if len(buffer) + len(para) + 2 <= max_chunk_size:
                buffer = buffer + "\n\n" + para if buffer else para
            else:
                if buffer:
                    merged.append(buffer)
                    buffer = ""
                # 如果单个段落就超过限制，回退到固定切分
                if len(para) > max_chunk_size:
                    start = 0
                    while start < len(para):
                        merged.append(para[start:start + max_chunk_size])
                        start += max_chunk_size - 50
                else:
                    buffer = para
        if buffer:
            merged.append(buffer)

        for idx, chunk_text in enumerate(merged):
            chunks.append(Chunk(
                text=chunk_text,
                source=doc["filename"],
                year=doc["year"],
                category=doc["category"],
                chunk_index=idx,
                char_count=len(chunk_text),
            ))
    return chunks


# ============================================================
# 分块策略 3：递归分块（LangChain 默认策略）
# ============================================================
def chunk_recursive(docs: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """
    按优先级递归切分：先尝试按标题(#)，再按段落(\\n\\n)，再按句子，最后按字符。

    这是 LangChain 的 RecursiveCharacterTextSplitter 的简化版。
    优先级分隔符：["\n# ", "\n\n", "\n", "。", "，", " "]

    优点：尽量在语义边界处切分
    缺点：实现稍复杂
    """
    separators = ["\n# ", "\n\n", "\n", "。", "，", ""]

    def _split_text(text: str, separators: list[str], chunk_size: int) -> list[str]:
        if len(text) <= chunk_size:
            return [text]

        # 找到能切分的分隔符
        for i, sep in enumerate(separators):
            if sep and sep in text:
                parts = text.split(sep)
                result = []
                buffer = ""
                for part in parts:
                    candidate = buffer + sep + part if buffer else part
                    if len(candidate) <= chunk_size:
                        buffer = candidate
                    else:
                        if buffer:
                            result.append(buffer)
                        # 如果单个 part 就超长，用下一个分隔符继续切
                        if len(part) > chunk_size:
                            result.extend(
                                _split_text(part, separators[i + 1:], chunk_size)
                            )
                            buffer = ""
                        else:
                            buffer = part
                if buffer:
                    result.append(buffer)
                return result

        # 所有分隔符都不行，按字符硬切
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

    chunks = []
    for doc in docs:
        parts = _split_text(doc["text"], separators, chunk_size)
        for idx, chunk_text in enumerate(parts):
            chunks.append(Chunk(
                text=chunk_text,
                source=doc["filename"],
                year=doc["year"],
                category=doc["category"],
                chunk_index=idx,
                char_count=len(chunk_text),
            ))
    return chunks


# ============================================================
# 分块统计（用于对比不同策略）
# ============================================================
def chunk_stats(chunks: list[Chunk], strategy_name: str) -> None:
    """打印分块统计信息"""
    sizes = [c.char_count for c in chunks]
    print(f"\n{'='*50}")
    print(f"分块策略: {strategy_name}")
    print(f"{'='*50}")
    print(f"  总分块数: {len(chunks)}")
    print(f"  平均大小: {sum(sizes)//len(sizes)} 字符")
    print(f"  最小: {min(sizes)} 字符")
    print(f"  最大: {max(sizes)} 字符")
    print(f"  中位数: {sorted(sizes)[len(sizes)//2]} 字符")

    # 按分类统计
    by_category = {}
    for c in chunks:
        by_category[c.category] = by_category.get(c.category, 0) + 1
    for cat, count in by_category.items():
        print(f"  {cat}: {count} 个分块")

    # 展示前3个分块样例
    print(f"\n  前3个分块样例:")
    for c in chunks[:3]:
        print(f"    {c}")


if __name__ == "__main__":
    data_dir = "/Users/boyang/Downloads/巴菲特致股东信"

    print("加载文档...")
    docs = load_all_documents(data_dir)
    print(f"共加载 {len(docs)} 篇文档")

    # 对比三种分块策略
    for strategy_fn, name in [
        (chunk_fixed, "固定大小 (500字, overlap=50)"),
        (chunk_by_paragraph, "段落分块 (max=800)"),
        (chunk_recursive, "递归分块 (500字, overlap=50)"),
    ]:
        chunks = strategy_fn(docs)
        chunk_stats(chunks, name)

    print("\n" + "="*50)
    print("提示：观察不同策略的分块数量和大小分布")
    print("好的分块应该：大小适中、语义完整、数量合理")
    print("="*50)

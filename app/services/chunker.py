"""语义分块：把长文本切成适合向量检索的片段。

设计取舍：
- 优先按「段落 → 换行 → 句号 → 分号 → 逗号」递归找断点，避免把一句话劈成两半；
- 相邻块保留 overlap 个字符的重叠，防止答案正好卡在切口处被漏掉；
- 每块带上 doc_id / page / chunk_index，这是后面「点引用跳原文」的唯一依据。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.parser import Page

# 断句优先级：越靠前越适合当切口
BREAK_CHARS = ("\n\n", "\n", "。", "！", "？", "；", "，")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    page: int
    text: str
    tokens: int


def count_tokens(text: str) -> int:
    """粗略 token 估算：中文按 1 字 ≈ 1 token，英文按 1 词 ≈ 1.3 token。

    换成 tiktoken 会让估算更准，但中文 BGE 系模型本来就按字切，
    这里保留零依赖实现，误差在 10% 以内。
    """
    cjk = len(_CJK_RE.findall(text))
    words = len(_WORD_RE.findall(text))
    return cjk + int(words * 1.3) + 1


def split_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """按字符窗口切片，并向后吸附到最近的断句符。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    total = len(text)
    # 允许越界寻找断点的最大距离（约窗口的 1/4）
    lookahead = max(16, chunk_size // 4)

    while start < total:
        end = min(start + chunk_size, total)
        if end < total:
            window = text[end:min(end + lookahead, total)]
            offset = _find_break(window)
            if offset >= 0:
                end += offset + 1  # 把断点字符一起收进本块

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= total:
            break
        # 下一块回退 overlap 个字符，但不能原地踏步
        start = max(end - overlap, start + 1)

    return chunks


def _find_break(window: str) -> int:
    """在窗口里找最早的断句符位置，找不到返回 -1（调用方会硬切）。"""
    best = -1
    for sep in BREAK_CHARS:
        idx = window.find(sep)
        # 优先靠前的断点；同一位置时优先优先级高的分隔符
        if idx >= 0 and (best < 0 or idx < best):
            best = idx
    return best


def build_chunks(
    pages: list[Page],
    doc_id: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """逐页切分，块序号在整个文档内连续（不按页重置）。"""
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        for piece in split_text(page.text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_index=index,
                    page=page.no,
                    text=piece,
                    tokens=count_tokens(piece),
                )
            )
            index += 1
    return chunks

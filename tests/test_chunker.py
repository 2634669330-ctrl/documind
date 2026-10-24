"""分块逻辑的单元测试：不依赖网络、不依赖模型。"""

from __future__ import annotations

import itertools

from app.services.chunker import build_chunks, count_tokens, split_text
from app.services.parser import Page


def test_count_tokens_handles_mixed_text() -> None:
    assert count_tokens("你好") == 2 + 1
    assert count_tokens("hello world") >= 3
    assert count_tokens("") == 1


def test_split_respects_size_and_covers_text() -> None:
    text = ("第一段内容，用来测试分块。" * 20) + ("\n\n第二段内容。" * 20)
    chunks = split_text(text, chunk_size=150, overlap=30)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.strip()
        # 允许在窗口尾部再多吸一段断点（约 chunk_size/4）
        assert len(chunk) <= 150 + 40
        assert chunk in text


def test_split_does_not_lose_boundaries() -> None:
    """相邻块应该有重叠，否则答案正好卡在切口时会被漏掉。"""
    text = "".join(f"句子{i}。" for i in range(200))
    chunks = split_text(text, chunk_size=120, overlap=20)
    assert len(chunks) >= 2
    for prev, nxt in itertools.pairwise(chunks):
        assert prev[-5:] in nxt or nxt in prev or prev in nxt


def test_short_text_returns_single_chunk() -> None:
    assert split_text("很短的一句话。", chunk_size=512) == ["很短的一句话。"]
    assert split_text("   ", chunk_size=512) == []


def test_build_chunks_keeps_page_and_sequence() -> None:
    pages = [Page(no=1, text="第一页内容。" * 60), Page(no=2, text="第二页内容。" * 60)]
    chunks = build_chunks(pages, doc_id="doc_test", chunk_size=100, overlap=20)

    assert len(chunks) >= 4
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))  # 全文档连续编号
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.doc_id == "doc_test" for c in chunks)
    assert all(c.tokens > 0 for c in chunks)

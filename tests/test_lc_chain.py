"""LangChain 编排路径的测试（用 FakeListChatModel，不联网、不需要 API Key）。"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("langchain_core", reason="未安装 langchain，跳过框架路径测试")

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.llm.lc_chain import astream_answer, build_chain
from app.llm.retriever import Hit


def _hits() -> list[Hit]:
    return [
        Hit(
            doc_id="doc1",
            filename="员工手册.txt",
            page=3,
            chunk_index=0,
            text="一线城市住宿标准不超过 600 元每晚。",
            score=0.82,
        )
    ]


def test_chain_streams_answer() -> None:
    fake = FakeListChatModel(responses=["一线城市住宿标准为 600 元/晚（来源 1）。"])

    async def collect() -> str:
        parts = []
        async for token in astream_answer("住宿标准是多少？", [], _hits(), llm=fake):
            parts.append(token)
        return "".join(parts)

    answer = asyncio.run(collect())
    assert "600" in answer


def test_prompt_contains_context_and_question() -> None:
    """Prompt 必须把检索片段和问题都带进去，否则模型无从回答。"""
    fake = FakeListChatModel(responses=["ok"])
    chain = build_chain(fake)

    messages = chain.first.invoke(
        {
            "context": "[1] 员工手册.txt 第 3 页\n住宿 600 元",
            "chat_history": "（无）",
            "question": "住宿标准？",
        }
    )
    rendered = "\n".join(str(m.content) for m in messages.to_messages())
    assert "住宿 600 元" in rendered
    assert "住宿标准？" in rendered

"""可选实现：用 LangChain（LCEL）跑同一条 RAG 链路。

为什么保留两条实现：
- 自研路径（app/llm/client.py + chains.py）：对流式、重试、超时、错误事件完全可控；
- LangChain 路径（本文件）：Prompt 模板 → 模型 → 输出解析串成一条管道，
  换模型、加中间步骤（改写问题、压缩上下文、加 rerank 节点）都只是改一行组合。

两条路径进出都是「token 流 + 引用列表」，接口层不需要区分。
设置 USE_LANGCHAIN=true 即可切换；没装 langchain 会自动回退到自研路径。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence

from app.core.config import settings
from app.core.prompts import SYSTEM_PROMPT, format_context, format_history
from app.llm.retriever import Hit

logger = logging.getLogger(__name__)

LC_USER_PROMPT = """【参考片段】
{context}

【历史对话】
{chat_history}

【问题】
{question}"""


def build_chain(llm=None):
    """Prompt | LLM | StrOutputParser。

    llm 参数用于测试时注入 FakeListChatModel，生产不传走 ChatOpenAI。
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", LC_USER_PROMPT),
        ]
    )

    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
            streaming=True,
        )

    return prompt | llm | StrOutputParser()


async def astream_answer(
    question: str,
    history: Sequence,
    hits: Sequence[Hit],
    llm=None,
) -> AsyncIterator[str]:
    """逐 token 产出回答文本（与自研路径的行为一致）。"""
    chain = build_chain(llm)
    payload = {
        "context": format_context(hits),
        "chat_history": format_history(history),
        "question": question,
    }
    async for chunk in chain.astream(payload):
        text = chunk if isinstance(chunk, str) else str(chunk)
        if text:
            yield text

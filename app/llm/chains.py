"""RAG 编排：把检索结果拼成 Prompt，驱动模型流式输出。

两条实现，行为对上层一致（都是「token 流」）：
- 默认：自研（Prompt 组装 + OpenAI 兼容客户端），流式 / 重试 / 超时完全可控；
- 可选：LangChain（LCEL 管道），USE_LANGCHAIN=true 时启用，见 app/llm/lc_chain.py。

要换成 LangChain 的 ConversationalRetrievalChain，也只需要在 stream_answer 里换分支，
路由层不用动。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence

from app.core.config import settings
from app.core.prompts import REFUSAL, SYSTEM_PROMPT, USER_PROMPT, format_context, format_history
from app.llm.retriever import Hit

logger = logging.getLogger(__name__)


def build_messages(question: str, history: Sequence, hits: Sequence[Hit]) -> list[dict]:
    """组装 OpenAI 格式的 messages。

    system 里放规则，user 里放「参考片段 + 历史 + 问题」——
    把资料放 user 而不是 system，是为了避免长上下文被某些模型当成系统指令削弱规则。
    """
    context = format_context(hits)
    history_text = format_history(history)
    user_content = USER_PROMPT.format(
        context=context,
        chat_history=history_text,
        question=question,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def stream_answer(question: str, history: Sequence, hits: Sequence[Hit]) -> AsyncIterator[str]:
    """统一的生成入口：按配置选择自研或 LangChain 实现。"""
    from app.llm.client import get_llm_client

    client = get_llm_client()

    # mock 模式（没配 Key）下没必要求助框架，直接用本地假回答
    if settings.use_langchain and not client.is_mock:
        try:
            from app.llm.lc_chain import astream_answer

            async for token in astream_answer(question, history, hits):
                yield token
            return
        except ImportError:
            logger.warning("USE_LANGCHAIN=true 但未安装 langchain，已回退到自研实现")

    async for token in client.stream_chat(build_messages(question, history, hits)):
        yield token


def blocked_answer() -> str:
    """检索不到可用片段时的固定回复（不进模型，省钱且绝不编造）。"""
    return REFUSAL

"""大模型客户端：OpenAI 兼容协议 + 流式输出 + 重试 + 用量统计。

为什么手写而不直接用 LangChain 的 ChatOpenAI：
- 流式 + 重试 + 超时的行为要可控（面试常问“模型接口挂了怎么办”）；
- 不依赖具体框架版本，换模型只改环境变量。

没有配置 LLM_API_KEY 时自动进入 mock 模式，方便本地 / CI 把链路跑通。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from collections.abc import AsyncIterator, Sequence

import httpx

from app.core.config import settings
from app.services.chunker import count_tokens

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """调用模型失败（含重试耗尽）。"""


class LLMClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=settings.llm_timeout_s)
        self._mock = settings.mock_llm or not settings.llm_api_key
        if self._mock:
            logger.warning("LLM 处于 mock 模式（未配置 LLM_API_KEY 或 MOCK_LLM=true），回答为本地模拟")

    @property
    def is_mock(self) -> bool:
        return self._mock

    # ------------------------------------------------------------------ 内部
    @property
    def _url(self) -> str:
        return f"{settings.llm_base_url.rstrip('/')}/chat/completions"

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages: Sequence[dict], stream: bool) -> dict:
        return {
            "model": settings.llm_model,
            "messages": list(messages),
            "temperature": settings.llm_temperature,
            "stream": stream,
        }

    # ------------------------------------------------------------------ 流式
    async def stream_chat(self, messages: Sequence[dict]) -> AsyncIterator[str]:
        if self._mock:
            async for token in self._mock_stream(messages):
                yield token
            return

        attempt = 0
        while True:
            yielded = False
            try:
                async with self._client.stream(
                    "POST", self._url, headers=self._headers, json=self._payload(messages, True)
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", "ignore")[:300]
                        raise LLMError(f"模型接口返回 {response.status_code}: {body}")

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            yielded = True
                            yield piece
                return
            except (httpx.HTTPError, LLMError) as exc:
                attempt += 1
                if yielded or attempt > settings.llm_max_retries:
                    raise LLMError(f"模型调用失败（已重试 {attempt - 1} 次）：{exc}") from exc
                wait = min(2 ** attempt, 8) + random.random()
                logger.warning("模型调用异常，%.1fs 后重试（第 %d 次）：%s", wait, attempt, exc)
                await asyncio.sleep(wait)

    # ------------------------------------------------------------------ 非流式
    async def complete(self, messages: Sequence[dict]) -> tuple[str, dict]:
        if self._mock:
            text = "".join([token async for token in self._mock_stream(messages)])
            return text, {"prompt_tokens": 0, "completion_tokens": count_tokens(text)}

        last_error: Exception | None = None
        for attempt in range(settings.llm_max_retries + 1):
            try:
                response = await self._client.post(
                    self._url, headers=self._headers, json=self._payload(messages, False)
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content, data.get("usage") or {}
            except (httpx.HTTPError, KeyError) as exc:
                last_error = exc
                await asyncio.sleep(min(2 ** attempt, 8))
        raise LLMError(f"模型调用失败：{last_error}")

    # ------------------------------------------------------------------ mock
    async def _mock_stream(self, messages: Sequence[dict]) -> AsyncIterator[str]:
        """离线演示模式：不调用模型，只展示「检索到的前两段原文」。

        这样没有 API Key 也能演示检索是否准确；
        输出会明确标注「未接入模型」，不会伪装成模型生成。
        """
        prompt = ""
        for msg in reversed(list(messages)):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        question = prompt
        match = re.search(r"【问题】\s*(.+)", prompt, re.S)
        if match:
            question = match.group(1).strip()

        context = ""
        ctx_match = re.search(r"【参考片段】\s*(.*?)\s*【历史对话】", prompt, re.S)
        if ctx_match:
            context = ctx_match.group(1).strip()

        # 只取前两段，每段截断，避免把整篇文档刷到屏幕上
        blocks = [b for b in re.split(r"\n\n(?=\[\d+\])", context) if b.strip()][:1]
        preview_lines = []
        for block in blocks:
            head, _, body = block.partition("\n")
            body = re.sub(r"\s+", " ", body).strip()
            if len(body) > 150:
                body = body[:150] + "…"
            preview_lines.append(f"{head.strip()}\n{body}")

        answer = (
            "【离线演示模式】未配置 LLM_API_KEY，这里不调用大模型，只展示检索到的原文。\n\n"
            f"你的问题：{question}\n\n"
            + "\n\n".join(preview_lines)
            + "\n\n（配置 Key 后，这里会变成模型基于上述片段的正式回答。）"
        )
        for char in answer:
            await asyncio.sleep(0.002)
            yield char


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client

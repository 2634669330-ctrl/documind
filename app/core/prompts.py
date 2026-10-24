"""Prompt 模板集中管理：调 Prompt 只改这一个文件，方便做 A/B 对比。

设计要点：
1. 角色设定 + 边界说明，把模型限定在“文档助手”身份里；
2. 明确「无依据就拒答」，这是抑制幻觉最重要的一条；
3. 要求标注（来源 N），与检索结果的编号一一对应，前端可点击溯源。
"""

from __future__ import annotations

from collections.abc import Iterable

REFUSAL = "根据现有文档无法回答。"

SYSTEM_PROMPT = """你是企业知识库的问答助手，只能依据【参考片段】回答。

规则：
1. 只使用参考片段中的信息，不得引入片段之外的知识或常识；
2. 若参考片段不足以回答问题，直接回复「根据现有文档无法回答」，不要猜测、不要编造；
3. 回答分点陈述，语言简洁准确；每个结论后用（来源 N）标注出处，N 为片段编号；
4. 只能引用真实存在的片段编号，不要虚构编号；
5. 如果用户问题与文档内容无关，礼貌说明并引导用户换个问法。"""

USER_PROMPT = """【参考片段】
{context}

【历史对话】
{chat_history}

【问题】
{question}"""


def format_context(hits: Iterable) -> str:
    """把检索结果编号后拼成上下文。

    编号必须从 1 开始、与返回给前端的 references 顺序一致，
    否则模型写的（来源 N）会指错地方。
    """
    lines = []
    for idx, hit in enumerate(hits, start=1):
        location = f"{hit.filename} 第 {hit.page} 页"
        lines.append(f"[{idx}] {location}\n{hit.text.strip()}")
    return "\n\n".join(lines)


def format_history(messages: Iterable, limit: int = 6) -> str:
    """最近 N 轮对话，用于指代消解（只取 user/assistant 的角色与内容）。"""
    recent = list(messages)[-limit:]
    if not recent:
        return "（无）"
    role_map = {"user": "用户", "assistant": "助手"}
    return "\n".join(f"{role_map.get(m.role, m.role)}：{m.content}" for m in recent)

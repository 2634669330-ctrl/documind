"""Streamlit 前端：上传文档 + 流式对话 + 引用溯源。

启动：streamlit run frontend/app.py
后端：默认 http://127.0.0.1:8000，可用环境变量 DOCUMIND_API 覆盖
"""

from __future__ import annotations

import json
import os

import requests
import streamlit as st

API = os.getenv("DOCUMIND_API", "http://127.0.0.1:8000").rstrip("/")

# 关键：关闭 requests 对系统代理的自动探测。
# 本机开着 VPN / 代理软件时，Windows 会把 127.0.0.1:8000 也转发给代理，
# 结果界面请求后端直接 502，演示当场翻车。
SESSION = requests.Session()
SESSION.trust_env = False

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------- SSE
def stream_chat(question: str, session_id: str | None):
    """逐事件读取后端 SSE：references 先到，token 持续追加，done 收尾。"""
    payload = {"question": question, "session_id": session_id}
    with SESSION.post(f"{API}/api/chat", json=payload, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        event, data = None, None
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
            elif line == "" and event:
                try:
                    payload_data = json.loads(data) if data else {}
                except json.JSONDecodeError:
                    payload_data = {}
                yield event, payload_data
                event, data = None, None


# --------------------------------------------------------------------------- 侧栏
with st.sidebar:
    st.header("📚 知识库")

    uploaded = st.file_uploader("上传文档", type=["pdf", "docx", "md", "txt"])
    if uploaded is not None and st.button("索引该文档", use_container_width=True):
        with st.spinner("解析、切分、向量化中…"):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            res = SESSION.post(f"{API}/api/upload", files=files, timeout=600)
        if res.ok:
            info = res.json()
            st.success(f"已索引 {info['filename']}：{info['chunks']} 块，{info['elapsed_ms']}ms")
        else:
            st.error(res.json().get("detail", res.text))

    st.divider()
    try:
        docs = SESSION.get(f"{API}/api/documents", timeout=10).json()
    except Exception:
        docs = {"total": 0, "items": []}

    st.caption(f"共 {docs.get('total', 0)} 份文档")
    for item in docs.get("items", []):
        col1, col2 = st.columns([4, 1])
        col1.write(f"**{item['filename']}**  \n{item['chunks']} 块 · {item['status']}")
        if col2.button("🗑", key=f"del_{item['id']}"):
            SESSION.delete(f"{API}/api/documents/{item['id']}", timeout=30)
            st.rerun()

    st.divider()
    if st.button("清空对话", use_container_width=True):
        st.session_state.history = []
        st.session_state.session_id = None
        st.rerun()

# --------------------------------------------------------------------------- 主区
st.title("📄 DocuMind")
st.caption("基于 RAG 的智能文档问答：答案附原文出处，文档里没有的会直接告诉你")

if "history" not in st.session_state:
    st.session_state.history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for cite in msg.get("citations", []):
            st.caption(f"▸ 来源 {cite['no']}　{cite['doc']} · 第 {cite['page']} 页")

question = st.chat_input("问点什么，比如：出差住宿标准是多少？")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer, citations = "", []
        try:
            for event, data in stream_chat(question, st.session_state.session_id):
                if event == "references":
                    citations = data or []
                elif event == "token":
                    answer += data.get("text", "")
                    placeholder.markdown(answer + "▌")
                elif event == "done":
                    st.session_state.session_id = data.get("session_id")
                elif event == "error":
                    st.error(data.get("message"))
        except Exception as exc:
            answer = f"请求失败：{exc}"
        placeholder.markdown(answer)

        for cite in citations:
            st.caption(f"▸ 来源 {cite['no']}　{cite['doc']} · 第 {cite['page']} 页")
            with st.expander(f"查看来源 {cite['no']} 片段"):
                st.write(cite["snippet"])

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )

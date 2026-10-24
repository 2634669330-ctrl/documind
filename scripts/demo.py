"""终端版演示脚本：一条命令跑完「上传 → 检索 → 问答 → 拒答」全流程。

用法（后端已启动的前提下）：
    python scripts/demo.py                 # 用默认地址 http://127.0.0.1:8000
    python scripts/demo.py http://127.0.0.1:8010

它会：
1. 检查服务健康状态；
2. 把 samples/ 下的示例文档上传索引（已存在的同名文档会跳过）；
3. 依次提问，打印回答、用时和引用来源；
4. 最后问一个文档里没有的问题，验证「不编造」。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
SAMPLES = Path(__file__).resolve().parents[1] / "samples"

QUESTIONS = [
    "一线城市出差住宿标准是多少？",
    "年假有几天？入职满 3 年呢？",
    "报销提交后多久能到账？",
    "支持上传哪些格式的文档？",
    "文档会上传到公网吗？",
    "公司股票代码是多少？",  # 故意问文档里没有的，验证拒答
]

BAR = "─" * 74


def line(text: str = "") -> None:
    print(text, flush=True)


def ask(client: httpx.Client, question: str, session_id: str | None) -> tuple[str, list, dict]:
    """发一次提问，把 SSE 事件拼回完整回答。"""
    answer, references, done = "", [], {}
    with client.stream("POST", "/api/chat", json={"question": question, "session_id": session_id}) as resp:
        event, data = None, None
        for raw in resp.iter_lines():
            row = raw.strip()
            if row.startswith("event:"):
                event = row[6:].strip()
            elif row.startswith("data:"):
                data = row[5:].strip()
            elif row == "" and event:
                payload = json.loads(data) if data else {}
                if event == "references":
                    references = payload
                elif event == "token":
                    answer += payload.get("text", "")
                elif event == "done":
                    done = payload
                event, data = None, None
    return answer, references, done


def main() -> int:
    line(BAR)
    line(f"DocuMind 演示脚本　目标服务：{BASE}")
    line(BAR)

    with httpx.Client(base_url=BASE, timeout=180) as client:
        # ---------- 1. 健康检查 ----------
        try:
            health = client.get("/health").json()
        except Exception as exc:
            line(f"✗ 连不上后端：{exc}")
            line("  请先双击「启动后端.cmd」，或执行 uvicorn app.main:app --port 8000")
            return 1

        llm_state = "mock（未配 Key）" if health["llm_mock"] else "已接入"
        line(
            f"✓ 服务正常 | 数据库={'OK' if health['database'] else '异常'}"
            f" | 向量库={'OK' if health['vector_store'] else '异常'}"
            f" | 模型={llm_state} | 向量={health['embedding_provider']}"
        )

        # ---------- 2. 上传示例文档 ----------
        existing = {d["filename"] for d in client.get("/api/documents").json()["items"]}
        files = sorted(path for path in SAMPLES.glob("*") if path.is_file())
        if not files:
            line("✗ samples/ 目录下没有文档")
            return 1

        for path in files:
            if path.name in existing:
                line(f"• 已存在，跳过：{path.name}")
                continue
            payload = path.read_bytes()
            resp = client.post("/api/upload", files={"file": (path.name, payload)})
            if resp.status_code != 200:
                line(f"✗ 上传失败 {path.name}：{resp.text}")
                continue
            info = resp.json()
            line(
                f"• 上传成功 {info['filename']} → {info['pages']} 页 / "
                f"{info['chunks']} 块 / {info['elapsed_ms']} ms"
            )

        docs = client.get("/api/documents").json()
        line(f"• 知识库现有 {docs['total']} 份文档")
        line()

        # ---------- 3. 逐题问答 ----------
        session_id: str | None = None
        for idx, question in enumerate(QUESTIONS, start=1):
            line(f"【问题 {idx}】{question}")
            started = time.perf_counter()
            answer, references, done = ask(client, question, session_id)
            session_id = done.get("session_id") or session_id
            cost = (time.perf_counter() - started) * 1000

            body = (answer.strip() or "（空）").replace("\n", "\n　　　　")
            line(f"　回答：{body}")
            if references:
                refs = "、".join(
                    f"[{ref['no']}] {ref['doc']} 第 {ref['page']} 页 片段 {ref.get('chunk_index', 0) + 1}"
                    for ref in references
                )
                line(f"　引用：{refs}")
            line(f"　用时：{cost:.0f} ms（服务端计时 {done.get('latency_ms', '-')} ms）")
            line()

        line(BAR)
        line(f"演示结束。会话 ID：{session_id or '-'}")
        line("接口文档：http://127.0.0.1:8000/docs")
        line("网页界面：双击「启动界面.cmd」后访问 http://localhost:8501")
        line(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""接口级冒烟测试：健康检查 → 上传 → 检索问答（SSE）→ 历史回查。

用 TestClient 走完整链路，模型是 mock、向量是 hash，
不依赖外网和模型权重，CI 里可以直接跑。
"""

from __future__ import annotations

import json


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    event, data = None, None
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
        elif line == "" and event:
            try:
                payload = json.loads(data) if data else {}
            except json.JSONDecodeError:
                payload = {}
            events.append((event, payload))
            event, data = None, None
    if event:  # 结尾没有空行的兜底
        events.append((event, json.loads(data) if data else {}))
    return events


def test_health(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["database"] is True
    assert body["vector_store"] is True
    assert body["llm_mock"] is True
    assert body["embedding_provider"] == "hash"


def test_upload_list_and_delete(client) -> None:
    content = ("差旅费报销细则。\n一线城市住宿标准不超过 600 元每晚。\n" * 12).encode("utf-8")
    resp = client.post(
        "/api/upload",
        files={"file": ("员工手册.txt", content, "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    upload = resp.json()
    assert upload["chunks"] >= 1
    assert upload["filename"] == "员工手册.txt"
    doc_id = upload["doc_id"]

    listing = client.get("/api/documents").json()
    assert listing["total"] >= 1
    assert any(item["id"] == doc_id for item in listing["items"])

    deleted = client.delete(f"/api/documents/{doc_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] == doc_id

    # 删除后不应再出现在列表里
    assert all(item["id"] != doc_id for item in client.get("/api/documents").json()["items"])


def test_chat_stream_and_history(client) -> None:
    content = "公司出差住宿标准：一线城市不超过 600 元每晚，二线城市 400 元每晚。".encode()
    up = client.post("/api/upload", files={"file": ("差旅制度.txt", content, "text/plain")})
    assert up.status_code == 200

    with client.stream("POST", "/api/chat", json={"question": "一线城市住宿标准是多少？"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())

    events = _parse_sse(body)
    names = [name for name, _ in events]
    assert "references" in names
    assert "token" in names
    assert "done" in names

    references = dict(events)["references"]
    assert len(references) >= 1
    assert references[0]["doc"] == "差旅制度.txt"

    answer = "".join(payload.get("text", "") for name, payload in events if name == "token")
    assert answer.strip()

    done = dict(events)["done"]
    session_id = done["session_id"]

    messages = client.get(f"/api/sessions/{session_id}/messages").json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["citations"]


def test_upload_rejects_unsupported_type(client) -> None:
    resp = client.post("/api/upload", files={"file": ("a.csv", b"a,b", "text/csv")})
    assert resp.status_code == 400


def test_chat_requires_question(client) -> None:
    resp = client.post("/api/chat", json={"question": ""})
    assert resp.status_code == 422  # pydantic 校验不过

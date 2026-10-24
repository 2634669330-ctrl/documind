"""SSE 事件拼装。

SSE 的报文格式很简单：`event: xxx` + `data: ...` + 空行。
前端用 EventSource 或 fetch + ReadableStream 都能消费。
"""

from __future__ import annotations

import json
from typing import Any


def sse(event: str, data: Any) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

"""问答接口：SSE 流式返回。

事件序列：
    references -> token * N -> done
异常时插入 error 事件，前端可据此提示重试。

注意点：
- 落库放在生成器里、用独立 Session：请求作用域的 Session 在响应结束时会被关闭；
- 先推 references 再推 token，前端可以立刻把“来源卡片”渲染出来，感知更快。
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_chunk_index, get_embedding_model, get_vector_store
from app.api.schemas import ChatRequest
from app.core.sse import sse
from app.db.models import ChatSession, Citation, Message
from app.db.session import SessionLocal, get_db
from app.llm.chains import blocked_answer, build_messages, stream_answer
from app.llm.client import LLMError
from app.llm.retriever import retrieve
from app.services.chunker import count_tokens

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat", summary="提问（SSE 流式）")
async def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # ---- 会话：没传 session_id 就新建 ----
    session_id = payload.session_id or uuid.uuid4().hex[:12]
    session = db.get(ChatSession, session_id)
    if session is None:
        session = ChatSession(id=session_id, title=question[:24])
        db.add(session)
        db.commit()

    # ---- 历史（不含本轮） ----
    history = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    # ---- 落库用户问题 ----
    user_message = Message(id=uuid.uuid4().hex[:12], session_id=session_id, role="user", content=question)
    db.add(user_message)
    db.commit()

    # ---- 检索（同步 CPU/IO，丢线程池） ----
    hits = await run_in_threadpool(
        retrieve,
        question,
        get_embedding_model(),
        get_vector_store(),
        get_chunk_index(),
    )
    references = [hit.to_reference(no) for no, hit in enumerate(hits, start=1)]
    logger.info("检索完成 session=%s hits=%d", session_id, len(hits))

    # 只组装一次 messages，prompt 用量与流式生成复用同一份
    messages = build_messages(question, history, hits) if hits else []
    prompt_tokens = count_tokens("".join(str(m.get("content", "")) for m in messages)) if messages else 0

    async def event_stream():
        started = time.perf_counter()
        yield sse("references", references)

        answer_parts: list[str] = []
        error: str | None = None

        if not hits:
            text = blocked_answer()
            answer_parts.append(text)
            yield sse("token", {"text": text})
        else:
            try:
                async for token in stream_answer(question, history, hits):
                    answer_parts.append(token)
                    yield sse("token", {"text": token})
            except LLMError as exc:
                error = str(exc)
                logger.error("生成失败 session=%s: %s", session_id, exc)
                yield sse("error", {"message": error})

        answer = "".join(answer_parts).strip()
        latency_ms = int((time.perf_counter() - started) * 1000)
        message_id = uuid.uuid4().hex[:12]

        # 用独立 Session 落库，避免请求级 Session 已关闭
        try:
            with SessionLocal() as write_db:
                assistant = Message(
                    id=message_id,
                    session_id=session_id,
                    role="assistant",
                    content=answer,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=count_tokens(answer),
                    latency_ms=latency_ms,
                )
                write_db.add(assistant)
                for ref in references:
                    write_db.add(
                        Citation(
                            message_id=message_id,
                            no=ref["no"],
                            doc_id=ref["doc_id"],
                            filename=ref["doc"],
                            page=ref["page"],
                            chunk_index=ref["chunk_index"],
                            score=ref["score"],
                            snippet=ref["snippet"],
                        )
                    )
                # 会话标题取第一轮问题的前 24 字
                sess = write_db.get(ChatSession, session_id)
                if sess is not None and sess.title == "新会话":
                    sess.title = question[:24]
                write_db.commit()
        except Exception:
            logger.exception("回答落库失败 message_id=%s", message_id)

        yield sse(
            "done",
            {
                "message_id": message_id,
                "session_id": session_id,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": count_tokens(answer),
                },
                "latency_ms": latency_ms,
                "error": error,
            },
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 让 nginx 不要缓冲 SSE
        },
    )

"""会话与历史消息查询。"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import MessageOut, SessionCreate, SessionOut
from app.db.models import ChatSession, Message
from app.db.session import get_db

router = APIRouter(prefix="/sessions", tags=["chat"])


@router.post("", response_model=SessionOut, summary="新建会话")
def create_session(payload: SessionCreate, db: Session = Depends(get_db)) -> SessionOut:
    session = ChatSession(id=uuid.uuid4().hex[:12], title=payload.title or "新会话")
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionOut.model_validate(session, from_attributes=True)


@router.get("", response_model=list[SessionOut], summary="会话列表")
def list_sessions(db: Session = Depends(get_db)) -> list[SessionOut]:
    rows = db.scalars(select(ChatSession).order_by(ChatSession.created_at.desc())).all()
    return [SessionOut.model_validate(row, from_attributes=True) for row in rows]


@router.get("/{session_id}/messages", response_model=list[MessageOut], summary="历史消息")
def list_messages(session_id: str, db: Session = Depends(get_db)) -> list[MessageOut]:
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    rows = (
        db.scalars(
            select(Message).filter(Message.session_id == session_id).order_by(Message.created_at.asc())
        ).all()
    )
    result: list[MessageOut] = []
    for row in rows:
        citations = [
            {
                "no": cite.no,
                "doc_id": cite.doc_id,
                "doc": cite.filename,
                "page": cite.page,
                "score": cite.score,
                "snippet": cite.snippet,
            }
            for cite in row.citations
        ]
        result.append(
            MessageOut(
                id=row.id,
                role=row.role,
                content=row.content,
                latency_ms=row.latency_ms,
                created_at=row.created_at,
                citations=citations,
            )
        )
    return result

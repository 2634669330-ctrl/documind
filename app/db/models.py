"""ORM 模型：文档、会话、消息、引用。

四张表的关系：
    Document  1 ── n  Citation        （引用指向某份文档的某页某块）
    ChatSession 1 ── n  Message       （一次会话里的多轮问答）
    Message  1 ── n  Citation         （一条回答可以引用多个片段）
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    """一份被索引过的文档。"""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), index=True)
    stored_path: Mapped[str] = mapped_column(String(512))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    pages: Mapped[int] = mapped_column(Integer, default=0)
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="indexed")  # indexing / indexed / failed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ChatSession(Base):
    """一次对话会话（前端左侧的会话列表）。"""

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(String(128), default="新会话")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    messages: Mapped[list[Message]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    """一条消息：role=user / assistant。"""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    session: Mapped[ChatSession] = relationship(back_populates="messages")
    citations: Mapped[list[Citation]] = relationship(
        back_populates="message", cascade="all, delete-orphan", order_by="Citation.no"
    )


class Citation(Base):
    """回答里的一条出处：哪个文件、第几页、第几块、相似度多少。

    doc_id / filename / page 都是「快照」，不建外键：
    文档删掉后历史回答里的出处仍然可读，避免级联删除把聊天记录弄残。
    """

    __tablename__ = "citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"), index=True)
    no: Mapped[int] = mapped_column(Integer)            # 对应 Prompt 里的 [N]
    doc_id: Mapped[str] = mapped_column(String(32))
    filename: Mapped[str] = mapped_column(String(255))
    page: Mapped[int] = mapped_column(Integer, default=1)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    snippet: Mapped[str] = mapped_column(Text, default="")

    message: Mapped[Message] = relationship(back_populates="citations")


Index("ix_messages_session_created", Message.session_id, Message.created_at)
Index("ix_citations_message_no", Citation.message_id, Citation.no)

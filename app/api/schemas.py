"""API 层的数据结构（Pydantic 模型）：请求体校验 + 响应序列化。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    elapsed_ms: int
    status: str = "indexed"


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    size_bytes: int
    pages: int
    chunks: int
    status: str
    created_at: datetime


class DocumentListOut(BaseModel):
    total: int
    items: list[DocumentOut]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, description="为空则自动新建会话")


class SessionCreate(BaseModel):
    title: str = Field(default="新会话", max_length=128)


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    latency_ms: int = 0
    created_at: datetime
    citations: list[dict] = Field(default_factory=list)


class HealthOut(BaseModel):
    status: str
    database: bool
    vector_store: bool
    llm_mock: bool
    embedding_provider: str
    chunks: int

"""文档管理：列表 / 删除。

删除要同时清理四处，否则会出现「列表里没了，但还能被检索到」的脏数据：
1. 向量库（Chroma where doc_id）
2. chunks.jsonl（BM25 语料）
3. 原始文件（data/uploads）
4. 数据库记录（Document + 关联 Citation）
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_chunk_index, get_vector_store
from app.api.schemas import DocumentListOut, DocumentOut
from app.db.models import Document
from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["document"])


@router.get("", response_model=DocumentListOut, summary="文档列表")
def list_documents(db: Session = Depends(get_db)) -> DocumentListOut:
    rows = db.scalars(select(Document).order_by(Document.created_at.desc())).all()
    total_chunks = db.scalar(select(func.coalesce(func.sum(Document.chunks), 0))) or 0
    logger.debug("文档列表：%d 条，共 %d 块", len(rows), total_chunks)
    return DocumentListOut(
        total=len(rows),
        items=[DocumentOut.model_validate(row) for row in rows],
    )


@router.delete("/{doc_id}", summary="删除文档及其索引")
def delete_document(doc_id: str, db: Session = Depends(get_db)) -> dict:
    document = db.get(Document, doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 同步端点由 FastAPI 放在线程池里执行，这里直接调用即可
    get_vector_store().delete_document(doc_id)
    get_chunk_index().remove_document(doc_id)

    stored = Path(document.stored_path)
    if stored.exists():
        stored.unlink(missing_ok=True)

    db.delete(document)
    db.commit()
    logger.info("文档已删除 doc_id=%s", doc_id)
    return {"deleted": doc_id}

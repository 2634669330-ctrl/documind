"""上传与索引：一次请求完成「落盘 → 解析 → 分块 → 向量化 → 入库 → 建 BM25 索引」。

每一步都是 CPU/IO 密集操作，用 run_in_threadpool 丢到线程池，
避免阻塞 event loop（FastAPI 是单事件循环，同步阻塞会拖垮其他请求）。
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_chunk_index, get_embedding_model, get_vector_store
from app.api.schemas import UploadResponse
from app.core.config import settings
from app.db.models import Document
from app.db.session import get_db
from app.services.chunker import build_chunks
from app.services.parser import ParseError, UnsupportedFormatError, parse_file

logger = logging.getLogger(__name__)

router = APIRouter(tags=["document"])


@router.post("/upload", response_model=UploadResponse, summary="上传并索引文档")
async def upload_document(
    file: UploadFile = File(..., description="PDF / DOCX / MD / TXT"),
    db: Session = Depends(get_db),
) -> UploadResponse:
    started = time.perf_counter()
    filename = file.filename or "unnamed"
    suffix = Path(filename).suffix.lower()

    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {suffix}，仅支持 {sorted(settings.allowed_extensions)}",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(payload) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {settings.max_upload_mb} MB 限制")

    settings.ensure_dirs()
    doc_id = uuid.uuid4().hex[:12]
    stored_path = settings.upload_dir / f"{doc_id}{suffix}"
    stored_path.write_bytes(payload)

    embedder = get_embedding_model()
    store = get_vector_store()
    chunk_index = get_chunk_index()

    try:
        parsed = await run_in_threadpool(parse_file, stored_path, filename)
        chunks = build_chunks(
            parsed.pages,
            doc_id=doc_id,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        if not chunks:
            raise HTTPException(status_code=422, detail="文档没有可用文本，可能是空文件")

        vectors = await run_in_threadpool(embedder.encode, [chunk.text for chunk in chunks])
        await run_in_threadpool(
            store.add,
            [f"{doc_id}:{chunk.chunk_index}" for chunk in chunks],
            vectors,
            [chunk.text for chunk in chunks],
            [
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )
        await run_in_threadpool(
            chunk_index.add,
            [
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                }
                for chunk in chunks
            ],
        )
    except (ParseError, UnsupportedFormatError) as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        stored_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        logger.exception("索引失败 doc_id=%s", doc_id)
        raise HTTPException(status_code=500, detail=f"索引失败：{exc}") from exc

    document = Document(
        id=doc_id,
        filename=filename,
        stored_path=str(stored_path),
        size_bytes=len(payload),
        pages=len(parsed.pages),
        chunks=len(chunks),
        status="indexed",
    )
    db.add(document)
    db.commit()

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info("索引完成 doc_id=%s file=%s chunks=%d %dms", doc_id, filename, len(chunks), elapsed_ms)
    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        pages=len(parsed.pages),
        chunks=len(chunks),
        elapsed_ms=elapsed_ms,
    )

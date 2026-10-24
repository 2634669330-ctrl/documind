"""FastAPI 入口：注册路由、CORS、健康检查、启动时建表。

启动：uvicorn app.main:app --reload --port 8000
文档：http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api import routes_chat, routes_documents, routes_sessions, routes_upload
from app.api.deps import get_chunk_index, get_vector_store
from app.api.schemas import HealthOut
from app.core.config import settings
from app.core.logging import setup_logging
from app.db.init_db import init_db
from app.db.session import engine

setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时做三件事：建目录、建表、打印当前配置（排查环境问题很有用）。"""
    settings.ensure_dirs()
    init_db()
    logger.info(
        "DocuMind 启动 | env=%s | llm=%s | embedding=%s | vector=%s/%s",
        settings.app_env,
        settings.llm_model if settings.llm_api_key else "mock",
        settings.embedding_provider,
        settings.chroma_mode,
        settings.chroma_collection,
    )
    yield
    logger.info("DocuMind 关闭")


def create_app() -> FastAPI:
    app = FastAPI(
        title="DocuMind",
        version="0.2.0",
        description="基于 RAG 的企业级智能文档知识库问答系统",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes_upload.router, prefix="/api")
    app.include_router(routes_documents.router, prefix="/api")
    app.include_router(routes_chat.router, prefix="/api")
    app.include_router(routes_sessions.router, prefix="/api")

    @app.get("/health", response_model=HealthOut, tags=["ops"], summary="健康检查")
    def health() -> HealthOut:
        db_ok = True
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            db_ok = False

        store_ok = get_vector_store().healthy()
        stats = get_chunk_index().stats()
        # mock = 没配 Key，或显式打开了 MOCK_LLM（后者用于离线演练，避免误烧额度）
        llm_mock = settings.mock_llm or not settings.llm_api_key
        return HealthOut(
            status="ok" if (db_ok and store_ok) else "degraded",
            database=db_ok,
            vector_store=store_ok,
            llm_mock=llm_mock,
            embedding_provider=settings.embedding_provider,
            chunks=stats["chunks"],
        )

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "name": settings.app_name,
            "version": "0.2.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()

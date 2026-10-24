"""数据库连接与会话管理。

SQLite 与 PostgreSQL 用同一套代码，靠 DATABASE_URL 切换：
    sqlite:///data/documind.db
    postgresql+psycopg://documind:documind@localhost:5432/documind
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import BASE_DIR, settings


def _ensure_sqlite_parent(url: str) -> None:
    """sqlite 文件所在目录不存在时先建目录，否则首次启动会报 unable to open database file。"""
    prefix = "sqlite:///"
    if url.startswith(prefix):
        raw = url[len(prefix):]
        if raw and raw != ":memory:":
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = BASE_DIR / path  # 与 config 的相对路径基准保持一致
            path.parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_parent(settings.database_url)

_is_sqlite = settings.database_url.startswith("sqlite")

engine: Engine = create_engine(
    settings.database_url,
    echo=False,
    future=True,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)

if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _sqlite_pragma(dbapi_connection, _record):
        """SQLite 默认不校验外键，打开它，级联删除才会生效。"""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI 依赖：每个请求一个 Session，请求结束自动关闭。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

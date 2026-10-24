"""建表脚本：python -m app.db.init_db

生产环境建议换成 Alembic 做版本化迁移，这里保持零依赖。
"""

from __future__ import annotations

import logging

from sqlalchemy import inspect

from app.core.logging import setup_logging
from app.db.models import Base
from app.db.session import engine

logger = logging.getLogger(__name__)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    tables = inspect(engine).get_table_names()
    logger.info("数据库已就绪，表：%s", ", ".join(sorted(tables)))


if __name__ == "__main__":
    setup_logging("INFO")
    init_db()

"""pytest 全局配置：在导入 app 之前把环境变量指向临时目录。

关键点：Settings 是导入时创建的缓存单例，所以环境变量必须在这里（早于 app 导入）设置。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / ".pytest-data"

if TEST_DATA.exists():
    shutil.rmtree(TEST_DATA)
(TEST_DATA / "data").mkdir(parents=True, exist_ok=True)

os.environ.update(
    {
        "APP_ENV": "test",
        "LOG_LEVEL": "WARNING",
        "DATA_DIR": str(TEST_DATA / "data"),
        "CHROMA_DIR": str(TEST_DATA / "chroma"),
        "DATABASE_URL": f"sqlite:///{TEST_DATA.as_posix()}/test.db",
        "EMBEDDING_PROVIDER": "hash",
        "MOCK_LLM": "true",
        "CHUNK_SIZE": "150",
        "CHUNK_OVERLAP": "30",
        "RETRIEVE_TOP_K": "10",
        "RERANK_TOP_N": "3",
        "MIN_RELEVANCE": "0.0",   # 测试用哈希向量，相似度本身偏低，不做硬阈值
        "ENABLE_BM25": "true",
    }
)

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def client():
    """整个测试会话共用一个 TestClient（会触发 lifespan：建表、建目录）。"""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client

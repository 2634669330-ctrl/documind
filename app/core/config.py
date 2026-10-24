"""全局配置：所有可调参数都来自环境变量或 .env，代码里不硬编码。

用法：
    from app.core.config import settings
    print(settings.llm_model)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（app/core/config.py -> app/core -> app -> 根）
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """pydantic-settings 会自动读取同名环境变量（大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 应用 ----------
    app_name: str = "DocuMind"
    app_env: str = "dev"                 # dev / prod
    log_level: str = "INFO"
    cors_origins: str = "*"              # 逗号分隔，* 表示全部放行

    # ---------- 大模型 ----------
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.2
    mock_llm: bool = False               # true：不调用真实模型，走本地假回答（离线演示用）
    use_langchain: bool = False          # true：改用 LangChain(LCEL) 编排（需 requirements-langchain.txt）

    # ---------- 向量化 ----------
    embedding_provider: Literal["hash", "local", "api"] = "hash"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_device: str = "cpu"        # cpu / cuda
    embedding_batch_size: int = 16
    embedding_api_model: str = "text-embedding-3-small"

    # ---------- 切分与检索 ----------
    chunk_size: int = 512                # 单位：token（中文近似按字符算）
    chunk_overlap: int = 64
    retrieve_top_k: int = 20             # 单路召回条数
    rerank_top_n: int = 5                # 重排后送入模型的条数
    enable_bm25: bool = True
    enable_rerank: bool = False          # 需要 sentence-transformers 里的 CrossEncoder
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    min_relevance: float = 0.15          # 低于该分数视为“文档里没有”

    # ---------- 存储 ----------
    data_dir: Path = Path("data")
    chroma_dir: Path = Path("data/chroma")
    chroma_collection: str = "documind_chunks"
    chroma_mode: Literal["persistent", "http"] = "persistent"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    database_url: str = "sqlite:///data/documind.db"
    max_upload_mb: int = 50

    # ---------- 派生属性 ----------
    @property
    def allowed_extensions(self) -> set[str]:
        return {".pdf", ".docx", ".md", ".markdown", ".txt"}

    @property
    def upload_dir(self) -> Path:
        return self._abs(self.data_dir) / "uploads"

    @property
    def index_dir(self) -> Path:
        return self._abs(self.data_dir) / "index"

    @property
    def chunks_file(self) -> Path:
        """全部 chunk 的落盘索引（BM25 与引用回查都依赖它）。"""
        return self.index_dir / "chunks.jsonl"

    @property
    def chroma_path(self) -> Path:
        return self._abs(self.chroma_dir)

    @property
    def cors_origin_list(self) -> list[str]:
        raw = (self.cors_origins or "").strip()
        if not raw or raw == "*":
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]

    def _abs(self, path: Path) -> Path:
        """相对路径统一相对项目根目录，避免受启动时 cwd 影响。"""
        return path if path.is_absolute() else (BASE_DIR / path)

    def ensure_dirs(self) -> None:
        for path in (self._abs(self.data_dir), self.upload_dir, self.index_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

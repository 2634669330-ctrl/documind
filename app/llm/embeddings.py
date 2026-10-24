"""向量化（Embedding）：三种实现，按 EMBEDDING_PROVIDER 切换。

- local：sentence-transformers 加载 BGE-m3，本地推理，效果最好（需 pip install -r requirements-ml.txt）
- api  ：调用 OpenAI 兼容的 /embeddings 接口（智谱、OpenAI、SiliconFlow 等）
- hash ：零依赖的确定性哈希向量，只用来把整条链路跑通（开发 / 单元测试），检索质量不具备参考价值

所有实现都对外暴露同样的两个成员：`dim` 和 `encode(list[str]) -> list[list[float]]`，
上层代码不需要关心用的是哪种。
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import lru_cache

from app.core.config import settings

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


class BaseEmbedder(ABC):
    name: str = "base"

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """把一批文本编码成向量，返回顺序与入参一致。"""


# --------------------------------------------------------------------------- hash
class HashEmbedder(BaseEmbedder):
    """字符 n-gram + 带符号哈希（hashing trick）。

    优点：纯标准库、确定性、毫秒级；缺点：只有字面相似度，不懂语义。
    仅用于本地跑通链路和测试断言，真实演示请切到 local。
    """

    name = "hash"

    def __init__(self, dim: int) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(t) for t in texts]

    def _encode_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for feature in self._features(text):
            digest = int.from_bytes(
                hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big"
            )
            idx = digest % self._dim
            sign = 1.0 if (digest >> 63) & 1 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _features(text: str) -> list[str]:
        chars = [c for c in text if not c.isspace()]
        grams = chars + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
        grams += [w.lower() for w in _WORD_RE.findall(text)]
        return grams


# ---------------------------------------------------------------------------- api
class ApiEmbedder(BaseEmbedder):
    """调用 OpenAI 兼容的 embedding 接口。"""

    name = "api"

    def __init__(self) -> None:
        import httpx

        self._client = httpx.Client(timeout=settings.llm_timeout_s)
        self._dim_value = settings.embedding_dim
        self._model = settings.embedding_api_model

    @property
    def dim(self) -> int:
        return self._dim_value

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch = max(1, settings.embedding_batch_size)
        for start in range(0, len(texts), batch):
            chunk = list(texts[start:start + batch])
            resp = self._client.post(
                f"{settings.llm_base_url.rstrip('/')}/embeddings",
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                json={"model": self._model, "input": chunk},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda item: item["index"])
            vectors.extend([item["embedding"] for item in data])
        return vectors


# -------------------------------------------------------------------------- local
class LocalBgeEmbedder(BaseEmbedder):
    """本地 BGE-m3：第一次调用时才加载模型（约 2GB 内存）。"""

    name = "local"

    def __init__(self) -> None:
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "本地向量化需要 sentence-transformers，请先执行："
                    "pip install -r requirements-ml.txt"
                ) from exc
            logger.info("加载向量模型 %s（device=%s）", settings.embedding_model, settings.embedding_device)
            self._model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
        return self._model

    @property
    def dim(self) -> int:
        return settings.embedding_dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, v)) for v in vectors]


# ------------------------------------------------------------------------- factory
@lru_cache
def get_embedder() -> BaseEmbedder:
    provider = settings.embedding_provider
    if provider == "local":
        embedder: BaseEmbedder = LocalBgeEmbedder()
    elif provider == "api":
        embedder = ApiEmbedder()
    else:
        if settings.app_env != "dev":
            logger.warning("hash 向量仅适用于开发环境，生产请切换 EMBEDDING_PROVIDER=local")
        embedder = HashEmbedder(settings.embedding_dim)
    logger.info("Embedding provider=%s dim=%d", embedder.name, embedder.dim)
    return embedder

"""向量库封装：只暴露 add / delete / search / count 四个动作。

默认用 ChromaDB 的嵌入式模式（PersistentClient，零运维）；
docker-compose 里跑独立 chroma 服务时，把 CHROMA_MODE=http 即可切到 HttpClient。
换 Milvus / Qdrant 只需要重写这个类，上层不用动。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self._client: Any = None
        self._collection: Any = None

    # ------------------------------------------------------------------ 内部
    def _ensure(self):
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("缺少依赖 chromadb，请先 pip install -r requirements.txt") from exc

        if settings.chroma_mode == "http":
            logger.info("连接 Chroma 服务 %s:%s", settings.chroma_host, settings.chroma_port)
            self._client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        else:
            settings.chroma_path.mkdir(parents=True, exist_ok=True)
            logger.info("使用 Chroma 本地持久化目录 %s", settings.chroma_path)
            self._client = chromadb.PersistentClient(path=str(settings.chroma_path))

        # 余弦距离：向量已归一化，等价于内积，检索更稳定
        try:
            self._collection = self._client.get_or_create_collection(
                name=settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            self._collection = self._client.get_or_create_collection(name=settings.chroma_collection)
        return self._collection

    # ------------------------------------------------------------------ 写入
    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        if not ids:
            return
        collection = self._ensure()
        collection.add(
            ids=list(ids),
            embeddings=[list(map(float, v)) for v in embeddings],
            documents=list(documents),
            metadatas=list(metadatas),
        )

    def delete_document(self, doc_id: str) -> None:
        """按文档删除全部向量（where 过滤，不需要记具体 chunk id）。"""
        collection = self._ensure()
        try:
            collection.delete(where={"doc_id": doc_id})
        except Exception as exc:
            logger.warning("删除向量失败 doc_id=%s: %s", doc_id, exc)

    # ------------------------------------------------------------------ 查询
    def search(self, embedding: Sequence[float], top_k: int) -> list[dict]:
        """返回 [{text, metadata, score}]，score 是余弦相似度（越大越相关）。"""
        collection = self._ensure()
        result = collection.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
        )

        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        hits: list[dict] = []
        for text, meta, dist in zip(docs, metas, dists, strict=False):
            hits.append(
                {
                    "text": text or "",
                    "metadata": meta or {},
                    "score": max(0.0, 1.0 - float(dist)),  # 余弦距离 -> 相似度
                }
            )
        return hits

    def count(self) -> int:
        try:
            return int(self._ensure().count())
        except Exception:
            return 0

    def healthy(self) -> bool:
        try:
            self._ensure()
            return True
        except Exception as exc:
            logger.warning("向量库不可用：%s", exc)
            return False

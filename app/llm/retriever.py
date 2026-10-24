"""检索层：向量召回 + BM25 召回 → 去重融合 →（可选）重排。

为什么两路都要：
- 向量召回擅长「同义不同词」（问“住宿标准”，文档写“差旅费报销细则”）；
- BM25 擅长「专有名词 / 编号 / 型号」（问“GB/T 19001 第 8.2 条”），向量模型对编号不敏感。

两路结果按 (doc_id, chunk_index) 去重；同一片段被两路同时召回时给一点加分，
否则向量分数和 BM25 分数不可比，排序会乱。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.llm.embeddings import BaseEmbedder
from app.llm.vectorstore import VectorStore

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

# 双路同时命中时的加分（经验值，0.05 足够让共现片段排到前面，又不会压过强相关）
_BOTH_ROUTES_BONUS = 0.05
# BM25 归一化后的权重，略低于向量相似度
_BM25_WEIGHT = 0.9


@dataclass
class Hit:
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    score: float
    source: str = "vector"  # vector / bm25 / hybrid

    def to_reference(self, no: int) -> dict:
        """转成前端需要的引用结构。"""
        return {
            "no": no,
            "doc_id": self.doc_id,
            "doc": self.filename,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
            "snippet": self.text[:200],
        }


def _tokenize(text: str) -> list[str]:
    """中文按字、英文按词。jieba 装了就用 jieba，没装也能跑。"""
    try:
        import jieba

        jieba.setLogLevel(60)  # 关掉 jieba 的启动日志
        tokens = [t.strip() for t in jieba.lcut_for_search(text.lower())]
        tokens = [t for t in tokens if t and not t.isspace()]
    except Exception:
        tokens = [w.lower() for w in _WORD_RE.findall(text)]
        tokens += _CJK_RE.findall(text)
    return tokens


class ChunkIndex:
    """chunk 原文的落盘索引（data/index/chunks.jsonl）。

    Chroma 里只有向量和 metadata，BM25 需要原文语料；引用回查、删除文档也要用它。
    体量到几十万块时应该换成 ES / OpenSearch，这里保持单文件简单可迁移。
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._records: list[dict] | None = None
        self._bm25: Any = None
        self._bm25_size = 0

    # ------------------------------------------------------------------ 读取
    def records(self) -> list[dict]:
        with self._lock:
            if self._records is None:
                self._records = self._load_from_disk()
            return self._records

    def _load_from_disk(self) -> list[dict]:
        if not self._path.exists():
            return []
        rows: list[dict] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    # ------------------------------------------------------------------ 写入
    def add(self, records: list[dict]) -> None:
        if not records:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                for row in records:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._records = None  # 下次读取时重新加载

    def remove_document(self, doc_id: str) -> None:
        with self._lock:
            rows = self._records if self._records is not None else self._load_from_disk()
            kept = [row for row in rows if row.get("doc_id") != doc_id]
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", encoding="utf-8") as fh:
                for row in kept:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._records = kept
            self._bm25 = None

    # ------------------------------------------------------------------ BM25
    def _ensure_bm25(self):
        rows = self.records()
        if self._bm25 is not None and self._bm25_size == len(rows):
            return self._bm25, rows
        if not rows:
            return None, rows

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:  # pragma: no cover
            logger.warning("未安装 rank_bm25，BM25 召回已跳过")
            return None, rows

        tokenized = [_tokenize(row.get("text", "")) for row in rows]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_size = len(rows)
        return self._bm25, rows

    def search_bm25(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        bm25, rows = self._ensure_bm25()
        if bm25 is None:
            return []

        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(rows[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def stats(self) -> dict:
        rows = self.records()
        return {
            "chunks": len(rows),
            "documents": len({row.get("doc_id") for row in rows}),
        }


# --------------------------------------------------------------------------- 重排
_CROSS_ENCODER: Any = None
_CROSS_ENCODER_NAME = ""


def _get_cross_encoder():
    global _CROSS_ENCODER, _CROSS_ENCODER_NAME
    if _CROSS_ENCODER is not None and settings.rerank_model == _CROSS_ENCODER_NAME:
        return _CROSS_ENCODER
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning("未安装 sentence-transformers，跳过重排（pip install -r requirements-ml.txt）")
        return None
    logger.info("加载重排模型 %s", settings.rerank_model)
    _CROSS_ENCODER = CrossEncoder(settings.rerank_model, max_length=512)
    _CROSS_ENCODER_NAME = settings.rerank_model
    return _CROSS_ENCODER


def rerank(question: str, hits: list[Hit], top_n: int) -> list[Hit]:
    if len(hits) <= 1:
        return hits[:top_n]
    encoder = _get_cross_encoder()
    if encoder is None:
        return hits[:top_n]

    pairs = [(question, hit.text) for hit in hits]
    scores = encoder.predict(pairs)
    for hit, score in zip(hits, scores, strict=False):
        # Cross-Encoder 输出是 logits，sigmoid 到 0~1 便于和阈值比较
        hit.score = float(1.0 / (1.0 + pow(2.718281828, -float(score))))
    return sorted(hits, key=lambda h: h.score, reverse=True)[:top_n]


# --------------------------------------------------------------------------- 主流程
def retrieve(
    question: str,
    embedder: BaseEmbedder,
    store: VectorStore,
    chunk_index: ChunkIndex,
    top_k: int | None = None,
) -> list[Hit]:
    """双路召回 + 融合。返回按相关性降序的片段列表（最多 rerank_top_n 条）。"""
    top_k = top_k or settings.retrieve_top_k
    merged: dict[tuple[str, int], Hit] = {}

    # 1) 向量召回
    try:
        query_vector = embedder.encode([question])[0]
        for item in store.search(query_vector, top_k):
            meta = item.get("metadata") or {}
            key = (str(meta.get("doc_id", "")), int(meta.get("chunk_index", -1)))
            merged[key] = Hit(
                doc_id=key[0],
                filename=str(meta.get("filename", "未知文档")),
                page=int(meta.get("page", 1)),
                chunk_index=key[1],
                text=item.get("text", ""),
                score=float(item.get("score", 0.0)),
                source="vector",
            )
    except Exception as exc:
        logger.warning("向量召回失败：%s", exc)

    # 2) BM25 召回
    if settings.enable_bm25:
        try:
            bm25_hits = chunk_index.search_bm25(question, top_k)
            max_raw = max((score for _, score in bm25_hits), default=0.0) or 1.0
            for row, raw in bm25_hits:
                key = (str(row.get("doc_id", "")), int(row.get("chunk_index", -1)))
                score = _BM25_WEIGHT * raw / max_raw
                if key in merged:  # 双路都命中
                    merged[key].score = min(1.0, merged[key].score + _BOTH_ROUTES_BONUS)
                    merged[key].source = "hybrid"
                else:
                    merged[key] = Hit(
                        doc_id=key[0],
                        filename=str(row.get("filename", "未知文档")),
                        page=int(row.get("page", 1)),
                        chunk_index=key[1],
                        text=str(row.get("text", "")),
                        score=score,
                        source="bm25",
                    )
        except Exception as exc:
            logger.warning("BM25 召回失败：%s", exc)

    # 3) 过滤 + 排序
    candidates = [hit for hit in merged.values() if hit.score >= settings.min_relevance]
    candidates.sort(key=lambda h: h.score, reverse=True)

    # 4) 可选重排
    if settings.enable_rerank and candidates:
        candidates = rerank(question, candidates[: top_k], settings.rerank_top_n)

    return candidates[: settings.rerank_top_n]

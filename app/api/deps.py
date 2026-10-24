"""依赖注入：把向量库、chunk 索引、Embedding、LLM 客户端做成进程内单例。

模型加载和大对象初始化都只做一次，避免每个请求重复加载。
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import settings
from app.llm.client import LLMClient, get_llm_client
from app.llm.embeddings import BaseEmbedder, get_embedder
from app.llm.retriever import ChunkIndex
from app.llm.vectorstore import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()


@lru_cache
def get_chunk_index() -> ChunkIndex:
    settings.ensure_dirs()
    return ChunkIndex(settings.chunks_file)


def get_embedding_model() -> BaseEmbedder:
    return get_embedder()


def get_llm() -> LLMClient:
    return get_llm_client()

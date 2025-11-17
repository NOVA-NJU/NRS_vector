"""向量存储的业务逻辑实现。"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from config import VectorSettings, get_settings
from models import DocumentPayload, DocumentUpsertResponse, VectorMatch, VectorSearchRequest, VectorSearchResponse

# TODO (feat/db-crud):
# 以下 VectorService 类当前使用内存字典作为向量和文档的临时存储。
# `feat/db-crud` 分支的目标是将其替换为与真实向量数据库（如 ChromaDB）的交互。
#
# 需要修改的关键部分包括：
# 1. `_init_client()`:
#    - 当前返回一个占位符字符串。
#    - 需要修改为初始化并返回一个真实的数据库客户端实例。
#
# 2. `upsert_document()`:
#    - 当前将文档和向量存储在 `self._documents` 和 `self._embeddings` 字典中。
#    - 需要修改为调用数据库客户端的 `upsert` 或 `add` 方法，将文档、向量和元数据持久化到数据库中。
#
# 3. `search()`:
#    - 当前在内存中的 `self._embeddings` 字典上进行暴力余弦相似度搜索。
#    - 需要修改为调用数据库客户端的 `query` 或 `search` 方法，利用数据库的索引进行高效检索。
#    - 内存缓存 `self._documents` 和 `self._embeddings` 届时可以移除。


class VectorService:
    """
    VectorService 封装了与向量数据库的交互逻辑，旨在保持 FastAPI 路由层的简洁和解耦。
    主要功能包括：
    - 初始化底层向量数据库客户端（当前为占位实现）。
    - 加载并管理 SentenceTransformer 嵌入模型。
    - 支持批量文本向量化，并进行归一化以便余弦相似度计算。
    - 文档的写入/更新（upsert），并将文档及其向量缓存于内存中。
    - 基于余弦相似度在缓存向量中进行检索，返回最相似的文档结果。
    属性:
        settings (VectorSettings): 向量服务相关配置。
        _client (Any): 向量数据库客户端实例（当前为字符串占位）。
        _embedder (SentenceTransformer): 文本嵌入模型实例。
        _documents (Dict[str, DocumentPayload]): 已缓存的文档内容。
        _embeddings (Dict[str, np.ndarray]): 已缓存的文档向量。
    方法:
        _init_client(): 初始化底层向量数据库客户端。
        _init_embedder(): 加载 SentenceTransformer 嵌入模型。
        _embed_texts(texts): 批量生成文本向量。
        upsert_document(payload): 写入或更新文档及其向量。
        search(request): 基于余弦相似度检索最相似文档。
    """
    """封装向量数据库交互，保持 FastAPI 路由层的简洁。"""


    def __init__(self, settings: VectorSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = self._init_client()
        self._embedder = self._init_embedder()
        self._documents: Dict[str, DocumentPayload] = {}
        self._embeddings: Dict[str, np.ndarray] = {}

    def _init_client(self) -> str:
        """初始化底层向量数据库（当前为占位实现）。"""
        # 真实实现应返回实际客户端，这里仅保留占位以保证可导入。
        return f"chroma://{self.settings.db_path}"

    def _init_embedder(self) -> SentenceTransformer:
        """加载 SentenceTransformer 模型，沿用 test.py 的逻辑。"""
        load_kwargs = {"token": self.settings.HF_TOKEN} if self.settings.HF_TOKEN else {}
        return SentenceTransformer(self.settings.embedding_model, **load_kwargs)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """批量生成文本向量，默认进行归一化方便余弦相似度计算。"""
        if not texts:
            return np.empty((0, self.settings.embedding_dim))
        return self._embedder.encode(texts, normalize_embeddings=True)

    async def upsert_document(self, payload: DocumentPayload) -> DocumentUpsertResponse:
        """写入或更新文档，并将 embedding 缓存在内存中。"""
        embedding = self._embed_texts([payload.text])[0]
        self._documents[payload.document_id] = payload
        self._embeddings[payload.document_id] = embedding
        return DocumentUpsertResponse(document_id=payload.document_id, status="stored")

    async def search(self, request: VectorSearchRequest) -> VectorSearchResponse:
        """使用余弦相似度在缓存向量中检索。"""
        if not self._embeddings:
            return VectorSearchResponse(results=[], query=request.query, top_k=request.top_k)

        query_embedding = self._embed_texts([request.query])[0]
        scored_matches: List[VectorMatch] = []
        for doc_id, doc_embedding in self._embeddings.items():
            score_tensor = cos_sim(query_embedding, doc_embedding)
            score = float(score_tensor.item())
            metadata = self._documents[doc_id].metadata
            scored_matches.append(VectorMatch(document_id=doc_id, score=score, metadata=metadata))

        scored_matches.sort(key=lambda match: match.score, reverse=True)
        top_results = scored_matches[: request.top_k]
        return VectorSearchResponse(results=top_results, query=request.query, top_k=request.top_k)


vector_service = VectorService()

"""向量存储的业务逻辑实现。"""
from __future__ import annotations

from typing import List

from config import VectorSettings, get_settings
from models import DocumentPayload, DocumentUpsertResponse, VectorMatch, VectorSearchRequest, VectorSearchResponse


class VectorService:
    """封装向量数据库交互，保持 FastAPI 路由层的简洁。"""

    def __init__(self, settings: VectorSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = self._init_client()

    def _init_client(self) -> str:
        """初始化底层向量数据库（当前为占位实现）。"""
        # 真实实现应返回实际客户端，这里仅保留占位以保证可导入。
        return f"chroma://{self.settings.db_path}"

    async def upsert_document(self, payload: DocumentPayload) -> DocumentUpsertResponse:
        """写入或更新文档的向量化内容。"""
        # 目前仅回传请求信息，后续可替换为真实数据库操作。
        return DocumentUpsertResponse(document_id=payload.document_id, status="stored")

    async def search(self, request: VectorSearchRequest) -> VectorSearchResponse:
        """通过配置的向量后端执行相似度搜索。"""
        dummy_results: List[VectorMatch] = [
            VectorMatch(document_id="placeholder-doc", score=0.0, metadata={"source": "stub"})
        ]
        return VectorSearchResponse(results=dummy_results[: request.top_k], query=request.query, top_k=request.top_k)


vector_service = VectorService()

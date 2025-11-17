"""向量相关操作的 API 路由。"""
from fastapi import APIRouter

from models import DocumentPayload, DocumentUpsertResponse, VectorSearchRequest, VectorSearchResponse
from services import vector_service

router = APIRouter()


@router.post("/search", response_model=VectorSearchResponse)
async def search_vectors(request: VectorSearchRequest) -> VectorSearchResponse:
    """根据查询语句执行语义相似度搜索。"""
    return await vector_service.search(request)


@router.post("/documents", response_model=DocumentUpsertResponse)
async def add_document(payload: DocumentPayload) -> DocumentUpsertResponse:
    """向向量库新增或更新文档。"""
    return await vector_service.upsert_document(payload)

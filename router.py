"""向量相关操作的 API 路由定义。

这个文件定义了所有与向量存储相关的 HTTP API 接口。
用户可以通过这些接口:
1. 搜索相似文档
2. 添加新文档到向量库
"""
from fastapi import APIRouter

from models import DocumentPayload, DocumentUpsertResponse, VectorSearchRequest, VectorSearchResponse, DocumentGetResponse, ClearDbResponse
from services import vector_service

# 创建一个 API 路由器
# 路由器就像一个接口组，把相关的接口组织在一起
router = APIRouter()


@router.post("/search", response_model=VectorSearchResponse)
async def search_vectors(request: VectorSearchRequest) -> VectorSearchResponse:
    """语义相似度搜索接口。
    
    这个接口接收用户的查询文本，返回最相似的文档列表。
    
    工作流程:
    1. 接收用户的查询请求（包含查询文本和想要多少个结果）
    2. 把查询文本转换成向量
    3. 在向量库中搜索最相似的文档
    4. 返回匹配结果
    
    参数:
        request: 搜索请求对象，包含:
            - query: 查询文本，例如 "图书馆几点开门"
            - top_k: 想要返回几个结果，例如 5
    
    返回:
        搜索结果对象，包含:
            - results: 匹配的文档列表（按相似度从高到低排序）
            - query: 原始查询文本
            - top_k: 返回的结果数量
    
    示例:
        POST /vectors/search
        {
            "query": "图书馆开放时间",
            "top_k": 3
        }
    """
    # 直接调用服务层的搜索方法，保持路由层简洁
    return await vector_service.search(request)


@router.get("/search/{document_id}", response_model=DocumentGetResponse)
async def get_document(document_id: str) -> DocumentGetResponse:
    return await vector_service.get_document_by_id(document_id)


@router.post("/documents", response_model=DocumentUpsertResponse)
async def add_document(payload: DocumentPayload) -> DocumentUpsertResponse:
    """文档添加/更新接口。
    
    这个接口用于向向量库中添加新文档或更新已有文档。
    
    工作流程:
    1. 接收文档数据（ID、文本内容、元数据）
    2. 把文本转换成向量
    3. 存储文档和向量
    4. 返回确认信息
    
    参数:
        payload: 文档数据对象，包含:
            - document_id: 文档的唯一标识（如果ID已存在则更新，不存在则新建）
            - text: 文档的文本内容
            - metadata: 文档的元数据（额外信息），可选
    
    返回:
        确认响应对象，包含:
            - document_id: 文档ID
            - status: 存储状态，例如 "stored"
            - detail: 详细信息（可选）
    
    示例:
        POST /vectors/documents
        {
            "document_id": "doc_001",
            "text": "南京大学图书馆上午8点开放",
            "metadata": {"source": "官网", "date": "2025-01-01"}
        }
    """
    # 直接调用服务层的文档写入方法
    return await vector_service.upsert_document(payload)


@router.post("/cleardb", response_model=ClearDbResponse)
async def clear_db() -> ClearDbResponse:
    return await vector_service.clear_db()

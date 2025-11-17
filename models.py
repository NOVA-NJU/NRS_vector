"""向量相关 API 的数据契约。"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentPayload(BaseModel):
    """描述单个文档及其元数据。"""

    document_id: str = Field(..., description="文档的唯一标识")
    text: str = Field(..., description="需要做嵌入的原始文本内容")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorSearchRequest(BaseModel):
    """相似度搜索请求体。"""

    query: str = Field(..., description="需要嵌入并匹配的自然语言查询")
    top_k: int = Field(5, ge=1, le=50, description="需要返回的结果数量")


class VectorMatch(BaseModel):
    """向量库返回的单条匹配结果。"""

    document_id: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorSearchResponse(BaseModel):
    """包含相似度匹配结果及其上下文的响应体。"""

    results: List[VectorMatch]
    query: str
    top_k: int


class DocumentUpsertResponse(BaseModel):
    """用于确认文档入库的响应。"""

    document_id: str
    status: str = "queued"
    detail: Optional[str] = None

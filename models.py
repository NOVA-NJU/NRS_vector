"""向量相关 API 的数据模型定义。

这个文件定义了 API 接口使用的所有数据结构（数据契约）。
使用 Pydantic 模型确保:
1. 数据类型正确
2. 必填字段不能缺失
3. 数据格式符合要求
4. 自动生成 API 文档

什么是数据契约?
就像买卖双方签订的合同，规定了数据的格式和内容。
前端/客户端必须按照这个格式发送数据，后端也按这个格式返回数据。
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentPayload(BaseModel):
    """文档数据模型，用于添加或更新文档。
    
    这个模型描述了一个文档应该包含哪些信息。
    当用户想要向向量库添加文档时，就要按照这个格式提供数据。
    """

    # document_id: 文档的唯一标识符
    # ... 表示这是必填字段，不能省略
    # description 会显示在 API 文档中，帮助使用者理解这个字段的含义
    document_id: str = Field(..., description="文档的唯一标识")
    
    # text: 文档的原始文本内容，这是最重要的字段
    # 这个文本会被转换成向量用于后续搜索
    text: str = Field(..., description="需要做嵌入的原始文本内容")
    
    # metadata: 文档的元数据（附加信息）
    # Dict[str, Any] 表示一个字典，key 是字符串，value 可以是任何类型
    # default_factory=dict 表示如果不提供这个字段，就用空字典作为默认值
    # 元数据可以存储任何额外信息，比如来源、作者、日期等
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorSearchRequest(BaseModel):
    """搜索请求模型，用于执行相似度搜索。
    
    用户发起搜索时需要提供:
    1. 想搜索什么（query）
    2. 想要几个结果（top_k）
    """

    # query: 用户的搜索查询文本
    # 例如: "图书馆几点开门" 或 "食堂今天吃什么"
    query: str = Field(..., description="需要嵌入并匹配的自然语言查询")
    
    # top_k: 想要返回的结果数量
    # 默认值是 5，表示返回最相似的 5 个文档
    # ge=1 表示 greater than or equal，最小值是 1
    # le=50 表示 less than or equal，最大值是 50
    top_k: int = Field(5, ge=1, le=50, description="需要返回的结果数量")


class VectorMatch(BaseModel):
    """单个匹配结果模型，表示一个相似的文档。
    
    搜索返回的每个结果都是一个 VectorMatch 对象，包含:
    1. 文档ID
    2. 相似度分数
    3. 文档的元数据
    """

    # document_id: 匹配到的文档的ID
    document_id: str
    
    # score: 相似度分数，范围通常是 0 到 1
    # 分数越高表示越相似
    # 例如: 0.95 表示非常相似，0.3 表示不太相似
    score: float
    
    # metadata: 文档的元数据
    # 搜索时会把存储时提供的元数据一起返回
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorSearchResponse(BaseModel):
    """搜索响应模型，包含所有匹配结果。
    
    这是搜索接口返回给用户的数据格式。
    包含了查询结果列表以及一些上下文信息。
    """

    # results: 匹配结果列表，按相似度从高到低排序
    # List[VectorMatch] 表示这是一个 VectorMatch 对象的列表
    results: List[VectorMatch]
    
    # query: 原始查询文本（回传给用户，方便确认）
    query: str
    
    # top_k: 请求的结果数量（回传给用户，方便确认）
    # 注意: 实际返回的结果数量可能小于 top_k
    # 例如: 向量库只有 3 个文档，但用户请求 top_k=5，则只返回 3 个
    top_k: int


class DocumentUpsertResponse(BaseModel):
    """文档写入响应模型，确认文档已成功存储。
    
    这是添加/更新文档接口返回给用户的确认信息。
    """

    # document_id: 文档ID（回传给用户确认）
    document_id: str
    
    # status: 操作状态
    # 默认是 "queued"（已排队）
    # 实际使用中可能是: "stored"（已存储）、"failed"（失败）等
    status: str = "queued"
    
    # detail: 详细信息（可选）
    # Optional[str] 表示这个字段可以是字符串或 None
    # 可以用来存储额外的说明信息或错误信息
    detail: Optional[str] = None

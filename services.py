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
        """初始化向量服务实例。
        
        参数:
            settings: 配置对象，如果不传则自动读取默认配置
        
        初始化步骤:
            1. 保存配置到 self.settings
            2. 初始化数据库客户端（当前为占位）
            3. 加载嵌入模型（SentenceTransformer）
            4. 创建内存缓存字典，用于存储文档和向量
        """
        # 如果没有传入配置，就用默认配置
        self.settings = settings or get_settings()
        
        # 初始化数据库客户端（目前只是返回一个字符串占位符）
        self._client = self._init_client()
        
        # 加载文本嵌入模型（把文字转换成向量的AI模型）
        self._embedder = self._init_embedder()
        
        # 创建两个字典用于在内存中缓存数据：
        # _documents: 存储文档内容，key是文档ID，value是文档对象
        self._documents: Dict[str, DocumentPayload] = {}
        # _embeddings: 存储文档对应的向量，key是文档ID，value是numpy数组
        self._embeddings: Dict[str, np.ndarray] = {}

    def _init_client(self) -> str:
        """初始化底层向量数据库客户端。
        
        返回:
            数据库连接字符串（当前为占位实现）
        
        注意:
            这是一个占位方法，真实场景中应该:
            1. 创建实际的数据库客户端对象（如 chromadb.Client()）
            2. 建立数据库连接
            3. 返回可用的客户端实例
            
            当前只返回一个字符串，方便开发阶段测试代码逻辑
        """
        # 拼接一个假的数据库连接地址，格式类似 "chroma://./chroma_db"
        # 真实实现应该是: return chromadb.Client(...)
        return f"chroma://{self.settings.db_path}"

    def _init_embedder(self) -> SentenceTransformer:
        """加载文本嵌入模型（把文字转换成数字向量的AI模型）。
        
        返回:
            SentenceTransformer 模型实例，用于后续的文本向量化
        
        工作流程:
            1. 检查配置中是否有 Hugging Face Token（用于下载私有模型）
            2. 如果有 token，就传给模型加载器
            3. 从 Hugging Face 下载或加载本地缓存的模型
        
        什么是嵌入模型?
            嵌入模型可以把文字转换成一串数字（向量）。
            相似意思的文字会被转换成相似的向量，这样计算机就能理解文字的语义。
            例如: "图书馆" 和 "library" 会被转换成接近的向量
        """
        # 如果配置里有 HF_TOKEN，就准备一个字典传递 token 参数
        # 否则传递空字典（不需要认证）
        load_kwargs = {"token": self.settings.HF_TOKEN} if self.settings.HF_TOKEN else {}
        
        # 加载模型，模型名称从配置中读取（如 "BAAI/bge-small-zh-v1.5"）
        # **load_kwargs 会把字典展开成关键字参数
        return SentenceTransformer(self.settings.embedding_model, **load_kwargs)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """把一批文本转换成向量（数字表示）。
        
        参数:
            texts: 要转换的文本列表，例如 ["你好", "再见"]
        
        返回:
            numpy 数组，每行是一个文本对应的向量
            例如: shape 为 (2, 512) 表示2个文本，每个512维向量
        
        什么是归一化?
            归一化就是把向量的长度统一调整到1。
            这样做的好处是，后续计算相似度时只需要关注方向，不用考虑长度。
            类似把所有箭头都调整成同样长度，只比较它们的指向。
        """
        # 如果传入的文本列表是空的，就返回一个空的向量数组
        if not texts:
            # np.empty 创建一个空数组，shape 是 (0, 维度)
            return np.empty((0, self.settings.embedding_dim))
        
        # 调用模型的 encode 方法，把文本列表转换成向量
        # normalize_embeddings=True 表示对向量进行归一化处理
        return self._embedder.encode(texts, normalize_embeddings=True)

    async def upsert_document(self, payload: DocumentPayload) -> DocumentUpsertResponse:
        """添加或更新一个文档到向量库。
        
        参数:
            payload: 文档数据，包含文档ID、文本内容和元数据
        
        返回:
            确认信息，告诉调用者文档已存储成功
        
        工作流程:
            1. 把文档的文本内容转换成向量（调用嵌入模型）
            2. 把文档内容存到 _documents 字典中
            3. 把对应的向量存到 _embeddings 字典中
            4. 返回成功响应
        
        什么是 upsert?
            upsert = update + insert 的组合词
            如果文档ID已存在就更新，不存在就插入，一个操作搞定两件事
        """
        # 步骤1: 把文档文本转换成向量
        # _embed_texts 返回数组，[0] 取出第一个（也是唯一一个）向量
        embedding = self._embed_texts([payload.text])[0]
        
        # 步骤2: 把文档对象存到字典中，key是文档ID
        self._documents[payload.document_id] = payload
        
        # 步骤3: 把向量也存到字典中，key同样是文档ID
        # 这样通过文档ID就能同时找到文档内容和它的向量
        self._embeddings[payload.document_id] = embedding
        
        # 步骤4: 返回一个响应对象，告诉调用者文档已存储
        return DocumentUpsertResponse(document_id=payload.document_id, status="stored")

    async def search(self, request: VectorSearchRequest) -> VectorSearchResponse:
        """根据查询语句搜索最相似的文档。
        
        参数:
            request: 搜索请求，包含查询文本和要返回的结果数量(top_k)
        
        返回:
            搜索结果响应，包含匹配的文档列表（按相似度从高到低排序）
        
        工作流程:
            1. 检查是否有已存储的文档，没有就返回空结果
            2. 把查询文本转换成向量
            3. 遍历所有已存储的文档向量，计算与查询向量的相似度
            4. 按相似度从高到低排序
            5. 返回前 top_k 个最相似的结果
        
        什么是余弦相似度?
            余弦相似度是衡量两个向量相似程度的方法。
            值在 -1 到 1 之间，越接近 1 表示越相似。
            就像比较两个箭头的方向是否一致，方向越接近就越相似。
        """
        # 步骤1: 如果向量库是空的（没有存储任何文档），直接返回空结果
        if not self._embeddings:
            return VectorSearchResponse(results=[], query=request.query, top_k=request.top_k)

        # 步骤2: 把用户的查询文本转换成向量
        # 例如: "图书馆几点开门" -> [0.123, 0.456, ...]
        query_embedding = self._embed_texts([request.query])[0]
        
        # 步骤3: 创建一个列表，用来存储所有文档的匹配结果
        scored_matches: List[VectorMatch] = []
        
        # 遍历所有已存储的文档向量
        for doc_id, doc_embedding in self._embeddings.items():
            # 计算查询向量和文档向量的余弦相似度
            # cos_sim 返回的是一个 tensor（张量），需要转换成普通数字
            score_tensor = cos_sim(query_embedding, doc_embedding)
            score = float(score_tensor.item())  # .item() 把 tensor 转成 Python 数字
            
            # 获取这个文档的元数据（额外信息）
            metadata = self._documents[doc_id].metadata
            
            # 把匹配结果添加到列表中
            scored_matches.append(VectorMatch(document_id=doc_id, score=score, metadata=metadata))

        # 步骤4: 按相似度分数从高到低排序
        # key=lambda match: match.score 表示按 score 字段排序
        # reverse=True 表示降序（从大到小）
        scored_matches.sort(key=lambda match: match.score, reverse=True)
        
        # 步骤5: 只取前 top_k 个结果
        # 例如: top_k=5 就只返回最相似的5个文档
        top_results = scored_matches[: request.top_k]
        
        # 返回搜索结果响应对象
        return VectorSearchResponse(results=top_results, query=request.query, top_k=request.top_k)


vector_service = VectorService()

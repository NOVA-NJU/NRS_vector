"""向量存储的业务逻辑实现。"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
#
# ✅ TODO (feat/embedder-chunker): 已完成！
# 文本分块（Chunking）功能已集成到 `upsert_document()` 方法中。
# 
# 实现内容：
# 1. 新增配置项（config.py）：
#    - enable_chunking: 是否启用分块（默认 False）
#    - chunk_size: 每块字符数（默认 500）
#    - chunk_overlap: 块间重叠字符数（默认 50）
#
# 2. 新增辅助方法：
#    - _chunk_text(): 使用 LangChain RecursiveCharacterTextSplitter 智能分块
#      * 优先按段落（双换行）分割
#      * 其次按句子（句号、问号、感叹号）分割
#      * 再按标点符号（分号、逗号）分割
#      * 最后按空格或字符级别分割
#      * 保持语义完整性，避免在句子中间切断
#
# 3. upsert_document() 增强：
#    - 自动判断文本长度是否需要分块
#    - 分块模式：为每个块生成子文档ID（parent_doc_chunk_N）
#    - 非分块模式：保持原有逻辑
#
# 4. 搜索时的考虑：
#    - 当前搜索会同时返回所有块的匹配结果
#    - 每个块独立计算相似度分数
#    - 调用方可通过 metadata.parent_doc 识别同一文档的不同块
#    - 未来可在 search() 方法中添加块聚合逻辑（如取最高分或平均分）


class VectorService:
    """
    VectorService 封装了与向量数据库的交互逻辑，旨在保持 FastAPI 路由层的简洁和解耦。
    主要功能包括：
    - 初始化底层向量数据库客户端（当前为占位实现）。
    - 加载并管理 SentenceTransformer 嵌入模型。
    - 支持批量文本向量化，并进行归一化以便余弦相似度计算。
    - 文档的写入/更新（upsert），支持长文本自动分块。
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
        _chunk_text(text, chunk_size, overlap): 将长文本切分成重叠的块。
        upsert_document(payload): 写入或更新文档及其向量（支持自动分块）。
        search(request): 基于余弦相似度检索最相似文档。
    """


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

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """将长文本切分成多个重叠的块（使用 LangChain 智能分块）。
        
        参数:
            text: 要切分的原始文本
            chunk_size: 每个块的字符数
            overlap: 相邻块之间的重叠字符数
        
        返回:
            文本块列表，每个块是一个字符串
        
        分块策略:
            使用 LangChain 的 RecursiveCharacterTextSplitter，按照自然语义边界切分：
            1. 优先按段落分割（双换行）
            2. 其次按句子分割（句号、问号、感叹号）
            3. 再按分号、逗号等标点分割
            4. 最后按空格分割
            5. 回退到字符级别分割
            
            这种方式比简单的固定长度切分更智能，能保持文本的语义完整性。
        
        优势:
            - 保留语义完整性：尽量不在句子中间切断
            - 适配中文：使用中文标点符号作为分隔符
            - 灵活性强：自动调整块大小以适应自然边界
        
        示例:
            text = "这是第一句。这是第二句。这是第三句。"
            结果可能是: ["这是第一句。", "这是第二句。这是第三句。"]
            （根据 chunk_size 和文本长度自动调整）
        """
        # 如果文本长度小于等于块大小，不需要分块，直接返回原文本
        if len(text) <= chunk_size:
            return [text]
        
        # 定义中文优化的分隔符列表
        # 优先级从高到低：段落 -> 句子 -> 短语 -> 词 -> 字符
        chinese_separators = [
            "\n\n",  # 双换行：段落分隔
            "\n",    # 单换行：行分隔
            "。",     # 句号：句子结束
            "！",     # 感叹号
            "？",     # 问号
            "；",     # 分号
            "，",     # 逗号
            " ",      # 空格：词分隔
            ""        # 空字符串：回退到字符级分割
        ]
        
        # 创建 LangChain 文本分割器
        # RecursiveCharacterTextSplitter 会递归尝试使用分隔符列表中的每个分隔符
        # 直到找到合适的切分点
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,           # 目标块大小
            chunk_overlap=overlap,           # 块间重叠大小
            length_function=len,             # 使用字符数作为长度度量
            separators=chinese_separators,   # 应用中文优化的分隔符
            keep_separator=True              # 保留分隔符（如句号）在文本中
        )
        
        # 执行文本分割
        chunks = text_splitter.split_text(text)
        
        # 过滤掉空白块
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    # TODO (feat/embedder-chunker):
    # 文本分块（Chunking）功能应该在这里集成！
    # 
    # 为什么需要分块?
    # - 长文档（如论文、书籍章节）如果整体向量化，会损失细节信息
    # - 嵌入模型通常有最大输入长度限制（如 512 个 token）
    # - 分块后的小段落能提供更精准的匹配结果
    #
    # 建议的实现位置和方式:
    # 1. 在 `upsert_document()` 方法内部，接收到 payload.text 后：
    #    - 如果文本超过阈值（如 500 字符），触发分块逻辑
    #    - 调用分块函数（如 _chunk_text()）将长文本切分成多个小块
    #    - 为每个块生成子文档ID（如 "doc_001_chunk_0", "doc_001_chunk_1"）
    #    - 分别存储每个块的向量和内容
    #
    # 2. 分块策略可选方案:
    #    - 固定长度分块: 每 N 个字符一块
    #    - 滑动窗口分块: 重叠一定比例，避免语义割裂
    #    - 语义分块: 按句子、段落等自然边界切分
    #    - 使用 LangChain 的 TextSplitter 工具
    #
    # 3. 搜索时的调整:
    #    - 在 `search()` 方法中，需要考虑同一文档的多个块可能都被匹配
    #    - 可以按原始文档ID聚合结果，取最高分或平均分
    #
    # 4. 建议新增的辅助方法:
    #    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    #        \"\"\"将长文本切分成多个重叠的块。\"\"\"
    #        # 实现固定长度 + 重叠的分块逻辑
    #        pass
    #
    # 5. 配置项建议 (config.py):
    #    - CHUNK_SIZE: 每块的字符数
    #    - CHUNK_OVERLAP: 块之间的重叠字符数
    #    - ENABLE_CHUNKING: 是否启用分块（布尔值）
    #
    # 示例实现框架:
    # async def upsert_document(self, payload: DocumentPayload) -> DocumentUpsertResponse:
    #     if self.settings.enable_chunking and len(payload.text) > self.settings.chunk_size:
    #         # 分块模式
    #         chunks = self._chunk_text(payload.text)
    #         for idx, chunk in enumerate(chunks):
    #             chunk_id = f"{payload.document_id}_chunk_{idx}"
    #             embedding = self._embed_texts([chunk])[0]
    #             self._documents[chunk_id] = DocumentPayload(
    #                 document_id=chunk_id,
    #                 text=chunk,
    #                 metadata={**payload.metadata, "parent_doc": payload.document_id, "chunk_index": idx}
    #             )
    #             self._embeddings[chunk_id] = embedding
    #     else:
    #         # 原有逻辑：整体向量化
    #         embedding = self._embed_texts([payload.text])[0]
    #         self._documents[payload.document_id] = payload
    #         self._embeddings[payload.document_id] = embedding
    #     return DocumentUpsertResponse(document_id=payload.document_id, status="stored")

    async def upsert_document(self, payload: DocumentPayload) -> DocumentUpsertResponse:
        """添加或更新一个文档到向量库。
        
        参数:
            payload: 文档数据，包含文档ID、文本内容和元数据
        
        返回:
            确认信息，告诉调用者文档已存储成功
        
        工作流程:
            1. 判断是否需要分块（根据配置和文本长度）
            2. 如果需要分块：
               - 切分文本为多个小块
               - 为每个块生成子文档ID和向量
               - 分别存储每个块
            3. 如果不需要分块：
               - 直接对整个文本生成向量
               - 存储文档和向量
            4. 返回成功响应
        
        什么是 upsert?
            upsert = update + insert 的组合词
            如果文档ID已存在就更新，不存在就插入，一个操作搞定两件事
        """
        # 判断是否需要启用分块
        # 条件：配置开启 且 文本长度超过阈值
        if self.settings.enable_chunking and len(payload.text) > self.settings.chunk_size:
            # ========== 分块模式 ==========
            # 步骤1: 调用分块函数，将长文本切分成多个小块
            chunks = self._chunk_text(
                payload.text,
                self.settings.chunk_size,
                self.settings.chunk_overlap
            )
            
            # 步骤2: 遍历每个文本块，分别处理
            for idx, chunk in enumerate(chunks):
                # 为每个块生成唯一的子文档ID
                # 格式: "原文档ID_chunk_块索引"
                # 例如: "doc_001_chunk_0", "doc_001_chunk_1"
                chunk_id = f"{payload.document_id}_chunk_{idx}"
                
                # 将当前块的文本转换成向量
                embedding = self._embed_texts([chunk])[0]
                
                # 创建子文档对象，保留原文档的元数据并添加分块信息
                chunk_payload = DocumentPayload(
                    document_id=chunk_id,
                    text=chunk,
                    metadata={
                        **payload.metadata,  # 保留原文档的所有元数据
                        "parent_doc": payload.document_id,  # 记录父文档ID
                        "chunk_index": idx,  # 记录这是第几个块
                        "total_chunks": len(chunks)  # 记录总共有多少个块
                    }
                )
                
                # 存储子文档和对应的向量
                self._documents[chunk_id] = chunk_payload
                self._embeddings[chunk_id] = embedding
        else:
            # ========== 非分块模式（原有逻辑）==========
            # 直接对整个文本进行向量化
            embedding = self._embed_texts([payload.text])[0]
            
            # 存储完整文档和向量
            self._documents[payload.document_id] = payload
            self._embeddings[payload.document_id] = embedding
        
        # 返回成功响应（无论是否分块，都返回原始文档ID）
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

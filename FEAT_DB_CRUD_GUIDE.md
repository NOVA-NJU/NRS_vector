# feat/db-crud 分支开发指南

## 📋 分支目标

将当前基于内存字典的临时存储替换为持久化的向量数据库（ChromaDB），实现真正的生产级数据持久化和高效检索。

**版本目标**: v0.3.0  
**预计完成时间**: 2-3 周  
**优先级**: 高（生产就绪的前置条件）

---

## 🎯 核心任务

### 1. ChromaDB 集成与初始化

#### 任务描述
在 `services.py` 中实现真正的 ChromaDB 客户端初始化，替换当前的占位字符串。

#### 当前代码
```python
def _init_client(self) -> str:
    """占位方法，返回假的连接字符串"""
    return f"chroma://{self.settings.db_path}"
```

#### 目标实现
```python
import chromadb
from chromadb.config import Settings as ChromaSettings

def _init_client(self) -> chromadb.Client:
    """初始化 ChromaDB 客户端"""
    # 使用持久化模式
    client = chromadb.PersistentClient(
        path=self.settings.db_path,
        settings=ChromaSettings(
            anonymized_telemetry=False,  # 关闭匿名遥测
            allow_reset=True,             # 开发模式允许重置
        )
    )
    return client
```

#### 实施步骤
1. **安装依赖**
   ```bash
   pip install chromadb
   pip freeze > requirements.txt
   ```

2. **更新类型注解**
   - `self._client` 类型从 `str` 改为 `chromadb.Client`
   - 添加 `from chromadb import Client` 导入

3. **创建或获取集合**
   ```python
   def _get_or_create_collection(self) -> chromadb.Collection:
       """获取或创建向量集合"""
       return self._client.get_or_create_collection(
           name="documents",
           metadata={"description": "NRS Vector document collection"}
       )
   ```

4. **测试连接**
   - 添加单元测试验证客户端初始化成功
   - 测试集合创建和基本操作

#### 注意事项
- ChromaDB 默认使用 SQLite 作为元数据存储
- `db_path` 目录需要写权限
- 考虑在 `.gitignore` 中添加 `chroma_db/` 目录

---

### 2. 文档存储（Upsert）迁移

#### 任务描述
将 `upsert_document()` 方法从内存字典存储迁移到 ChromaDB。

#### 当前逻辑
```python
# 非分块模式
self._documents[payload.document_id] = payload
self._embeddings[payload.document_id] = embedding

# 分块模式
for idx, chunk in enumerate(chunks):
    chunk_id = f"{payload.document_id}_chunk_{idx}"
    self._documents[chunk_id] = chunk_payload
    self._embeddings[chunk_id] = embedding
```

#### 目标实现
```python
async def upsert_document(self, payload: DocumentPayload) -> DocumentUpsertResponse:
    """添加或更新文档到 ChromaDB"""
    collection = self._get_or_create_collection()
    
    if self.settings.enable_chunking and len(payload.text) > self.settings.chunk_size:
        # 分块模式
        chunks = self._chunk_text(
            payload.text,
            self.settings.chunk_size,
            self.settings.chunk_overlap
        )
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{payload.document_id}_chunk_{idx}"
            embedding = self._embed_texts([chunk])[0]
            
            ids.append(chunk_id)
            embeddings.append(embedding.tolist())  # NumPy 数组转列表
            documents.append(chunk)
            metadatas.append({
                **payload.metadata,
                "parent_doc": payload.document_id,
                "chunk_index": idx,
                "total_chunks": len(chunks)
            })
        
        # 批量插入所有块
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    else:
        # 非分块模式
        embedding = self._embed_texts([payload.text])[0]
        
        collection.upsert(
            ids=[payload.document_id],
            embeddings=[embedding.tolist()],
            documents=[payload.text],
            metadatas=[payload.metadata]
        )
    
    return DocumentUpsertResponse(document_id=payload.document_id, status="stored")
```

#### 实施步骤
1. **适配 ChromaDB API**
   - 研究 `collection.upsert()` 方法的参数格式
   - 注意 embeddings 需要是列表格式（`numpy.ndarray.tolist()`）

2. **元数据序列化**
   - ChromaDB 元数据支持的类型：str, int, float, bool
   - 复杂对象需要序列化为 JSON 字符串

3. **批量操作优化**
   - 分块模式下一次性 upsert 所有块，减少 I/O
   - 考虑添加进度回调（处理大文档时）

4. **错误处理**
   - 捕获 `chromadb.errors.UniqueConstraintError`（虽然 upsert 理论上不会报错）
   - 添加重试机制（数据库锁、网络问题）

5. **移除内存缓存**
   - 删除 `self._documents` 和 `self._embeddings` 字典
   - 如需缓存，考虑使用 LRU Cache

---

### 3. 语义搜索迁移

#### 任务描述
将 `search()` 方法从内存暴力搜索迁移到 ChromaDB 的向量索引查询。

#### 当前逻辑
```python
# 遍历所有文档向量，计算余弦相似度
for doc_id, doc_embedding in self._embeddings.items():
    score = float(cos_sim(query_embedding, doc_embedding).item())
    scored_matches.append(VectorMatch(...))

# 排序并返回 top_k
scored_matches.sort(key=lambda m: m.score, reverse=True)
top_results = scored_matches[:request.top_k]
```

#### 目标实现
```python
async def search(self, request: VectorSearchRequest) -> VectorSearchResponse:
    """基于 ChromaDB 进行语义搜索"""
    collection = self._get_or_create_collection()
    
    # 生成查询向量
    query_embedding = self._embed_texts([request.query])[0]
    
    # ChromaDB 查询
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=request.top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # 转换结果格式
    matches = []
    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        distance = results['distances'][0][i]
        
        # ChromaDB 返回的是距离（越小越好），需转换为相似度（越大越好）
        # 对于 L2 距离: similarity = 1 / (1 + distance)
        # 对于归一化向量的余弦距离: similarity = 1 - distance
        score = 1 - distance  # 假设使用余弦距离
        
        metadata = results['metadatas'][0][i] if results['metadatas'] else {}
        
        matches.append(VectorMatch(
            document_id=doc_id,
            score=score,
            metadata=metadata
        ))
    
    return VectorSearchResponse(
        results=matches,
        query=request.query,
        top_k=request.top_k
    )
```

#### 实施步骤
1. **理解距离度量**
   - ChromaDB 默认使用 L2 距离（欧几里得距离）
   - 可配置为余弦距离、内积等
   - 确认与当前余弦相似度的对应关系

2. **配置集合度量**
   ```python
   collection = client.get_or_create_collection(
       name="documents",
       metadata={"hnsw:space": "cosine"}  # 使用余弦距离
   )
   ```

3. **分数转换**
   - 余弦距离范围 [0, 2]，相似度范围 [-1, 1]
   - 转换公式: `similarity = 1 - (distance / 2)`
   - 或者: `similarity = 1 - distance`（如果向量已归一化）

4. **元数据过滤（可选增强）**
   ```python
   results = collection.query(
       query_embeddings=[query_embedding.tolist()],
       n_results=request.top_k,
       where={"category": "校园服务"}  # 元数据过滤
   )
   ```

5. **性能测试**
   - 对比内存搜索和 ChromaDB 搜索的性能
   - 大规模数据（10k+ 文档）下的响应时间

---

### 4. 文档块结果聚合（增强功能）

#### 任务描述
实现同一父文档的多个块的结果聚合逻辑。

#### 问题场景
```
搜索结果:
1. doc_001_chunk_2 (score: 0.87)
2. doc_001_chunk_0 (score: 0.85)
3. doc_002_chunk_1 (score: 0.82)
4. doc_001_chunk_1 (score: 0.80)
```

用户可能期望看到：
```
搜索结果:
1. doc_001 (最高分: 0.87, 平均分: 0.84, 匹配块: 3)
2. doc_002 (最高分: 0.82, 匹配块: 1)
```

#### 实现方案

**方案 A: 取最高分**
```python
def aggregate_chunks_max(matches: List[VectorMatch]) -> List[VectorMatch]:
    """按父文档聚合，取最高分"""
    parent_docs = {}
    
    for match in matches:
        parent_id = match.metadata.get("parent_doc")
        if not parent_id:
            # 非分块文档，直接保留
            parent_docs[match.document_id] = match
        else:
            # 分块文档，取最高分
            if parent_id not in parent_docs or match.score > parent_docs[parent_id].score:
                parent_docs[parent_id] = VectorMatch(
                    document_id=parent_id,
                    score=match.score,
                    metadata={
                        **match.metadata,
                        "matched_chunks": [match.document_id],
                        "aggregation_method": "max"
                    }
                )
            else:
                # 记录匹配的块ID
                parent_docs[parent_id].metadata["matched_chunks"].append(match.document_id)
    
    return sorted(parent_docs.values(), key=lambda m: m.score, reverse=True)
```

**方案 B: 平均分**
```python
def aggregate_chunks_avg(matches: List[VectorMatch]) -> List[VectorMatch]:
    """按父文档聚合，计算平均分"""
    parent_scores = {}
    
    for match in matches:
        parent_id = match.metadata.get("parent_doc", match.document_id)
        
        if parent_id not in parent_scores:
            parent_scores[parent_id] = {
                "scores": [],
                "chunks": [],
                "metadata": match.metadata
            }
        
        parent_scores[parent_id]["scores"].append(match.score)
        parent_scores[parent_id]["chunks"].append(match.document_id)
    
    results = []
    for parent_id, data in parent_scores.items():
        avg_score = sum(data["scores"]) / len(data["scores"])
        results.append(VectorMatch(
            document_id=parent_id,
            score=avg_score,
            metadata={
                **data["metadata"],
                "matched_chunks": len(data["chunks"]),
                "aggregation_method": "average"
            }
        ))
    
    return sorted(results, key=lambda m: m.score, reverse=True)
```

**方案 C: 加权分数**（推荐）
```python
def aggregate_chunks_weighted(matches: List[VectorMatch]) -> List[VectorMatch]:
    """按父文档聚合，使用加权分数"""
    parent_scores = {}
    
    for match in matches:
        parent_id = match.metadata.get("parent_doc", match.document_id)
        
        if parent_id not in parent_scores:
            parent_scores[parent_id] = {
                "max_score": match.score,
                "avg_score": match.score,
                "count": 1,
                "chunks": [match.document_id],
                "metadata": match.metadata
            }
        else:
            data = parent_scores[parent_id]
            data["max_score"] = max(data["max_score"], match.score)
            data["avg_score"] = (data["avg_score"] * data["count"] + match.score) / (data["count"] + 1)
            data["count"] += 1
            data["chunks"].append(match.document_id)
    
    results = []
    for parent_id, data in parent_scores.items():
        # 加权公式: 70% 最高分 + 30% 平均分
        weighted_score = 0.7 * data["max_score"] + 0.3 * data["avg_score"]
        
        results.append(VectorMatch(
            document_id=parent_id,
            score=weighted_score,
            metadata={
                **data["metadata"],
                "matched_chunks": data["count"],
                "max_score": data["max_score"],
                "avg_score": data["avg_score"],
                "aggregation_method": "weighted"
            }
        ))
    
    return sorted(results, key=lambda m: m.score, reverse=True)
```

#### 实施建议
1. **添加配置选项**
   ```python
   # config.py
   chunk_aggregation: str = "weighted"  # "max" | "avg" | "weighted" | "none"
   ```

2. **在 search() 方法中调用**
   ```python
   async def search(self, request: VectorSearchRequest) -> VectorSearchResponse:
       # ... ChromaDB 查询 ...
       
       if self.settings.chunk_aggregation != "none":
           matches = self._aggregate_chunks(matches, method=self.settings.chunk_aggregation)
       
       return VectorSearchResponse(...)
   ```

3. **提供原始结果选项**
   - 添加 API 参数 `aggregate_chunks: bool = True`
   - 让用户选择是否需要聚合

---

## 🔧 配置变更

### config.py 新增配置项

```python
class VectorSettings(BaseSettings):
    # ... 现有配置 ...
    
    # ========== ChromaDB 相关配置 ==========
    
    # collection_name: 向量集合名称
    # 多个集合可用于隔离不同领域的数据
    collection_name: str = "documents"
    
    # distance_metric: 向量距离度量方式
    # cosine: 余弦距离（推荐，与当前实现一致）
    # l2: 欧几里得距离
    # ip: 内积
    distance_metric: str = "cosine"
    
    # ========== 搜索增强配置 ==========
    
    # chunk_aggregation: 文档块聚合方式
    # none: 不聚合，返回所有匹配的块
    # max: 取最高分
    # avg: 取平均分
    # weighted: 加权分数（70% 最高分 + 30% 平均分）
    chunk_aggregation: str = "weighted"
    
    # ========== 性能优化配置 ==========
    
    # enable_cache: 是否启用查询结果缓存
    enable_cache: bool = False
    
    # cache_ttl: 缓存过期时间（秒）
    cache_ttl: int = 300
```

---

## 🧪 测试策略

### 1. 单元测试

创建 `tests/test_chromadb_integration.py`：

```python
import pytest
from services import VectorService
from models import DocumentPayload, VectorSearchRequest

@pytest.fixture
def vector_service():
    """创建测试用的 VectorService 实例"""
    service = VectorService()
    yield service
    # 清理：删除测试数据库
    service._client.delete_collection("documents")

def test_client_initialization(vector_service):
    """测试 ChromaDB 客户端初始化"""
    assert vector_service._client is not None
    assert hasattr(vector_service._client, 'get_or_create_collection')

def test_document_upsert(vector_service):
    """测试文档插入"""
    payload = DocumentPayload(
        document_id="test_001",
        text="这是一个测试文档",
        metadata={"category": "测试"}
    )
    
    response = await vector_service.upsert_document(payload)
    assert response.status == "stored"
    
    # 验证文档已存储
    collection = vector_service._get_or_create_collection()
    result = collection.get(ids=["test_001"])
    assert len(result['ids']) == 1

def test_document_search(vector_service):
    """测试语义搜索"""
    # 插入测试文档
    await vector_service.upsert_document(DocumentPayload(
        document_id="doc_001",
        text="南京大学图书馆",
        metadata={}
    ))
    
    # 搜索
    request = VectorSearchRequest(query="图书馆", top_k=5)
    response = await vector_service.search(request)
    
    assert len(response.results) > 0
    assert response.results[0].document_id == "doc_001"

def test_chunking_with_chromadb(vector_service):
    """测试分块存储和搜索"""
    # 启用分块
    vector_service.settings.enable_chunking = True
    vector_service.settings.chunk_size = 200
    
    # 插入长文本
    long_text = "这是第一段内容。" * 50  # 生成长文本
    await vector_service.upsert_document(DocumentPayload(
        document_id="long_doc",
        text=long_text,
        metadata={}
    ))
    
    # 验证生成了多个块
    collection = vector_service._get_or_create_collection()
    chunks = collection.get(where={"parent_doc": "long_doc"})
    assert len(chunks['ids']) > 1

def test_chunk_aggregation(vector_service):
    """测试块聚合功能"""
    # 准备测试数据...
    # 测试不同聚合策略的结果...
    pass
```

### 2. 集成测试

创建 `tests/test_api_with_db.py`：

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_full_workflow():
    """测试完整的文档存储和搜索流程"""
    # 1. 添加文档
    response = client.post("/vectors/documents", json={
        "document_id": "api_test_001",
        "text": "南京大学计算机系人工智能实验室",
        "metadata": {"source": "测试"}
    })
    assert response.status_code == 200
    
    # 2. 搜索文档
    response = client.post("/vectors/search", json={
        "query": "人工智能",
        "top_k": 5
    })
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) > 0
    assert results[0]["document_id"] == "api_test_001"
```

### 3. 性能测试

创建 `tests/test_performance.py`：

```python
import time
from services import VectorService
from models import DocumentPayload, VectorSearchRequest

def test_large_scale_upsert():
    """测试大规模文档插入性能"""
    service = VectorService()
    
    start_time = time.time()
    for i in range(1000):
        await service.upsert_document(DocumentPayload(
            document_id=f"perf_test_{i}",
            text=f"这是性能测试文档 {i}，包含一些测试内容。",
            metadata={"index": i}
        ))
    end_time = time.time()
    
    print(f"插入 1000 个文档耗时: {end_time - start_time:.2f} 秒")
    assert end_time - start_time < 60  # 应该在 1 分钟内完成

def test_search_performance():
    """测试搜索性能"""
    service = VectorService()
    
    # 预先插入 1000 个文档...
    
    start_time = time.time()
    for i in range(100):
        await service.search(VectorSearchRequest(query=f"测试查询 {i}", top_k=10))
    end_time = time.time()
    
    print(f"100 次搜索耗时: {end_time - start_time:.2f} 秒")
    assert end_time - start_time < 10  # 平均每次搜索 < 100ms
```

---

## 📦 依赖管理

### requirements.txt 更新

```txt
# 现有依赖
fastapi==0.121.2
uvicorn==0.38.0
pydantic==2.12.4
pydantic-settings==2.6.0
sentence-transformers==2.2.2
langchain-text-splitters==0.3.4

# 新增依赖
chromadb==0.5.20          # 向量数据库
numpy==1.24.3             # 数值计算（可能已包含在 sentence-transformers 中）

# 可选依赖（用于性能优化）
redis==5.0.1              # 查询结果缓存
```

### 安装命令

```bash
# 基础安装
pip install chromadb==0.5.20

# 可选：启用 GPU 支持（如果有 NVIDIA GPU）
pip install chromadb[gpu]

# 可选：启用分布式部署
pip install chromadb[distributed]
```

---

## 🚀 开发流程建议

### 阶段 1: 基础集成（第 1 周）

1. **创建分支**
   ```bash
   git checkout -b feat/db-crud
   ```

2. **安装 ChromaDB**
   ```bash
   pip install chromadb
   pip freeze > requirements.txt
   git add requirements.txt
   git commit -m "chore: add chromadb dependency"
   ```

3. **实现客户端初始化**
   - 修改 `_init_client()` 方法
   - 添加 `_get_or_create_collection()` 方法
   - 测试连接

4. **更新配置**
   - 在 `config.py` 中添加 ChromaDB 相关配置
   - 测试配置加载

### 阶段 2: 核心功能迁移（第 1-2 周）

5. **迁移 upsert_document()**
   - 实现非分块模式的 ChromaDB 存储
   - 实现分块模式的批量存储
   - 编写单元测试
   - 移除 `self._documents` 和 `self._embeddings`

6. **迁移 search()**
   - 实现 ChromaDB 查询
   - 调整分数转换逻辑
   - 验证搜索结果正确性

7. **测试与调试**
   - 运行现有的 `test_chunking.py`
   - 确保功能与内存版本一致
   - 修复发现的问题

### 阶段 3: 增强功能（第 2-3 周）

8. **实现块聚合**
   - 实现三种聚合策略
   - 添加配置选项
   - 编写测试

9. **性能优化**
   - 批量操作优化
   - 添加查询缓存（可选）
   - 性能测试与调优

10. **文档更新**
    - 更新 README.md
    - 添加 ChromaDB 使用说明
    - 编写迁移指南

### 阶段 4: 测试与合并（第 3 周）

11. **全面测试**
    - 单元测试覆盖率 > 80%
    - 集成测试
    - 性能测试

12. **代码审查与合并**
    ```bash
    git push origin feat/db-crud
    # 创建 Pull Request
    # 代码审查
    # 合并到 dev 分支
    ```

---

## ⚠️ 潜在风险与解决方案

### 风险 1: ChromaDB 学习曲线

**风险**: 团队不熟悉 ChromaDB API，可能导致实现延期。

**解决方案**:
- 先阅读 [ChromaDB 官方文档](https://docs.trychroma.com/)
- 创建独立的 POC（概念验证）项目，熟悉 API
- 参考 ChromaDB 官方示例代码

### 风险 2: 数据迁移困难

**风险**: 现有内存数据无法平滑迁移到 ChromaDB。

**解决方案**:
- 由于当前是内存存储，重启即丢失，无需迁移
- 提供数据导入脚本，方便批量导入测试数据
- 文档中说明数据不向后兼容

### 风险 3: 性能不达预期

**风险**: ChromaDB 查询性能可能不如预期。

**解决方案**:
- 进行性能基准测试（Benchmark）
- 调优 ChromaDB 索引参数（HNSW 配置）
- 考虑添加查询结果缓存（Redis）
- 如性能严重不足，评估其他向量数据库（Milvus、Qdrant）

### 风险 4: 分块聚合逻辑复杂

**风险**: 聚合逻辑可能引入新的 bug。

**解决方案**:
- 先实现最简单的 "max" 策略
- 编写详细的单元测试
- 提供配置选项关闭聚合功能
- 逐步迭代优化聚合算法

### 风险 5: 兼容性问题

**风险**: ChromaDB 版本更新可能破坏兼容性。

**解决方案**:
- 锁定 ChromaDB 版本号（如 `chromadb==0.5.20`）
- 订阅 ChromaDB 更新通知
- 在升级前充分测试

---

## 📚 参考资源

### 官方文档
- [ChromaDB 官方文档](https://docs.trychroma.com/)
- [ChromaDB GitHub 仓库](https://github.com/chroma-core/chroma)
- [ChromaDB Python 客户端 API](https://docs.trychroma.com/reference/Client)

### 示例项目
- [ChromaDB Getting Started](https://docs.trychroma.com/getting-started)
- [LangChain + ChromaDB 集成](https://python.langchain.com/docs/integrations/vectorstores/chroma)

### 相关技术
- [HNSW 算法原理](https://arxiv.org/abs/1603.09320)（ChromaDB 使用的索引算法）
- [向量数据库对比](https://zilliz.com/comparison)

---

## 🎯 成功标准

feat/db-crud 分支开发完成需满足以下条件：

1. ✅ **功能完整性**
   - 所有现有功能（向量化、分块、搜索）正常工作
   - 数据持久化到磁盘，服务重启后数据不丢失
   - 支持文档的添加、更新、搜索

2. ✅ **性能要求**
   - 单次文档插入 < 100ms
   - 单次搜索查询（1000 个文档）< 200ms
   - 支持至少 10k+ 文档规模

3. ✅ **代码质量**
   - 单元测试覆盖率 > 80%
   - 所有测试通过
   - 代码符合 PEP 8 规范
   - 完整的中文注释

4. ✅ **文档完善**
   - README.md 更新 ChromaDB 使用说明
   - API 文档更新
   - 提供数据导入脚本示例

5. ✅ **向后兼容**
   - API 接口不变
   - 配置文件向后兼容
   - 测试脚本无需修改

---

## 📞 技术支持

如遇到问题，可参考以下资源：

1. **ChromaDB 官方社区**
   - [Discord](https://discord.gg/MMeYNTmh3x)
   - [GitHub Discussions](https://github.com/chroma-core/chroma/discussions)

2. **项目内部讨论**
   - 创建 GitHub Issue 讨论技术细节
   - 代码审查时提出疑问

3. **AI 辅助开发**
   - 使用 GitHub Copilot 生成代码框架
   - 遇到问题时咨询 AI 助手

---

**文档版本**: v1.0  
**创建日期**: 2025-11-17  
**预计开始日期**: 2025-11-18  
**目标完成日期**: 2025-12-08

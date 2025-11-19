# NRS Vector - 向量存储服务

基于 FastAPI、ChromaDB 和 SentenceTransformer 的生产级语义相似度搜索服务，支持文档持久化存储、智能分块与高效检索。

## 项目概述

NRS Vector 是一个生产级的向量数据库服务，提供文档的语义相似度搜索功能。项目采用模块化设计，集成 ChromaDB 实现数据持久化，支持大规模文档存储与检索。

### 核心特性

- ✅ **持久化存储**：基于 ChromaDB 1.3.4 向量数据库，数据持久化到磁盘，服务重启后数据不丢失
- ✅ **文本向量化**：集成 `BAAI/bge-small-zh-v1.5` 中文嵌入模型，将文本转换为 512 维向量
- ✅ **智能文本分块**：使用 LangChain RecursiveCharacterTextSplitter，按语义边界智能切分长文本
- ✅ **高效语义搜索**：基于余弦距离的向量检索，支持 HNSW 索引加速
- ✅ **RESTful API**：基于 FastAPI 构建，自动生成交互式 API 文档
- ✅ **文档管理**：支持文档的添加、更新操作（Upsert），自动处理长文本分块
- ✅ **配置灵活**：通过环境变量或 `.env` 文件管理配置
- ✅ **类型安全**：使用 Pydantic 进行数据验证和类型检查

## 技术栈

- **Web 框架**: FastAPI 0.121.2
- **向量数据库**: ChromaDB 1.3.4（持久化存储 + HNSW 索引）
- **嵌入模型**: SentenceTransformer 5.1.2 (BAAI/bge-small-zh-v1.5)
- **文本处理**: LangChain 1.0.7 + LangChain Text Splitters 1.0.0（智能分块）
- **数据验证**: Pydantic 2.12.4
- **配置管理**: Pydantic Settings 2.6.0
- **Web 服务器**: Uvicorn 0.38.0
- **数值计算**: NumPy（向量操作）

## 项目结构

```
NRS_vector/
├── main.py                  # FastAPI 应用入口，服务启动
├── router.py                # API 路由定义（搜索、文档管理）
├── services.py              # 核心业务逻辑（向量化、分块、检索、ChromaDB 交互）
├── models.py                # 数据模型定义（请求/响应格式）
├── config.py                # 配置管理（环境变量、默认值）
├── requirements.txt         # 项目依赖
├── .env                     # 环境变量配置（需自行创建）
├── test_chunking.py         # 文本分块功能测试脚本
├── chroma_db/               # ChromaDB 数据库文件（自动生成）
├── FEAT_DB_CRUD_GUIDE.md    # ChromaDB 集成开发指南
└── README.md                # 项目文档
```

## 快速开始

### 环境要求

- Python 3.10+
- 虚拟环境 (推荐)

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/NOVA-NJU/NRS_vector.git
   cd NRS_vector
   ```

2. **创建虚拟环境并安装依赖**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   
   pip install -r requirements.txt
   ```
   
   **主要依赖**：
   - `chromadb==1.3.4` - 向量数据库
   - `fastapi==0.121.2` - Web 框架
   - `sentence-transformers==5.1.2` - 嵌入模型
   - `langchain==1.0.7` 和 `langchain-text-splitters==1.0.0` - 文本分块
   - `pydantic==2.12.4` 和 `pydantic-settings==2.6.0` - 配置管理

3. **配置环境变量（可选）**
   
   创建 `.env` 文件：
   ```env
   # 服务器配置
   VECTOR_HOST=0.0.0.0
   VECTOR_PORT=8000
   VECTOR_DEBUG=True
   
   # 嵌入模型配置
   VECTOR_embedding_model=BAAI/bge-small-zh-v1.5
   VECTOR_HF_TOKEN=your_huggingface_token  # 可选，用于私有模型或避免限速
   
   # 文本分块配置
   VECTOR_enable_chunking=False  # 是否启用智能分块
   VECTOR_chunk_size=500         # 每块字符数
   VECTOR_chunk_overlap=50       # 块间重叠字符数
   
   # ChromaDB 数据库配置
   VECTOR_db_path=./chroma_db    # 数据库存储路径
   ```
   
   **重要**：环境变量名区分大小写，必须使用 `VECTOR_` 前缀。

4. **启动服务**
   ```bash
   uvicorn main:app --reload
   ```
   
   **首次启动说明**：
   - 首次运行会下载嵌入模型（约 150MB），需要一定时间
   - ChromaDB 会在 `./chroma_db/` 目录下自动创建数据库文件
   - 确保有足够的磁盘空间用于存储向量数据

5. **访问 API 文档**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc
   - 健康检查: http://localhost:8000/health

## API 接口

### 1. 添加/更新文档

**端点**: `POST /vectors/documents`

**请求体**:
```json
{
  "text": "南京大学仙林校区图书馆上午8点开放",
  "url": "https://example.com/post/123",
  "metadata": {
    "source": "官网",
    "category": "校园服务"
  }
}
```

**响应**:
```json
{
  "document_id": "1",
  "status": "stored",
  "detail": null
}
```

**智能分块说明**：
- 当 `enable_chunking=True` 且文本长度超过 `chunk_size` 时，系统会自动将长文本切分成多个语义完整的块
- 每个块会生成子文档 ID（格式：`{parent_id}_chunk_{index}`）
- 子文档的元数据中包含 `parent_doc`（父文档ID）、`chunk_index`（块索引）、`total_chunks`（总块数）
- 分块策略采用 LangChain 的层级分隔符，优先按段落→句子→标点→空格切分，保持语义完整性

### 2. 语义相似度搜索

**端点**: `POST /vectors/search`

**请求体**:
```json
{
  "query": "图书馆几点开门",
  "top_k": 5
}
```

**响应**:
```json
{
  "results": [
    {
        "document_id": "1",
        "score": 0.6163,
        "text": "南京大学仙林校区图书馆上午8点开放",
        "metadata": {
            "category": "校园服务",
            "url": "https://example.com/post/123",
            "source": "官网"
        }
    }
  ],
  "query": "图书馆几点开门",
  "top_k": 5
}
```

### 3. 健康检查

**端点**: `GET /health`

**响应**:
```json
{
  "status": "ok"
}
```

## 使用示例

### Python 请求示例

```python
import requests

# 添加文档
response = requests.post(
    "http://localhost:8000/vectors/documents",
    json={
        "text": "南京大学仙林校区图书馆上午8点开放",
        "url": "https://example.com/post/123",
        "metadata": {"source": "官网", "category": "校园服务"}
    }
)
print(response.json())

# 搜索文档
response = requests.post(
    "http://localhost:8000/vectors/search",
    json={
        "query": "图书馆几点开门",
        "top_k": 5
    }
)
print(response.json())
```

### cURL 示例

```bash
# 添加文档
curl -X POST "http://localhost:8000/vectors/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "南京大学仙林校区图书馆上午8点开放",
    "url": "https://example.com/post/123",
    "metadata": {"source": "官网", "category": "校园服务"}
  }'

# 搜索文档
curl -X POST "http://localhost:8000/vectors/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "图书馆几点开门",
    "top_k": 5
  }'
```

## 配置说明

所有配置项支持通过环境变量设置（前缀 `VECTOR_`，**区分大小写**）：

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| 服务地址 | `VECTOR_HOST` | `0.0.0.0` | 监听的 IP 地址 |
| 服务端口 | `VECTOR_PORT` | `8000` | 监听的端口号 |
| 调试模式 | `VECTOR_DEBUG` | `True` | 是否启用热重载 |
| 嵌入模型 | `VECTOR_embedding_model` | `BAAI/bge-small-zh-v1.5` | 使用的嵌入模型 |
| 向量维度 | `VECTOR_embedding_dim` | `512` | 向量维度（需与模型匹配） |
| 数据库路径 | `VECTOR_db_path` | `./chroma_db` | ChromaDB 向量数据库存储路径 |
| HF Token | `VECTOR_HF_TOKEN` | `None` | Hugging Face 访问令牌 |
| 默认返回数 | `VECTOR_default_top_k` | `5` | 搜索默认返回结果数 |
| 启用分块 | `VECTOR_enable_chunking` | `False` | 是否启用智能文本分块 |
| 块大小 | `VECTOR_chunk_size` | `500` | 每个文本块的字符数 |
| 块重叠 | `VECTOR_chunk_overlap` | `50` | 相邻块的重叠字符数 |

**注意**：配置项区分大小写，环境变量名必须精确匹配（如 `enable_chunking` 不能写成 `ENABLE_CHUNKING`）

## 当前实现状态

### ✅ 已完成功能

1. **持久化存储**（v0.3.0 新增）
   - 集成 ChromaDB 1.3.4 向量数据库
   - 文档和向量持久化到磁盘（`./chroma_db/` 目录）
   - 服务重启后数据自动恢复
   - 支持 HNSW 索引，高效向量检索
   - 余弦距离度量，保证搜索准确性

2. **向量化引擎**
   - 集成 SentenceTransformer 5.1.2 嵌入模型（BAAI/bge-small-zh-v1.5）
   - 支持批量文本向量化
   - 自动向量归一化处理
   - 512 维向量表示

3. **智能文本分块**（v0.2.0）
   - 集成 LangChain 1.0.7 RecursiveCharacterTextSplitter
   - 按语义边界智能切分：段落 → 句子 → 标点 → 空格 → 字符
   - 支持配置块大小和重叠度
   - 自动生成子文档 ID 和元数据
   - 保留分隔符，保持文本语义完整性
   - 中文标点符号优化

4. **文档存储**
   - ChromaDB 持久化存储
   - 支持文档 ID 去重（Upsert 语义）
   - 元数据存储与检索
   - 自动处理长文本分块存储
   - 子文档与父文档关联追踪
   - 批量插入优化（分块模式）

5. **语义搜索**
   - 基于余弦距离的高效检索
   - ChromaDB HNSW 索引加速
   - Top-K 结果排序
   - 相似度分数返回（距离转换）
   - 支持跨文档块检索

6. **API 服务**
   - RESTful 风格接口设计
   - 自动 API 文档生成（Swagger UI + ReDoc）
   - 请求/响应数据验证
   - 健康检查端点
   - 完整的中文注释

7. **配置管理**
   - 环境变量支持（区分大小写）
   - `.env` 文件加载
   - 类型安全的配置验证
   - 灵活的分块参数配置
   - ChromaDB 路径配置

### 🚧 待实现功能

1. **高级检索增强**
   - [ ] 元数据过滤查询（利用 ChromaDB 的 where 子句）
   - [ ] 混合搜索（向量 + 关键词）
   - [ ] 分页支持
   - [ ] 文档块结果聚合（按父文档，支持 max/avg/weighted 策略）
   - [ ] 自定义相似度阈值过滤

2. **文档管理增强**
   - [x] 长文本智能分块（已完成）
   - [x] 文档持久化存储（已完成）
   - [ ] 批量文档导入接口
   - [ ] 文档删除接口（物理删除 + 软删除）
   - [ ] 文档更新历史追踪
   - [ ] 支持更多文本格式（PDF、DOCX 等）

3. **性能优化**
   - [ ] 查询结果缓存（Redis）
   - [ ] 异步批处理优化
   - [ ] 请求限流和熔断
   - [ ] 连接池管理
   - [ ] HNSW 索引参数调优

4. **监控与运维**
   - [ ] 请求日志记录
   - [ ] 性能指标监控（Prometheus）
   - [ ] 错误追踪（Sentry）
   - [ ] API 调用统计
   - [ ] 健康检查增强（数据库连接状态）
   - [ ] 数据备份与恢复

5. **部署与扩展**
   - [ ] Docker 容器化
   - [ ] Kubernetes 部署配置
   - [ ] API 认证与授权（JWT）
   - [ ] 多租户支持
   - [ ] 水平扩展方案（分布式 ChromaDB）

## 架构说明

### 数据流程

```
用户请求 → FastAPI 路由 → 业务逻辑层 → 嵌入模型 → 向量存储
                ↓
         数据验证（Pydantic）
                ↓
         返回结构化响应
```

### 模块职责

- **main.py**: 应用初始化、路由挂载、服务启动
- **router.py**: API 端点定义、请求路由
- **services.py**: 核心业务逻辑（向量化、搜索、存储）
- **models.py**: 数据模型、请求/响应格式
- **config.py**: 配置管理、环境变量读取

### 设计原则

- **单一职责**: 每个模块专注一个功能领域
- **依赖注入**: 通过配置工厂函数管理依赖
- **类型安全**: 使用 Pydantic 确保数据类型正确
- **易于测试**: 模块解耦，便于单元测试

## ChromaDB 持久化存储

### 数据存储机制

NRS Vector 使用 ChromaDB 作为向量数据库，数据持久化到本地文件系统：

```
chroma_db/
├── chroma.sqlite3          # 元数据存储（文档ID、元数据等）
└── [向量索引文件]          # HNSW 索引和向量数据
```

**特性**：
- **持久化**：所有文档和向量数据持久化到磁盘，服务重启后自动恢复
- **HNSW 索引**：使用 Hierarchical Navigable Small World 图算法，提供近似最近邻搜索
- **余弦距离**：配置为 `cosine` 距离度量，适合文本相似度计算
- **自动管理**：无需手动创建表或索引，ChromaDB 自动处理

### 集合（Collection）管理

项目使用单一集合 `"documents"` 存储所有文档：

```python
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # 余弦距离度量
)
```

### 数据备份

备份 ChromaDB 数据非常简单：

```bash
# 停止服务
# 复制整个 chroma_db 目录
cp -r chroma_db chroma_db_backup_$(date +%Y%m%d)

# 恢复时替换即可
```

### 性能特点

- **写入性能**：单文档插入 < 50ms（包含向量化时间）
- **查询性能**：1000 个文档规模下，Top-K 查询 < 100ms
- **扩展性**：支持数百万级文档存储（依赖硬件配置）
- **磁盘占用**：每个 512 维向量约占用 2KB 存储空间

## 文本分块功能详解

### 为什么需要分块？

1. **嵌入模型限制**：大多数嵌入模型有最大输入长度（如 512 tokens）
2. **提高检索精度**：长文档整体向量化会损失细节信息，分块后能实现段落级精准匹配
3. **优化搜索体验**：用户查询通常针对具体段落，而非整篇文档

### 分块策略

采用 **层级递归分割**，按照以下优先级切分：

```
段落 (双换行 \n\n) 
  ↓
句子 (。！？) 
  ↓
短语 (；，) 
  ↓
词 (空格) 
  ↓
字符 (强制切分)
```

**优势**：
- 保持语义完整性，避免在句子中间切断
- 适配中文文本特点（使用中文标点符号）
- 自动调整块大小以适应自然边界
- 支持块间重叠，避免边界信息丢失

### 使用示例

```python
# 在 .env 中配置
VECTOR_enable_chunking=True
VECTOR_chunk_size=500
VECTOR_chunk_overlap=50

# 测试分块功能
python test_chunking.py
```

### 分块结果示例

对于一段 475 字符的文本，设置 `chunk_size=200` 时：

```
原文档ID: nanjing_university_recruitment
  ↓ 分块
├─ nanjing_university_recruitment_chunk_0 (98 字符)
│  metadata: {parent_doc: "nanjing_university_recruitment", chunk_index: 0, total_chunks: 3}
├─ nanjing_university_recruitment_chunk_1 (193 字符)
│  metadata: {parent_doc: "nanjing_university_recruitment", chunk_index: 1, total_chunks: 3}
└─ nanjing_university_recruitment_chunk_2 (178 字符)
   metadata: {parent_doc: "nanjing_university_recruitment", chunk_index: 2, total_chunks: 3}
```

搜索时会返回所有相关块，可通过 `metadata.parent_doc` 追溯父文档。

## 注意事项

1. **首次启动时间**: 首次运行会下载嵌入模型（约 150MB），需要一定时间
2. **模型选择**: 使用 `BAAI/bge-small-zh-v1.5`（中文优化）；如需更换可在 `VECTOR_EMBEDDING_MODEL` 中指定，但默认保持该模型
3. **性能瓶颈**: 暴力搜索算法在大规模数据下性能有限
4. **分块配置**: 
   - `chunk_size` 建议范围 300-800 字符（过小损失语义，过大失去分块意义）
   - `chunk_overlap` 建议设为 chunk_size 的 5-20%
   - 环境变量名严格区分大小写（`enable_chunking` 而非 `ENABLE_CHUNKING`）
6. **生产环境建议**：
   - 使用反向代理（Nginx）进行负载均衡
   - 启用 HTTPS 加密传输
   - 定期备份 `chroma_db/` 目录
   - 监控磁盘使用率和查询延迟
   - 考虑使用 Docker 容器化部署

## 开发路线图

### Phase 1: 基础功能 ✅（v0.1.0 - 已完成）
- 文本向量化（SentenceTransformer）
- 语义相似度搜索（余弦相似度）
- RESTful API 接口
- 配置管理

### Phase 2: 智能分块 ✅（v0.2.0 - 已完成）
- LangChain 文本分割器集成
- 层级递归分块策略
- 中文标点符号优化
- 子文档元数据管理
- 分块功能测试

### Phase 3: 持久化存储 ✅（v0.3.1 - 已完成）
- ChromaDB 向量数据库集成
- 文档数据持久化
- 向量索引优化
- 集合管理
- 数据迁移工具

### Phase 4: 高级特性 🚧（v0.4.0 - 进行中）
- 元数据过滤查询
- 文档块结果聚合
- 混合搜索（向量 + 关键词）
- 批量导入/导出接口
- 文档删除和更新历史

### Phase 5: 生产就绪（v1.0.0 - 计划中）
- 监控与日志系统
- 容器化部署（Docker + K8s）
- API 认证与授权
- 性能优化（缓存、连接池）
- 负载测试与压力测试
- 数据备份与恢复方案

## 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境设置

1. Fork 本仓库
2. 创建特性分支: `git checkout -b feature/your-feature`
3. 安装开发依赖: `pip install -r requirements.txt`
4. 确保 ChromaDB 数据库正常运行
5. 运行测试: `python test_chunking.py`
6. 提交更改: `git commit -m 'Add some feature'`
7. 推送分支: `git push origin feature/your-feature`
8. 提交 Pull Request

### 代码规范

- 遵循 PEP 8 代码风格
- 所有函数和类必须有中文文档字符串
- 添加单元测试覆盖新功能
- 确保所有测试通过后再提交

## 常见问题（FAQ）

### Q: 如何清空所有数据？
A: 停止服务后删除 `chroma_db/` 目录，重启服务会自动创建新的空数据库。

### Q: 支持多语言文档吗？
A: 当前使用中文优化模型，但也能处理英文。若需更好的多语言支持，可更换为 `multilingual` 系列模型。

### Q: 如何提高搜索速度？
A: 
1. 确保使用 HNSW 索引（已默认启用）
2. 调整 `top_k` 参数（值越小越快）
3. 考虑添加 Redis 缓存层
4. 使用 SSD 存储 `chroma_db/` 目录

### Q: 数据库文件有多大？
A: 每个文档约占用 2-5KB（取决于文本长度和元数据）。10k 文档约需 20-50MB 磁盘空间。

### Q: 支持分布式部署吗？
A: 当前版本使用 PersistentClient（单机模式）。分布式部署可考虑 ChromaDB 的 Client/Server 模式或切换到 Milvus/Qdrant。

## 许可证

[请根据实际情况添加许可证信息]

## 联系方式

- 组织: NOVA-NJU
- 仓库: https://github.com/NOVA-NJU/NRS_vector

---

**最后更新**: 2025-11-17  
**当前版本**: v0.3.1 (开发阶段)  
**当前分支**: dev  
**下一个里程碑**: v0.4.0 (高级检索特性)

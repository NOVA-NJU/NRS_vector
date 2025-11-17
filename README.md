# NRS Vector - 向量存储服务

基于 FastAPI 和 SentenceTransformer 的语义相似度搜索服务，支持文档向量化存储与检索。

## 项目概述

NRS Vector 是一个轻量级的向量数据库服务，提供文档的语义相似度搜索功能。项目采用模块化设计，易于扩展和维护。

### 核心特性

- ✅ **文本向量化**：基于 `BAAI/bge-small-zh-v1.5` 中文嵌入模型，将文本转换为 512 维向量
- ✅ **语义搜索**：使用余弦相似度进行文档检索，返回最相关的结果
- ✅ **RESTful API**：基于 FastAPI 构建，自动生成交互式 API 文档
- ✅ **文档管理**：支持文档的添加、更新操作（Upsert）
- ✅ **配置灵活**：通过环境变量或 `.env` 文件管理配置
- ✅ **类型安全**：使用 Pydantic 进行数据验证和类型检查

## 技术栈

- **Web 框架**: FastAPI 0.121.2
- **嵌入模型**: SentenceTransformer (BAAI/bge-small-zh-v1.5)
- **数据验证**: Pydantic 2.12.4
- **配置管理**: Pydantic Settings 2.6.0
- **Web 服务器**: Uvicorn 0.38.0

## 项目结构

```
NRS_vector/
├── main.py          # FastAPI 应用入口，服务启动
├── router.py        # API 路由定义（搜索、文档管理）
├── services.py      # 核心业务逻辑（向量化、检索）
├── models.py        # 数据模型定义（请求/响应格式）
├── config.py        # 配置管理（环境变量、默认值）
├── requirements.txt # 项目依赖
├── .env            # 环境变量配置（需自行创建）
└── test.py         # 测试脚本
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

3. **配置环境变量（可选）**
   
   创建 `.env` 文件：
   ```env
   VECTOR_HOST=0.0.0.0
   VECTOR_PORT=8000
   VECTOR_DEBUG=True
   VECTOR_HF_TOKEN=your_huggingface_token  # 可选，用于私有模型或避免限速
   VECTOR_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
   VECTOR_DB_PATH=./chroma_db
   ```

4. **启动服务**
   ```bash
   uvicorn main:app --reload
   ```

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
  "document_id": "doc_001",
  "text": "南京大学仙林校区图书馆上午8点开放",
  "metadata": {
    "source": "官网",
    "category": "校园服务"
  }
}
```

**响应**:
```json
{
  "document_id": "doc_001",
  "status": "stored",
  "detail": null
}
```

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
      "document_id": "doc_001",
      "score": 0.8756,
      "metadata": {
        "source": "官网",
        "category": "校园服务"
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
        "document_id": "doc_001",
        "text": "南京大学仙林校区图书馆上午8点开放",
        "metadata": {"source": "官网"}
    }
)
print(response.json())

# 搜索文档
response = requests.post(
    "http://localhost:8000/vectors/search",
    json={
        "query": "图书馆几点开门",
        "top_k": 3
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
    "document_id": "doc_001",
    "text": "南京大学仙林校区图书馆上午8点开放",
    "metadata": {"source": "官网"}
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

所有配置项支持通过环境变量设置（前缀 `VECTOR_`）：

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| 服务地址 | `VECTOR_HOST` | `0.0.0.0` | 监听的 IP 地址 |
| 服务端口 | `VECTOR_PORT` | `8000` | 监听的端口号 |
| 调试模式 | `VECTOR_DEBUG` | `True` | 是否启用热重载 |
| 嵌入模型 | `VECTOR_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 使用的嵌入模型 |
| 向量维度 | `VECTOR_EMBEDDING_DIM` | `512` | 向量维度（需与模型匹配） |
| 数据库路径 | `VECTOR_DB_PATH` | `./chroma_db` | 向量数据库存储路径（当前未启用） |
| HF Token | `VECTOR_HF_TOKEN` | `None` | Hugging Face 访问令牌 |
| 默认返回数 | `VECTOR_DEFAULT_TOP_K` | `5` | 搜索默认返回结果数 |

## 当前实现状态

### ✅ 已完成功能

1. **向量化引擎**
   - 集成 SentenceTransformer 嵌入模型
   - 支持批量文本向量化
   - 自动向量归一化处理

2. **文档存储**
   - 内存级文档缓存
   - 支持文档 ID 去重（Upsert 语义）
   - 元数据存储与检索

3. **语义搜索**
   - 基于余弦相似度的检索算法
   - Top-K 结果排序
   - 相似度分数返回

4. **API 服务**
   - RESTful 风格接口设计
   - 自动 API 文档生成
   - 请求/响应数据验证
   - 健康检查端点

5. **配置管理**
   - 环境变量支持
   - `.env` 文件加载
   - 类型安全的配置验证

### 🚧 待实现功能

1. **持久化存储**
   - [ ] 集成 ChromaDB 向量数据库
   - [ ] 文档数据持久化
   - [ ] 向量索引优化

2. **高级检索**
   - [ ] 元数据过滤
   - [ ] 混合搜索（向量 + 关键词）
   - [ ] 分页支持

3. **文档处理**
   - [ ] 长文本分块（Chunking）
   - [ ] 批量文档导入
   - [ ] 文档删除接口

4. **性能优化**
   - [ ] 向量缓存策略
   - [ ] 异步批处理
   - [ ] 请求限流

5. **监控与日志**
   - [ ] 请求日志记录
   - [ ] 性能指标监控
   - [ ] 错误追踪

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

## 注意事项

1. **首次启动时间**: 首次运行会下载嵌入模型（约 150MB），需要一定时间
2. **内存使用**: 当前所有文档和向量存储在内存中，重启后数据丢失
3. **模型选择**: `bge-small-zh-v1.5` 针对中文优化，如需处理英文可更换模型
4. **性能瓶颈**: 暴力搜索算法在大规模数据下性能有限，建议集成向量数据库

## 开发路线图

### Phase 1: 基础功能 ✅（已完成）
- 文本向量化
- 语义相似度搜索
- RESTful API 接口

### Phase 2: 持久化与优化 🚧（进行中）
- ChromaDB 集成
- 向量索引
- 性能优化

### Phase 3: 高级特性（计划中）
- 长文本分块
- 混合搜索
- 批量导入/导出

### Phase 4: 生产就绪（计划中）
- 监控与日志
- 容器化部署
- API 认证

## 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境设置

1. Fork 本仓库
2. 创建特性分支: `git checkout -b feature/your-feature`
3. 提交更改: `git commit -m 'Add some feature'`
4. 推送分支: `git push origin feature/your-feature`
5. 提交 Pull Request

## 许可证

[请根据实际情况添加许可证信息]

## 联系方式

- 组织: NOVA-NJU
- 仓库: https://github.com/NOVA-NJU/NRS_vector

---

**最后更新**: 2025-11-17  
**版本**: v0.1.0 (开发阶段)
"""提供独立向量存储服务的 FastAPI 应用。

这是整个向量服务的入口文件，负责:
1. 创建 FastAPI 应用实例
2. 挂载路由（API 接口）
3. 提供健康检查接口
4. 启动 Web 服务器
"""
from fastapi import FastAPI
from config import settings
import uvicorn
from router import router as vector_router

# 创建 FastAPI 应用实例
# title: 应用名称，会显示在自动生成的 API 文档中
# version: 版本号
app = FastAPI(title="Vector Store Service", version="0.1.0")

# 挂载向量相关的路由
# prefix="/vectors": 所有路由都会以 /vectors 开头，例如 /vectors/search
# tags=["vectors"]: 在 API 文档中给这些接口打上 "vectors" 标签，方便分类
app.include_router(vector_router, prefix="/vectors", tags=["vectors"])


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """健康检查接口，用于监控服务是否正常运行。
    
    这个接口通常被:
    - Docker 容器编排工具（如 Kubernetes）用来检查服务是否存活
    - 负载均衡器用来判断是否应该向这个服务发送流量
    - 监控系统用来确认服务状态
    
    返回:
        简单的状态字典，表示服务正常运行
    """
    return {"status": "ok"}

# 这个 if 语句确保只有直接运行这个文件时才会启动服务器
# 如果是被其他模块导入，则不会自动启动
if __name__ == "__main__":
    # 使用 uvicorn 启动 FastAPI 应用
    # "main:app": 表示从 main 模块导入 app 对象
    # host: 监听的 IP 地址，从配置文件读取
    # port: 监听的端口号，从配置文件读取
    # reload: 是否启用热重载（代码改动后自动重启），从配置文件读取
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
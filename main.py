"""提供独立向量存储服务的 FastAPI 应用。"""
from fastapi import FastAPI
from config import settings
import uvicorn
from router import router as vector_router

app = FastAPI(title="Vector Store Service", version="0.1.0")
app.include_router(vector_router, prefix="/vectors", tags=["vectors"])


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """供编排器探测服务状态的健康检查接口。"""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
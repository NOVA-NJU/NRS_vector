"""向量存储模块的配置辅助方法。"""
from pydantic_settings import BaseSettings


class VectorSettings(BaseSettings):
    """保存向量数据库与嵌入模型的可调参数。"""

    db_path: str = "data/vector_store"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    default_top_k: int = 5

    class Config:
        env_prefix = "VECTOR_"


def get_settings() -> VectorSettings:
    """返回缓存的配置实例，便于依赖注入。"""
    return VectorSettings()  # BaseSettings 会缓存环境变量读取

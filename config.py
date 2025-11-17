"""向量存储模块的配置辅助方法。"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os


class VectorSettings(BaseSettings):
    """保存向量数据库与嵌入模型的可调参数。"""

    db_path: str = "./chroma_db"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    default_top_k: int = 5

    # 【新增字段】
    # 将其定义为 Optional[str] 或 str | None，表示它可以是字符串或 None（如果没有设置）。
    HF_TOKEN: Optional[str] = None

    # 2. 核心调整：使用 model_config 替换 Config
    model_config = SettingsConfigDict(
        # env_prefix: 查找所有以 VECTOR_ 开头的环境变量
        env_prefix="VECTOR_", 
        
        # env_file: 指定加载 .env 文件
        env_file=".env",
        
        # case_sensitive: 默认为 False。将其设置为 True 确保变量名大小写匹配。
        case_sensitive=True, 
    )


def get_settings() -> VectorSettings:
    """返回缓存的配置实例，便于依赖注入。"""
    return VectorSettings()  # BaseSettings 会缓存环境变量读取

settings = get_settings()

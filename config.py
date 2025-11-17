"""向量存储模块的配置管理。

这个文件负责管理整个应用的配置参数。
配置可以来自:
1. 环境变量（优先级最高）
2. .env 文件
3. 代码中定义的默认值（优先级最低）

为什么要用配置文件?
- 不同环境（开发、测试、生产）可能需要不同的配置
- 敏感信息（如 Token）不应该硬编码在代码中
- 修改配置不需要改代码，更灵活
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os


class VectorSettings(BaseSettings):
    """向量服务配置类，定义所有可配置的参数。
    
    使用 Pydantic BaseSettings 的好处:
    1. 自动从环境变量读取配置
    2. 自动进行类型转换和验证
    3. 支持默认值
    4. 配置集中管理，便于维护
    """

    # ========== 数据库相关配置 ==========
    
    # db_path: 向量数据库的存储路径
    # 默认在当前目录的 chroma_db 文件夹下
    db_path: str = "./chroma_db"
    
    # ========== 嵌入模型相关配置 ==========
    
    # embedding_model: 使用的嵌入模型名称
    # 默认使用 BAAI 的中文小型模型，速度快、效果好
    # 可以改成其他模型，比如 "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    
    # embedding_dim: 嵌入向量的维度
    # bge-small-zh-v1.5 模型输出 512 维向量
    # 不同模型的维度可能不同，需要对应修改
    embedding_dim: int = 512
    
    # HF_TOKEN: Hugging Face 访问令牌（Token）
    # Optional[str] 表示可以是字符串或 None（不提供也可以）
    # 什么时候需要 Token?
    # - 访问私有模型时
    # - 避免下载模型时的速率限制
    # 如何设置? 在 .env 文件中添加: VECTOR_HF_TOKEN=your_token_here
    HF_TOKEN: Optional[str] = None
    
    # ========== 搜索相关配置 ==========
    
    # default_top_k: 默认返回的搜索结果数量
    # 如果用户搜索时没有指定 top_k，就用这个默认值
    default_top_k: int = 5
    
    # ========== 文本分块相关配置 ==========
    
    # enable_chunking: 是否启用文本分块功能
    # True: 长文本会被自动切分成多个小块分别存储
    # False: 文本不分块，整体向量化（当前默认行为）
    # 什么时候需要分块?
    # - 处理长文档（如论文、书籍章节）时
    # - 想要更精准的段落级匹配时
    # - 文本超过模型最大输入长度时
    enable_chunking: bool = False
    
    # chunk_size: 每个文本块的字符数
    # 默认 500 字符，这个长度适合大多数场景
    # 太小: 语义信息不完整，搜索效果差
    # 太大: 失去分块的意义，匹配不够精准
    # 建议范围: 300-800 字符
    chunk_size: int = 500
    
    # chunk_overlap: 相邻文本块之间的重叠字符数
    # 默认 50 字符，大约 10% 的重叠
    # 为什么需要重叠?
    # - 避免重要信息被切分到两个块的边界处
    # - 确保跨块的语义完整性
    # - 提高搜索召回率
    # 建议范围: chunk_size 的 5-20%
    chunk_overlap: int = 50
    
    # ========== 服务器相关配置 ==========
    
    # HOST: 服务监听的 IP 地址
    # "0.0.0.0" 表示监听所有网络接口，允许外部访问
    # "127.0.0.1" 表示只监听本地，只能从本机访问
    HOST: str = "0.0.0.0"
    
    # PORT: 服务监听的端口号
    # 默认 8000，可以改成其他未被占用的端口
    PORT: int = 8000
    
    # DEBUG: 是否启用调试模式
    # True: 启用热重载（代码改动后自动重启）和详细错误信息
    # False: 生产模式，性能更好但错误信息较少
    DEBUG: bool = True

    # ========== Pydantic 配置 ==========
    
    # model_config: Pydantic v2 的配置方式
    # 这个配置字典告诉 Pydantic 如何读取环境变量
    model_config = SettingsConfigDict(
        # env_prefix: 环境变量的前缀
        # 例如: 要设置 db_path，环境变量名应该是 VECTOR_db_path 或 VECTOR_DB_PATH
        # 这样可以避免不同服务的环境变量冲突
        env_prefix="VECTOR_", 
        
        # env_file: 指定 .env 文件的路径
        # 应用启动时会自动从这个文件读取配置
        # .env 文件格式: KEY=VALUE，每行一个配置
        env_file=".env",
        
        # case_sensitive: 环境变量名是否区分大小写
        # True: VECTOR_DB_PATH 和 VECTOR_db_path 是不同的变量
        # False: 大小写不敏感，都会被识别
        case_sensitive=True, 
    )


def get_settings() -> VectorSettings:
    """获取配置实例的工厂函数。
    
    返回:
        VectorSettings 配置对象
    
    为什么要用函数而不是直接创建对象?
    1. 便于依赖注入（Dependency Injection）
    2. 方便测试时替换配置
    3. 未来可以在这里添加缓存逻辑
    
    注意:
    BaseSettings 内部已经实现了缓存机制，多次调用不会重复读取环境变量。
    第一次调用时读取并缓存，后续调用直接返回缓存的值。
    """
    return VectorSettings()

# 创建一个全局配置实例，供其他模块导入使用
# 例如: from config import settings
settings = get_settings()

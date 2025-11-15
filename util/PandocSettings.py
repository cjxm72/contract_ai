from dotenv import load_dotenv
from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings
from typing import Optional

# load_dotenv()

class OpenAISettings(BaseSettings):
    """OpenAI兼容接口配置（含SiliconFlow等第三方服务）"""
    # API密钥（必填，从.env加载）
    api_key: str = Field(
        alias="OPENAI_API_KEY",  # 对应.env中的键名
        description="OpenAI兼容接口的API密钥"
    )

    # 接口基础URL（必填，需是合法URL格式）
    base_url: str = Field(
        alias="OPENAI_BASE_URL",
        description="OpenAI兼容接口的基础URL"
    )

    # 模型名称（支持动态切换）
    model: str = Field(
        alias="OPENAI_MODEL",
        description="使用的模型名称（需与接口支持的模型匹配）"
    )

    # 可选参数：超时时间（秒）
    timeout: int = Field(
        180,
        description="API请求超时时间（秒）"
    )

    # 可选参数：最大重试次数
    max_retries: int = Field(
        2,
        description="API请求失败后的最大重试次数"
    )

    use_local_model : bool = Field(
        True,
        description="是否使用本地模型"
    )
    timeout_seconds:int = Field(
        600,
        description="本地超时时间，如果超时则会使用在线模型，s为单位"
    )
    max_tokens:int = Field(
        4096,
        description="生成最大长度tokens"
    )
    class Config:
        # 指定.env文件路径（默认项目根目录）
        env_file = "../.env"
        env_file_encoding = "utf-8"
        # 允许未定义的环境变量（不强制校验）
        extra = "ignore"


# 实例化配置（自动加载.env中的OPENAI_*变量）
openai_settings = OpenAISettings()

# 示例：在代码中使用配置
if __name__ == "__main__":
    print(f"API密钥: {openai_settings.api_key[:4]}***")  # 隐藏部分密钥
    print(f"接口地址: {openai_settings.base_url}")
    print(f"使用模型: {openai_settings.model}")
    print(f"超时设置: {openai_settings.timeout}秒")

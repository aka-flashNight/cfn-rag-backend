from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    全局配置对象，支持从环境变量 / .env 读取。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "FastAPI AI Scaffold"
    debug: bool = False

    # 通用 OpenAI 兼容 LLM 配置（默认为 Gemini OpenAI 兼容端点）
    llm_api_key: str | None = None
    llm_api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model_name: str = "gemini-2.5-flash"

    # Agent / LangGraph：调试时打印完整 prompt 与模型输出（与 nodes 中 CFN_AGENT_DEBUG_LLM 一致）
    cfn_agent_debug_llm: bool = Field(
        default=False,
        validation_alias="CFN_AGENT_DEBUG_LLM",
    )


@lru_cache
def get_settings() -> Settings:
    """
    获取全局 Settings 单例，供 FastAPI Depends 注入使用。
    """

    return Settings()

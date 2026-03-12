from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    """
    获取全局 Settings 单例，供 FastAPI Depends 注入使用。
    """

    return Settings()

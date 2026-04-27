from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# 部署 Profile 类型常量（供其他模块 import）
# ---------------------------------------------------------------------------

DeploymentProfile = Literal["local", "server"]
CacheBackendType = Literal["memory", "redis"]
DbBackendType = Literal["sqlite", "postgres"]
VectorBackendType = Literal["llamaindex_local", "qdrant"]
CheckpointBackendType = Literal["sqlite", "postgres", "redis"]

# 各字段的 local / server 默认值对照表
_PROFILE_DEFAULTS: dict[str, dict[str, str]] = {
    "cache_backend":       {"local": "memory",             "server": "redis"},
    "db_backend":          {"local": "sqlite",             "server": "postgres"},
    "vector_backend":      {"local": "llamaindex_local",   "server": "qdrant"},
    "checkpoint_backend":  {"local": "sqlite",             "server": "postgres"},
}


class Settings(BaseSettings):
    """
    全局配置对象，支持从环境变量 / .env 读取。

    路线四新增双 Profile 支持：
    - CFN_PROFILE=local   → 全部本地资源（默认，零依赖）
    - CFN_PROFILE=server  → Redis + Qdrant + Postgres + 后台 Worker

    后端选择（cache_backend / db_backend / vector_backend / checkpoint_backend）
    可由 env var 显式覆盖；未设置时按 profile 自动推导。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # 基础字段
    # ------------------------------------------------------------------

    app_name: str = "FastAPI AI Scaffold"
    debug: bool = False

    # 通用 OpenAI 兼容 LLM 配置
    llm_api_key: str | None = None
    llm_api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model_name: str = "gemini-2.5-flash"

    # Agent / LangGraph 调试开关
    cfn_agent_debug_llm: bool = Field(
        default=False,
        validation_alias="CFN_AGENT_DEBUG_LLM",
    )
    cfn_agent_debug_latency: bool = Field(
        default=False,
        validation_alias="CFN_AGENT_DEBUG_LATENCY",
    )

    # ------------------------------------------------------------------
    # 路线四：部署 Profile
    # ------------------------------------------------------------------

    deployment_profile: DeploymentProfile = Field(
        default="local",
        validation_alias="CFN_PROFILE",
    )

    # ---- 存储后端（None 表示"按 profile 自动推导"）----

    cache_backend: CacheBackendType | None = Field(
        default=None,
        validation_alias="CFN_CACHE_BACKEND",
    )
    db_backend: DbBackendType | None = Field(
        default=None,
        validation_alias="CFN_DB_BACKEND",
    )
    vector_backend: VectorBackendType | None = Field(
        default=None,
        validation_alias="CFN_VECTOR_BACKEND",
    )
    checkpoint_backend: CheckpointBackendType | None = Field(
        default=None,
        validation_alias="CFN_CHECKPOINT_BACKEND",
    )

    # ---- Server Profile 外部服务连接 URL ----

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias="CFN_REDIS_URL",
    )
    postgres_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/cfn_rag",
        validation_alias="CFN_POSTGRES_URL",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        validation_alias="CFN_QDRANT_URL",
    )
    qdrant_collection: str = Field(
        default="cfn_game",
        validation_alias="CFN_QDRANT_COLLECTION",
    )
    worker_broker_url: str = Field(
        default="redis://localhost:6379/1",
        validation_alias="CFN_WORKER_BROKER_URL",
    )
    checkpoint_db_path: str = Field(
        default="",
        validation_alias="CFN_LANGGRAPH_CHECKPOINT_DB",
    )

    # ------------------------------------------------------------------
    # 便捷方法：获取"有效"后端选择（None → 按 profile 推导）
    # ------------------------------------------------------------------

    def effective(self, field_name: str) -> str:
        """返回某个后端字段的有效值：若已显式设置则用设置值，否则按 profile 推导。"""
        raw = getattr(self, field_name, None)
        if raw is not None:
            return raw
        profile = self.deployment_profile
        mapping = _PROFILE_DEFAULTS.get(field_name, {})
        return mapping.get(profile, mapping.get("local", "sqlite"))

    # ------------------------------------------------------------------
    # 便捷派生属性
    # ------------------------------------------------------------------

    @property
    def is_server_profile(self) -> bool:
        return self.deployment_profile == "server"

    @property
    def is_local_profile(self) -> bool:
        return self.deployment_profile == "local"

    @property
    def use_redis(self) -> bool:
        return self.effective("cache_backend") == "redis"

    @property
    def use_postgres(self) -> bool:
        return self.effective("db_backend") == "postgres"

    @property
    def use_qdrant(self) -> bool:
        return self.effective("vector_backend") == "qdrant"


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------

@lru_cache
def get_settings() -> Settings:
    """获取全局 Settings 单例，供 FastAPI Depends 注入使用。"""
    return Settings()

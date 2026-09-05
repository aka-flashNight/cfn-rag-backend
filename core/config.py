"""全局配置（仅 local 本地打包路线）。

对应 docs/v3-developer/01-总体架构与核心决策.md §8。Server Profile 的全部
配置项（Redis/Qdrant/Postgres/arq/后端选择）随该路线整体下线。
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置对象，支持从环境变量 / .env 读取。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LLM（前端可按请求覆盖 api_key/base/model；不再按消息存库）
    # ------------------------------------------------------------------

    llm_api_key: str = ""
    llm_api_base: str = "https://api.deepseek.com/v1"
    llm_model_name: str = "deepseek-v4-flash-vision-exp"
    llm_proxy_url: str = ""  # 仅作用于 LLM 客户端（httpx 客户端级），不改进程环境变量（修 E3）

    # ------------------------------------------------------------------
    # 资源路径
    # ------------------------------------------------------------------

    cfn_resources_dir: str = ""  # resources 目录（自动探测兜底，见 services/game_data/paths.py）
    cfn_game_project_dir: str = ""  # 游戏项目根（立绘 manifest 所在，见 07）

    # ------------------------------------------------------------------
    # 行为
    # ------------------------------------------------------------------

    draft_keep_turns: int = 3  # 岔开话题草案保留回合数（连续 N 次 ask 未触碰任务工具即删除）
    subagent_task_max_rounds: int = 4
    subagent_search_max_rounds: int = 3
    merge_grace_ms: int = 1500  # 正文流完后等子 Agent 的宽限
    subagent_timeout_s: int = 120

    # ------------------------------------------------------------------
    # 调试开关（沿用环境变量别名；LatencyTracker 依赖 latency 开关）
    # ------------------------------------------------------------------

    cfn_agent_debug_llm: bool = False
    cfn_agent_debug_latency: bool = False


@lru_cache
def get_settings() -> Settings:
    """获取全局 Settings 单例，供 FastAPI Depends 注入使用。"""
    return Settings()

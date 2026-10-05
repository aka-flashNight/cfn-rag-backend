"""
MCP Server 应用上下文：延迟加载 GameDataRegistry。

MCP Server 作为独立进程启动，共享 services/ 模块但不需要 FastAPI。
上下文负责加载游戏数据并暴露给 tool/resource 层使用。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from services.game_data.paths import find_resources_directory
from services.game_data.registry import GameDataRegistry

logger = logging.getLogger(__name__)


class AppContext:
    """MCP Server 应用上下文：持有 GameDataRegistry 单例。"""

    def __init__(self) -> None:
        self._registry: Optional[GameDataRegistry] = None
        self._resources_dir: Optional[Path] = None
        self._ready = False

    @property
    def registry(self) -> GameDataRegistry:
        if self._registry is None:
            raise RuntimeError("GameDataRegistry 尚未初始化，请先调用 init()")
        return self._registry

    @property
    def resources_dir(self) -> Path:
        if self._resources_dir is None:
            raise RuntimeError("资源目录尚未定位，请先调用 init()")
        return self._resources_dir

    @property
    def ready(self) -> bool:
        return self._ready

    def init(self) -> None:
        """
        初始化游戏数据上下文：
        1. 定位资源根目录（resources/ 或 CrazyFlashNight/）
        2. 创建并加载 GameDataRegistry
        """
        try:
            self._resources_dir = find_resources_directory()
            logger.info("游戏资源根目录: %s", self._resources_dir)
        except FileNotFoundError as e:
            logger.warning("未找到游戏资源目录: %s，MCP Server 将以降级模式运行", e)
            self._ready = False
            return

        data_root = self._resources_dir / "data"
        if not data_root.is_dir():
            logger.warning("游戏 data 目录不存在: %s，MCP Server 将以降级模式运行", data_root)
            self._ready = False
            return

        self._registry = GameDataRegistry.create(data_root=data_root)
        try:
            self._registry.load_all()
            logger.info(
                "GameDataRegistry 加载完成，物品: %d 种，关卡: %d 个，任务: %d 个",
                len(self._registry.items.items),
                len(self._registry.stages._stage_infos),
                len(self._registry.tasks.list_all_tasks()),
            )
            self._ready = True
        except Exception as e:
            logger.exception("GameDataRegistry 加载失败: %s", e)
            self._ready = False


# 模块级单例
_ctx: Optional[AppContext] = None


def get_app_context() -> AppContext:
    global _ctx
    if _ctx is None:
        _ctx = AppContext()
        _ctx.init()
    return _ctx


def reset_app_context() -> None:
    """重置上下文（用于测试）。"""
    global _ctx
    _ctx = None

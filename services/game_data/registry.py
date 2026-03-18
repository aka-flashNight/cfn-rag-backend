from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .crafting_registry import CraftingRegistry
from .item_registry import ItemRegistry
from .kshop_registry import KShopRegistry
from .paths import get_game_data_root
from .shop_registry import ShopRegistry
from .stage_registry import StageRegistry
from .task_registry import TaskRegistry
from .task_text_registry import TaskTextRegistry


@dataclass
class GameDataRegistry:
    """
    游戏数据统一注册中心。
    启动时初始化加载（建议后台线程），提供只读查询 API。
    """

    data_root: Path
    items: ItemRegistry
    tasks: TaskRegistry
    task_texts: TaskTextRegistry
    stages: StageRegistry
    shops: ShopRegistry
    kshop: KShopRegistry
    crafting: CraftingRegistry

    @classmethod
    def create(cls, *, data_root: Optional[Path] = None) -> "GameDataRegistry":
        root = Path(data_root).resolve() if data_root is not None else get_game_data_root()
        return cls(
            data_root=root,
            items=ItemRegistry(data_root=root),
            tasks=TaskRegistry(data_root=root),
            task_texts=TaskTextRegistry(data_root=root),
            stages=StageRegistry(data_root=root),
            shops=ShopRegistry(data_root=root),
            kshop=KShopRegistry(data_root=root),
            crafting=CraftingRegistry(data_root=root),
        )

    def load_all(self) -> None:
        # 顺序：items 先加载便于后续校验/筛选
        self.items.load()
        self.tasks.load()
        self.task_texts.load()
        self.stages.load()
        self.shops.load()
        self.kshop.load()
        self.crafting.load()


# 全局单例（启动时初始化）
GAME_DATA: GameDataRegistry | None = None


def init_game_data_registry(*, data_root: Optional[Path] = None) -> GameDataRegistry:
    global GAME_DATA
    reg = GameDataRegistry.create(data_root=data_root)
    reg.load_all()
    GAME_DATA = reg
    return reg


def get_game_data_registry() -> GameDataRegistry:
    if GAME_DATA is None:
        # 允许在非启动路径下懒加载（便于脚本/测试），但生产推荐在 startup 中初始化
        return init_game_data_registry()
    return GAME_DATA


"""npc 测试共享夹具：真实游戏数据注册表（一次性加载）。"""

from __future__ import annotations

import pytest

from services.game_data.registry import (
    GameDataRegistry,
    get_game_data_registry,
    init_game_data_registry,
)


@pytest.fixture(scope="session")
def game_data() -> GameDataRegistry:
    init_game_data_registry()
    return get_game_data_registry()

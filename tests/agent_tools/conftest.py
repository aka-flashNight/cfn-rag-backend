"""agent_tools 测试共享夹具：真实游戏数据注册表（一次性加载）。"""

from __future__ import annotations

import pytest

from services.agent_tools.handlers import build_validation_ctx
from services.agent_tools.validator import DraftValidationContext
from services.game_data.registry import (
    GameDataRegistry,
    get_game_data_registry,
    init_game_data_registry,
)

NPC_NAME = "铁匠"


@pytest.fixture(scope="session")
def game_data() -> GameDataRegistry:
    init_game_data_registry()
    return get_game_data_registry()


@pytest.fixture
def vctx(game_data) -> DraftValidationContext:
    """阶段 1、好感 0 的校验上下文（V7 区间 [9000, 27000]）。"""
    return build_validation_ctx(npc_name=NPC_NAME, player_progress=1, npc_affinity=0)

"""services/npc：NPC 状态管理（单例 + 锁 + 防抖落盘）。"""

from services.npc.manager import (
    NPCManager,
    NPCState,
    get_npc_manager,
    get_npc_state_path,
    set_npc_manager,
)

__all__ = [
    "NPCManager",
    "NPCState",
    "get_npc_manager",
    "get_npc_state_path",
    "set_npc_manager",
]

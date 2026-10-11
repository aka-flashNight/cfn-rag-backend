"""游戏内嵌集成协议 schema（`/api/integration/*`，见开发计划 §附 B）。

槽位投影（内存态）由编排/任务/记忆/提示词层消费；
character/progress/scene 字段已按最终形状冻结，player 为开放段（深层形状归后端消费方按需取用）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from services.integration.state import SLOT_KEY_PATTERN


class BindCharacter(BaseModel):
    """角色信息投影（游戏存档 → exe；至少 name，允许游戏侧扩展展示性字段）。"""

    model_config = ConfigDict(extra="allow")

    name: str


class BindProgress(BaseModel):
    """进度投影（只读）：供对话联动判定任务完成态（FR-8）。"""

    model_config = ConfigDict(extra="allow")

    tasks_finished: list[int] = Field(default_factory=list)
    task_chains_progress: dict[str, Any] = Field(default_factory=dict)


class BindScene(BaseModel):
    """场景投影（只读）：玩家当前所处地图/关卡。

    非战斗时 stageName 是上一次战斗关卡的残留值，消费方须用 inCombat 门控；
    stageFlag 为帧标签（与 map_definition locations[].sceneName 对齐）。
    """

    model_config = ConfigDict(extra="allow")

    stageFlag: str = ""
    stageName: str = ""
    entrance: str = ""
    inCombat: bool = False


class BindRequest(BaseModel):
    slot_key: str = Field(..., pattern=SLOT_KEY_PATTERN, description="当前存档槽位标识")
    character: BindCharacter | None = Field(default=None, description="角色信息投影")
    progress: BindProgress | None = Field(default=None, description="进度投影")
    scene: BindScene | None = Field(default=None, description="场景投影")
    player: dict[str, Any] | None = Field(
        default=None, description="玩家段投影（开放段：等级/称号/技能/装备/基建等，按需取用）"
    )


class BindResponse(BaseModel):
    ok: bool = True
    mode: str
    frontend_url: str


class UnbindResponse(BaseModel):
    ok: bool = True
    mode: str


class ShutdownResponse(BaseModel):
    ok: bool = True
    status: str = "shutting_down"


class IntegrationStatusResponse(BaseModel):
    ok: bool = True
    mode: str
    launch_mode: str
    bound_slot: str | None = None
    frontend_url: str
    character: dict | None = None
    progress: dict | None = None
    scene: dict | None = None
    player: dict | None = None
    stage: int | None = Field(
        default=None, description="由绑定存档投影推导的主线阶段（1-7）；不可判定＝null"
    )
    stage_name: str | None = Field(default=None, description="阶段名（废城/堕落城/…）；同上")

"""游戏内嵌集成协议 schema（`/api/integration/*`，见开发计划 §附 B）。

阶段一只记录槽位投影（内存态）；character/progress 字段已按最终形状冻结，
阶段二由编排/任务/记忆层消费。
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


class BindRequest(BaseModel):
    slot_key: str = Field(..., pattern=SLOT_KEY_PATTERN, description="当前存档槽位标识")
    character: BindCharacter | None = Field(default=None, description="角色信息投影")
    progress: BindProgress | None = Field(default=None, description="进度投影")


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

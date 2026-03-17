from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union


@dataclass(frozen=True)
class StageConfig:
    """
    主线阶段配置。

    预留字段方便后续补充：例如推荐等级区间、难度系数等。
    """

    name: str
    min_level: Optional[int] = None
    max_level: Optional[int] = None


# 主线阶段基础配置：1-6 阶段名称
PROGRESS_STAGE_CONFIG: Dict[int, StageConfig] = {
    1: StageConfig(name="废城"),
    2: StageConfig(name="堕落城"),
    3: StageConfig(name="荒漠军阀"),
    4: StageConfig(name="黑铁会总堂"),
    5: StageConfig(name="诺亚"),
    6: StageConfig(name="雪山"),
}


def get_progress_stage_config(stage: Union[int, None]) -> Optional[StageConfig]:
    """
    根据阶段数值获取阶段配置。

    预留统一入口，后续可以在这里集中处理边界值或别名。
    """
    if stage is None:
        return None
    if not isinstance(stage, int):
        return None
    return PROGRESS_STAGE_CONFIG.get(stage)


def get_progress_stage_name(stage: Union[int, None]) -> Optional[str]:
    """
    获取主线阶段名称（例如：废城 / 堕落城 / 荒漠军阀 ...）。
    """
    cfg = get_progress_stage_config(stage)
    return cfg.name if cfg is not None else None



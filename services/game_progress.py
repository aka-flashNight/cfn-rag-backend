from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union


@dataclass(frozen=True)
class StageConfig:
    """
    主线阶段配置。

    - name: 阶段名称（废城 / 堕落城 / 荒漠军阀 / 黑铁会总堂 / 诺亚 / 雪山）
    - min_level / max_level: 推荐等级区间
    - stage_name: 对应的主要地图区域名（stages 子目录）
    - main_task_min_id / main_task_max_id: 对应主线任务 id 范围（含端点）
    """

    name: str
    min_level: Optional[int] = None
    max_level: Optional[int] = None
    stage_name: Optional[str] = None
    main_task_min_id: Optional[int] = None
    main_task_max_id: Optional[int] = None


# 主线阶段基础配置：1-6 阶段
# 注：后续如果主线扩展，可只调整对应阶段的 id 范围上限。
PROGRESS_STAGE_CONFIG: Dict[int, StageConfig] = {
    # 1：废城，1-11级，stages 名：基地门口，主线 id 范围 0-21
    1: StageConfig(
        name="废城",
        min_level=1,
        max_level=11,
        stage_name="基地门口",
        main_task_min_id=0,
        main_task_max_id=21,
    ),
    # 2：堕落城，12-15级，stages 名：基地车库，主线 id 范围 22-28
    2: StageConfig(
        name="堕落城",
        min_level=12,
        max_level=15,
        stage_name="基地车库",
        main_task_min_id=22,
        main_task_max_id=28,
    ),
    # 3：荒漠军阀，16-19级，stages 名：基地房顶，主线 id 范围 29-36
    3: StageConfig(
        name="荒漠军阀",
        min_level=16,
        max_level=19,
        stage_name="基地房顶",
        main_task_min_id=29,
        main_task_max_id=36,
    ),
    # 4：黑铁会总堂，20-28级，stages 名：黑铁会总部，主线 id 范围 37-74
    4: StageConfig(
        name="黑铁会总堂",
        min_level=20,
        max_level=28,
        stage_name="黑铁会总部",
        main_task_min_id=37,
        main_task_max_id=74,
    ),
    # 5：诺亚，29-40级，stages 名：诺亚前线基地深处，主线 id 范围 75-77（暂定）
    5: StageConfig(
        name="诺亚",
        min_level=29,
        max_level=40,
        stage_name="诺亚前线基地深处",
        main_task_min_id=75,
        main_task_max_id=77,
    ),
    # 6：雪山，40-50级，stages 名：雪山，主线 id 范围 77-77（暂定）
    6: StageConfig(
        name="雪山",
        min_level=40,
        max_level=50,
        stage_name="雪山",
        main_task_min_id=77,
        main_task_max_id=77,
    ),
}


# 有效的关卡大区（stages 根目录下的有效子目录）
VALID_STAGE_ROOTS = {
    "基地门口",
    "基地车库",
    "基地房顶",
    "地下2层",
    "副本任务",
    "试炼场深处",
    "黑铁会总部",
    "诺亚前线基地深处",
    "雪山",
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


def get_progress_stage_level_range(stage: Union[int, None]) -> Optional[tuple[int, int]]:
    """
    获取阶段对应的推荐等级区间 (min_level, max_level)。
    """
    cfg = get_progress_stage_config(stage)
    if cfg is None or cfg.min_level is None or cfg.max_level is None:
        return None
    return cfg.min_level, cfg.max_level


def get_progress_stage_main_task_range(stage: Union[int, None]) -> Optional[tuple[int, int]]:
    """
    获取阶段对应的主线任务 id 范围 (min_id, max_id)。

    该范围可用于：
    - 在生成 agent 任务时选择“贴近玩家进度”的前置主线；
    - 过滤高于玩家进度的主线 id，避免越级任务。
    """
    cfg = get_progress_stage_config(stage)
    if cfg is None or cfg.main_task_min_id is None or cfg.main_task_max_id is None:
        return None
    return cfg.main_task_min_id, cfg.main_task_max_id


def is_valid_stage_root(stage_root: str) -> bool:
    """
    判断给定的 stages 子目录名是否为“有效关卡大区”。

    只有以下几类被视为可参与 agent 任务生成与筛选：
    - 六个主线阶段对应的大区：基地门口 / 基地车库 / 基地房顶 / 黑铁会总部 / 诺亚前线基地深处 / 雪山
    - 以及会穿插多个进度或仅支线的：地下2层 / 副本任务 / 试炼场深处
    其他 stages 子目录一律视为“无效”。
    """
    return stage_root in VALID_STAGE_ROOTS


# 少数关卡大区目录名较抽象，需附带叙事地图说明；其余目录名对 LLM 已足够清晰，直接沿用原名。
_STAGE_ROOT_REGION_HINT_EXTRA: Dict[str, str] = {
    "基地门口": "基地门口，主要区域为废城，主要敌人为僵尸",
    "基地车库": "基地车库，主要区域为堕落城，主要敌人为盗贼",
    "基地房顶": "基地房顶，主要区域为荒漠，主要敌人为军阀",
    "地下2层": "地下2层，主要区域为禁区(天网机器人)与诺亚外围(诺亚造物)",
    "黑铁会总部": "从高到低是黑铁会总堂-黑龙/火凤/翅虎堂-外围",
    "诺亚前线基地深处": "主要敌人为大量诺亚造物或少量诺亚高层",
}


def stage_root_region_hint(stage_root: str) -> str:
    """
    返回 stages 子目录（关卡大区）对应的地图/叙事区域说明，供 prepare_task_context 等写入关卡对象。

    - 对抽象目录名附加「主要区域」说明；
    - 黑铁会总部、诺亚前线基地深处、雪山、副本任务、试炼场深处等返回目录名本身。
    """
    return _STAGE_ROOT_REGION_HINT_EXTRA.get(stage_root, stage_root)



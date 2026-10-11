from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


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


# 主线阶段基础配置：1-7 阶段
# 注：后续如果主线扩展，可只调整对应阶段的 id 范围上限。
PROGRESS_STAGE_CONFIG: Dict[int, StageConfig] = {
    # 1：废城，1-11级，stages 名：基地门口，主线 id 范围 0-21
    1: StageConfig(
        name="废城",
        min_level=1,
        max_level=12,
        stage_name="基地门口",
        main_task_min_id=0,
        main_task_max_id=21,
    ),
    # 2：堕落城，12-15级，stages 名：基地车库，主线 id 范围 22-28
    2: StageConfig(
        name="堕落城",
        min_level=11,
        max_level=17,
        stage_name="基地车库",
        main_task_min_id=22,
        main_task_max_id=28,
    ),
    # 3：荒漠军阀，16-19级，stages 名：基地房顶，主线 id 范围 29-36
    3: StageConfig(
        name="荒漠军阀",
        min_level=15,
        max_level=22,
        stage_name="基地房顶",
        main_task_min_id=29,
        main_task_max_id=36,
    ),
    # 4：黑铁会总堂，20-28级，stages 名：黑铁会总部，主线 id 范围 37-74
    4: StageConfig(
        name="黑铁会总堂",
        min_level=20,
        max_level=31,
        stage_name="黑铁会总部",
        main_task_min_id=37,
        main_task_max_id=74,
    ),
    # 5：诺亚，29-40级，stages 名：诺亚前线基地深处，主线 id 范围 75-175（暂定）
    5: StageConfig(
        name="诺亚",
        min_level=29,
        max_level=40,
        stage_name="诺亚前线基地深处",
        main_task_min_id=75,
        main_task_max_id=175,
    ),
    # 6：雪山，40-50级，stages 名：雪山，主线 id 范围 77-295（暂定）
    6: StageConfig(
        name="雪山",
        min_level=40,
        max_level=50,
        stage_name="雪山",
        main_task_min_id=77,
        main_task_max_id=295,
    ),
    # 7：已通关，50-100级，stages 名：已通关，主线 id 范围 77-300（暂定）
    7: StageConfig(
        name="已通关",
        min_level=50,
        max_level=100,
        stage_name="副本任务",
        main_task_min_id=77,
        main_task_max_id=300,
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


# ---------------------------------------------------------------------------
# 存档投影 → 阶段推导（存档绑定时自动判定，不再依赖前端手选）
# ---------------------------------------------------------------------------
# 口径来源（游戏仓 tasks1.json 实证，2026-10-11 核实）：
# task_chains_progress["主线"] ＝ 最后一个已完成主线任务的 chain 值，只增不减；
# 节点任务完成后链值即跳到下一节点值，按「完成该任务后进入下一节点」的界读。

#: 主线链值终点：末任务「挫其锋芒」（id 77，目标关卡「深入禁区」）完成后链值为 780。
#: 主线扩篇时需同步此值与其后的等级档判定。
MAINLINE_CHAIN_COMPLETE = 780

#: 主线链值 → 阶段 的节点边界（升序，含端点）：链值未超过该界＝停在该节点。
#: （21/27/36/67 分别为节点任务链值减一，即任务未完成时的峰值链值。）
MAINLINE_STAGE_BOUNDARIES: tuple[tuple[int, int], ...] = (
    (21, 1),  # 未完成「一劳永逸」（id 21，地铁站）＝废城
    (27, 2),  # 未完成「狂野的玫瑰」（id 27，摇滚公园）＝堕落城
    (36, 3),  # 未完成「不安之影」（id 36，军阀前线基地）＝军阀
    (67, 4),  # 未完成「不败传说的破灭！」（id 67，黑铁会总堂）＝黑铁会总堂
)


def _mainline_chain_value(progress: Any) -> Optional[int]:
    """从进度投影取主线链值；段缺失/非数值/负数＝不可判（None）。"""
    if not isinstance(progress, dict):
        return None
    chains = progress.get("task_chains_progress")
    if not isinstance(chains, dict):
        return None
    raw = chains.get("主线")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = int(raw)
    return value if value >= 0 else None


def _player_level(player: Any) -> Optional[int]:
    """从玩家投影取等级；段缺失/非数值/非正数＝不可判（None）。"""
    if not isinstance(player, dict):
        return None
    raw = player.get("level")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    level = int(raw)
    return level if level > 0 else None


def derive_progress_stage(progress: Any, player: Any = None) -> Optional[int]:
    """
    由绑定存档投影推导主线阶段（1-7）；数据不足以判定时返回 None（由调用方回落请求值）。

    - 主线未完成（链值 < 780）：按节点边界归入 1-4，超过 67 未到终点＝诺亚(5)；
    - 主线已全完成（链值 ≥ 780）：按等级模拟——<40 诺亚(5)、40-49 雪山(6)、≥50 已通关(7)；
      等级缺失按 5（诺亚）保守回落。
    """
    chain = _mainline_chain_value(progress)
    if chain is None:
        return None
    if chain >= MAINLINE_CHAIN_COMPLETE:
        level = _player_level(player)
        if level is None or level < 40:
            return 5
        if level < 50:
            return 6
        return 7
    for bound, stage in MAINLINE_STAGE_BOUNDARIES:
        if chain <= bound:
            return stage
    return 5  # 68 ≤ 链值 < 780：诺亚段（直至「深入禁区」完成）




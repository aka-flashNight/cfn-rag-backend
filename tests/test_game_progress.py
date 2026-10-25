"""存档投影 → 主线阶段推导单测（节点边界按游戏仓 tasks1.json 实证）。

节点任务链值：id 21 地铁站→#22、id 27 摇滚公园→#28、id 36 军阀前线基地→#37、
id 67 黑铁会总堂→#68、id 77 深入禁区→#780（终点）。
边界语义：完成节点任务后链值跳到下一节点，即「未完成节点任务＝停在原节点」。
"""

from __future__ import annotations

import pytest

from services.game_progress import derive_progress_stage


def _progress(chain: int | float | None) -> dict:
    chains = {} if chain is None else {"主线": chain}
    return {"tasks_finished": [], "task_chains_progress": chains}


def _player(level: int | None = None) -> dict:
    return {} if level is None else {"level": level}


@pytest.mark.parametrize("chain,expected", [
    (0, 1), (1, 1), (21, 1),          # 未完成地铁站（id 21）＝废城
    (22, 2), (27, 2),                 # 完成地铁站、未完成摇滚公园（id 27）＝堕落城
    (28, 3), (36, 3),                 # 完成摇滚公园、未完成军阀前线基地（id 36）＝军阀
    (37, 4), (67, 4),                 # 完成军阀前线基地、未完成黑铁会总堂（id 67）＝黑铁会总堂
    (68, 5), (77, 5), (779, 5),       # 完成黑铁会总堂、主线未全完成＝诺亚
])
def test_chain_boundaries(chain, expected):
    assert derive_progress_stage(_progress(chain), _player(1)) == expected


def test_complete_mainline_falls_back_to_level():
    """链值 ≥780（「深入禁区」完成）后按等级模拟：<40 诺亚、40-49 雪山、≥50 已通关。"""
    for chain in (780, 781, 900):
        assert derive_progress_stage(_progress(chain), _player(38)) == 5
        assert derive_progress_stage(_progress(chain), _player(39)) == 5  # 39 空档归诺亚口径
        assert derive_progress_stage(_progress(chain), _player(40)) == 6
        assert derive_progress_stage(_progress(chain), _player(49)) == 6
        assert derive_progress_stage(_progress(chain), _player(50)) == 7
        assert derive_progress_stage(_progress(chain), _player(99)) == 7


def test_complete_mainline_level_unknown_falls_back_to_noah():
    assert derive_progress_stage(_progress(780), None) == 5
    assert derive_progress_stage(_progress(780), _player(None)) == 5
    assert derive_progress_stage(_progress(780), {"level": 0}) == 5
    assert derive_progress_stage(_progress(780), {"level": "40"}) == 5


@pytest.mark.parametrize("progress", [
    None,
    "not-a-dict",
    {},
    {"task_chains_progress": None},
    {"task_chains_progress": {"主线": None}},
    {"task_chains_progress": {"主线": "abc"}},
    {"task_chains_progress": {"主线": -1}},
    {"task_chains_progress": {"主线": True}},
])
def test_undecidable_progress_returns_none(progress):
    """数据不足以判定时返回 None（调用方回落请求携带的 progress_stage）。"""
    assert derive_progress_stage(progress, _player(10)) is None


def test_float_chain_value_tolerated():
    """JSON 反序列化的 22.0 等价整数链值。"""
    assert derive_progress_stage(_progress(22.0), None) == 2

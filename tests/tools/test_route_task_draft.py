"""route_task_draft 描述按「当前可切磋」条件拼接：

- 不可切磋（含未传参）时，描述与静态版本一致，且不出现「切磋/对战」字样；
- 可切磋时多一句「切磋/对战也能路由」，其余部分逐字不变。
"""

from __future__ import annotations

from services.tools.route.route_task_draft import (
    build_description,
    tool,
)


def _description(**kwargs) -> str:
    return tool.to_openai_tool(**kwargs)["function"]["description"]


def test_static_description_has_no_challenge_words():
    desc = _description()
    assert desc == build_description()
    assert "切磋" not in desc
    assert "对战" not in desc


def test_not_challengeable_keeps_static_description():
    assert _description(player_can_challenge=False) == build_description()
    assert _description(player_can_challenge=None) == build_description()


def test_challengeable_appends_clause_only():
    static = _description()
    challengeable = _description(player_can_challenge=True)
    assert "切磋、对战" in challengeable
    assert "note 里写明他要切磋/对战即可" in challengeable
    # 首尾两段逐字保留：只在中间插入一句
    assert challengeable.startswith(
        "把本轮交给后台任务系统，为玩家拟定一份**新的**委托草案。"
    )
    assert challengeable.endswith("无需调用工具时正常输出台词。")
    assert len(challengeable) > len(static)

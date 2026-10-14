"""DialogueRunner 三轮固定流程测试（阶段四 B4-2）。

覆盖：三轮工具表逐轮收窄、查询用满后换头且 system 无查询残留、
判断不动即结束、编辑打回后的修正轮、提示词纯净（edit-only 版无查询表述、
委派材料空块不出现）。
"""

from __future__ import annotations

import json

from services.dialogue.overrides import load_overrides
from services.integration.save_paths import agent_save_override_paths
from services.subagents.dialogue_runner import DialogueRunner
from services.subagents.prompts import build_dialogue_system, build_dialogue_user
from services.tools.base import ToolRegistry
from services.tools.query.search_npc_dialogues import tool as search_tool
from services.tools.task.edit_npc_dialogues import tool as edit_tool
from tests.dialogue.conftest import NPC, SLOT
from tests.orchestrator.conftest import FakeLLM, tc


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(search_tool)
    registry.register(edit_tool)
    return registry


def _launch(fake: FakeLLM, registry: ToolRegistry, game_data, **kwargs):
    return DialogueRunner.launch(
        llm=fake,
        registry=registry,
        npc_name=NPC,
        profile_block=f"你现在扮演游戏角色「{NPC}」（阵营：测试阵营）。",
        relationship_block="当前好感度 30 / 100，关系「熟悉」。",
        source_trigger="关键信息",
        reason="他替我找回了丢失的龙鳞。",
        game_data=game_data,
        **kwargs,
    )


def _tool_names(req) -> list[str]:
    return [t["function"]["name"] for t in (req.tools or [])]


_FULL_EDIT_LINES = [
    {"name": NPC, "text": "炉子今天烧得格外旺。"},
    {"name": "$PC", "text": "给我打把刀。"},
    {"name": NPC, "text": "龙鳞的事，我记下了。"},
]


async def test_three_round_flow_and_search_removal(dialogue_env, bound_slot):
    """三轮流程：查 → 查 → 改；查询用满后工具表收窄且 system 无查询残留。"""
    bound_slot()
    fake = FakeLLM()
    fake.add_chat(tool_calls=[tc("search_npc_dialogues", {"scopes": ["daily_all"]}, "c1")])
    fake.add_chat(tool_calls=[tc("search_npc_dialogues", {
        "scopes": ["keyword"], "keywords": ["龙鳞"],
    }, "c2")])
    fake.add_chat(tool_calls=[tc("edit_npc_dialogues", {
        "edits": [{"number": 1, "lines": _FULL_EDIT_LINES}],
    }, "c3")])

    handle = _launch(fake, _registry(), dialogue_env.game_data)
    event = await handle.wait_final(timeout=5)
    assert event.kind == "final"
    assert event.payload["changed"] is True
    assert len(event.payload["applied"]) == 1

    reqs = fake.chat_requests
    assert len(reqs) == 3
    assert _tool_names(reqs[0]) == ["search_npc_dialogues"]
    assert _tool_names(reqs[1]) == ["search_npc_dialogues", "edit_npc_dialogues"]
    assert _tool_names(reqs[2]) == ["edit_npc_dialogues"]
    # 摘掉查询后：system 里不得再出现任何查询/范围相关表述
    system_r3 = reqs[2].messages[0]["content"]
    assert "search_npc_dialogues" not in system_r3
    assert "查询" not in system_r3
    assert "范围" not in system_r3
    # 第 1/2 轮的 system 含查询流程（注册与提示词同步）
    assert "search_npc_dialogues" in reqs[0].messages[0]["content"]

    dialogue_path, _ = agent_save_override_paths(dialogue_env.data_root, SLOT)
    entries = load_overrides(dialogue_path)
    assert len(entries) == 1
    assert entries[0].lines[0].text == "炉子今天烧得格外旺。"
    # 触发来源经 ctx 透传进覆盖层条目
    assert entries[0].reason == "他替我找回了丢失的龙鳞。"
    assert entries[0].source_trigger == "关键信息"


async def test_no_change_decision_ends_run(dialogue_env, bound_slot):
    """判断不动即结束：第 1 轮直接输出文本＝本次「不动」，零落盘。"""
    bound_slot()
    fake = FakeLLM()
    fake.add_chat(content="他只是顺手帮了个小忙，不值得为此改口。")

    handle = _launch(fake, _registry(), dialogue_env.game_data)
    event = await handle.wait_final(timeout=5)
    assert event.kind == "final"
    assert event.payload["changed"] is False
    assert "不值得" in event.payload["note"]
    assert len(fake.chat_requests) == 1

    dialogue_path, _ = agent_save_override_paths(dialogue_env.data_root, SLOT)
    assert not dialogue_path.exists()


async def test_rejected_edit_fixed_next_round(dialogue_env, bound_slot):
    """编辑打回后的修正轮：第 2 轮被打回不落盘，第 3 轮修正重交后落盘。"""
    bound_slot()
    fake = FakeLLM()
    fake.add_chat(tool_calls=[tc("search_npc_dialogues", {"scopes": ["daily_all"]}, "c1")])
    # 第 2 轮：提交行数少于原组 → 打回
    fake.add_chat(tool_calls=[tc("edit_npc_dialogues", {
        "edits": [{"number": 1, "lines": [{"name": NPC, "text": "只剩一句。"}]}],
    }, "c2")])
    # 第 3 轮：按打回原因补齐原行后重交
    fake.add_chat(tool_calls=[tc("edit_npc_dialogues", {
        "edits": [{"number": 1, "lines": _FULL_EDIT_LINES}],
    }, "c3")])

    handle = _launch(fake, _registry(), dialogue_env.game_data)
    event = await handle.wait_final(timeout=5)
    assert event.kind == "final"
    assert event.payload["changed"] is True

    reqs = fake.chat_requests
    assert len(reqs) == 3
    # 第 2 轮查询仅用 1 次，仍可同时查/改；第 3 轮只能改
    assert _tool_names(reqs[1]) == ["search_npc_dialogues", "edit_npc_dialogues"]
    assert _tool_names(reqs[2]) == ["edit_npc_dialogues"]
    # 第 2 轮工具结果里是打回原因原文（模型据此修正）
    tool_msgs = [m for m in reqs[2].messages if m.get("role") == "tool"]
    rejected_payload = json.loads(tool_msgs[-1]["content"])
    assert rejected_payload["status"] == "rejected"
    assert any("不能删除原有信息" in r for r in rejected_payload["rejected"])
    # 第 3 轮 system 已换头（第 3 轮不再下发查询工具，提示词同步摘除）
    assert "search_npc_dialogues" not in reqs[2].messages[0]["content"]

    dialogue_path, _ = agent_save_override_paths(dialogue_env.data_root, SLOT)
    assert len(load_overrides(dialogue_path)) == 1


def test_edit_only_system_has_no_search_wording():
    """提示词纯净：allow_search=False 版零查询表述，两版本身份与台词块一致。"""
    full = build_dialogue_system(
        npc_name=NPC, profile_block="资料块", relationship_block="关系块", allow_search=True,
    )
    stripped = build_dialogue_system(
        npc_name=NPC, profile_block="资料块", relationship_block="关系块", allow_search=False,
    )
    assert "search_npc_dialogues" in full
    for word in ("search_npc_dialogues", "查询", "范围"):
        assert word not in stripped
    # 身份块与台词写法两版本逐字一致
    assert "资料块" in stripped and "关系块" in stripped
    assert "【台词写法】" in full and "【台词写法】" in stripped
    style_full = full.split("【台词写法】")[1]
    style_stripped = stripped.split("【台词写法】")[1]
    assert style_full == style_stripped


def test_user_prompt_omits_empty_blocks():
    """委派材料空块整块不出现（提示词纯净：不描述缺席状态）。"""
    user = build_dialogue_user(source_trigger="重要任务", reason="任务完成了。")
    assert "触发类型：重要任务" in user
    assert "发生了什么：任务完成了。" in user
    assert "方向指示" not in user
    assert "相关任务" not in user
    assert "发布该任务时的对话记录" not in user
    full_user = build_dialogue_user(
        source_trigger="重要任务", reason="任务完成了。", instruction="语气更敬重些",
        task_block="任务详情", publish_block="摘要与对话",
    )
    assert "改写的方向指示：语气更敬重些" in full_user
    assert "【相关任务】" in full_user
    assert "【发布该任务时的对话记录】" in full_user

"""DialogueRunner 三轮固定流程测试（阶段四 B4-2）。

覆盖：三轮工具表逐轮收窄、查询用满后换头且 system 无查询残留、
判断不动即结束、编辑打回后的修正轮、角色扮演资料（说话方式／脑补禁令进 system，
聊天专用的输出协议不进）、提示词纯净（edit-only 版无查询表述、
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
    # 摘掉查询后：system 里不得再出现查询流程的成串原文。不用裸词「范围」——
    # 说话方式第 4 条（「合理范围内话多」）本身就含这两字。
    system_r3 = reqs[2].messages[0]["content"]
    assert "search_npc_dialogues" not in system_r3
    assert "查询" not in system_r3
    for word in ("范围可多选", "最多查两次", "日常全量"):
        assert word not in system_r3
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
    # 断言用查询流程里的成串原文，不用裸词「范围」：说话方式第 4 条（「合理范围内话多」）
    # 与场景块（「均在你视线范围内」）都含这两字，裸词判不了「查询表述已摘除」。
    assert "search_npc_dialogues" in full
    assert "范围可多选" in full and "最多查两次" in full
    for word in ("search_npc_dialogues", "查询", "范围可多选", "最多查两次", "日常全量"):
        assert word not in stripped
    # 身份块与台词写法两版本逐字一致
    assert "资料块" in stripped and "关系块" in stripped
    assert "【台词写法】" in full and "【台词写法】" in stripped
    style_full = full.split("【台词写法】")[1]
    style_stripped = stripped.split("【台词写法】")[1]
    assert style_full == style_stripped


def test_edit_strategy_block_verbatim_in_both_heads():
    """「改哪些、怎么改」三条策略：两个 system 头逐字一致、排在台词写法之前、无查询表述。"""
    full = build_dialogue_system(npc_name=NPC, allow_search=True)
    stripped = build_dialogue_system(npc_name=NPC, allow_search=False)

    def _strategy(text: str) -> str:
        return text.split("【改哪些、怎么改】")[1].split("【台词写法】")[0]

    assert _strategy(full) == _strategy(stripped)
    assert "查询" not in _strategy(stripped)
    # 三条策略的关键口径（用户裁定 2026-10-16）
    for sentence in (
        "优先调整原有对话中的语气，不改变对话信息的情况下微调对话语气，"
        "符合当前你和玩家的关系或者你当前对玩家的态度。",
        "当语气确实不符合当前你的态度/你们的关系时，可以较大批量地修改对话，"
        "尤其是日常对话和近期任务对话。",
        "如果语气基本符合现在的情况，确实没有需要改的情况下，则考虑新增合适的对话。",
        "优先找出并修改和该信息、事件、任务有关联的对话，任何关联都可以，"
        "如果还是没有，则优先追加新日常对话，而不是改无关对话。",
        "也可以同时改相关对话并新增新日常对话。",
        "日常对话是经常会触发的可重复对话，注意普适性，体现人物性格，不要把话说的像第一次说。",
        "任务对话仅可触发一次，不会重复触发，无需担心此问题。",
    ):
        assert sentence in stripped


def test_dialogue_system_requires_pc_placeholder_for_player_lines():
    """真模型爱把玩家行的说话人写成中文「玩家」，两个 system 头都必须写明落盘用 $PC。"""
    for allow_search in (True, False):
        text = build_dialogue_system(npc_name=NPC, allow_search=allow_search)
        assert "玩家那行的说话人写 $PC，不要写「玩家」，也不要写他的名字。" in text


def test_dialogue_system_carries_roleplay_material():
    """角色扮演资料补齐：世界观/说话方式/脑补禁令都在，且与聊天主 Agent 同一份文本。"""
    from services.orchestrator.prompts import (
        ANTI_HALLUCINATION_RULE,
        SPEECH_STYLE_RULES,
    )

    setting = "【世界观背景概要】占位\n你所在场景：武器店。"
    built = build_dialogue_system(
        npc_name=NPC, profile_block="资料块", relationship_block="关系块",
        setting_block=setting,
    )
    assert setting in built
    assert SPEECH_STYLE_RULES.rstrip() in built
    assert ANTI_HALLUCINATION_RULE in built
    # 只属于聊天主 Agent 的输出协议（meta JSON / 工具）不进来
    assert "【对话输出规则】" not in built
    assert "meta JSON" not in built


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

"""search_npc_dialogues / edit_npc_dialogues 两个工具的测试（阶段四 B4-4 工具侧）。

覆盖：多选生效、任务近期与任务全量冲突取近期、已完成与 Agent 任务排除、
关键词只在未选范围里搜（并集去重）、编号跨轮可用、编辑打回不落盘、
未绑定存档降级、二次修改就地更新同一条目、同轮先查后改的 pipeline 顺序、
工具描述的逐字口径锁（多选声明 + 任务全量注释）。
"""

from __future__ import annotations

import json

from services.dialogue.overrides import load_overrides
from services.dialogue.task_text import load_task_text_overrides
from services.integration.save_paths import agent_save_override_paths
from services.tools.base import ToolContext, ToolRegistry, _inherit_ctx
from services.tools.query.search_npc_dialogues import tool as search_tool
from services.tools.task.edit_npc_dialogues import tool as edit_tool
from tests.dialogue.conftest import NPC, SLOT


def _tc(name: str, args: dict | None = None, call_id: str = "c0") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {}, ensure_ascii=False)},
    }


def _ctx(env, **kwargs) -> ToolContext:
    return ToolContext(npc_name=NPC, game_data=env.game_data, **kwargs)


def _override_paths(env):
    return agent_save_override_paths(env.data_root, SLOT)


# ---------------------------------------------------------------------------
# 查询侧
# ---------------------------------------------------------------------------

def test_multi_scope_select(dialogue_env, bound_slot):
    """多选生效：日常全量 + 任务近期同选，两桶都进结果并统一编号。"""
    bound_slot()
    res = search_tool.run(
        {"scopes": ["daily_all", "task_recent"]}, _ctx(dialogue_env),
    )
    text = res.result_json
    # 日常 2 组（id 3 没有测试使说话被门槛排除）＋ 主线 5 组 ＋ 支线 1 组
    assert "日常对话 组 id 1" in text
    assert "日常对话 组 id 2" in text
    assert "组 id 3" not in text
    assert "$MAIN_GET_5" in text
    assert "$SIDE_GET_1" in text
    # 已完成任务与 Agent 任务一律排除
    assert "$FIN_GET" not in text
    assert "$AGENT_GET" not in text
    # 统一编号 1..8，映射表与文本一致
    assert "【编号 8】" in text
    assert "【编号 9】" not in text
    assert len(res.dialogue_index) == 8
    assert res.dialogue_index[0]["kind"] == "daily"
    assert res.dialogue_index[-1]["key"] == "$SIDE_GET_1"


def test_player_line_renders_as_pc_placeholder(dialogue_env, bound_slot):
    """玩家行按落盘占位符列出：模型「原样抄回」就写对 $PC，不必靠后端改写。"""
    bound_slot()
    res = search_tool.run({"scopes": ["daily_all"]}, _ctx(dialogue_env))
    text = res.result_json
    assert "$PC：给我打把刀。" in text
    assert "玩家：" not in text


def test_edit_tool_name_field_states_pc_placeholder():
    """说话人字段的描述必须给出生成者看得懂的取值（曾经写「或玩家」，模型就填「玩家」）。"""
    desc = edit_tool.parameters_schema["properties"]["edits"]["items"]["properties"][
        "lines"]["items"]["properties"]["name"]["description"]
    assert "固定填 $PC" in desc


def test_task_recent_wins_over_task_all(dialogue_env, bound_slot):
    """任务近期与任务全量同选时采用任务近期（6 个主线只取最近 5 组）。"""
    bound_slot()
    res = search_tool.run(
        {"scopes": ["task_recent", "task_all"]}, _ctx(dialogue_env),
    )
    text = res.result_json
    assert "$MAIN_GET_5" in text
    assert "$MAIN_GET_6" not in text


def test_keyword_searches_beyond_selected(dialogue_env, bound_slot):
    """关键词只在未被选中范围覆盖的组里搜：命中并入、不重复列组。"""
    bound_slot()
    # 任务近期已含 $MAIN_GET_1..5；「龙鳞」还命中日常组 2、$MAIN_GET_1（已被
    # 近期覆盖）与 $MAIN_GET_6（未覆盖）——关键词桶只带进未覆盖的两组，
    # 已被覆盖的 $MAIN_GET_1 不重复编号。
    res = search_tool.run(
        {"scopes": ["task_recent", "keyword"], "keywords": ["龙鳞"]},
        _ctx(dialogue_env),
    )
    text = res.result_json
    assert text.count("$MAIN_GET_6") == 1
    assert text.count("$MAIN_GET_1") == 1
    assert "龙鳞可不好弄" in text  # 日常组 2 未被已选范围覆盖，关键词命中并入
    keys = [entry["key"] for entry in res.dialogue_index]
    assert len(keys) == len(set(keys))


def test_keyword_hit_daily_group(dialogue_env, bound_slot):
    """只选关键词范围时，日常组与任务组都能命中。"""
    bound_slot()
    res = search_tool.run(
        {"scopes": ["keyword"], "keywords": ["龙鳞"]}, _ctx(dialogue_env),
    )
    text = res.result_json
    assert "龙鳞可不好弄" in text      # 日常组 2
    assert "$MAIN_GET_1" in text       # 任务组
    kinds = {entry["kind"] for entry in res.dialogue_index}
    assert kinds == {"daily", "task"}


def test_scope_description_verbatim():
    """工具描述锁逐字口径：多选声明 + 任务全量的完整注释（用户裁定 2026-10-13）。"""
    blob = json.dumps(search_tool.to_openai_tool(), ensure_ascii=False)
    assert "这里是多选，不是单选" in blob
    assert (
        "不推荐。除非是非常重大的关系变化、导致所有对话都必须全部修改时，"
        "才选中任务全量；否则不建议选。"
    ) in blob
    # 定义与冲突口径也要给全：光写「不推荐」会让模型觉得这个选项莫名其妙
    assert "该 NPC 说话的全部未完成任务对话组；与 task_recent 冲突，两者都选时采用 task_recent。" in blob
    edit_blob = json.dumps(edit_tool.to_openai_tool(), ensure_ascii=False)
    assert "不需要改的行请原样抄回，不能删除原有信息" in edit_blob
    assert "每行填说话人 name 与台词 text，情绪可选填 emotion" in edit_blob
    # Char 由后端拼、Title 由后端查：模型这一侧只出现情绪标签，且必须写明可以留空
    assert "从你的可用情绪标签里选一个；语气平实、没有表情的行留空" in edit_blob
    assert '"title"' not in edit_blob
    assert "char" not in edit_blob


# ---------------------------------------------------------------------------
# 修改侧
# ---------------------------------------------------------------------------

def _search_then_edit_ctx(env) -> tuple[ToolContext, dict]:
    """跑一次查询，返回（带编号映射的 ctx, 编号→条目 映射）。"""
    res = search_tool.run({"scopes": ["daily_all"]}, _ctx(env))
    ctx = _inherit_ctx(_ctx(env), res)
    return ctx, {e["number"]: e for e in res.dialogue_index}


def test_index_usable_across_rounds(dialogue_env, bound_slot):
    """编号跨轮可用：查询的映射经 _inherit_ctx 传到下一轮的修改工具。"""
    bound_slot()
    ctx, _ = _search_then_edit_ctx(dialogue_env)
    res = edit_tool.run({"edits": [{
        "number": 1,
        "lines": [
            {"name": NPC, "text": "炉子今天歇火。"},
            {"name": "$PC", "text": "给我打把刀。"},
            {"name": NPC, "text": "明天再来。"},
        ],
    }]}, ctx)
    payload = json.loads(res.result_json)
    assert payload["status"] == "ok"
    assert len(payload["applied"]) == 1

    dialogue_path, _ = _override_paths(dialogue_env)
    entries = load_overrides(dialogue_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.action == "replace" and entry.group_id == "1"
    assert entry.base_dialogue_id == "1"
    lines = [line.to_subdialogue() for line in entry.lines]
    assert lines[0]["Text"] == "炉子今天歇火。"
    # 既有行 Title/Char 沿用原行；玩家行占位符原样保留
    assert lines[1]["Name"] == "$PC" and lines[1]["Title"] == "$PC_TITLE"
    # 组尾新增行：Title 取说话人名，Char 取原组最后出现的值
    assert lines[2]["Title"] == NPC and lines[2]["Char"] == NPC


def test_edit_rejected_nothing_persisted(dialogue_env, bound_slot):
    """非法行整组打回：原因带修正方向与重交提示，覆盖层文件不生成。"""
    bound_slot()
    ctx, _ = _search_then_edit_ctx(dialogue_env)
    res = edit_tool.run({"edits": [{
        "number": 1,
        "lines": [
            {"name": NPC, "text": "今天炉子烧得正旺。"},
            {"name": "旁人", "text": "替别人说话。"},
        ],
    }]}, ctx)
    payload = json.loads(res.result_json)
    assert payload["status"] == "rejected"
    assert any("不在允许范围内" in r and "修正：" in r for r in payload["rejected"])
    assert "重新提交" in payload["retry_hint"]
    assert "不要重复提交" in payload["retry_hint"]
    dialogue_path, text_path = _override_paths(dialogue_env)
    assert not dialogue_path.exists()
    assert not text_path.exists()


def test_unbound_save_degrades(dialogue_env):
    """未绑定存档：查询按基础数据走（可读），修改被 fail-closed 拒绝且不写盘。"""
    res = search_tool.run({"scopes": ["daily_all"]}, _ctx(dialogue_env))
    assert "日常对话 组 id 1" in res.result_json

    ctx = _inherit_ctx(_ctx(dialogue_env), res)
    edit_res = edit_tool.run({"edits": [{
        "number": 1,
        "lines": [
            {"name": NPC, "text": "改了也白改。"},
            {"name": "$PC", "text": "给我打把刀。"},
        ],
    }]}, ctx)
    payload = json.loads(edit_res.result_json)
    assert payload["status"] == "rejected"
    assert any("未绑定存档" in r for r in payload["rejected"])
    # 没有共享侧落点：data_root 下不得生出任何覆盖层
    assert not (dialogue_env.data_root / "rag").exists()


def test_second_edit_updates_same_entry(dialogue_env, bound_slot):
    """二次修改就地更新同一条目：agent_* 组再次修改不产生重复覆盖记录。"""
    bound_slot()
    ctx = _ctx(dialogue_env)
    # 第一次：新增一组日常对话
    res1 = edit_tool.run({
        "edits": [],
        "new_groups": [{"lines": [
            {"name": NPC, "text": "新组第一句。"},
            {"name": "$PC", "text": "好。"},
        ]}],
    }, ctx)
    assert json.loads(res1.result_json)["status"] == "ok"

    # 第二次查询拿到 agent_1 的编号，再改它
    res2 = search_tool.run({"scopes": ["daily_all"]}, ctx)
    agent_entry = next(e for e in res2.dialogue_index if e["key"].startswith("agent_"))
    ctx2 = _inherit_ctx(ctx, res2)
    res3 = edit_tool.run({"edits": [{
        "number": agent_entry["number"],
        "lines": [
            {"name": NPC, "text": "新组第一句改过了。"},
            {"name": "$PC", "text": "好。"},
            {"name": NPC, "text": "再补一句。"},
        ],
    }]}, ctx2)
    assert json.loads(res3.result_json)["status"] == "ok"

    dialogue_path, _ = _override_paths(dialogue_env)
    entries = load_overrides(dialogue_path)
    assert len(entries) == 1
    assert entries[0].group_id == agent_entry["key"]
    assert entries[0].action == "append"
    assert entries[0].base_dialogue_id is None
    assert [line.text for line in entries[0].lines] == [
        "新组第一句改过了。", "好。", "再补一句。",
    ]


def test_task_dialogue_edit_writes_text_overlay(dialogue_env, bound_slot):
    """任务对话修改写 task_text_overrides.json（$KEY 整份替换）。"""
    bound_slot()
    res = search_tool.run({"scopes": ["task_recent"]}, _ctx(dialogue_env))
    entry = next(e for e in res.dialogue_index if e["key"] == "$MAIN_GET_1")
    ctx = _inherit_ctx(_ctx(dialogue_env), res)
    edit_res = edit_tool.run({"edits": [{
        "number": entry["number"],
        "lines": [
            {"name": NPC, "text": "主线1的接取话，龙鳞记得带来。"},
            {"name": NPC, "text": "路上当心。"},
        ],
    }]}, ctx)
    assert json.loads(edit_res.result_json)["status"] == "ok"

    _, text_path = _override_paths(dialogue_env)
    mapping = load_task_text_overrides(text_path)
    assert set(mapping.keys()) == {"$MAIN_GET_1"}
    items = mapping["$MAIN_GET_1"]
    assert [item["text"] for item in items] == ["主线1的接取话，龙鳞记得带来。", "路上当心。"]


async def test_same_batch_search_before_edit(dialogue_env, bound_slot):
    """同轮「先查后改」：edit 排在 search 前面提交，pipeline 仍保证查询先执行。"""
    bound_slot()
    registry = ToolRegistry()
    registry.register(search_tool)
    registry.register(edit_tool)
    results = await registry.dispatch_batch([
        _tc("edit_npc_dialogues", {"edits": [{
            "number": 1,
            "lines": [
                {"name": NPC, "text": "顺序无关，先查后改。"},
                {"name": "$PC", "text": "给我打把刀。"},
            ],
        }]}, call_id="c_edit"),
        _tc("search_npc_dialogues", {"scopes": ["daily_all"]}, call_id="c_search"),
    ], _ctx(dialogue_env))
    by_id = {call_id: tr for call_id, tr, _ctx_after in results}
    assert json.loads(by_id["c_edit"].result_json)["status"] == "ok"
    dialogue_path, _ = _override_paths(dialogue_env)
    assert load_overrides(dialogue_path)[0].lines[0].text == "顺序无关，先查后改。"

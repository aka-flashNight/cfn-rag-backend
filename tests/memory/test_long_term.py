"""跨对话长期记忆（B3-2/B3-3/B3-4）：建表校验、作用域隔离、编号分配、
注入块拼接、聊天侧三形态工具（含满 20 条时的禁词锁）、摘要侧两形态工具。"""

from __future__ import annotations

import json
import sqlite3

import pytest

from services.integration.state import get_integration_state
from services.memory.long_term import (
    CHAT_TOOL_NAME,
    SUMMARY_TOOL_NAME,
    apply_chat_tool,
    apply_summary_tool,
    build_chat_tool,
    build_summary_tool,
    format_block,
    format_summary_block,
    load_snapshot,
    normalize_content,
    resolve_enabled,
    scope_key,
)
from services.memory.store import (
    LTM_MAX_CHARS,
    LTM_MAX_FACT_ENTRIES,
    STANDALONE_SCOPE,
    MemoryStore,
)

NPC = "铁匠"


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(db_path=tmp_path / "memory.db")
    yield s
    s.close()


@pytest.fixture
def bound_slot():
    """绑定一个存档槽位（切到 embedded 口径），用例结束后解绑。"""
    state = get_integration_state()

    def _bind(slot_key: str):
        state.bind(slot_key)
        return slot_key

    yield _bind
    state.unbind()


def _tool_blob(tool: dict) -> str:
    """工具定义的可见文本（名字＋描述＋参数描述），锁禁词用。"""
    return json.dumps(tool, ensure_ascii=False)


async def _snapshot(store, slot_key=None, npc: str = NPC):
    return await load_snapshot(store, slot_key=slot_key, npc=npc)


# ---------------------------------------------------------------------------
# 建表校验与作用域
# ---------------------------------------------------------------------------

def test_creates_table_on_existing_db_without_it(tmp_path):
    """已有 db 但没有长期记忆表 → 初始化时自动建表，且不碰既有表。"""
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE legacy_marker (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    store = MemoryStore(db_path=db_path)
    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    store.close()

    assert "long_term_memory" in tables
    assert "legacy_marker" in tables


def test_scope_key_falls_back_to_standalone():
    """未绑定存档时用独立作用域，与各存档互不混淆。"""
    assert scope_key(None) == STANDALONE_SCOPE
    assert scope_key("") == STANDALONE_SCOPE
    assert scope_key("  ") == STANDALONE_SCOPE
    assert scope_key("slot_1") == "slot_1"


def test_resolve_enabled_is_forced_on_when_embedded(bound_slot):
    """游戏内打开恒定启用、不受理关闭；非游戏打开按请求值，缺省关闭。"""
    assert resolve_enabled(None) is False
    assert resolve_enabled(False) is False
    assert resolve_enabled(True) is True

    bound_slot("slot_1")
    assert resolve_enabled(False) is True
    assert resolve_enabled(None) is True


def test_normalize_content_truncates_at_limit():
    """后端校验宽松：500 字以下都能存，长了截断。"""
    assert normalize_content("  你好  ") == "你好"
    assert normalize_content(None) == ""
    long_text = "字" * (LTM_MAX_CHARS + 80)
    assert len(normalize_content(long_text)) == LTM_MAX_CHARS


async def test_entries_isolated_between_scopes(store):
    """每个存档用自己的长期记忆；未绑定那份独立。"""
    await store.ltm_save_base("slot_1", NPC, "存档一的印象", "s1")
    await store.ltm_append_fact("slot_1", NPC, "存档一的事件", "s1")
    await store.ltm_save_base("slot_2", NPC, "存档二的印象", "s2")
    await store.ltm_save_base(STANDALONE_SCOPE, NPC, "非游戏打开的印象", "s0")

    snap1 = await _snapshot(store, "slot_1")
    snap2 = await _snapshot(store, "slot_2")
    snap0 = await _snapshot(store, None)

    assert snap1.base == "存档一的印象"
    assert [f.content for f in snap1.facts] == ["存档一的事件"]
    assert snap2.base == "存档二的印象"
    assert snap2.facts == []
    assert snap0.base == "非游戏打开的印象"
    assert snap0.slot_key == STANDALONE_SCOPE


async def test_entry_no_is_per_scope_and_npc(store):
    """编号是作用域内某 NPC 的直观序号，从 1 起、按顺序递增。"""
    for i in range(1, 4):
        await store.ltm_append_fact("slot_1", NPC, f"事件{i}", "s")
    await store.ltm_append_fact("slot_1", "药师", "另一个 NPC 的事件", "s")
    await store.ltm_append_fact("slot_2", NPC, "另一个存档的事件", "s")

    snap = await _snapshot(store, "slot_1", NPC)
    assert [f.entry_no for f in snap.facts] == [1, 2, 3]
    assert (await _snapshot(store, "slot_1", "药师")).facts[0].entry_no == 1
    assert (await _snapshot(store, "slot_2", NPC)).facts[0].entry_no == 1


async def test_base_is_single_row_overwritten_in_place(store):
    """基础长期记忆每个作用域只有一条，重复写是覆盖不是新增。"""
    first = await store.ltm_save_base("slot_1", NPC, "旧印象", "s1")
    second = await store.ltm_save_base("slot_1", NPC, "新印象", "s2")

    assert second.id == first.id
    assert await store.ltm_get_base("slot_1", NPC) == "新印象"
    assert await store.ltm_fact_count("slot_1", NPC) == 0


async def test_records_source_session(store):
    """存长期记忆时带上来源窗口 id。"""
    entry = await store.ltm_append_fact("slot_1", NPC, "某件大事", "sess-abc")
    assert entry.source_session_id == "sess-abc"

    base = await store.ltm_save_base("slot_1", NPC, "整体印象", "sess-xyz")
    assert base.source_session_id == "sess-xyz"


# ---------------------------------------------------------------------------
# 注入块
# ---------------------------------------------------------------------------

async def test_format_block_omitted_when_scope_empty(store):
    snap = await _snapshot(store, "slot_1")
    assert snap.is_empty
    assert format_block(snap) == ""
    assert format_summary_block(snap) == ""


async def test_format_block_gives_numbered_facts(store):
    await store.ltm_save_base("slot_1", NPC, "你是个信得过的老主顾。", "s")
    await store.ltm_append_fact("slot_1", NPC, "玩家替我找回了祖传铁锤。", "s")
    await store.ltm_append_fact("slot_1", NPC, "玩家答应保守矿洞的秘密。", "s")
    snap = await _snapshot(store, "slot_1")

    block = format_block(snap)
    assert "基础印象：你是个信得过的老主顾。" in block
    assert "1. 玩家替我找回了祖传铁锤。" in block
    assert "2. 玩家答应保守矿洞的秘密。" in block
    assert "关键记录（编号在调整时引用）：" in block


async def test_format_block_skips_missing_sections(store):
    """只有基础印象 / 只有关键记录时，缺的那段整行不出现。"""
    await store.ltm_save_base("slot_1", NPC, "只有印象。", "s")
    only_base = format_block(await _snapshot(store, "slot_1"))
    assert "关键记录（编号在调整时引用）：" not in only_base

    await store.ltm_save_base("slot_2", NPC, "", "s")
    await store.ltm_append_fact("slot_2", NPC, "只有事件。", "s")
    only_fact = format_block(await _snapshot(store, "slot_2"))
    assert "基础印象：" not in only_fact
    assert "1. 只有事件。" in only_fact


async def test_summary_block_isolates_from_rolling_summary(store):
    """摘要轮的长期记忆是独立块，且明确写清不参与摘要撰写。"""
    await store.ltm_save_base("slot_1", NPC, "整体印象。", "s")
    await store.ltm_append_fact("slot_1", NPC, "关键事件。", "s")
    block = format_summary_block(await _snapshot(store, "slot_1"))

    assert block.startswith("【长期记忆（跨对话窗口保留）】")
    assert "与本次要撰写的对话摘要是两套独立的内容" in block
    assert "长期记忆不参与对话摘要的撰写" in block
    assert "不要把长期记忆里的信息写进摘要" in block
    assert "基础印象：整体印象。" in block
    assert "1. 关键事件。" in block


# ---------------------------------------------------------------------------
# 聊天侧工具三形态
# ---------------------------------------------------------------------------

async def test_chat_tool_form_without_base_requires_both_fields(store):
    """尚无基础印象：必须同时给出基础印象与一条关键记录。"""
    tool = build_chat_tool(await _snapshot(store, "slot_1"))
    fn = tool["function"]
    assert fn["name"] == CHAT_TOOL_NAME
    assert fn["parameters"]["required"] == ["base_content", "content"]
    assert "必须同时给出两项" in fn["description"]
    # 首次写入只给 100 字，留足后续补充空间；已有印象时按 300 字
    assert "例如15字以内" in fn["description"]
    assert "总字数不得超过100 字" in fn["description"]
    assert "300" not in fn["description"]


async def test_chat_tool_form_with_room_makes_entry_no_optional(store):
    """有基础印象、关键记录未满：编号可选，填了＝融合改写，留空＝新记一条。"""
    await store.ltm_save_base("slot_1", NPC, "整体印象。", "s")
    tool = build_chat_tool(await _snapshot(store, "slot_1"))
    fn = tool["function"]
    assert fn["parameters"]["required"] == ["content"]
    assert "entry_no" in fn["parameters"]["properties"]
    assert "不要另记一条" in fn["description"]
    # 陌生状态下的首次任务是唯一例外，之后只记有长期影响的任务
    assert (
        "普通小委托都不记（除非是陌生状态下的首次任务，可记录一次。"
        "已有任何一条任务/委托的长期记忆后，不再记忆不重要小任务，"
        "只可记录关键、重要、有长期影响的任务）。"
    ) in fn["description"]


async def test_chat_tool_form_when_full_only_accepts_entry_no(store):
    """满 20 条：只剩调整能力，编号必填。"""
    await store.ltm_save_base("slot_1", NPC, "整体印象。", "s")
    for i in range(LTM_MAX_FACT_ENTRIES):
        await store.ltm_append_fact("slot_1", NPC, f"事件{i}", "s")
    snap = await _snapshot(store, "slot_1")
    assert snap.is_full

    fn = build_chat_tool(snap)["function"]
    assert fn["parameters"]["required"] == ["entry_no", "content"]
    assert "改写" in fn["description"]


async def test_full_form_never_hints_the_append_branch(store):
    """满 20 条时模型视角下只有调整能力：整份工具定义零处提及另一种用法。"""
    await store.ltm_save_base("slot_1", NPC, "整体印象。", "s")
    for i in range(LTM_MAX_FACT_ENTRIES):
        await store.ltm_append_fact("slot_1", NPC, f"事件{i}", "s")
    blob = _tool_blob(build_chat_tool(await _snapshot(store, "slot_1")))

    for forbidden in ("追加", "新记一条", "留空", "新增一条", "再记一条"):
        assert forbidden not in blob
    # 参数表里也只有编号与内容两项
    assert set(json.loads(blob)["function"]["parameters"]["properties"]) == {"entry_no", "content"}


async def test_chat_tool_is_not_in_global_registry(store):
    """长期记忆工具不进全局注册表（注册表做不到同名按状态换形）。"""
    from services.tools.base import get_tool_registry

    registry = get_tool_registry()
    assert registry.get(CHAT_TOOL_NAME) is None
    assert registry.get(SUMMARY_TOOL_NAME) is None


# ---------------------------------------------------------------------------
# 聊天侧工具执行
# ---------------------------------------------------------------------------

def _call(name: str, args: dict) -> dict:
    return {"id": "c0", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


async def test_apply_chat_tool_writes_base_and_first_fact(store):
    snap = await _snapshot(store, "slot_1")
    result = await apply_chat_tool(
        store, _call(CHAT_TOOL_NAME, {"base_content": "整体印象。", "content": "第一件大事。"}),
        snapshot=snap, session_id="s1",
    )

    assert result == "已记录关键记录 1"
    assert snap.base == "整体印象。"
    assert await store.ltm_get_base("slot_1", NPC) == "整体印象。"
    assert (await _snapshot(store, "slot_1")).facts[0].content == "第一件大事。"


async def test_apply_chat_tool_appends_without_entry_no(store):
    await store.ltm_save_base("slot_1", NPC, "整体印象。", "s")
    snap = await _snapshot(store, "slot_1")

    await apply_chat_tool(store, _call(CHAT_TOOL_NAME, {"content": "事件甲。"}),
                          snapshot=snap, session_id="s1")
    await apply_chat_tool(store, _call(CHAT_TOOL_NAME, {"content": "事件乙。"}),
                          snapshot=snap, session_id="s1")

    assert [f.content for f in snap.facts] == ["事件甲。", "事件乙。"]
    assert [f.entry_no for f in snap.facts] == [1, 2]


async def test_apply_chat_tool_merges_into_existing_entry(store):
    await store.ltm_save_base("slot_1", NPC, "整体印象。", "s")
    await store.ltm_append_fact("slot_1", NPC, "玩家接了我的委托。", "s")
    snap = await _snapshot(store, "slot_1")

    result = await apply_chat_tool(
        store, _call(CHAT_TOOL_NAME, {"entry_no": 1, "content": "玩家接了并完成我的委托。"}),
        snapshot=snap, session_id="s2",
    )

    assert result == "已改写关键记录 1"
    assert snap.facts[0].content == "玩家接了并完成我的委托。"
    assert await store.ltm_fact_count("slot_1", NPC) == 1


async def test_apply_chat_tool_ignores_unknown_entry_no(store):
    await store.ltm_save_base("slot_1", NPC, "整体印象。", "s")
    snap = await _snapshot(store, "slot_1")

    result = await apply_chat_tool(
        store, _call(CHAT_TOOL_NAME, {"entry_no": 9, "content": "凭空改写。"}),
        snapshot=snap, session_id="s",
    )

    assert result is None
    assert snap.facts == []
    assert await store.ltm_fact_count("slot_1", NPC) == 0


async def test_apply_chat_tool_refuses_unnumbered_write_when_full(store):
    """满 20 条时没有编号的调用不落库（工具描述里也不存在这种用法）。"""
    await store.ltm_save_base("slot_1", NPC, "整体印象。", "s")
    for i in range(LTM_MAX_FACT_ENTRIES):
        await store.ltm_append_fact("slot_1", NPC, f"事件{i}", "s")
    snap = await _snapshot(store, "slot_1")

    result = await apply_chat_tool(store, _call(CHAT_TOOL_NAME, {"content": "第 21 条。"}),
                                   snapshot=snap, session_id="s")

    assert result is None
    assert snap.fact_count == LTM_MAX_FACT_ENTRIES


async def test_apply_chat_tool_can_rewrite_least_important_when_full(store):
    """满 20 条仍能记新事：把不重要的那条改写成新的。"""
    await store.ltm_save_base("slot_1", NPC, "整体印象。", "s")
    for i in range(LTM_MAX_FACT_ENTRIES):
        await store.ltm_append_fact("slot_1", NPC, f"事件{i}", "s")
    snap = await _snapshot(store, "slot_1")

    result = await apply_chat_tool(
        store, _call(CHAT_TOOL_NAME, {"entry_no": 3, "content": "玩家救了整个村子。"}),
        snapshot=snap, session_id="s",
    )

    assert result == "已改写关键记录 3"
    assert snap.fact_count == LTM_MAX_FACT_ENTRIES
    assert snap.fact_by_no(3).content == "玩家救了整个村子。"


async def test_apply_chat_tool_ignores_other_tools_and_blank_content(store):
    snap = await _snapshot(store, "slot_1")
    assert await apply_chat_tool(store, _call("route_task_draft", {"note": "x"}),
                                 snapshot=snap, session_id="s") is None
    assert await apply_chat_tool(store, _call(CHAT_TOOL_NAME, {"content": "   "}),
                                 snapshot=snap, session_id="s") is None
    assert await apply_chat_tool(store, _call(CHAT_TOOL_NAME, {"entry_no": "abc", "content": ""}),
                                 snapshot=snap, session_id="s") is None
    assert snap.is_empty


async def test_apply_chat_tool_tolerates_malformed_arguments(store):
    snap = await _snapshot(store, "slot_1")
    broken = {"id": "c0", "type": "function",
              "function": {"name": CHAT_TOOL_NAME, "arguments": "{不是 JSON"}}
    assert await apply_chat_tool(store, broken, snapshot=snap, session_id="s") is None
    assert snap.is_empty


# ---------------------------------------------------------------------------
# 摘要侧工具两形态
# ---------------------------------------------------------------------------

async def test_summary_tool_two_forms(store):
    await store.ltm_save_base("slot_1", NPC, "整体印象。", "s")
    with_base = build_summary_tool(await _snapshot(store, "slot_1"))["function"]
    without_base = build_summary_tool(await _snapshot(store, "slot_2"))["function"]

    assert with_base["name"] == SUMMARY_TOOL_NAME
    assert with_base["description"].startswith("更新你对这个玩家的基础印象")
    assert "没有明显变化就不调用" in with_base["description"]
    assert "必须完整保留" in with_base["description"]
    assert "不复述" in with_base["description"]
    assert "（300 字以内）" in with_base["description"]

    assert without_base["description"].startswith("记录你对这个玩家的基础印象")
    assert "根据本次给出的对话内容写" in without_base["description"]
    assert "最大100 字以内" in without_base["description"]
    assert without_base["parameters"]["required"] == ["content"]


async def test_apply_summary_tool_writes_base_once(store):
    assert await apply_summary_tool(
        store, [_call(SUMMARY_TOOL_NAME, {"content": "第一次印象。"})],
        slot_key="slot_1", npc=NPC, session_id="s1",
    ) is True
    assert await store.ltm_get_base("slot_1", NPC) == "第一次印象。"

    # 没有调用＝基础印象保持原样（无明显变化的那一轮）
    assert await apply_summary_tool(store, [], slot_key="slot_1", npc=NPC, session_id="s2") is False
    assert await store.ltm_get_base("slot_1", NPC) == "第一次印象。"

    # 空内容同样不动
    assert await apply_summary_tool(
        store, [_call(SUMMARY_TOOL_NAME, {"content": "  "})],
        slot_key="slot_1", npc=NPC, session_id="s3",
    ) is False
    assert await store.ltm_get_base("slot_1", NPC) == "第一次印象。"


async def test_apply_summary_tool_updates_in_place(store):
    await store.ltm_save_base("slot_1", NPC, "旧印象。", "s1")
    await apply_summary_tool(
        store, [_call(SUMMARY_TOOL_NAME, {"content": "新印象。"})],
        slot_key="slot_1", npc=NPC, session_id="s2",
    )
    assert await store.ltm_get_base("slot_1", NPC) == "新印象。"
    rows = await store.ltm_list("slot_1", NPC)
    assert len(rows) == 1

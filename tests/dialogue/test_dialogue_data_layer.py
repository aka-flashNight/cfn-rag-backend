"""阶段四 B4-4 数据层正式用例（两个覆盖层 + 合并视图 + 写入守卫 + 版本号）。

与工具侧用例（``test_dialogue_tools.py``）的分工：那边锁「模型看到的文本」和工具契约，
这里锁数据层自身的口径——

- 跳过规则：已完成任务与 Agent 委托的任务对话都不进候选，NPC 不说话的组也不进；
- 读的是合并后视图：改过之后再看，删行/改行按保序对齐保留原行字段；
- 修／加／不动三种产出；
- 二次修改就地更新同一条目、``agent_<n>`` 按 NPC 作用域不撞；
- 逐条校验打回（打回的那组不落盘，同批其它组照常生效）；
- 写盘原子性与 ``dialogue_version`` 递增；
- 读盘 fail-closed：坏文件绝不让半份数据糊掉基础对话。

全部用例只碰 tmp_path 下的假数据根与假槽位，不触碰真实存档。
"""

from __future__ import annotations

import json

import pytest
from services.dialogue import overrides as overlay
from services.dialogue import task_text as text_overlay
from services.dialogue.editor import DialogueEditor
from services.dialogue.models import (
    MAX_LINES_PER_GROUP,
    MAX_LINE_TEXT_CHARS,
    PLAYER_CHAR,
    PLAYER_NAME,
    PLAYER_TITLE,
    DialogueGroup,
    DialogueLine,
    player_defaults,
    sanitize_text,
)
from services.dialogue.sources import DialogueSources
from services.dialogue.view import (
    DialogueView,
    build_view,
    is_agent_task_id,
    normalize_finished_ids,
    reset_sources_cache,
)
from services.integration.save_paths import agent_save_override_paths
from tests.dialogue.conftest import NPC, SLOT

OTHER_NPC = "铁匠"  # 只在覆盖层里出现，基础数据里没有它的日常对话文件

_TASK_KEYS = [f"$MAIN_GET_{i}" for i in range(1, 7)]


def _view(env, *, slot_key: str | None = SLOT, finished=(900,)) -> DialogueView:
    """显式给槽位与已完成投影，不依赖全局绑定（数据层用例要能单独跑）。"""
    return DialogueView(
        data_root=env.data_root,
        game_data=env.game_data,
        slot_key=slot_key,
        finished_ids=list(finished),
        sources=DialogueSources(env.data_root),
    )


def _paths(rooted, slot_key: str = SLOT) -> tuple:
    """覆盖层路径。参数可以是夹具给的 env，也可以是构造好的 view（都带 data_root）。"""
    return agent_save_override_paths(rooted.data_root, slot_key)


def _write_overlays(env, dialogue_entries, text_mapping, slot_key: str = SLOT) -> None:
    dialogue_path, text_path = _paths(env, slot_key)
    if dialogue_entries is not None:
        overlay.save_overrides(dialogue_path, dialogue_entries)
    if text_mapping is not None:
        text_overlay.save_task_text_overrides(text_path, text_mapping)


def _line_entry(group_id: str, text: str, *, npc: str = NPC, action: str = "replace") -> overlay.DialogueOverride:
    return overlay.DialogueOverride(
        npc=npc,
        action=action,
        group_id=group_id,
        lines=[DialogueLine(name=npc, title=npc, char=npc, text=text)],
    )


def _edit(number: int, lines: list[dict]) -> dict:
    return {"number": number, "lines": lines}


def _say(text: str, name: str = NPC) -> dict:
    return {"name": name, "text": text}


def _daily_editor(view: DialogueView, npc: str = NPC) -> DialogueEditor:
    """按「日常全量」的编号构造编辑器（工具侧用例已锁过编号本身，这里只取编号）。"""
    return DialogueEditor(view, view.select(npc, ["daily_all"]).index_entries())


# ---------------------------------------------------------------------------
# 跳过规则：哪些组根本不进候选
# ---------------------------------------------------------------------------

def test_finished_and_agent_task_dialogues_are_skipped(dialogue_env):
    """已完成任务与 Agent 委托的任务对话都被跳过（改了玩家也看不到）。"""
    view = _view(dialogue_env)
    candidates = view.task_candidates(NPC)
    ids = {cand.task_id for cand in candidates}

    assert 900 not in ids          # 已完成任务（投影 tasks_finished 命中）
    assert 200001 not in ids       # Agent 委托（id 落在 200001–300000）
    assert ids == {101, 102, 103, 104, 105, 106, 201}

    # 同一份数据，取消「已完成」投影后 900 就回来了——排除的判据确实来自投影
    assert 900 in {c.task_id for c in _view(dialogue_env, finished=()).task_candidates(NPC)}


def test_agent_task_id_bounds():
    """Agent 委托 id 区间的边界（与 TaskRegistry.list_agent_tasks 同口径）。"""
    assert is_agent_task_id(200000) is False
    assert is_agent_task_id(200001) is True
    assert is_agent_task_id(300000) is True
    assert is_agent_task_id(300001) is False
    assert is_agent_task_id("200001") is True
    assert is_agent_task_id(None) is False
    assert is_agent_task_id("主线任务") is False


def test_finished_projection_accepts_both_shapes():
    """完成投影两种形状：id 数组与 id→次数表（次数为 0 不算完成）。"""
    assert normalize_finished_ids([900, "901", None]) == {900, 901}
    assert normalize_finished_ids({"900": 2, "901": 0, "902": "1"}) == {900, 902}
    assert normalize_finished_ids(None) == set()
    assert normalize_finished_ids("900") == set()


def test_group_without_npc_speech_is_skipped(dialogue_env):
    """硬门槛：组内没有该 NPC 说话就不在候选里（基础数据 id 3 那组是旁人说话）。"""
    view = _view(dialogue_env)
    selection = view.select(NPC, ["daily_all"])

    assert [entry["key"] for entry in selection.index_entries()] == ["1", "2"]
    assert "组 id 3" not in selection.render()
    # 门槛看的是行里的说话人，不是块名：别人说话、本人不开口的组不算本人台词
    assert view.sources.group(NPC, "3").speaks(NPC) is False


# ---------------------------------------------------------------------------
# 读的是合并后视图
# ---------------------------------------------------------------------------

def test_daily_groups_read_merged_view_with_overlay_last(dialogue_env):
    """日常对话＝基础 ∪ 本存档覆盖层：基础组原序，改写就地替换，新增组追加在末尾。"""
    _write_overlays(
        dialogue_env,
        [
            _line_entry("2", "覆盖层新说的第二组台词。"),
            _line_entry("9", "新增组台词。", action="append"),
            _line_entry("1", "别的 NPC 的覆盖不影响本 NPC", npc=OTHER_NPC),
        ],
        None,
    )
    groups = {g.group_id: g for g in _view(dialogue_env).daily_groups(NPC)}

    assert list(groups) == ["1", "2", "3", "9"]            # 基础组原序，新增组追加在末尾
    assert groups["1"].lines[0].text == "今天炉子烧得正旺。"  # 未被覆盖的组保持基础数据
    assert groups["2"].lines[0].text == "覆盖层新说的第二组台词。"
    assert groups["2"].from_overlay is True and groups["1"].from_overlay is False
    assert groups["9"].task_requirement == 0                # 新增组无主线门槛，始终可用
    assert groups["3"].lines[0].text == "这组没有测试使说话。"  # 进合并视图，只是不进候选


def test_npc_line_samples_take_all_daily_plus_task(dialogue_env):
    """完成 NPC 的台词参考：日常全量 + 任务取前 task_limit 条，只取本人行。

    任务样本含已完成任务（过往台词），但 Agent 委托永远排除。
    """
    _write_overlays(
        dialogue_env,
        [_line_entry("2", "覆盖层改写后的第二组台词。")],
        None,
    )
    view = _view(dialogue_env)

    daily = ["今天炉子烧得正旺。", "覆盖层改写后的第二组台词。"]
    tasks = [f"主线{i}的接取话，龙鳞记得带来。" for i in range(1, 7)]

    assert view.npc_line_samples(NPC) == daily + tasks
    assert view.npc_line_samples(NPC, task_limit=2) == daily + tasks[:2]
    assert view.npc_line_samples("查无此人") == []

    long_samples = view.npc_line_samples(NPC, task_limit=20)
    assert "已完成任务的接取话。" in long_samples        # 已完成任务也作为过往台词
    assert "委托任务的接取话。" not in long_samples      # Agent 委托永不参考
    assert "给我打把刀。" not in long_samples            # 玩家行
    assert "这组没有测试使说话。" not in long_samples    # 旁人说的行


def test_task_dialogue_reads_merged_text_and_keeps_extra_fields(dialogue_env):
    """任务对话读合并值：覆盖层有就整份用它，且条目附加字段（插图）不丢。"""
    _write_overlays(dialogue_env, None, {
        "$MAIN_GET_1": [{
            "name": NPC, "title": NPC, "char": NPC,
            "text": "改写过的接取话。", "imageurl": "art/main1.png",
        }],
    })
    cand = next(c for c in _view(dialogue_env).task_candidates(NPC) if c.text_key == "$MAIN_GET_1")

    assert [line.text for line in cand.group.lines] == ["改写过的接取话。"]
    assert cand.group.lines[0].extra == {"imageurl": "art/main1.png"}

    # 改写这条任务对话时，附加字段要跟着原行一起留回来（丢了就是删信息）
    view = _view(dialogue_env)
    selection = view.select(NPC, ["task_all"])
    entry = next(e for e in selection.index_entries() if e["key"] == "$MAIN_GET_1")
    outcome = DialogueEditor(view, selection.index_entries()).apply(
        npc=NPC, edits=[_edit(entry["number"], [_say("改写过的接取话。"), _say("路上当心。")])],
    )
    assert outcome.wrote_task_text is True
    _, text_path = _paths(dialogue_env)
    items = text_overlay.load_task_text_overrides(text_path)["$MAIN_GET_1"]
    assert items[0]["imageurl"] == "art/main1.png"
    assert [item["text"] for item in items] == ["改写过的接取话。", "路上当心。"]


def test_deleting_lines_is_accepted(dialogue_env):
    """删行是合法输出：行数少于原组不再打回，按提交的行落盘。"""
    view = _view(dialogue_env)
    editor = _daily_editor(view)
    first = editor.apply(npc=NPC, edits=[_edit(1, [
        _say("今天炉子烧得正旺。"), _say("给我打把刀。", PLAYER_NAME), _say("明天再来。"),
    ])])
    assert first.wrote_dialogue is True

    # 换一个视图实例（模拟下一次委派重新读盘）：此时该组已有 3 行
    again = _daily_editor(_view(dialogue_env))
    second = again.apply(npc=NPC, edits=[_edit(1, [
        _say("今天炉子烧得正旺。"),
    ])])
    assert second.changed is True and second.rejected == []
    dialogue_path, _ = _paths(dialogue_env)
    entry = overlay.load_overrides(dialogue_path)[0]
    assert [line.text for line in entry.lines] == ["今天炉子烧得正旺。"]


def test_deleted_middle_line_keeps_extra_fields_on_the_right_line(dialogue_env):
    """删掉前一行后，后一行的附加字段（插图）必须跟着真正的原行走，不能错位。"""
    _write_overlays(dialogue_env, None, {
        "$MAIN_GET_1": [
            {"name": NPC, "title": NPC, "char": NPC, "text": "先坐下喝一杯。"},
            {
                "name": NPC, "title": NPC, "char": NPC,
                "text": "龙鳞记得带来。", "imageurl": "art/quest1.png",
            },
        ],
    })
    view = _view(dialogue_env)
    selection = view.select(NPC, ["task_all"])
    entry = next(e for e in selection.index_entries() if e["key"] == "$MAIN_GET_1")
    outcome = DialogueEditor(view, selection.index_entries()).apply(
        npc=NPC, edits=[_edit(entry["number"], [_say("龙鳞记得带回来。")])],
    )
    assert outcome.wrote_task_text is True and outcome.rejected == []

    _, text_path = _paths(dialogue_env)
    items = text_overlay.load_task_text_overrides(text_path)["$MAIN_GET_1"]
    assert len(items) == 1
    assert items[0]["text"] == "龙鳞记得带回来。"
    assert items[0]["imageurl"] == "art/quest1.png"


def test_normalized_speaker_still_matches_origin_for_extra_fields(dialogue_env):
    """先归一化再对齐：玩家写成「玩家」、NPC 省略说话人时，原行附加字段仍跟着走。"""
    _write_overlays(dialogue_env, None, {
        "$MAIN_GET_1": [
            {
                "name": PLAYER_NAME, "title": PLAYER_TITLE, "char": PLAYER_CHAR,
                "text": "龙鳞我带来了。", "imageurl": "art/player.png",
            },
            {
                "name": NPC, "title": NPC, "char": NPC,
                "text": "放桌上吧。", "imageurl": "art/npc.png",
            },
        ],
    })
    view = _view(dialogue_env)
    selection = view.select(NPC, ["task_all"])
    entry = next(e for e in selection.index_entries() if e["key"] == "$MAIN_GET_1")
    outcome = DialogueEditor(view, selection.index_entries()).apply(
        npc=NPC,
        edits=[_edit(entry["number"], [
            {"name": "玩家", "text": "龙鳞我带来了。"},
            {"name": "", "text": "放桌上吧。"},
        ])],
    )
    assert outcome.rejected == []

    _, text_path = _paths(dialogue_env)
    items = text_overlay.load_task_text_overrides(text_path)["$MAIN_GET_1"]
    assert [item["name"] for item in items] == [PLAYER_NAME, NPC]
    assert [item.get("imageurl") for item in items] == ["art/player.png", "art/npc.png"]


# ---------------------------------------------------------------------------
# 修／加／不动三种产出
# ---------------------------------------------------------------------------

def test_three_outcomes(dialogue_env):
    """修＝replace 已有组；加＝append 新组；不动＝什么都不落、连版本号都不 bump。"""
    view = _view(dialogue_env)
    editor = _daily_editor(view)
    sync_path = view.data_root / "rag" / "sync_state.json"

    fixed = editor.apply(npc=NPC, edits=[_edit(1, [
        _say("炉子今天歇火。"), _say("给我打把刀。", PLAYER_NAME),
    ])])
    assert fixed.wrote_dialogue is True and fixed.wrote_task_text is False
    assert fixed.changed is True

    added = _daily_editor(view).apply(npc=NPC, edits=[], new_groups=[
        {"lines": [_say("新组第一句。"), _say("好。", PLAYER_NAME)]},
    ])
    assert added.wrote_dialogue is True

    dialogue_path, text_path = _paths(dialogue_env)
    entries = overlay.load_overrides(dialogue_path)
    assert [(e.npc, e.group_id, e.action) for e in entries] == [
        (NPC, "1", "replace"), (NPC, "agent_1", "append"),
    ]
    assert entries[0].base_dialogue_id == "1"
    assert entries[1].base_dialogue_id is None
    assert not text_path.exists()                      # 没碰任务对话就不生成那个文件
    assert json.loads(sync_path.read_text("utf-8"))["dialogue_version"] == 2

    # 不动：没有有效产出 → 不写盘、不 bump 版本号（子 Agent 判断「不需要改」时不该惊动游戏侧）
    before = sync_path.read_text("utf-8")
    idle = _daily_editor(view).apply(npc=NPC, edits=[], new_groups=[])
    assert idle.changed is False and idle.rejected == []
    assert sync_path.read_text("utf-8") == before
    assert len(overlay.load_overrides(dialogue_path)) == 2


def test_new_group_lines_get_defaults_from_npc(dialogue_env):
    """新增组的 Title/Char 由后端补齐：玩家行走 AS2 占位符，NPC 行用名字。"""
    view = _view(dialogue_env)
    outcome = _daily_editor(view).apply(npc=NPC, edits=[], new_groups=[
        {"lines": [_say("新组第一句。"), _say("好。", PLAYER_NAME)]},
    ])
    assert outcome.wrote_dialogue is True

    dialogue_path, _ = _paths(dialogue_env)
    lines = [line.to_subdialogue() for line in overlay.load_overrides(dialogue_path)[0].lines]
    assert lines[0] == {"Name": NPC, "Title": NPC, "Char": NPC, "Text": "新组第一句。"}
    assert lines[1] == {"Name": PLAYER_NAME, "Title": PLAYER_TITLE, "Char": PLAYER_CHAR, "Text": "好。"}
    assert player_defaults() == (PLAYER_TITLE, PLAYER_CHAR)


def test_player_speaker_alias_normalized_to_pc(dialogue_env):
    """模型把玩家写成中文「玩家」时后端归一成 $PC：改已有组与新增组都不打回，落盘走占位符。"""
    view = _view(dialogue_env)
    outcome = _daily_editor(view).apply(npc=NPC, edits=[_edit(1, [
        _say("炉子今天歇火。"), _say("给我打把刀。", "玩家"),
    ])], new_groups=[
        {"lines": [_say("新组第一句。"), _say("好。", "玩家")]},
    ])
    assert outcome.rejected == []

    dialogue_path, _ = _paths(dialogue_env)
    lines = [
        line.to_subdialogue()
        for entry in overlay.load_overrides(dialogue_path)
        for line in entry.lines
    ]
    assert [line["Name"] for line in lines] == [NPC, PLAYER_NAME, NPC, PLAYER_NAME]
    assert lines[1]["Title"] == PLAYER_TITLE and lines[1]["Char"] == PLAYER_CHAR


@pytest.mark.parametrize("alias", ["$PC_TITLE", "（$PC）", "$pc"])
def test_player_speaker_pc_variants_normalized_to_pc(dialogue_env, alias):
    """说话人里含 $PC 但多写了前后缀时也收敛成 $PC——模型会把称号／立绘占位符当名字。"""
    view = _view(dialogue_env)
    outcome = _daily_editor(view).apply(npc=NPC, edits=[_edit(1, [
        _say("炉子今天歇火。"), _say("给我打把刀。", alias),
    ])], new_groups=[
        {"lines": [_say("新组第一句。"), _say("好。", alias)]},
    ])
    assert outcome.rejected == []

    dialogue_path, _ = _paths(dialogue_env)
    lines = [
        line.to_subdialogue()
        for entry in overlay.load_overrides(dialogue_path)
        for line in entry.lines
    ]
    assert [line["Name"] for line in lines] == [NPC, PLAYER_NAME, NPC, PLAYER_NAME]
    assert lines[1]["Title"] == PLAYER_TITLE and lines[1]["Char"] == PLAYER_CHAR


# ---------------------------------------------------------------------------
# 二次修改就地更新同一条目 / agent_<n> 不撞
# ---------------------------------------------------------------------------

def test_upsert_updates_in_place_and_keeps_provenance():
    """同 (npc, group_id) 再次写入＝就地更新：不追加第二条，且没给的字段沿用旧值。"""
    entries = overlay.upsert_override(
        [], npc=NPC, group_id="1", action="replace",
        lines=[DialogueLine(name=NPC, title=NPC, char=NPC, text="第一版。")],
        base_dialogue_id="1", reason="关系变了", source_task_id=900, source_trigger="重要任务",
    )
    first = entries[0]
    again = overlay.upsert_override(
        entries, npc=NPC, group_id="1", action="replace",
        lines=[DialogueLine(name=NPC, title=NPC, char=NPC, text="第二版。")],
    )

    assert len(again) == 1
    updated = again[0]
    assert [line.text for line in updated.lines] == ["第二版。"]
    # 空值不覆盖已有留痕（同一组被多次改写时，最初的触发来源仍查得到）
    assert updated.base_dialogue_id == "1"
    assert updated.reason == "关系变了"
    assert updated.source_task_id == 900
    assert updated.source_trigger == "重要任务"
    assert updated.created_at == first.created_at


def test_repeat_edit_same_group_never_doubles_up(dialogue_env):
    """反复改同一基础组：覆盖层始终一条 replace，合并视图里也只有一组。"""
    view = _view(dialogue_env)
    for text in ("第一版。", "第二版。", "第三版。"):
        outcome = _daily_editor(view).apply(npc=NPC, edits=[_edit(1, [
            _say(text), _say("给我打把刀。", PLAYER_NAME),
        ])])
        assert outcome.wrote_dialogue is True

    dialogue_path, _ = _paths(dialogue_env)
    entries = overlay.load_overrides(dialogue_path)
    assert len(entries) == 1 and entries[0].group_id == "1"
    assert _view(dialogue_env).daily_groups(NPC)[0].lines[0].text == "第三版。"
    # 同 group_id 重复应用不产生重复组（后写的赢）
    merged = overlay.merge_groups(
        view.sources.groups(NPC),
        [_line_entry("1", "甲版。"), _line_entry("1", "乙版。")],
        NPC,
    )
    assert len(merged) == 3
    assert next(g for g in merged if g.group_id == "1").lines[0].text == "乙版。"


def test_agent_group_ids_do_not_collide(dialogue_env):
    """``agent_<n>`` 按 NPC 作用域自增：跨 NPC 各自从 1 起，跨存档也各自独立。"""
    view = _view(dialogue_env)
    ids: list[str] = []
    for npc, count in ((NPC, 2), (OTHER_NPC, 2)):
        for _ in range(count):
            outcome = _daily_editor(view, npc).apply(npc=npc, edits=[], new_groups=[
                {"lines": [_say("新组台词。", npc)]},
            ])
            assert outcome.wrote_dialogue is True
            ids.append(outcome.applied[0].split("日常对话组 ")[1].split("：")[0])
    assert ids == ["agent_1", "agent_2", "agent_1", "agent_2"]

    dialogue_path, _ = _paths(dialogue_env)
    entries = overlay.load_overrides(dialogue_path)
    assert len(entries) == 4 and len({(e.npc, e.group_id) for e in entries}) == 4
    # 各自的合并视图只认自己的组：铁匠读不到测试使的 agent_1
    assert [g.group_id for g in _view(dialogue_env).daily_groups(NPC)] == ["1", "2", "3", "agent_1", "agent_2"]
    assert [g.group_id for g in _view(dialogue_env).daily_groups(OTHER_NPC)] == ["agent_1", "agent_2"]


def test_allocate_group_id_scans_base_and_overlay():
    """分配口径＝该 NPC 的基础组 id ∪ 覆盖层已分配的 ``agent_*``，且与纯数字 id 永不相撞。"""
    base = [
        DialogueGroup(npc=NPC, group_id="1"),
        DialogueGroup(npc=NPC, group_id="agent_7"),   # 基础数据里真出现 agent_7 也不重号
        DialogueGroup(npc=OTHER_NPC, group_id="agent_9"),  # 别的 NPC 不参与本 NPC 的计数
    ]
    assert overlay.allocate_group_id(NPC, base, []) == "agent_8"
    assert overlay.allocate_group_id(NPC, base, [_line_entry("agent_3", "旧记录。")]) == "agent_8"
    assert overlay.allocate_group_id(NPC, base, [_line_entry("agent_12", "旧记录。")]) == "agent_13"
    assert overlay.allocate_group_id(NPC, [], []) == "agent_1"


def test_overlays_are_isolated_per_slot(dialogue_env):
    """覆盖层按存档隔离：换槽位＝换文件，另一个存档看不到本存档的改写。"""
    _daily_editor(_view(dialogue_env)).apply(npc=NPC, edits=[_edit(1, [
        _say("只属于这个存档的改写。"), _say("给我打把刀。", PLAYER_NAME),
    ])])

    other_slot = "slot_dialogue_other"
    assert _view(dialogue_env, slot_key=other_slot).daily_groups(NPC)[0].lines[0].text == "今天炉子烧得正旺。"
    assert _view(dialogue_env).daily_groups(NPC)[0].lines[0].text == "只属于这个存档的改写。"

    other_path, _ = _paths(dialogue_env, other_slot)
    assert not other_path.exists()
    assert overlay.load_overrides(_paths(dialogue_env)[0])[0].group_id == "1"


def test_view_pins_slot_even_after_rebind(dialogue_env, bound_slot):
    """写侧落点用视图自己的槽位，不重读全局绑定：委派期间换档也不会串档。"""
    bound_slot()  # 全局绑到 SLOT
    pinned = build_view(game_data=dialogue_env.game_data, slot_key="slot_pinned")

    outcome = DialogueEditor(pinned, pinned.select(NPC, ["daily_all"]).index_entries()).apply(
        npc=NPC, edits=[_edit(1, [
            _say("钉在原存档的改写。"), _say("给我打把刀。", PLAYER_NAME),
        ])])
    assert outcome.changed is True

    pinned_path, _ = _paths(dialogue_env, "slot_pinned")
    assert overlay.load_overrides(pinned_path)[0].lines[0].text == "钉在原存档的改写。"
    assert not _paths(dialogue_env)[0].exists()


# ---------------------------------------------------------------------------
# 校验失败路径（逐条打回，不落盘）
# ---------------------------------------------------------------------------

def _rejected_text(view: DialogueView, edits: list[dict], new_groups: list[dict] | None = None) -> str:
    """只提交一组，返回唯一的打回原因（同时确认什么都没落盘）。"""
    existed_before = _paths(view)[0].exists()
    outcome = DialogueEditor(view, view.select(NPC, ["daily_all", "task_all"]).index_entries()).apply(
        npc=NPC, edits=edits, new_groups=new_groups or [],
    )
    assert outcome.changed is False
    assert _paths(view)[0].exists() is existed_before     # 打回不产生覆盖层文件
    assert len(outcome.rejected) == 1
    return outcome.rejected[0]


def test_group_level_rejections(dialogue_env):
    """整组打回的三种入口：没台词、超行数上限、越界说话人。"""
    view = _view(dialogue_env)

    assert "没有提交任何台词" in _rejected_text(view, [_edit(1, [])])
    too_many = _rejected_text(view, [_edit(1, [
        _say(f"第{i}句。") for i in range(MAX_LINES_PER_GROUP + 1)
    ])])
    assert f"超过单组上限 {MAX_LINES_PER_GROUP} 行" in too_many
    outsider = _rejected_text(view, [_edit(1, [
        _say("今天炉子烧得正旺。"), _say("给我打把刀。", PLAYER_NAME), _say("替别人说话。", "旁人"),
    ])])
    assert "不在允许范围内" in outsider and "旁人" in outsider


def test_line_level_rejections(dialogue_env):
    """逐行校验：超长/空台词整组打回，并给出具体行号与修正方向。"""
    view = _view(dialogue_env)

    long_text = _rejected_text(view, [_edit(1, [
        _say("很长" * MAX_LINE_TEXT_CHARS), _say("给我打把刀。", PLAYER_NAME),
    ])])
    assert f"超过上限 {MAX_LINE_TEXT_CHARS} 字" in long_text and "第 1 行" in long_text
    assert "修正：" in long_text

    # 空 text 不再在入口被静默丢掉：明确指到行号打回
    blank = _rejected_text(view, [_edit(1, [_say(""), _say("今天炉子烧得正旺。")])])
    assert "第 1 行台词为空" in blank
    assert "修正：" in blank

    # 整组都是空行时同样指到第一行，而不是含糊的「没有提交任何台词」
    assert "第 1 行台词为空" in _rejected_text(view, [_edit(1, [_say(""), _say("   ")])])


def test_unknown_or_stale_numbers(dialogue_env):
    """编号非法、编号不在本次查询结果、编号指向基础数据里已不存在的组——都打回不落盘。"""
    view = _view(dialogue_env)

    assert "缺编号或编号非法" in _rejected_text(view, [_edit("不是数字", [
        _say("今天炉子烧得正旺。"), _say("给我打把刀。", PLAYER_NAME),
    ])])
    assert "不在本次查询结果里" in _rejected_text(view, [_edit(99, [_say("凭空的改写。")])])

    stale = DialogueEditor(view, [{"number": 1, "kind": "daily", "npc": NPC, "key": "77"}])
    outcome = stale.apply(npc=NPC, edits=[_edit(1, [_say("指向已不存在的组。")])])
    assert outcome.changed is False
    assert any("在基础数据里已不存在" in text for text in outcome.rejected)


def test_missing_npc_name_rejects_whole_batch(dialogue_env):
    """缺 NPC 名整批拒绝：覆盖层按 NPC 块名落盘，不知道是谁就不能写。"""
    view = _view(dialogue_env)
    outcome = DialogueEditor(view, view.select(NPC, ["daily_all"]).index_entries()).apply(
        npc="  ", edits=[_edit(1, [_say("改写。"), _say("给我打把刀。", PLAYER_NAME)])],
    )
    assert outcome.changed is False
    assert outcome.rejected == ["缺少 NPC 名，未做任何修改。"]


def test_partial_batch_applies_good_and_rejects_bad(dialogue_env):
    """一批里好组照常生效、坏组单独打回，打回原因逐条回给模型修正。"""
    view = _view(dialogue_env)
    editor = DialogueEditor(view, view.select(NPC, ["daily_all", "task_all"]).index_entries())
    task_entry = next(e for e in editor.index.values() if e["key"] == "$MAIN_GET_2")
    outcome = editor.apply(
        npc=NPC,
        edits=[
            _edit(1, [_say("炉子今天歇火。"), _say("给我打把刀。", PLAYER_NAME)]),
            _edit(2, []),                                                  # 空台词＝整组打回
            _edit(task_entry["number"], [_say("主线2的接取话。")]),
        ],
        new_groups=[{"lines": [_say("新增一组台词。")]}],
    )

    assert len(outcome.applied) == 3 and len(outcome.rejected) == 1
    assert any("没有提交任何台词" in text for text in outcome.rejected)
    dialogue_path, text_path = _paths(dialogue_env)
    assert [e.group_id for e in overlay.load_overrides(dialogue_path)] == ["1", "agent_1"]
    mapping = text_overlay.load_task_text_overrides(text_path)
    assert list(mapping) == ["$MAIN_GET_2"]
    assert mapping["$MAIN_GET_2"][0]["text"] == "主线2的接取话。"
    # 打回的那组保持基础数据原样，没被半份改动污染
    assert _view(dialogue_env).daily_groups(NPC)[1].lines[0].text == "龙鳞可不好弄。"


# ---------------------------------------------------------------------------
# 写盘 + 版本号
# ---------------------------------------------------------------------------

def test_save_bumps_dialogue_version_and_preserves_other_keys(dialogue_env, monkeypatch):
    """落盘后 dialogue_version 递增（游戏侧据此使对话缓存失效），其它键原样保留。"""
    import services.agent_tools.task_tools as task_tools

    sync_path = dialogue_env.data_root / "rag" / "sync_state.json"
    sync_path.parent.mkdir(parents=True, exist_ok=True)
    sync_path.write_text(json.dumps({"task_version": 7, "text_version": 3}), encoding="utf-8")

    view = _view(dialogue_env)
    for text in ("第一版。", "第二版。"):
        _daily_editor(view).apply(npc=NPC, edits=[_edit(1, [
            _say(text), _say("给我打把刀。", PLAYER_NAME),
        ])])

    doc = json.loads(sync_path.read_text("utf-8"))
    assert doc["dialogue_version"] == 2
    assert doc["task_version"] == 7 and doc["text_version"] == 3   # 同一文件，别的键不能丢

    # 版本号 bump 失败绝不能报错：覆盖层已经写成了，报错只会让模型重交、把追加行写重
    def _boom(_data_root):
        raise OSError("sync_state 不可写")

    monkeypatch.setattr(task_tools, "bump_dialogue_version", _boom)
    outcome = _daily_editor(view).apply(npc=NPC, edits=[], new_groups=[
        {"lines": [_say("新增一组。")]}])
    assert outcome.changed is True and outcome.rejected == []
    dialogue_path, _ = _paths(dialogue_env)
    # 前面两次改的是同一组（1 条 replace），这里再加一条 append
    assert [e.group_id for e in overlay.load_overrides(dialogue_path)] == ["1", "agent_1"]
    assert json.loads(sync_path.read_text("utf-8"))["dialogue_version"] == 2   # 版本号没 bump 上去


def test_atomic_write_leaves_no_temp_files(dialogue_env):
    """原子写：覆盖层目录里不留 .tmp 残片，两份覆盖层各自独立落盘。"""
    view = _view(dialogue_env)
    editor = DialogueEditor(view, view.select(NPC, ["daily_all", "task_all"]).index_entries())
    task_entry = next(e for e in editor.index.values() if e["key"] == "$MAIN_GET_3")
    editor.apply(npc=NPC, edits=[
        _edit(1, [_say("日常改写。"), _say("给我打把刀。", PLAYER_NAME)]),
        _edit(task_entry["number"], [_say("任务改写。")]),
    ])

    slot_dir = _paths(dialogue_env)[0].parent
    names = sorted(p.name for p in slot_dir.iterdir())
    assert names == ["dialogue_overrides.json", "task_text_overrides.json"]
    for name in names:
        json.loads((slot_dir / name).read_text("utf-8"))            # 落盘即合法 JSON


def test_save_refuses_illegal_slot_key(dialogue_env, bound_slot):
    """槽位键非法（含路径分隔符）＝拒绝写盘并复位产出，绝不回退到共享侧。"""
    bound_slot()
    view = _view(dialogue_env, slot_key="../evil")
    outcome = _daily_editor(view).apply(npc=NPC, edits=[_edit(1, [
        _say("改了也不该落盘。"), _say("给我打把刀。", PLAYER_NAME),
    ])])

    assert outcome.changed is False
    assert any("存档槽位键非法" in text for text in outcome.rejected)
    assert not (dialogue_env.data_root / "evil").exists()
    assert not (dialogue_env.data_root / "rag" / "agent_saves").exists()


# ---------------------------------------------------------------------------
# 读盘 fail-closed：坏文件不能糊掉基础对话
# ---------------------------------------------------------------------------

def test_broken_dialogue_overlay_reads_as_empty(tmp_path):
    """文件不存在＝无增量；坏 JSON／顶层非对象／缺数组都按空处理。"""
    missing = tmp_path / "dialogue_overrides.json"
    assert overlay.load_overrides(missing) == []

    broken = tmp_path / "b.json"
    broken.write_text("{不是合法 JSON", encoding="utf-8")
    assert overlay.load_overrides(broken) == []

    not_object = tmp_path / "c.json"
    not_object.write_text("[]", encoding="utf-8")
    assert overlay.load_overrides(not_object) == []

    no_array = tmp_path / "d.json"
    no_array.write_text(json.dumps({"overrides": {"npc": NPC}}), encoding="utf-8")
    assert overlay.load_overrides(no_array) == []


def test_invalid_overlay_entries_are_dropped(tmp_path):
    """字段不全、action 非法、无有效台词的条目逐条忽略；合法条目不受牵连。"""
    path = tmp_path / "dialogue_overrides.json"
    path.write_text(json.dumps({"overrides": [
        {"npc": "", "action": "replace", "group_id": "1", "subdialogues": [{"Name": NPC, "Text": "无主"}]},
        {"npc": NPC, "action": "delete", "group_id": "1", "subdialogues": [{"Name": NPC, "Text": "非法动作"}]},
        {"npc": NPC, "action": "replace", "group_id": "", "subdialogues": [{"Name": NPC, "Text": "无 id"}]},
        {"npc": NPC, "action": "replace", "group_id": "1", "subdialogues": []},
        {"npc": NPC, "action": "replace", "group_id": "2", "subdialogues": [
            {"Name": NPC, "Title": NPC, "Char": NPC, "Text": "合法一条。"},
            {"Name": NPC, "Title": NPC, "Char": NPC, "Text": "   "},          # 空文本行忽略
        ]},
        "不是对象",
    ]}, ensure_ascii=False), encoding="utf-8")

    entries = overlay.load_overrides(path)
    assert [(e.npc, e.group_id) for e in entries] == [(NPC, "2")]
    assert [line.text for line in entries[0].lines] == ["合法一条。"]


def test_replace_of_vanished_group_is_ignored(dialogue_env):
    """覆盖层引用了基础数据里已不存在的组＝记警告并忽略该条（不凭空造组）。"""
    _write_overlays(dialogue_env, [
        _line_entry("88", "指向不存在的组。"),
        _line_entry("2", "有效改写。"),
    ], None)
    groups = _view(dialogue_env).daily_groups(NPC)

    assert [g.group_id for g in groups] == ["1", "2", "3"]
    assert groups[1].lines[0].text == "有效改写。"


def test_broken_task_text_overlay_reads_as_empty(dialogue_env):
    """任务文本覆盖层同样 fail-closed：坏文件＝没有覆盖，基础文本照常读。"""
    _, text_path = _paths(dialogue_env)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("{坏文件", encoding="utf-8")

    assert text_overlay.load_task_text_overrides(text_path) == {}
    assert [c.text_key for c in _view(dialogue_env).task_candidates(NPC)] == _TASK_KEYS + ["$SIDE_GET_1"]

    text_path.write_text(json.dumps({
        "不带美元符号的键": [{"name": NPC, "text": "不该被采信。"}],
        "$BAD_NOT_LIST": "不是数组",
        "$BAD_EMPTY": [],
        "$BAD_ITEMS": ["不是对象"],
        "$MAIN_GET_1": [{"name": NPC, "title": NPC, "char": NPC, "text": "合法覆盖。"}],
    }, ensure_ascii=False), encoding="utf-8")
    mapping = text_overlay.load_task_text_overrides(text_path)
    assert list(mapping) == ["$MAIN_GET_1"]
    assert _view(dialogue_env).task_candidates(NPC)[0].group.lines[0].text == "合法覆盖。"


def test_set_key_refuses_non_text_keys_and_empty_lines():
    """只收 ``$`` 开头的文本 key；空台词不写入（非文本 key 会污染游戏侧文本表）。"""
    line = DialogueLine(name=NPC, title=NPC, char=NPC, text="台词。")

    assert text_overlay.set_key({}, "不带美元符号", [line]) == {}
    assert text_overlay.set_key({"$KEEP": []}, "$", [line]) == {"$KEEP": []}
    assert text_overlay.set_key({}, "$NEW_KEY", []) == {}

    written = text_overlay.set_key({}, "$NEW_KEY", [line])
    assert written == {"$NEW_KEY": [{"name": NPC, "title": NPC, "char": NPC, "text": "台词。"}]}
    assert text_overlay.merged_value("$NEW_KEY", [{"text": "基础值"}], written) == written["$NEW_KEY"]
    assert text_overlay.merged_value("$OTHER", [{"text": "基础值"}], written) == [{"text": "基础值"}]


# ---------------------------------------------------------------------------
# 条数上限与控制字符
# ---------------------------------------------------------------------------

def test_enforce_cap_drops_oldest_appends_only():
    """超上限先淘汰最旧的 append，replace 不淘汰；append 淘汰完仍超限时保留全部不丢数据。"""
    entries = [
        _line_entry(str(i), f"改写台词{i}。") if i % 3 == 0 else
        _line_entry(f"agent_{i}", f"新增台词{i}。", action="append")
        for i in range(overlay.MAX_OVERRIDES_PER_FILE + 1)
    ]
    trimmed = overlay.enforce_cap(entries)
    kept_ids = {e.group_id for e in trimmed}

    assert len(trimmed) == overlay.MAX_OVERRIDES_PER_FILE
    assert "agent_1" not in kept_ids                        # 最旧的 append 先走
    assert entries[0].group_id in kept_ids                  # replace 一条都不淘汰
    assert len([e for e in trimmed if e.action == "replace"]) == \
        len([e for e in entries if e.action == "replace"])
    assert trimmed[-1].group_id == entries[-1].group_id     # 淘汰从头部按顺序走

    only_replaces = [
        _line_entry(str(i), f"改写台词{i}。") for i in range(overlay.MAX_OVERRIDES_PER_FILE + 5)
    ]
    assert len(overlay.enforce_cap(only_replaces)) == len(only_replaces)  # 无可淘汰项就不丢数据
    assert len(overlay.enforce_cap(entries[:50])) == 50                   # 未超限原样返回


def test_control_characters_never_reach_disk(dialogue_env):
    """XML 非法字符在写盘前被滤净（宿主 XmlDocument 会因 \\x00 解析失败）。"""
    assert sanitize_text("  你好\x00\x08\n世界  ") == "你好\n世界"
    assert sanitize_text(None) == ""
    assert sanitize_text("保留\t制表") == "保留\t制表"

    view = _view(dialogue_env)
    outcome = _daily_editor(view).apply(npc=NPC, edits=[_edit(1, [
        _say("今天炉子歇火了\x00\x0b。"), _say("给我打把刀。", PLAYER_NAME),
    ])])
    assert outcome.wrote_dialogue is True
    dialogue_path, _ = _paths(dialogue_env)
    text = overlay.load_overrides(dialogue_path)[0].lines[0].text
    assert "\x00" not in text and "\x0b" not in text
    assert text == "今天炉子歇火了。"


def test_sources_cache_resets_between_data_roots(dialogue_env):
    """共享读取器按 data_root 缓存，测试内可清；同一目录内基础 XML 变了要重解析。"""
    reset_sources_cache()
    view = build_view(game_data=dialogue_env.game_data, slot_key=SLOT)
    assert [g.group_id for g in view.sources.groups(NPC)] == ["1", "2", "3"]

    extra = DialogueSources(dialogue_env.data_root)
    assert extra.npc_names() == [NPC]
    assert extra.group(NPC, "999") is None
    assert [g.group_id for g in extra.iter_all_groups()] == ["1", "2", "3"]

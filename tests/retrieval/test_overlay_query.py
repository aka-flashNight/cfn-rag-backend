"""查询期覆盖层替换（作者裁定 2026-10-17）：检索命中之后才把改写文本换进去。

索引只装基础语料（覆盖层不进向量库，见 test_task_overlay.py），所以分词与打分始终按
基础文本算，节点 id 与分数在替换前后保持不变；被换的是「给模型看的正文」。
覆盖层新增的内容（append 的新对话组、Agent 新发布的任务）不在索引里，这里也不补。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.dialogue import overrides as dialogue_overlay
from services.dialogue import task_text as task_text_overlay
from services.dialogue.models import DialogueLine
from services.integration.state import get_integration_state
from services.retrieval import loader as loader_mod
from services.retrieval.hybrid import RetrievalBundle, ScoredNode
from services.retrieval.overlay import apply_slot_overrides, reset_cache_for_tests
from services.retrieval.store import Node

NPC = "铁匠"
PLAYER = "$PC"
SLOT = "slot_overlay_query"

_LIST_XML = "<list><items>npc_smith.xml</items></list>"

_NPC_XML = """<root>
  <Dialogues>
    <Name>铁匠</Name>
    <Dialogue id="1">
      <TaskRequirement>0</TaskRequirement>
      <SubDialogue><Name>铁匠</Name><Title>铁匠</Title><Char>铁匠</Char><Text>今天炉子烧得正旺。</Text></SubDialogue>
      <SubDialogue><Name>$PC</Name><Title>$PC_TITLE</Title><Char>$PC_CHAR</Char><Text>给我打把刀。</Text></SubDialogue>
    </Dialogue>
    <Dialogue id="2">
      <TaskRequirement>0</TaskRequirement>
      <SubDialogue><Name>铁匠</Name><Title>铁匠</Title><Char>铁匠#微笑</Char><Text>龙鳞可不好弄。</Text></SubDialogue>
    </Dialogue>
  </Dialogues>
</root>
"""


def _say(text: str, name: str = NPC) -> DialogueLine:
    return DialogueLine(name=name, title=name, char=name, text=text)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _scored(node_id: str, text: str, node_type: str, character: str) -> ScoredNode:
    return ScoredNode(
        node=Node(id=node_id, text=text, type=node_type, character=character),
        dense_score=0.42,
        fused_score=0.31,
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_cache_for_tests()
    yield
    reset_cache_for_tests()


@pytest.fixture
def resources(tmp_path, monkeypatch) -> Path:
    """tmp 资源树（基础对话 + 一条任务两段台词），并把 loader 的资源根指过来。"""
    dialogues_dir = tmp_path / "data" / "dialogues"
    dialogues_dir.mkdir(parents=True)
    (dialogues_dir / "list.xml").write_text(_LIST_XML, encoding="utf-8")
    (dialogues_dir / "npc_smith.xml").write_text(_NPC_XML, encoding="utf-8")
    _write_json(
        tmp_path / "data" / "task" / "mainline_tasks.json",
        {"tasks": [{"id": 101, "get_conversation": "$T101_GET", "finish_conversation": "$T101_FINISH"}]},
    )
    _write_json(
        tmp_path / "data" / "task" / "text" / "mainline_text.json",
        {
            "$T101_GET": [
                {"name": NPC, "text": "拿废铁来。"},
                {"name": NPC, "text": "动作快点。"},
            ],
            "$T101_FINISH": [{"name": NPC, "text": "干得不错。"}],
        },
    )
    monkeypatch.setattr(loader_mod, "_resources_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def bound(resources):
    """绑定到 resources 下的测试槽，返回该槽两个覆盖层文件的写入器。"""
    state = get_integration_state()
    state.bind(slot_key=SLOT)
    dialogue_path, text_path = (
        resources / "data" / "rag" / "agent_saves" / SLOT / "dialogue_overrides.json",
        resources / "data" / "rag" / "agent_saves" / SLOT / "task_text_overrides.json",
    )

    def _write(overrides: list[dialogue_overlay.DialogueOverride] | None = None,
               task_mapping: dict[str, list[dict]] | None = None) -> None:
        if overrides is not None:
            dialogue_overlay.save_overrides(dialogue_path, overrides)
        if task_mapping is not None:
            task_text_overlay.save_task_text_overrides(text_path, task_mapping)

    yield _write
    state.unbind()


def _bundle(dialogue_ids=(), task_ids=()) -> RetrievalBundle:
    """按节点 id 造一个检索结果（对话与任务各进一个池）。"""
    base_text = {
        "dialogue-1": "今天炉子烧得正旺。\n给我打把刀。",
        "dialogue-2": "龙鳞可不好弄。",
        "task-101-get-0": "拿废铁来。",
        "task-101-get-1": "动作快点。",
        "task-101-finish-0": "干得不错。",
    }
    return RetrievalBundle(
        npc_dialogue=[
            _scored(node_id, base_text[node_id], "dialogue", NPC.lower()) for node_id in dialogue_ids
        ],
        npc_task=[
            _scored(node_id, base_text[node_id], "task", NPC.lower()) for node_id in task_ids
        ],
    )


def _by_id(bundle: RetrievalBundle) -> dict[str, ScoredNode]:
    return {sn.node.id: sn for sn in (*bundle.npc_dialogue, *bundle.npc_task)}


# ---------------------------------------------------------------------------
# 日常对话覆盖层
# ---------------------------------------------------------------------------

def test_unbound_bundle_is_untouched(resources):
    bundle = _bundle(dialogue_ids=["dialogue-1"])
    assert apply_slot_overrides(bundle) is bundle
    assert bundle.npc_dialogue[0].node.text == "今天炉子烧得正旺。\n给我打把刀。"


def test_replaced_group_swaps_text_but_keeps_id_and_scores(bound):
    bound(overrides=[dialogue_overlay.DialogueOverride(
        npc=NPC, action="replace", group_id="1",
        lines=[_say("炉子歇了，明天再来。"), _say("给我打把刀。", PLAYER)],
    )])
    bundle = apply_slot_overrides(_bundle(dialogue_ids=["dialogue-1", "dialogue-2"]))

    picked = _by_id(bundle)
    assert set(picked) == {"dialogue-1", "dialogue-2"}
    assert picked["dialogue-1"].node.text == "炉子歇了，明天再来。"
    assert picked["dialogue-2"].node.text == "龙鳞可不好弄。"  # 未改写的组保持基础文本
    # 分数与节点 id 不因替换而漂移：打分始终按基础语料算
    assert picked["dialogue-1"].dense_score == 0.42
    assert picked["dialogue-1"].fused_score == 0.31


def test_group_reduced_to_player_lines_drops_node(bound):
    """改写后整组只剩玩家台词＝没有 NPC 可归的正文，该节点退出结果而不是留下空行。"""
    bound(overrides=[dialogue_overlay.DialogueOverride(
        npc=NPC, action="replace", group_id="1", lines=[_say("给我打把刀。", PLAYER)],
    )])
    bundle = apply_slot_overrides(_bundle(dialogue_ids=["dialogue-1", "dialogue-2"]))
    assert set(_by_id(bundle)) == {"dialogue-2"}


def test_appended_group_is_not_in_results(bound):
    """覆盖层新增的组不在索引里，查询期也无从替换——这是把高频写盘挡在索引外的已知代价。"""
    bound(overrides=[dialogue_overlay.DialogueOverride(
        npc=NPC, action="append", group_id="agent_1", lines=[_say("新增的一组台词。")],
    )])
    bundle = apply_slot_overrides(_bundle(dialogue_ids=["dialogue-1"]))
    assert [sn.node.id for sn in bundle.npc_dialogue] == ["dialogue-1"]


def test_unknown_group_id_keeps_base_text(bound):
    """覆盖层引用了基础数据没有的组 id：对应不到节点 id，整批原样返回。"""
    bound(overrides=[dialogue_overlay.DialogueOverride(
        npc=NPC, action="replace", group_id="999", lines=[_say("凭空的改写。")],
    )])
    bundle = apply_slot_overrides(_bundle(dialogue_ids=["dialogue-1"]))
    assert bundle.npc_dialogue[0].node.text == "今天炉子烧得正旺。\n给我打把刀。"


# ---------------------------------------------------------------------------
# 任务文本覆盖层
# ---------------------------------------------------------------------------

def test_task_text_override_replaces_by_line_index(bound):
    bound(task_mapping={"$T101_GET": [
        {"name": NPC, "text": "先把废铁搬过来。"},
        {"name": NPC, "text": "手脚麻利些。"},
    ]})
    picked = _by_id(apply_slot_overrides(_bundle(task_ids=["task-101-get-0", "task-101-get-1"])))
    assert picked["task-101-get-0"].node.text == "先把废铁搬过来。"
    assert picked["task-101-get-1"].node.text == "手脚麻利些。"
    assert picked["task-101-get-0"].node.id == "task-101-get-0"


def test_task_text_shortened_drops_lines_beyond_override(bound):
    """整段换短了：原序号行在新数组里不存在，该节点退出结果，不去借别行的文本。"""
    bound(task_mapping={"$T101_GET": [{"name": NPC, "text": "只留一句。"}]})
    bundle = apply_slot_overrides(_bundle(task_ids=["task-101-get-0", "task-101-get-1"]))
    assert [sn.node.id for sn in bundle.npc_task] == ["task-101-get-0"]
    assert bundle.npc_task[0].node.text == "只留一句。"


def test_task_text_turned_player_line_drops_node(bound):
    bound(task_mapping={"$T101_GET": [{"name": PLAYER, "title": "$PC_TITLE", "char": "$PC_CHAR",
                                      "text": "玩家代说的一句。"}]})
    bundle = apply_slot_overrides(_bundle(task_ids=["task-101-get-0"]))
    assert bundle.npc_task == []


# ---------------------------------------------------------------------------
# 其它池 / 失败口径
# ---------------------------------------------------------------------------

def test_non_dialogue_pools_pass_through(bound):
    bound(overrides=[dialogue_overlay.DialogueOverride(
        npc=NPC, action="replace", group_id="1", lines=[_say("改写。")],
    )])
    bundle = _bundle(dialogue_ids=["dialogue-1"])
    bundle.world_lore = [_scored("lore-1", "世界观条目", "lore", None)]
    apply_slot_overrides(bundle)
    assert bundle.world_lore[0].node.text == "世界观条目"


def test_broken_overlay_file_falls_back_to_base_text(bound, resources):
    """覆盖层文件坏了按空处理：检索退回基础文本，绝不因覆盖失败打断回合。"""
    bound(overrides=[dialogue_overlay.DialogueOverride(
        npc=NPC, action="replace", group_id="1", lines=[_say("改写。")],
    )])
    (resources / "data" / "rag" / "agent_saves" / SLOT / "dialogue_overrides.json").write_text(
        "{ not json", encoding="utf-8"
    )
    bundle = apply_slot_overrides(_bundle(dialogue_ids=["dialogue-1"]))
    assert bundle.npc_dialogue[0].node.text == "今天炉子烧得正旺。\n给我打把刀。"


def test_none_bundle_returns_none():
    assert apply_slot_overrides(None) is None

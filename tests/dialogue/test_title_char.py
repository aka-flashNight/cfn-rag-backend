"""Title / Char 的后端补足额（模型只给 name / emotion / text）。

覆盖：Title 四级回落（本组原值 → 基础对话数据 → NPC 设定首称号 → 说话人名）、
同人多 Title 取票数最多、情绪归一（白名单 / 超长 / 串进整段 Char）、
Char 只在有合法情绪时拼后缀（抄回的行不被削掉既有表情）、玩家行挂 $PC_CHAR、
任务文本写盘与日常对话写盘同一口径。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.agent_tools.task_tools import write_confirmed_agent_task_files
from services.dialogue.models import compose_char, normalize_emotion
from services.dialogue.sources import resolve_speaker_title, speaker_emotions
from services.dialogue.view import reset_sources_cache, shared_sources
from services.game_data.models import Task
from services.integration.state import get_integration_state
from services.npc.manager import NPCManager, NPCState, set_npc_manager
from services.tools.base import ToolContext, _inherit_ctx
from services.tools.query.search_npc_dialogues import tool as search_tool
from services.tools.task.edit_npc_dialogues import tool as edit_tool

NPC = "铁匠"
SLOT = "slot_title_char"

_LIST_XML = "<list><items>npc_title.xml</items></list>"

_NPC_XML = """<root>
  <Dialogues>
    <Name>铁匠</Name>
    <Dialogue id="1">
      <TaskRequirement>0</TaskRequirement>
      <SubDialogue><Name>铁匠</Name><Title>炉边老板</Title><Char>铁匠#严肃</Char><Text>废铁收来。</Text></SubDialogue>
      <SubDialogue><Name>$PC</Name><Title>$PC_TITLE</Title><Char>$PC_CHAR</Char><Text>给我打把刀。</Text></SubDialogue>
      <SubDialogue><Name>学徒</Name><Title>打下手的</Title><Char>学徒</Char><Text>师父说得对。</Text></SubDialogue>
      <SubDialogue><Name>路人甲</Name><Title></Title><Char>路人甲</Char><Text>路过一句。</Text></SubDialogue>
    </Dialogue>
    <Dialogue id="2">
      <TaskRequirement>0</TaskRequirement>
      <SubDialogue><Name>学徒</Name><Title>小学徒</Title><Char>学徒</Char><Text>这句在前。</Text></SubDialogue>
      <SubDialogue><Name>学徒</Name><Title>小学徒</Title><Char>学徒</Char><Text>这句在后。</Text></SubDialogue>
    </Dialogue>
  </Dialogues>
</root>
"""


@pytest.fixture
def env(tmp_path):
    """最小数据根：两个日常组 ＋ 假注册表；NPC 设定里首称号与基础数据故意不同。"""
    data_root = tmp_path / "data"
    dialogues = data_root / "dialogues"
    dialogues.mkdir(parents=True)
    (dialogues / "list.xml").write_text(_LIST_XML, encoding="utf-8")
    (dialogues / "npc_title.xml").write_text(_NPC_XML, encoding="utf-8")

    tasks = [Task(
        id=101, title="$MAIN_TITLE_1", chain="主线#1",
        get_conversation="$MAIN_GET_1", get_npc=NPC,
    )]
    texts = {
        "$MAIN_TITLE_1": "收废铁",
        "$MAIN_GET_1": [{
            "name": NPC, "title": "炉边老板", "char": NPC, "text": "废铁收来，别掺沙。",
        }],
    }
    game_data = SimpleNamespace(
        data_root=data_root,
        tasks=SimpleNamespace(list_all_tasks=lambda: tasks),
        task_texts=SimpleNamespace(
            all=lambda: texts, resolve_str=lambda key: texts.get(key, key),
        ),
    )

    reset_sources_cache()
    set_npc_manager(NPCManager(
        state={
            # 基础数据用「炉边老板」，设定里首称号是「警长」——玩家看到的铭牌取基础数据
            NPC: NPCState(titles=["警长", "铁匠"], emotions=["严肃", "微笑"]),
            # 基础数据里没有这个人，只能回落到设定首称号
            "路人甲": NPCState(titles=["过客"], emotions=[]),
        },
        state_path=tmp_path / "npc_state_db.json",
    ))
    yield SimpleNamespace(data_root=data_root, game_data=game_data)
    set_npc_manager(None)
    reset_sources_cache()


@pytest.fixture
def bound_slot():
    state = get_integration_state()
    state.bind(SLOT, progress={"tasks_finished": []})
    yield SLOT
    state.unbind()


@pytest.fixture
def unbound():
    """任务写盘用例走共享文件：状态单例可能被别的模块留在已绑定态。"""
    state = get_integration_state()
    state.unbind()
    yield state
    state.unbind()


def _ctx(env) -> ToolContext:
    return ToolContext(npc_name=NPC, game_data=env.game_data)


# ---------------------------------------------------------------------------
# Title 四级回落
# ---------------------------------------------------------------------------

def test_title_resolution_levels(env):
    sources = shared_sources(Path(env.data_root))
    # 1. 本次改动范围内该说话人已有的 Title 最权威
    assert resolve_speaker_title(NPC, sources, own="旧铭牌") == "旧铭牌"
    # 2. 基础对话数据优先于 NPC 设定的首称号（两者实测并不总相等）
    assert resolve_speaker_title(NPC, sources) == "炉边老板"
    # 同一说话人有多个 Title 时取出现最多的（小学徒×2 > 打下手的×1）
    assert resolve_speaker_title("学徒", sources) == "小学徒"
    # 3. 基础数据查不到才回落到设定首称号
    assert resolve_speaker_title("路人甲", sources) == "过客"
    # 4. 全都查不到用说话人名兜底，绝不留空
    assert resolve_speaker_title("查无此人", sources) == "查无此人"
    # 玩家行的 Title 是占位符，不进基础数据表
    assert resolve_speaker_title("$PC", sources) == "$PC"


def test_emotion_normalization(env):
    assert speaker_emotions(NPC) == ["严肃", "微笑"]
    assert speaker_emotions("查无此人") == []
    # 白名单内保留；白名单外削成无表情（免得拼出游戏里没有的帧标签）
    assert normalize_emotion("严肃", speaker_emotions(NPC)) == "严肃"
    assert normalize_emotion("暴怒", speaker_emotions(NPC)) == ""
    # 模型把整串 Char 塞进来时只取末段，否则拼出「名#名#情绪」
    assert normalize_emotion("铁匠#微笑", speaker_emotions(NPC)) == "微笑"
    # 整句话与空值都不是情绪标签
    assert normalize_emotion("他皱着眉头看了你一眼", speaker_emotions(NPC)) == ""
    assert normalize_emotion("", speaker_emotions(NPC)) == ""
    # 没登记情绪的 NPC 放行短标签，不至于把台词全削成无表情
    assert normalize_emotion("得意", speaker_emotions("查无此人")) == "得意"
    assert compose_char(NPC, "严肃") == "铁匠#严肃"
    assert compose_char(NPC, "") == NPC


# ---------------------------------------------------------------------------
# 日常对话写盘
# ---------------------------------------------------------------------------

def _edit_ctx(env):
    res = search_tool.run({"scopes": ["daily_all"]}, _ctx(env))
    return _inherit_ctx(_ctx(env), res), {e["number"]: e for e in res.dialogue_index}


def _daily_lines(env, group_id: str = "1") -> list[dict]:
    from services.dialogue.overrides import load_overrides
    from services.integration.save_paths import agent_save_override_paths

    path, _ = agent_save_override_paths(env.data_root, SLOT)
    entry = next(e for e in load_overrides(path) if e.group_id == group_id)
    return [line.to_subdialogue() for line in entry.lines]


def test_copied_row_keeps_char_and_title(env, bound_slot):
    """原样抄回的行：Title 与既有表情都不动，情绪给了才换表情。"""
    ctx, index = _edit_ctx(env)
    number = next(n for n, e in index.items() if e["kind"] == "daily" and e["key"] == "1")
    res = edit_tool.run({"edits": [{
        "number": number,
        "lines": [
            {"name": NPC, "text": "废铁收来。"},
            {"name": "$PC", "text": "给我打把刀。"},
            {"name": "学徒", "text": "师父说得对。"},
            {"name": "路人甲", "text": "路过一句。"},
        ],
    }]}, ctx)
    assert json.loads(res.result_json)["status"] == "ok"
    lines = _daily_lines(env)
    assert lines[0]["Title"] == "炉边老板"
    assert lines[0]["Char"] == "铁匠#严肃"       # 没给情绪＝沿用原表情，不被削平
    assert lines[1] == {
        "Name": "$PC", "Title": "$PC_TITLE", "Char": "$PC_CHAR", "Text": "给我打把刀。",
    }
    assert lines[2]["Title"] == "打下手的"       # 本组原值优先于全局票数
    assert lines[2]["Char"] == "学徒"
    # 基础数据里这行 Title 是空的 → 回落到 NPC 设定的首称号，不是拿名字当称号
    assert lines[3]["Title"] == "过客"


def test_emotion_composes_char(env, bound_slot):
    """给了合法情绪＝后端拼 Char；玩家行的表情挂在 $PC_CHAR 上。"""
    ctx, index = _edit_ctx(env)
    number = next(n for n, e in index.items() if e["kind"] == "daily" and e["key"] == "1")
    res = edit_tool.run({"edits": [{
        "number": number,
        "lines": [
            {"name": NPC, "emotion": "暴怒", "text": "废铁收来。"},
            {"name": "$PC", "emotion": "犹豫", "text": "给我打把刀。"},
            {"name": "学徒", "emotion": "微笑", "text": "师父说得对。"},
            {"name": "路人甲", "text": "路过一句。"},
            {"name": NPC, "emotion": "微笑", "text": "再打一把。"},
        ],
    }]}, ctx)
    assert json.loads(res.result_json)["status"] == "ok"
    lines = _daily_lines(env)
    # 「暴怒」不在白名单：这条情绪作废，Char 沿用原值
    assert lines[0]["Char"] == "铁匠#严肃"
    assert lines[1]["Char"] == "$PC_CHAR#犹豫"
    assert lines[1]["Title"] == "$PC_TITLE"
    # 学徒没登记情绪白名单 → 短标签放行，不至于全削成无表情
    assert lines[2]["Char"] == "学徒#微笑"
    assert lines[4]["Char"] == "铁匠#微笑"


def test_new_row_title_from_base_data(env, bound_slot):
    """组尾新增行：Title 取本组该说话人的原值，Char 取同说话人末次出现的值。"""
    ctx, index = _edit_ctx(env)
    number = next(n for n, e in index.items() if e["kind"] == "daily" and e["key"] == "1")
    res = edit_tool.run({"edits": [{
        "number": number,
        "lines": [
            {"name": NPC, "text": "废铁收来。"},
            {"name": "$PC", "text": "给我打把刀。"},
            {"name": "学徒", "text": "师父说得对。"},
            {"name": "路人甲", "text": "路过一句。"},
            {"name": NPC, "emotion": "微笑", "text": "明天再来。"},
            {"name": "学徒", "text": "我也走了。"},
        ],
    }]}, ctx)
    assert json.loads(res.result_json)["status"] == "ok"
    lines = _daily_lines(env)
    assert lines[4]["Title"] == "炉边老板" and lines[4]["Char"] == "铁匠#微笑"
    # 本组里学徒的 Title 是「打下手的」，比全局票数「小学徒」更权威
    assert lines[5]["Title"] == "打下手的" and lines[5]["Char"] == "学徒"


# ---------------------------------------------------------------------------
# 任务文本写盘
# ---------------------------------------------------------------------------

def test_task_text_item_title_and_char(env, unbound):
    """任务对话：模型给的 title 一律不读，Title 后端查、Char 后端拼。"""
    _, task_id = write_confirmed_agent_task_files(
        draft={
            "title": "收废铁",
            "description": "帮我收点废铁。",
            "npc_name": NPC,
            "get_dialogue": [
                # 模型即便仍塞了 title，也按后端查出来的写
                {"name": NPC, "title": "瞎写的称号", "emotion": "严肃", "text": "废铁收来。"},
                {"name": "$PC", "emotion": "犹豫", "text": "我尽量。"},
                {"name": "学徒", "text": "师父说得对。"},
            ],
            "finish_dialogue": [{"name": NPC, "text": "干得不错。"}],
        },
        npc_name_fallback=NPC,
        game_data=env.game_data,
    )
    text_path = env.data_root / "task" / "text" / "agent_text.json"
    doc = json.loads(text_path.read_text(encoding="utf-8"))
    get_items = doc[f"$AGENT_GET_{task_id}"]
    assert get_items[0] == {
        "name": NPC, "title": "炉边老板", "char": "铁匠#严肃", "text": "废铁收来。",
    }
    assert get_items[1] == {
        "name": "$PC", "title": "$PC_TITLE", "char": "$PC_CHAR#犹豫", "text": "我尽量。",
    }
    assert get_items[2] == {
        "name": "学徒", "title": "小学徒", "char": "学徒", "text": "师父说得对。",
    }
    # 没给情绪＝Char 就是说话人名，不拼空后缀
    assert doc[f"$AGENT_FINISH_{task_id}"][0]["char"] == NPC

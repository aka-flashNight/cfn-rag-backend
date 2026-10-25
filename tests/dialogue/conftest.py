"""对话工具/子 Agent 测试夹具：最小对话数据根 + 假游戏数据注册表 + 槽位绑定。

数据根结构（tmp_path 下）：
- data/dialogues/list.xml 登记一个 NPC 文件；
- npc_test.xml 含「测试使」三个日常组（id 1/2 有本人说话，id 3 没有 → 不进候选）；
- 假 game_data：6 个未完成任务（主线#1..#6，各带接取对话）＋支线#1 ＋
  已完成任务 900（投影 tasks_finished 命中 → 排除）＋Agent 任务 200001（排除）。
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from services.game_data.models import Task
from services.integration.state import get_integration_state

NPC = "测试使"
SLOT = "slot_dialogue_test"

_LIST_XML = "<list><items>npc_test.xml</items></list>"

_NPC_XML = """<root>
  <Dialogues>
    <Name>测试使</Name>
    <Dialogue id="1">
      <TaskRequirement>0</TaskRequirement>
      <SubDialogue><Name>测试使</Name><Title>测试使</Title><Char>测试使</Char><Text>今天炉子烧得正旺。</Text></SubDialogue>
      <SubDialogue><Name>$PC</Name><Title>$PC_TITLE</Title><Char>$PC_CHAR</Char><Text>给我打把刀。</Text></SubDialogue>
    </Dialogue>
    <Dialogue id="2">
      <TaskRequirement>0</TaskRequirement>
      <SubDialogue><Name>测试使</Name><Title>测试使</Title><Char>测试使#微笑</Char><Text>龙鳞可不好弄。</Text></SubDialogue>
    </Dialogue>
    <Dialogue id="3">
      <TaskRequirement>0</TaskRequirement>
      <SubDialogue><Name>旁人</Name><Title>旁人</Title><Char>旁人</Char><Text>这组没有测试使说话。</Text></SubDialogue>
    </Dialogue>
  </Dialogues>
</root>
"""


def _line(text: str, name: str = NPC) -> dict:
    return {"name": name, "title": name, "char": name, "text": text}


def _build_tasks() -> list[Task]:
    tasks: list[Task] = []
    for i in range(1, 7):
        tasks.append(Task(
            id=100 + i,
            title=f"$MAIN_TITLE_{i}",
            chain=f"主线#{i}",
            get_conversation=f"$MAIN_GET_{i}",
            get_npc=NPC,
        ))
    tasks.append(Task(
        id=201, title="$SIDE_TITLE_1", chain="支线#1",
        get_conversation="$SIDE_GET_1", get_npc=NPC,
    ))
    # 已完成任务（投影 tasks_finished 命中，候选必须排除）
    tasks.append(Task(
        id=900, title="$FIN_TITLE", chain="主线#9",
        get_conversation="$FIN_GET", get_npc=NPC,
    ))
    # Agent 委托（id 段 200001–300000，候选必须排除）
    tasks.append(Task(
        id=200001, title="$AGENT_TITLE", chain="委托",
        get_conversation="$AGENT_GET", get_npc=NPC,
    ))
    return tasks


def _build_texts() -> dict:
    texts: dict = {}
    for i in range(1, 7):
        texts[f"$MAIN_TITLE_{i}"] = f"龙鳞委托{i}"
        texts[f"$MAIN_GET_{i}"] = [_line(f"主线{i}的接取话，龙鳞记得带来。")]
    texts["$SIDE_TITLE_1"] = "支线杂事"
    texts["$SIDE_GET_1"] = [_line("支线的接取话。")]
    texts["$FIN_TITLE"] = "已完成任务"
    texts["$FIN_GET"] = [_line("已完成任务的接取话。")]
    texts["$AGENT_TITLE"] = "委托任务"
    texts["$AGENT_GET"] = [_line("委托任务的接取话。")]
    return texts


@pytest.fixture
def dialogue_env(tmp_path):
    data_root = tmp_path / "data"
    dialogues_dir = data_root / "dialogues"
    dialogues_dir.mkdir(parents=True)
    (dialogues_dir / "list.xml").write_text(_LIST_XML, encoding="utf-8")
    (dialogues_dir / "npc_test.xml").write_text(_NPC_XML, encoding="utf-8")

    tasks = _build_tasks()
    texts = _build_texts()
    game_data = SimpleNamespace(
        data_root=data_root,
        tasks=SimpleNamespace(list_all_tasks=lambda: tasks),
        task_texts=SimpleNamespace(all=lambda: texts),
    )
    return SimpleNamespace(data_root=data_root, game_data=game_data)


@pytest.fixture
def bound_slot():
    """绑定存档槽位（进度投影含已完成任务 900），用例结束后解绑。"""
    state = get_integration_state()

    def _bind(progress: dict | None = None) -> str:
        state.bind(SLOT, progress=progress or {"tasks_finished": [900]})
        return SLOT

    yield _bind
    state.unbind()

"""插话日常链路的任务发布能力块（与聊天主 Agent 共用同一份文本）。

覆盖：发布硬约束 / 商店报酬 / 切磋可行性三块注入与关闭、工具判据的切磋拼接、
路由纪律「一轮最多派一件事」，以及 draft 路由 schema 的条件覆写。
"""

from __future__ import annotations

from types import SimpleNamespace

from services.dialogue_interject.context import DailyBlocks
from services.dialogue_interject.prompts import (
    build_daily_system_prompt,
    build_daily_tool_guide,
)
from services.dialogue_interject.tools import DailyToolRuntime
from services.npc.manager import NPCState
from tests.dialogue_interject import fixtures as fx

STATE = NPCState(
    favorability=3,
    relationship_level="熟悉",
    sex="男",
    emotions=["普通"],
    faction="A兵团",
    titles=["军需官"],
)

CONSTRAINT_HEAD = "【任务发布硬约束】"
SHOP_NONE = "你不经营商店"
CHALLENGE_NONE = "你没有切磋关卡"
CHALLENGE_OK = "在对话情景合适时可以考虑提出跟玩家切磋一场"
CHALLENGE_WEAK = "玩家当前的实力还暂时不能挑战你"


def _prompt(**over) -> str:
    kwargs: dict = {"npc_name": fx.NPC, "state": STATE}
    kwargs.update(over)
    return build_daily_system_prompt(**kwargs)


# ---------------------------------------------------------------------------
# 三块注入
# ---------------------------------------------------------------------------

def test_publish_blocks_injected_and_mechanism_free():
    system = _prompt()
    assert CONSTRAINT_HEAD in system
    assert SHOP_NONE in system
    assert CHALLENGE_NONE in system
    # 插话口径：不出现「后台」这类系统视角措辞
    assert "后台" not in system


def test_shop_hint_lists_types_when_npc_runs_a_shop():
    system = _prompt(has_shop=True, shop_reward_types=["武器", "药剂"])
    assert SHOP_NONE not in system
    assert "你经营商店" in system
    assert "[武器、药剂]" in system


def test_challenge_hint_three_states():
    assert CHALLENGE_NONE in _prompt()
    weak = _prompt(
        state=NPCState(**{**STATE.__dict__, "challenge": "试炼场"}),
        player_can_challenge=False,
    )
    assert CHALLENGE_WEAK in weak
    ok = _prompt(
        state=NPCState(**{**STATE.__dict__, "challenge": "试炼场"}),
        player_can_challenge=True,
    )
    assert CHALLENGE_OK in ok
    assert CHALLENGE_WEAK not in ok


def test_publish_disabled_drops_all_three_blocks():
    system = _prompt(task_publish_disabled=True, has_shop=True, player_can_challenge=True)
    assert CONSTRAINT_HEAD not in system
    assert "你经营商店" not in system
    assert CHALLENGE_OK not in system
    assert "你当前不能发布委托" in system


# ---------------------------------------------------------------------------
# 工具判据
# ---------------------------------------------------------------------------

def test_tool_guide_mentions_sparring_only_when_challengeable():
    assert "切磋、对战" not in build_daily_tool_guide()
    with_spar = build_daily_tool_guide(player_can_challenge=True)
    assert "玩家提出想跟你切磋、对战、比划或直接动手，也走这个入口。" in with_spar


def test_tool_guide_carries_single_dispatch_discipline():
    assert "一轮最多只派一件事" in build_daily_tool_guide()


# ---------------------------------------------------------------------------
# draft 路由 schema 条件覆写
# ---------------------------------------------------------------------------

class _FakeRegistry:
    def __init__(self) -> None:
        self.names: list[str] = []

    def get_openai_tools(self, names=None):
        self.names = list(names or [])
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"STATIC:{name}",
                    "parameters": {},
                },
            }
            for name in self.names
        ]


def _runtime(*, player_can_challenge: bool) -> DailyToolRuntime:
    registry = _FakeRegistry()
    blocks = DailyBlocks(
        system_prompt="SYS",
        registry=registry,
        npc_state=STATE,
        npc_states={},
        pending_draft_row=None,
        player_can_challenge=player_can_challenge,
    )
    return DailyToolRuntime(SimpleNamespace(), blocks)


def _descriptions(runtime: DailyToolRuntime) -> dict[str, str]:
    return {
        item["function"]["name"]: item["function"]["description"]
        for item in runtime.tools() or []
    }


def test_draft_schema_gets_challenge_clause_when_challengeable():
    desc = _descriptions(_runtime(player_can_challenge=True))
    assert "切磋、对战" in desc["route_task_draft"]
    assert desc["route_search"] == "STATIC:route_search"


def test_draft_schema_stays_static_when_not_challengeable():
    desc = _descriptions(_runtime(player_can_challenge=False))
    assert desc["route_task_draft"] == "STATIC:route_task_draft"

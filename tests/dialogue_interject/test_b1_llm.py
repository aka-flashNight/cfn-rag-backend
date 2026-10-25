"""B1：日常稳定块、JSONL 提示词、真 LLM 适配器（全部用假客户端）。"""

from __future__ import annotations

from functools import partial

import pytest

from services.dialogue_interject.context import DailyBlocks, assemble_daily_blocks, build_user_context
from services.dialogue_interject.llm import InterjectLLMUnavailable, stream_daily_llm
from services.dialogue_interject.orchestrator import InterjectOrchestrator
from services.dialogue_interject.prompts import (
    build_daily_system_prompt,
    build_messages,
    tool_matrix_for,
)
from services.llm import LLMConfig, StreamEvent
from services.npc.manager import NPCState
from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import collect_events, make_request


STATE = NPCState(
    favorability=7,
    relationship_level="熟悉",
    sex="男",
    emotions=["普通", "微笑", "冷漠"],
    faction="A兵团",
    titles=["军需官"],
    appearance="旧皮夹克，左眉有一道疤。",
)


class FakeNPCManager:
    async def get(self, npc_name: str) -> NPCState:
        return STATE

    async def all_states(self) -> dict[str, NPCState]:
        return {fx.NPC: STATE}


class FakeGameData:
    map_registry = None


class FakeLLMClient:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.requests: list = []

    async def chat_stream(self, request):
        self.requests.append(request)
        for chunk in self.chunks:
            yield StreamEvent(kind="content", text=chunk)


def test_daily_system_prompt_has_stable_blocks_and_jsonl_rules():
    system = build_daily_system_prompt(
        npc_name=fx.NPC,
        state=STATE,
        scene_block="【场景】旧仓库",
        same_faction_block="同阵营角色：\n「老赵」（A兵团）",
        progress_desc="当前主线进度：阶段 2（堕落城）。",
    )
    assert "【世界观背景概要】" in system
    assert "【输出格式（最高优先级）】" in system
    assert '{"add":"台词","emo":"情绪"}' in system
    assert '{"closing":"结束语","emo":"情绪"}' in system
    assert "135" in system
    assert f"「{fx.NPC}」" in system
    assert "普通、微笑、冷漠" in system
    assert "【场景】旧仓库" in system
    assert "【同阵营角色】" in system
    assert "【你的形象】" in system
    assert "好感度是 7" in system

    # 不能混入终端的 meta 行 / 说话爆发块；B2 的工具说明必须在
    for forbidden in ("meta", "说话爆发"):
        assert forbidden not in system
    assert "route_search" in system
    assert "route_task_draft" in system
    assert "本轮只能对话" not in system
    assert "本轮没有可用工具" not in system


def test_daily_user_context_is_un_numbered_and_request_scoped():
    context = build_user_context(make_request(fx.daily_request()))
    assert "【本组已播历史】" in context
    assert f"{fx.NPC}：又是你。" in context
    assert "【原对话剩余台词" in context
    assert "[1]" not in context  # 日常不编号
    assert "有事说事。" in context
    assert "【玩家刚刚说的话】" in context
    assert "最近怎么样？" in context


def test_task_user_context_keeps_numbering():
    context = build_user_context(make_request(fx.task_request()))
    assert "[1] Andy Law：你先别急着答应。" in context
    assert "【任务上下文（任务前）】" in context


def test_build_messages_is_exactly_system_plus_request_context():
    request = make_request(fx.daily_request())
    messages = build_messages(request, daily_system="SYS")
    assert [item["role"] for item in messages] == ["system", "user"]
    assert messages[0]["content"] == "SYS"
    assert "最近怎么样？" in messages[1]["content"]


async def test_assemble_daily_blocks_uses_npc_state_and_emotion_whitelist(memory_store):
    blocks = await assemble_daily_blocks(
        make_request(fx.daily_request()),
        npc_manager=FakeNPCManager(),
        game_data=FakeGameData(),
        memory=memory_store,
        rag_enabled=False,
    )
    assert isinstance(blocks, DailyBlocks)
    assert blocks.npc_emotions == ("普通", "微笑", "冷漠")
    assert f"「{fx.NPC}」" in blocks.system_prompt
    assert "好感度是 7" in blocks.system_prompt


async def test_pending_claim_block_reports_inventory_shortage_on_reply_turn(memory_store):
    from services.dialogue_interject.item_claims import ITEM_CLAIMS

    async def _blocks(counts):
        return await assemble_daily_blocks(
            make_request(fx.daily_request(player_item_counts=counts)),
            npc_manager=FakeNPCManager(),
            game_data=FakeGameData(),
            memory=memory_store,
            rag_enabled=False,
        )

    # 索取还没发生时（NPC 发问那一轮）不提示玩家持有情况。
    blocks = await _blocks({})
    assert blocks.pending_claim_block == ""

    ITEM_CLAIMS.create(
        fx.DAILY_DIALOGUE_ID, npc=fx.NPC, item_name="普通hp药剂", quantity=2, turn_no=1,
    )
    blocks = await _blocks({"普通hp药剂": 1})
    assert "【待处理的索取】" in blocks.pending_claim_block
    assert "玩家看起来拥有该物品的数量不足。" in blocks.pending_claim_block

    ITEM_CLAIMS.close(fx.DAILY_DIALOGUE_ID)
    ITEM_CLAIMS.create(
        fx.DAILY_DIALOGUE_ID, npc=fx.NPC, item_name="普通hp药剂", quantity=1, turn_no=1,
    )
    blocks = await _blocks({})
    assert "玩家看起来没有这个物品。" in blocks.pending_claim_block

    ITEM_CLAIMS.close(fx.DAILY_DIALOGUE_ID)
    ITEM_CLAIMS.create(
        fx.DAILY_DIALOGUE_ID, npc=fx.NPC, item_name="普通hp药剂", quantity=1, turn_no=1,
    )
    blocks = await _blocks({"普通hp药剂": 1})
    assert "玩家看起来" not in blocks.pending_claim_block


async def test_stream_daily_llm_filters_content_and_exposes_no_tools():
    client = FakeLLMClient(['{"add":"你', '好","emo":"普通"}\n'])
    request = make_request(fx.daily_request())
    blocks = DailyBlocks(system_prompt="SYS", npc_emotions=("普通",))
    chunks = [
        chunk async for chunk in stream_daily_llm(
            request,
            client=client,
            blocks=blocks,
            llm_config=LLMConfig(
                api_key="test-key", api_base="http://unit.test/v1", model_name="test-model",
            ),
        )
    ]
    assert "".join(chunks) == '{"add":"你好","emo":"普通"}\n'
    assert len(client.requests) == 1
    chat_request = client.requests[0]
    assert chat_request.tools is None
    assert chat_request.send_image is False
    assert chat_request.purpose == "chat"
    assert chat_request.messages[0] == {"role": "system", "content": "SYS"}
    assert "最近怎么样？" in chat_request.messages[1]["content"]
    assert "[1]" not in chat_request.messages[1]["content"]


async def test_stream_daily_llm_without_api_key_raises_typed_error(monkeypatch):
    from services.dialogue_interject import llm as llm_module

    monkeypatch.setattr(
        llm_module, "resolve_llm_config", lambda *_args, **_kwargs: LLMConfig(api_key=""),
    )
    request = make_request(fx.daily_request())
    with pytest.raises(InterjectLLMUnavailable) as excinfo:
        chunks = [
            chunk async for chunk in stream_daily_llm(
                request,
                client=FakeLLMClient([]),
                blocks=DailyBlocks(system_prompt="SYS"),
            )
        ]
    assert excinfo.value.code == "llm_not_configured"


async def test_real_adapter_feeds_orchestrator_first_line_before_done():
    client = FakeLLMClient([
        '{"add":"还能怎么样，老样子。","emo":"普通"}\n',
        '{"closing":"说吧。","emo":"普通"}\n',
    ])
    blocks = DailyBlocks(system_prompt="SYS", npc_emotions=("普通",))
    factory = partial(
        stream_daily_llm,
        client=client,
        blocks=blocks,
        llm_config=LLMConfig(
            api_key="test-key", api_base="http://unit.test/v1", model_name="test-model",
        ),
    )
    orchestrator = InterjectOrchestrator(
        make_request(fx.daily_request()), session_id="s", stream_factory=factory,
    )
    events = await collect_events(orchestrator)
    names = [event.event for event in events]
    assert names == ["status", "line", "line", "done"]
    assert events[1].data["text"] == "还能怎么样，老样子。"
    assert events[1].data["seq"] == 0
    assert names.index("line") < names.index("done")


def test_tool_matrix_b2_daily_has_tools():
    assert "route_search" in tool_matrix_for("daily")
    assert "route_task_draft" in tool_matrix_for("daily")
    assert "update_long_term_memory" in tool_matrix_for("daily")
    assert "route_task_confirm" not in tool_matrix_for("daily")
    assert "route_task_confirm" in tool_matrix_for("daily", has_draft=True)
    assert tool_matrix_for("task") == ("route_search", "update_long_term_memory")

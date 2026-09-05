"""orchestrator e2e 测试夹具：Fake LLM（脚本化流）+ 可注入依赖环境。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from services.game_data.registry import get_game_data_registry, init_game_data_registry
from services.llm.client import ChatResult, StreamEvent
from services.memory.store import MemoryStore
from services.npc.manager import NPCManager, NPCState
from services.orchestrator.turn import OrchestratorDeps
from services.tools import get_tool_registry

NPC_NAME = "铁匠"


# ---------------------------------------------------------------------------
# Fake LLM：chat_stream 脚本（meta 行 + 正文）与 chat 脚本（工具调用）分离排队
# ---------------------------------------------------------------------------

class FakeLLM:
    """按入队顺序弹出脚本；流式与工具循环互不干扰（orchestrator/子 Agent 共享实例）。"""

    def __init__(self) -> None:
        self._streams: list[dict[str, Any]] = []
        self._chats: list[dict[str, Any]] = []
        self.stream_requests: list[Any] = []
        self.chat_requests: list[Any] = []

    # -- 脚本登记 --

    def add_stream(self, text: str = "", meta: str | None = None, usage: dict | None = None) -> None:
        self._streams.append({"meta": meta, "text": text, "usage": usage})

    def add_chat(self, content: str = "", tool_calls: list[dict] | None = None,
                 usage: dict | None = None, delay: float = 0.0) -> None:
        self._chats.append({"content": content, "tool_calls": tool_calls, "usage": usage, "delay": delay})

    # -- LLMClient 接口形状 --

    def chat_stream(self, req: Any):
        self.stream_requests.append(req)
        script = self._streams.pop(0)

        async def _gen():
            if script["meta"]:
                yield StreamEvent(kind="content", text=script["meta"] + "\n")
            text = script["text"]
            step = 3
            for i in range(0, len(text), step):
                yield StreamEvent(kind="content", text=text[i:i + step])
                await asyncio_sleep(0)
            if script["usage"]:
                yield StreamEvent(kind="usage", usage=script["usage"])
            yield StreamEvent(kind="finish", usage=script["usage"])

        return _gen()

    async def chat(self, req: Any) -> ChatResult:
        self.chat_requests.append(req)
        script = self._chats.pop(0)
        if script["delay"]:
            await asyncio_sleep(script["delay"])
        return ChatResult(
            content=script["content"],
            tool_calls=script["tool_calls"] or [],
            usage=script["usage"],
        )


async def asyncio_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def tc(name: str, args: dict[str, Any], call_id: str = "c0") -> dict[str, Any]:
    """构造 openai 形态的 tool_call（脚本用）。"""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------

@dataclass
class Env:
    deps: OrchestratorDeps
    store: MemoryStore
    npc_manager: NPCManager
    fake: FakeLLM
    session_id: str
    tmp_path: Path
    game_data: Any = field(default=None)


@pytest.fixture
def env(tmp_path) -> Env:
    store = MemoryStore(db_path=tmp_path / "memory.db")
    npc_manager = NPCManager(
        state={
            NPC_NAME: NPCState(
                favorability=30,
                relationship_level="熟悉",
                emotions=["普通", "微笑", "生气"],
                faction="A兵团",
                titles=["铁匠"],
            ),
            "药师": NPCState(favorability=10, emotions=["普通"], faction="A兵团", titles=["药师"]),
        },
        state_path=tmp_path / "npc_state_db.json",
    )
    init_game_data_registry()
    game_data = get_game_data_registry()
    fake = FakeLLM()
    deps = OrchestratorDeps(
        memory=store,
        npc_manager=npc_manager,
        registry=get_tool_registry(),
        game_data=game_data,
        engine=None,  # Tier-1 检索降级为空（检索引擎另有测试）
        llm_factory=lambda cfg: fake,
        merge_grace_ms=200,
        subagent_timeout_s=5,
        task_max_rounds=4,
        search_max_rounds=3,
        draft_keep_turns=3,
        history_limit=20,
        enable_summary=False,
    )
    # 同步创建会话（MemoryStore 同步方法直接调用）
    session_id = store.create_session_sync(npc_name=NPC_NAME, title="测试会话").session_id
    return Env(
        deps=deps,
        store=store,
        npc_manager=npc_manager,
        fake=fake,
        session_id=session_id,
        tmp_path=tmp_path,
        game_data=game_data,
    )


VALID_DRAFT: dict[str, Any] = {
    "draft_id": "testd001",
    "task_type": "资源收集",
    "title": "食材收集委托",
    "npc_name": NPC_NAME,
    "rewards": [{"item_name": "金币", "count": 20000}],
}

MERGE_REPLY_TASK = "给你安排了个收集食材的活，报酬是金币，你接还是不接？"


async def collect(orchestrator) -> list:
    return [ev async for ev in orchestrator.run() if ev.event != "keep_alive"]


def contents_of(events: list) -> list[str]:
    return [e.data.get("delta", "") for e in events if e.event == "content"]


def phases_of(events: list) -> list[tuple[str, str]]:
    return [
        (e.data.get("agent", ""), e.data.get("phase", ""))
        for e in events
        if e.event == "agent_status"
    ]


def notices_of(events: list) -> list[str]:
    return [e.data.get("text", "") for e in events if e.event == "system_notice"]


def compressed_kinds(events: list) -> list[str]:
    """事件类型序列（连续 content 合并为一个），便于断言结构。"""
    out: list[str] = []
    for e in events:
        if e.event == "content" and out and out[-1] == "content":
            continue
        out.append(e.event)
    return out

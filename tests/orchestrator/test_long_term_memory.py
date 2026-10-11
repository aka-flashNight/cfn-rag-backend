"""跨对话长期记忆在编排层的落地（B3-3/B3-4）＋已完成委托拼接（任务发布→完成）。

口径：
- 开关是请求级字段 long_term_memory_enabled；游戏内（embedded 口径）恒定启用、不受理关闭；
- 关闭＝长期记忆块与长期记忆工具都不出现（「无」＝完全不提）；
- 启用＝按 (作用域, NPC) 读快照，块进 user 侧共享上下文，工具按作用域当前状态定形；
- 工具不在全局注册表里，模型发了调用由编排层就地落库，失败不阻断本轮对话。
"""

from __future__ import annotations

import json

import pytest

from services.integration.save_paths import shared_task_paths
from services.integration.state import get_integration_state
from services.memory.long_term import CHAT_TOOL_NAME
from services.memory.store import LTM_MAX_FACT_ENTRIES, STANDALONE_SCOPE
from services.orchestrator.prompts import LTM_TASK_DONE_HINT
from services.orchestrator.prompts import build_finished_task_block as wrap_done_block
from services.orchestrator.prompts import build_user_shared_core
from services.orchestrator.turn import TurnOrchestrator
from services.tools.base import ROUTE_TOOLS_NEW
from tests.orchestrator.conftest import NPC_NAME, collect, contents_of, tc

BASE_TEXT = "你是常来铺子里的老主顾，说话直，办事靠得住。"
FACT_TEXT = "你替我找回了祖传铁锤。"


@pytest.fixture
def bound_slot():
    state = get_integration_state()

    def _bind(slot_key: str, **kwargs):
        state.bind(slot_key, **kwargs)
        return slot_key

    yield _bind
    state.unbind()


def _orch(env, query: str, **kwargs) -> TurnOrchestrator:
    return TurnOrchestrator(
        session_id=env.session_id, npc_name=NPC_NAME, query=query, deps=env.deps, **kwargs,
    )


def _shared_core(env) -> str:
    return env.fake.stream_requests[0].messages[1]["content"]


def _chat_tool_names(env) -> list[str]:
    return [t["function"]["name"] for t in (env.fake.stream_requests[0].tools or [])]


def _long_term_tool(env) -> dict | None:
    tools = env.fake.stream_requests[0].tools or []
    return next((t for t in tools if t["function"]["name"] == CHAT_TOOL_NAME), None)


async def _seed(env, *, base: str = "", facts: tuple[str, ...] = (), scope: str = STANDALONE_SCOPE):
    if base:
        await env.store.ltm_save_base(scope, NPC_NAME, base, env.session_id)
    for fact in facts:
        await env.store.ltm_append_fact(scope, NPC_NAME, fact, env.session_id)


# ---------------------------------------------------------------------------
# 开关
# ---------------------------------------------------------------------------

async def test_disabled_by_default_leaves_no_trace(env):
    """不传开关＝关闭：没有长期记忆块，也没有长期记忆工具。"""
    await _seed(env, base=BASE_TEXT, facts=(FACT_TEXT,))
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯，聊什么都行。")
    orch = _orch(env, "你好")
    await collect(orch)

    assert orch._ctx.long_term_enabled is False
    assert orch._ctx.long_term is None
    assert CHAT_TOOL_NAME not in _chat_tool_names(env)
    assert "长期记忆" not in _shared_core(env)
    assert BASE_TEXT not in _shared_core(env)


async def test_explicit_off_same_as_default(env):
    await _seed(env, base=BASE_TEXT, facts=(FACT_TEXT,))
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    orch = _orch(env, "你好", long_term_memory_enabled=False)
    await collect(orch)

    assert orch._ctx.long_term is None
    assert _long_term_tool(env) is None


async def test_enabled_injects_block_and_tool(env):
    await _seed(env, base=BASE_TEXT, facts=(FACT_TEXT, "你答应过不把我的秘密说出去。"))
    env.fake.add_stream(meta='{"emo":"微笑","fav":0}', text="又见面了。")
    orch = _orch(env, "你好", long_term_memory_enabled=True)
    await collect(orch)

    assert orch._ctx.long_term_enabled is True
    assert orch._ctx.long_term.slot_key == STANDALONE_SCOPE
    core = _shared_core(env)
    assert "【长期记忆（跨对话窗口保留）】" in core
    assert f"基础印象：{BASE_TEXT}" in core
    assert f"1. {FACT_TEXT}" in core
    assert "2. 你答应过不把我的秘密说出去。" in core

    # 工具排在路由表之后（同一个工具名，形态由作用域当前状态决定）
    assert _chat_tool_names(env) == [*ROUTE_TOOLS_NEW, CHAT_TOOL_NAME]
    fn = _long_term_tool(env)["function"]
    assert fn["parameters"]["required"] == ["content"]


async def test_enabled_empty_scope_offers_tool_without_block(env):
    """作用域里一条都没有：不注入空块，但要给「同时补基础印象」那一形态的工具。"""
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="你好。")
    orch = _orch(env, "你好", long_term_memory_enabled=True)
    await collect(orch)

    assert "【长期记忆" not in _shared_core(env)
    fn = _long_term_tool(env)["function"]
    assert fn["parameters"]["required"] == ["base_content", "content"]


async def test_full_scope_switches_tool_to_rewrite_only(env):
    """满 20 条：工具只剩调整能力，整份定义零处提及另一种用法。"""
    await _seed(env, base=BASE_TEXT, facts=tuple(f"事件{i}" for i in range(LTM_MAX_FACT_ENTRIES)))
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    orch = _orch(env, "你好", long_term_memory_enabled=True)
    await collect(orch)

    tool = _long_term_tool(env)
    fn = tool["function"]
    assert fn["parameters"]["required"] == ["entry_no", "content"]
    blob = json.dumps(tool, ensure_ascii=False)
    for forbidden in ("追加", "新记一条", "留空"):
        assert forbidden not in blob
    # 注入块照旧给全量编号，供改写时引用
    assert f"{LTM_MAX_FACT_ENTRIES}. 事件{LTM_MAX_FACT_ENTRIES - 1}" in _shared_core(env)


async def test_embedded_forces_on_and_scopes_to_slot(env, bound_slot):
    """游戏内打开恒定启用（前端不给关闭入口，后端也不受理关闭），且用本存档那一份。"""
    await _seed(env, base="非游戏打开那一份的印象。")
    await _seed(env, base="存档一的印象。", facts=("存档一的事件。",), scope="slot_1")
    bound_slot("slot_1")

    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    orch = _orch(env, "你好", long_term_memory_enabled=False)
    await collect(orch)

    assert orch._ctx.long_term_enabled is True
    assert orch._ctx.long_term.slot_key == "slot_1"
    core = _shared_core(env)
    assert "基础印象：存档一的印象。" in core
    assert "1. 存档一的事件。" in core
    assert "非游戏打开那一份的印象。" not in core


# ---------------------------------------------------------------------------
# 模型调用就地落库
# ---------------------------------------------------------------------------

async def test_model_call_writes_base_and_fact(env):
    env.fake.add_stream(
        meta='{"emo":"微笑","fav":2}',
        text="这事我记下了。",
        tool_calls=[tc(CHAT_TOOL_NAME, {"base_content": BASE_TEXT, "content": FACT_TEXT})],
    )
    orch = _orch(env, "帮我把铁锤找回来了", long_term_memory_enabled=True)
    events = await collect(orch)

    # 对话照常完成，工具调用不外泄给玩家
    assert "".join(contents_of(events)) == "这事我记下了。"
    assert await env.store.ltm_get_base(STANDALONE_SCOPE, NPC_NAME) == BASE_TEXT
    facts = await env.store.ltm_list(STANDALONE_SCOPE, NPC_NAME)
    assert [f.content for f in facts if f.kind == "fact"] == [FACT_TEXT]
    assert facts[-1].source_session_id == env.session_id


async def test_model_rewrite_call_updates_in_place(env):
    await _seed(env, base=BASE_TEXT, facts=(FACT_TEXT,))
    env.fake.add_stream(
        meta='{"emo":"普通","fav":0}',
        text="好。",
        tool_calls=[tc(CHAT_TOOL_NAME, {"entry_no": 1, "content": "你替我找回铁锤，还修好了炉子。"})],
    )
    await collect(_orch(env, "还修了炉子", long_term_memory_enabled=True))

    rows = await env.store.ltm_list(STANDALONE_SCOPE, NPC_NAME)
    facts = [r for r in rows if r.kind == "fact"]
    assert len(facts) == 1
    assert facts[0].content == "你替我找回铁锤，还修好了炉子。"


async def test_ignored_when_disabled_even_if_model_forces_call(env):
    """关闭态不下发这个工具；模型硬发也不落库。"""
    env.fake.add_stream(
        meta='{"emo":"普通","fav":0}',
        text="嗯。",
        tool_calls=[tc(CHAT_TOOL_NAME, {"base_content": "x", "content": "y"})],
    )
    await collect(_orch(env, "你好"))

    assert await env.store.ltm_list(STANDALONE_SCOPE, NPC_NAME) == []


async def test_write_failure_does_not_break_the_turn(env, monkeypatch):
    """记忆是附加能力：写库炸了也照常把话说完。"""
    async def _boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(env.store, "ltm_save_base", _boom)
    monkeypatch.setattr(env.store, "ltm_append_fact", _boom)

    env.fake.add_stream(
        meta='{"emo":"普通","fav":0}',
        text="话还是要说完。",
        tool_calls=[tc(CHAT_TOOL_NAME, {"base_content": BASE_TEXT, "content": FACT_TEXT})],
    )
    events = await collect(_orch(env, "你好", long_term_memory_enabled=True))
    assert "".join(contents_of(events)) == "话还是要说完。"


# ---------------------------------------------------------------------------
# 已完成委托的实时拼接
# ---------------------------------------------------------------------------

def _write_task_files(data_root, tasks: list[dict], texts: dict) -> None:
    tasks_path, text_path = shared_task_paths(data_root)
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(json.dumps({"tasks": tasks}, ensure_ascii=False), encoding="utf-8")
    text_path.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")


async def test_finished_task_block_reaches_prompt(env, bound_slot, monkeypatch):
    """当前存档里这个 NPC 发布的委托被完成 → 本轮就把完成对话拼给 NPC。"""
    data_root = env.tmp_path / "data"
    monkeypatch.setattr(env.game_data, "data_root", data_root)
    _write_task_files(
        data_root,
        [{"id": 200001, "title": "$AGENT_TITLE_200001", "get_npc": NPC_NAME,
          "finish_conversation": "$AGENT_FINISH_200001"}],
        {"$AGENT_TITLE_200001": "取回祖传铁锤",
         "$AGENT_FINISH_200001": [{"name": "$PC", "text": "带回来了。"},
                                   {"name": NPC_NAME, "text": "这份情我记着。"}]},
    )
    bound_slot("slot_1", progress={"tasks_finished": [200001]})

    env.fake.add_stream(meta='{"emo":"微笑","fav":3}', text="多谢你。")
    orch = _orch(env, "铁锤给你", long_term_memory_enabled=True)
    await collect(orch)

    assert orch._ctx.finished_task_block.startswith("【你发布的委托已被玩家完成】")
    core = _shared_core(env)
    assert "此前你发布的委托「取回祖传铁锤」已被玩家完成。\n以下对话已经发生：" in core
    assert "玩家：带回来了。" in core
    # 已有关键记录时才要求把「已发布」改成「已完成」
    assert LTM_TASK_DONE_HINT not in core


async def test_finished_task_hint_only_when_facts_exist(env, bound_slot, monkeypatch):
    data_root = env.tmp_path / "data"
    monkeypatch.setattr(env.game_data, "data_root", data_root)
    _write_task_files(
        data_root,
        [{"id": 200001, "title": "$AGENT_TITLE_200001", "get_npc": NPC_NAME}],
        {"$AGENT_TITLE_200001": "取回祖传铁锤"},
    )
    bound_slot("slot_1", progress={"tasks_finished": [200001]})
    await _seed(env, base=BASE_TEXT, facts=("我发布了取回铁锤的委托。",), scope="slot_1")

    env.fake.add_stream(meta='{"emo":"微笑","fav":0}', text="多谢。")
    await collect(_orch(env, "办妥了", long_term_memory_enabled=True))

    assert LTM_TASK_DONE_HINT in _shared_core(env)


async def test_no_finished_projection_leaves_block_empty(env, bound_slot, monkeypatch):
    data_root = env.tmp_path / "data"
    monkeypatch.setattr(env.game_data, "data_root", data_root)
    bound_slot("slot_1", progress={"tasks_finished": []})

    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    orch = _orch(env, "你好")
    await collect(orch)

    assert orch._ctx.finished_task_block == ""
    assert "已被玩家完成" not in _shared_core(env)


# ---------------------------------------------------------------------------
# 块序与包装（纯函数层）
# ---------------------------------------------------------------------------

def test_block_order_session_then_finished_then_long_term():
    core = build_user_shared_core(
        player_identity="一名 A 兵团佣兵。",
        favorability=30,
        relationship_level="熟悉",
        finished_task_block="【已完成委托块】",
        long_term_block="【长期记忆块】",
        retrieved_context="检索到的资料",
        summary_str="早期摘要",
        history_str="玩家：你好",
    )
    assert core.index("【已完成委托块】") < core.index("【长期记忆块】")
    # 摘要与长期记忆同属跨轮稀疏层，两块相邻；RAG 跨轮全变，排在它们之后
    assert core.index("【长期记忆块】") < core.index("早期摘要")
    assert core.index("早期摘要") < core.index("检索到的资料")
    assert core.index("检索到的资料") < core.index("玩家：你好")


def test_finished_block_wrapper_appends_hint_only_with_facts():
    """没有关键记录时「把已发布改成已完成」没有可执行对象，整句不拼。"""
    assert wrap_done_block("", has_long_term_facts=True) == ""
    assert wrap_done_block("【块】", has_long_term_facts=False) == "【块】"
    assert wrap_done_block("【块】", has_long_term_facts=True) == f"【块】\n{LTM_TASK_DONE_HINT}"

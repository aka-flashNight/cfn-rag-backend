"""任务前插话「临时新增奖励」：额度硬校验、载荷、注册条件、执行链与主干保护。

全部离线：假回执、假 game_data，不调真实模型、不写任何游戏仓文件。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from services.dialogue_interject.actions import ACTIONS, ActionResult
from services.dialogue_interject.context import DailyBlocks
from services.dialogue_interject.item_claims import ITEM_CLAIMS
from services.dialogue_interject.item_value import gift_value_cap
from services.dialogue_interject.llm import _status_stage_for
from services.dialogue_interject.orchestrator import InterjectOrchestrator
from services.dialogue_interject.prompts import (
    build_task_system_prompt,
    build_task_tool_guide,
)
from services.dialogue_interject.schemas import (
    ActionPayload,
    StreamAction,
    TaskRewardAdjustPayload,
)
from services.dialogue_interject.task_reward import (
    ADJUST_TOOL,
    MAX_EXTRA_REWARDS,
    TASK_NUMBERED_FOLLOWUP,
    affinity_multiplier,
    check_extra_rewards,
    parse_adjust_arguments,
    reward_basis,
    reward_cap,
    task_reward_enabled,
    task_reward_tool_schema,
)
from services.dialogue_interject.tools import DailyToolRuntime
from services.npc.manager import NPCState
from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import collect_events, make_request


class _Registry:
    def get_openai_tools(self, names=None):
        return [
            {"type": "function", "function": {"name": name, "parameters": {}}}
            for name in (names or [])
        ]


class _Items:
    def get_price(self, name):
        return {"普通hp药剂": 100, "稀有芯片": 5000}.get(name, 0)

    def get_by_name(self, name):
        return None


class _GameData:
    items = _Items()


def _blocks(**kwargs) -> DailyBlocks:
    return DailyBlocks(
        system_prompt="SYS",
        tool_free_system_prompt="FREE_SYS",
        npc_state=SimpleNamespace(
            faction="A兵团", titles=[], favorability=60, challenge=None,
        ),
        game_data=_GameData(),
        registry=_Registry(),
        session_id=fx.TASK_DIALOGUE_ID,
        slot_key="slot-1",
        progress_stage=1,
        turn_no=1,
        **kwargs,
    )


def _call(rewards) -> dict:
    return {
        "type": "function",
        "id": "call_reward",
        "function": {
            "name": ADJUST_TOOL,
            "arguments": {"extra_rewards": list(rewards)},
        },
    }


async def _execute(
    runtime: DailyToolRuntime,
    call: dict,
    *,
    success: bool = True,
    error: str = "",
    resolve: bool = True,
):
    """跑工具并模拟 Host 回执；返回 (结果, 发出的 action 载荷或 None)。"""
    task = asyncio.create_task(runtime.execute([call]))
    action_data = None
    while not task.done():
        try:
            progress = await asyncio.wait_for(runtime.progress_queue.get(), timeout=0.02)
        except asyncio.TimeoutError:
            continue
        if progress.action is not None and action_data is None:
            action_data = progress.action.data
            if resolve:
                ACTIONS.resolve(ActionResult(
                    action_id=action_data["action_id"],
                    success=success,
                    error=error,
                ))
    while not runtime.progress_queue.empty():
        progress = runtime.progress_queue.get_nowait()
        if progress.action is not None and action_data is None:
            action_data = progress.action.data
    return await task, action_data


# ---------------------------------------------------------------------------
# 额度与硬校验
# ---------------------------------------------------------------------------


def test_affinity_multiplier_matches_reward_validator():
    assert [affinity_multiplier(x) for x in (10, 20, 50, 80)] == [0.90, 1.00, 1.10, 1.20]


def test_reward_cap_uses_gift_value_cap_formula():
    for stage, favorability in ((1, 60), (2, 30), (3, 60), (3, 80)):
        assert reward_cap(stage=stage, favorability=favorability) == gift_value_cap(
            stage=stage, favorability=favorability,
        )
    # 指挥窗口示例：阶段 3 / 好感度 60 的实算值，载荷由程序算而非写死。
    assert reward_cap(stage=3, favorability=60) == 99000


def test_reward_basis_uses_chinese_field_names():
    basis = reward_basis(stage=3, favorability=60)
    assert basis == {
        "任务类型": "传话",
        "阶段": 3,
        "好感度": 60,
        "好感度系数": 1.10,
        "阶段基础上限": 30000,
        "讨价还价放大倍数": 1.0,
    }


def test_extra_rewards_hard_checks():
    cap = reward_cap(stage=1, favorability=60)  # 33000
    ok = check_extra_rewards(_GameData(), ["金币#500", "普通hp药剂#2"], cap=cap)
    assert ok.ok and ok.reason == "" and ok.total == 700
    assert ok.rewards == ("金币#500", "普通hp药剂#2")

    assert check_extra_rewards(_GameData(), [], cap=cap).reason == "空清单"
    assert check_extra_rewards(
        _GameData(), ["金币#1", "经验值#2", "技能点#1"], cap=cap,
    ).reason == f"超过 {MAX_EXTRA_REWARDS} 样"
    assert check_extra_rewards(_GameData(), ["金币#0"], cap=cap).reason == "数量必须是正整数"
    assert check_extra_rewards(_GameData(), ["金币#-3"], cap=cap).reason == "数量必须是正整数"
    assert check_extra_rewards(_GameData(), ["没有井号"], cap=cap).reason == "数量必须是正整数"
    assert "无法估价" in check_extra_rewards(
        _GameData(), ["未知物品#1"], cap=cap,
    ).reason
    over = check_extra_rewards(_GameData(), ["金币#500000"], cap=cap)
    assert not over.ok and over.reason == "总额超过上限" and over.rewards == ()


def test_extra_rewards_expression_is_normalized():
    cap = reward_cap(stage=1, favorability=60)
    ok = check_extra_rewards(_GameData(), [" 金币 # 500 "], cap=cap)
    assert ok.ok and ok.rewards == ("金币#500",)


def test_parse_adjust_arguments_accepts_json_string_and_dict():
    assert parse_adjust_arguments({"function": {"arguments": {
        "extra_rewards": ["金币#1"],
    }}}) == ["金币#1"]
    assert parse_adjust_arguments({"function": {"arguments":
        '{"extra_rewards":["金币#2"]}',
    }}) == ["金币#2"]
    assert parse_adjust_arguments({"function": {"arguments": "{"}}) == []
    assert parse_adjust_arguments({"function": {}}) == []


# ---------------------------------------------------------------------------
# 载荷形状：零新增 SSE 事件 / 端点 / 请求字段
# ---------------------------------------------------------------------------


def test_action_payload_shape_and_chinese_keys():
    cap = reward_cap(stage=1, favorability=60)
    payload = TaskRewardAdjustPayload.model_validate({
        "action_id": "act-1",
        "task_id": 50004,
        "extra_rewards": ["金币#500", "普通hp药剂#2"],
        "额度上限": cap,
        "额度依据": reward_basis(stage=1, favorability=60),
    })
    wire = StreamAction(payload).to_event()
    assert wire.event == "action"
    assert wire.data == {
        "action_id": "act-1",
        "kind": "task_reward_adjust",
        "task_id": 50004,
        "extra_rewards": ["金币#500", "普通hp药剂#2"],
        "额度上限": cap,
        "额度依据": reward_basis(stage=1, favorability=60),
    }


def test_item_grant_payload_is_unchanged():
    payload = ActionPayload(
        action_id="act-0", npc=fx.NPC, item_name="普通hp药剂",
        quantity=2, requested_quantity=5,
    )
    assert payload.to_wire() == {
        "action_id": "act-0",
        "kind": "item_grant",
        "npc": fx.NPC,
        "item_name": "普通hp药剂",
        "quantity": 2,
        "requested_quantity": 5,
    }


def test_task_reward_payload_rejects_invalid_task_id():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TaskRewardAdjustPayload.model_validate({
            "action_id": "a", "task_id": 0,
            "extra_rewards": ["金币#1"],
            "额度上限": 1, "额度依据": {},
        })


# ---------------------------------------------------------------------------
# 注册条件
# ---------------------------------------------------------------------------


def test_task_reward_enabled_requires_task_pre_without_action_or_claim():
    request = make_request(fx.task_request())
    assert task_reward_enabled(request, None) is True
    assert task_reward_enabled(request, object()) is False

    post = fx.task_request()
    post["dialogue"]["kind"] = "task_post"
    assert task_reward_enabled(make_request(post), None) is False

    daily = make_request(fx.daily_request())
    assert task_reward_enabled(daily, None) is False

    with_action = make_request(fx.task_request(action={
        "kind": "item_take_accept", "action_id": "act-2",
        "claim_id": "claim-2", "item_name": "普通hp药剂", "quantity": 1,
    }))
    assert task_reward_enabled(with_action, None) is False


def test_tool_is_registered_only_for_task_pre():
    request = make_request(fx.task_request())
    names = [
        item["function"]["name"]
        for item in (DailyToolRuntime(request, _blocks()).tools() or [])
    ]
    assert ADJUST_TOOL in names

    post = fx.task_request()
    post["dialogue"]["kind"] = "task_post"
    post_names = [
        item["function"]["name"]
        for item in (DailyToolRuntime(make_request(post), _blocks()).tools() or [])
    ]
    assert ADJUST_TOOL not in post_names

    daily_names = [
        item["function"]["name"]
        for item in (DailyToolRuntime(make_request(fx.daily_request()), _blocks()).tools() or [])
    ]
    assert ADJUST_TOOL not in daily_names

    with_action = make_request(fx.task_request(action={
        "kind": "item_take_accept", "action_id": "act-3",
        "claim_id": "claim-3", "item_name": "普通hp药剂", "quantity": 1,
    }))
    assert ADJUST_TOOL not in [
        item["function"]["name"]
        for item in (DailyToolRuntime(with_action, _blocks()).tools() or [])
    ]

    ITEM_CLAIMS.create(
        fx.TASK_DIALOGUE_ID, npc=fx.NPC, item_name="普通hp药剂", quantity=1, turn_no=1,
    )
    assert ADJUST_TOOL not in [
        item["function"]["name"]
        for item in (DailyToolRuntime(request, _blocks()).tools() or [])
    ]


def test_tool_schema_limits_extra_rewards_to_two():
    schema = task_reward_tool_schema()
    assert schema["function"]["name"] == ADJUST_TOOL
    params = schema["function"]["parameters"]["properties"]["extra_rewards"]
    assert params["maxItems"] == 2 and params["minItems"] == 1


# ---------------------------------------------------------------------------
# 提示词按状态注入
# ---------------------------------------------------------------------------


def test_task_tool_guide_only_mentions_reward_when_enabled():
    guide = build_task_tool_guide(item_state="give_take", task_reward_enabled=True)
    assert ADJUST_TOOL in guide
    assert "只新增奖励" in guide
    assert "最多 2 样" in guide
    assert ADJUST_TOOL not in build_task_tool_guide(item_state="give_take")

    system = build_task_system_prompt(
        npc_name=fx.NPC, state=NPCState(), item_state="give_take",
        task_reward_enabled=True,
    )
    assert ADJUST_TOOL in system
    tool_free = build_task_system_prompt(
        npc_name=fx.NPC, state=NPCState(), item_state="give_take",
        task_reward_enabled=True, include_tool_guide=False,
    )
    assert ADJUST_TOOL not in tool_free


def test_reward_tool_status_is_waiting_in_task_mode():
    call = {"type": "function", "id": "c1", "function": {
        "name": ADJUST_TOOL, "arguments": "{}",
    }}
    assert _status_stage_for([call], mode="task") == "waiting"
    assert _status_stage_for([call], mode="daily") == ""


# ---------------------------------------------------------------------------
# 执行链：只新增、不发超额 action、后端零写盘
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_emits_action_with_chinese_basis_and_numbered_followup():
    request = make_request(fx.task_request())
    runtime = DailyToolRuntime(request, _blocks())
    result, payload = await _execute(runtime, _call(["金币#500", "普通hp药剂#2"]))

    assert payload is not None
    assert payload["kind"] == "task_reward_adjust"
    assert payload["task_id"] == 200124
    assert payload["extra_rewards"] == ["金币#500", "普通hp药剂#2"]
    assert payload["额度上限"] == reward_cap(stage=1, favorability=60)
    assert payload["额度依据"] == reward_basis(stage=1, favorability=60)
    assert "已经在原有奖励之外新增" in result.result_block
    assert "原有奖励不变" in result.result_block
    assert "按【原始剩余台词（按编号no）】" in result.instruction
    assert result.route_kind == "task_reward_adjust"
    assert result.system_prompt == "FREE_SYS"
    # 后端零写盘：不记物品账本、不记赠送明细
    assert ITEM_CLAIMS.gift_detail_block(fx.TASK_DIALOGUE_ID) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("rewards", [
    ["金币#1", "经验值#2", "技能点#1"],  # 超过 2 样
    ["金币#500000"],                     # 超过额度
    ["未知物品#1"],                      # 无法估价
    ["金币#0"],                          # 数量非正
    [],                                  # 空清单
])
async def test_invalid_or_over_cap_never_emits_action(rewards):
    request = make_request(fx.task_request())
    runtime = DailyToolRuntime(request, _blocks())
    result, payload = await _execute(runtime, _call(rewards))

    assert payload is None
    assert "没有新增奖励" in result.result_block
    assert "原有奖励不变" in result.result_block or "这次没有新增奖励" in result.result_block


@pytest.mark.asyncio
async def test_failed_receipt_does_not_claim_success():
    request = make_request(fx.task_request())
    runtime = DailyToolRuntime(request, _blocks())
    result, payload = await _execute(
        runtime, _call(["金币#500"]), success=False, error="任务目录不可写",
    )
    assert payload is not None
    assert "任务目录不可写" in result.result_block
    assert "没有新增奖励" in result.result_block
    assert "已经" not in result.result_block


@pytest.mark.asyncio
async def test_receipt_timeout_does_not_replay_action(monkeypatch):
    import services.dialogue_interject.tools as tools_module

    monkeypatch.setattr(tools_module, "ACTION_WAIT_TIMEOUT_S", 0.05)
    request = make_request(fx.task_request())
    runtime = DailyToolRuntime(request, _blocks())
    result, payload = await _execute(
        runtime, _call(["金币#500"]), resolve=False,
    )
    assert payload is not None
    assert "游戏侧没有确认" in result.result_block
    assert "没有新增奖励" in result.result_block
    assert "已经" not in result.result_block
    # 工具结果回来后仍要回到按编号收尾的主干
    assert "按【原始剩余台词（按编号no）】" in result.instruction


@pytest.mark.asyncio
async def test_missing_task_id_never_emits_action():
    payload_request = fx.task_request()
    payload_request["task_context"] = {"completed": None, "upcoming": None}
    request = make_request(payload_request)
    runtime = DailyToolRuntime(request, _blocks())
    result, payload = await _execute(runtime, _call(["金币#500"]))
    assert payload is None
    assert "没有读到任务编号" in result.result_block


@pytest.mark.asyncio
async def test_task_post_never_emits_action_even_if_tool_called():
    post = fx.task_request()
    post["dialogue"]["kind"] = "task_post"
    runtime = DailyToolRuntime(make_request(post), _blocks())
    result, payload = await _execute(runtime, _call(["金币#500"]))
    assert payload is None
    assert "当前不能新增奖励" in result.result_block


# ---------------------------------------------------------------------------
# 主干保护：新增奖励轮用 end 也不丢原剩余
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reward_round_end_does_not_drop_original_remaining():
    payload = TaskRewardAdjustPayload.model_validate({
        "action_id": "act-9", "task_id": 50004,
        "extra_rewards": ["金币#500"],
        "额度上限": reward_cap(stage=1, favorability=60),
        "额度依据": reward_basis(stage=1, favorability=60),
    })

    async def _stream(_request):
        yield '{"type":"add","text":"这点报酬我添上了。","emo":"普通"}\n'
        yield StreamAction(payload)
        yield '{"type":"end"}\n'

    orchestrator = InterjectOrchestrator(
        make_request(fx.task_request()),
        session_id="s",
        stream_factory=_stream,
    )
    events = await collect_events(orchestrator)
    plans = [event.data for event in events if event.event == "plan"]
    assert plans == [{"mute": False, "resume": 1, "steps": []}]
    assert [event.data["text"] for event in events if event.event == "line"] == [
        "这点报酬我添上了。",
    ]


def test_numbered_followup_constant_matches_task_item_round():
    assert TASK_NUMBERED_FOLLOWUP == (
        "\n按【原始剩余台词（按编号no）】从 1 开始逐条处理，"
        "最后用 continue 或 end 收尾；没有原剩余时用 end。"
    )

"""P6：物品动作请求 schema、旁路事件形状与 action-result 端点。"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from services.dialogue_interject import api as interject_api
from services.dialogue_interject.actions import is_loopback_host
from services.dialogue_interject.schemas import (
    ActionPayload,
    ActionResultRequest,
    InterjectRequest,
    ItemClaimPayload,
    StreamAction,
    StreamItemClaim,
)
from tests.dialogue_interject import fixtures as fx


def _daily(**overrides) -> dict:
    payload = fx.daily_request(**overrides)
    return payload


def _task(**overrides) -> dict:
    payload = fx.task_request(**overrides)
    return payload


def test_item_take_action_requires_claim_item_and_quantity():
    with pytest.raises(ValidationError):
        InterjectRequest.model_validate(_daily(action={
            "kind": "item_take_accept",
            "action_id": "a1",
        }))
    with pytest.raises(ValidationError):
        InterjectRequest.model_validate(_daily(action={
            "kind": "item_take_decline",
            "action_id": "a1",
            "claim_id": "c1",
            "item_name": "M4A1",
        }))
    ok = InterjectRequest.model_validate(_daily(action={
        "kind": "item_take_accept",
        "action_id": "a1",
        "claim_id": "c1",
        "item_name": "M4A1",
        "quantity": 1,
        "item_level": 6,
    }))
    assert ok.action is not None and ok.action.item_level == 6


def test_item_level_only_for_accept_and_no_item_fields_for_task_actions():
    with pytest.raises(ValidationError):
        InterjectRequest.model_validate(_daily(action={
            "kind": "item_take_decline",
            "action_id": "a1",
            "claim_id": "c1",
            "item_name": "M4A1",
            "quantity": 1,
            "item_level": 3,
        }))
    with pytest.raises(ValidationError):
        InterjectRequest.model_validate(_task(action={
            "kind": "task_accept",
            "action_id": "a1",
            "item_level": 3,
        }))
    with pytest.raises(ValidationError):
        InterjectRequest.model_validate(_task(action={
            "kind": "task_decline",
            "action_id": "a1",
            "claim_id": "c1",
        }))


@pytest.mark.parametrize("request_factory", [_daily, _task], ids=["daily", "task"])
@pytest.mark.parametrize("kind", ["task_accept", "task_decline"])
def test_task_actions_allowed_in_daily_and_task_mode(request_factory, kind):
    request = InterjectRequest.model_validate(request_factory(action={
        "kind": kind,
        "action_id": "a1",
    }))
    assert request.action is not None and request.action.kind == kind


def test_item_take_action_requires_daily_mode():
    with pytest.raises(ValidationError):
        InterjectRequest.model_validate(_task(action={
            "kind": "item_take_accept",
            "action_id": "a1",
            "claim_id": "c1",
            "item_name": "M4A1",
            "quantity": 1,
        }))


def test_player_item_names_normalized_and_daily_only():
    names = [f"物品{i}" for i in range(250)] + ["物品0", "长" * 120]
    request = InterjectRequest.model_validate(_daily(player_item_names=names))
    assert len(request.player_item_names) == 200
    assert len(set(request.player_item_names)) == 200
    assert all(len(name) <= 100 for name in request.player_item_names)

    with pytest.raises(ValidationError):
        InterjectRequest.model_validate(_task(player_item_names=["M4A1"]))


def test_player_item_counts_normalized_and_daily_only():
    counts = {"负数": -2, " 长" + "字" * 120: 3}
    for i in range(250):
        counts[f"物品{i}"] = i
    request = InterjectRequest.model_validate(_daily(player_item_counts=counts))
    assert len(request.player_item_counts) == 200
    assert request.player_item_counts["负数"] == 0
    assert all(len(name) <= 100 for name in request.player_item_counts)

    with pytest.raises(ValidationError):
        InterjectRequest.model_validate(_task(player_item_counts={"M4A1": 1}))


def test_stream_action_and_claim_shapes():
    action = StreamAction(ActionPayload(
        action_id="a1", npc=fx.NPC, item_name="普通hp药剂",
        quantity=2, requested_quantity=5,
    ))
    assert isinstance(action, str) and action == ""
    event = action.to_event()
    assert event.event == "action"
    assert event.data == {
        "action_id": "a1",
        "kind": "item_grant",
        "npc": fx.NPC,
        "item_name": "普通hp药剂",
        "quantity": 2,
        "requested_quantity": 5,
    }

    claim = StreamItemClaim(ItemClaimPayload(
        state="pending", claim_id="c1", npc=fx.NPC,
        item_name="普通hp药剂", quantity=3,
    ))
    assert claim.to_event().event == "item_claim"
    assert claim.to_event().data["state"] == "pending"


def test_action_result_request_shape():
    payload = ActionResultRequest.model_validate({
        "action_id": "a1", "success": True, "actual_quantity": 2, "overflow_money": 40,
    })
    assert payload.actual_quantity == 2
    with pytest.raises(ValidationError):
        ActionResultRequest.model_validate({"action_id": "a1", "success": True, "actual_quantity": -1})


def test_action_result_endpoint_accepts_loopback_and_is_idempotent():
    app = FastAPI()
    app.include_router(interject_api.router, prefix="/game")
    client = TestClient(app)

    body = {"action_id": "act-endpoint-1", "success": True, "actual_quantity": 2}
    first = client.post("/game/dialogue-interject/action-result", json=body)
    second = client.post("/game/dialogue-interject/action-result", json=body)

    assert first.status_code == 200
    assert first.json() == {"ok": True, "applied": False}
    assert second.json() == {"ok": True, "applied": False}


def test_loopback_host_check():
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert is_loopback_host("localhost")
    assert is_loopback_host("testclient")
    assert not is_loopback_host("10.0.0.5")

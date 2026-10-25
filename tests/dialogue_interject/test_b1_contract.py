"""B1 契约：reference 越界 422、turn_id 幂等、finalize 回执与并发冲突。"""

from __future__ import annotations

import logging
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from services.dialogue_interject import api as interject_api
from services.dialogue_interject.schemas import FinalizeRequest, InterjectRequest
from services.dialogue_interject.sessions import ensure_session, finalize
from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import make_request


@pytest.fixture(autouse=True)
def _reset_factories():
    yield
    interject_api.set_stream_factories(None, None)


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(interject_api.router, prefix="/game")
    return TestClient(app)


def _with_reference(payload: dict, reference: list[dict]) -> dict:
    data = {**payload}
    data["dialogue"] = {**payload["dialogue"], "reference_continuation": reference}
    return data


# ---------------------------------------------------------------------------
# reference_continuation 越界（协议 5.1 / 实施指引 3.1）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["edit", "remove"])
def test_reference_index_beyond_current_n_is_rejected(kind):
    reference = (
        {"kind": "edit", "index": 9, "name": fx.NPC, "text": "越界改写"}
        if kind == "edit" else {"kind": "remove", "index": 9}
    )
    payload = _with_reference(fx.task_request(), [reference])
    with pytest.raises(ValidationError) as excinfo:
        InterjectRequest.model_validate(payload)
    assert "reference_continuation" in str(excinfo.value)
    assert "1..4" in str(excinfo.value)


def test_reference_index_equal_n_is_accepted():
    payload = _with_reference(fx.task_request(), [
        {"kind": "edit", "index": 4, "name": fx.NPC, "text": "合法改写"},
        {"kind": "remove", "index": 4},
    ])
    request = InterjectRequest.model_validate(payload)
    assert request.dialogue.reference_continuation[0].index == 4


def test_reference_index_zero_rejected_by_field():
    payload = _with_reference(fx.task_request(), [
        {"kind": "remove", "index": 0},
    ])
    with pytest.raises(ValidationError):
        InterjectRequest.model_validate(payload)


def test_reference_out_of_range_rejected_over_http(memory_store):
    payload = _with_reference(fx.daily_request(), [
        {"kind": "edit", "index": 3, "name": fx.NPC, "text": "日常剩余只有 2 条"},
    ])
    response = _client().post("/game/dialogue-interject", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# turn_id 幂等（协议 6.5）
# ---------------------------------------------------------------------------


async def test_same_turn_id_reuses_session_and_does_not_duplicate(memory_store):
    request = make_request(fx.daily_request())
    first = await ensure_session(request, store=memory_store)
    second = await ensure_session(request, store=memory_store)

    assert first.session_id == second.session_id
    assert first.turn_duplicate is False
    assert second.turn_duplicate is True
    assert memory_store.dialogue_turn_count_sync(first.session_id) == 1
    assert [row.seq for row in memory_store.dialogue_ledger_read_sync(first.session_id)] == [0]

    # 本轮 $PC 只应经 finalize 写一次；重发/重复 finalize 都不追加
    finalize_request = FinalizeRequest.model_validate(fx.F01_FINALIZE)
    first_finalize = await finalize(finalize_request, store=memory_store)
    second_finalize = await finalize(finalize_request, store=memory_store)
    assert first_finalize.appended == 4
    assert second_finalize.appended == 0
    assert memory_store.count_messages_sync(first.session_id) == 5
    ledger = memory_store.dialogue_ledger_read_sync(first.session_id)
    assert [row.seq for row in ledger] == [0, 1, 2, 3, 4]
    assert sum(1 for row in ledger if row.seq == 1) == 1


async def test_different_turn_ids_register_two_turns_same_session(memory_store):
    first = await ensure_session(make_request(fx.daily_request()), store=memory_store)
    second = await ensure_session(make_request(fx.daily_turn_b_request()), store=memory_store)

    assert first.session_id == second.session_id
    assert second.turn_duplicate is False
    assert memory_store.dialogue_turn_count_sync(first.session_id) == 2
    assert memory_store.dialogue_turn_status_sync(
        first.session_id, fx.DAILY_TURN_A,
    ) == "started"


async def test_finalize_marks_turn_finalized_idempotently(memory_store):
    await ensure_session(make_request(fx.daily_request()), store=memory_store)
    payload = FinalizeRequest.model_validate(fx.F01_FINALIZE)

    first = await finalize(payload, store=memory_store)
    assert first.ok is True
    assert memory_store.dialogue_turn_status_sync(
        first.session_id, fx.DAILY_TURN_A,
    ) == "finalized"

    second = await finalize(payload, store=memory_store)
    assert second.appended == 0
    assert memory_store.dialogue_turn_count_sync(first.session_id) == 1
    assert memory_store.count_messages_sync(first.session_id) == 5


# ---------------------------------------------------------------------------
# finalize 并发冲突：记录并 409，不静默吞错
# ---------------------------------------------------------------------------


def test_concurrent_finalize_conflict_returns_409(memory_store, monkeypatch, caplog):
    async def boom(*_args, **_kwargs):
        raise sqlite3.IntegrityError("UNIQUE constraint failed: ledger.seq")

    monkeypatch.setattr(memory_store, "dialogue_played_append", boom)
    # session 需先存在，finalize 才走到对账
    info, _ = memory_store.dialogue_session_ensure_sync(
        fx.DAILY_DIALOGUE_ID, fx.NPC, "并发测试", None,
    )

    with caplog.at_level(logging.ERROR, logger="services.dialogue_interject.sessions"):
        response = _client().post(
            "/game/dialogue-interject/finalize", json=fx.F01_FINALIZE,
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "concurrent_finalize_conflict"
    assert "并发冲突" in caplog.text
    assert info.session_id  # session 已存在，冲突只影响本次对账

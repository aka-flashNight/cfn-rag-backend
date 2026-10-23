from __future__ import annotations

import asyncio
import time

import pytest

from services.dialogue_interject.item_claims import (
    CLAIM_TIMEOUT_S,
    ClaimExistsError,
    ItemClaimStore,
)
from services.dialogue_interject.item_ledger import (
    format_ledger_block,
    load_item_ledger_block,
    record_item_ledger,
)


def test_claim_state_machine():
    store = ItemClaimStore()
    assert store.pending("s1") is None
    claim = store.create("s1", npc="Andy Law", item_name="M4A1", quantity=1, turn_no=1)
    assert store.pending("s1") is claim
    with pytest.raises(ClaimExistsError):
        store.create("s1", npc="Andy Law", item_name="M4A1", quantity=1, turn_no=1)
    assert store.close("s1", "wrong") is None
    assert store.close("s1", claim.claim_id) is claim
    assert store.pending("s1") is None


def test_claim_expires():
    store = ItemClaimStore()
    claim = store.create("s1", npc="Andy Law", item_name="M4A1", quantity=1, turn_no=1)
    claim.created_at = time.time() - CLAIM_TIMEOUT_S - 1
    assert store.pending("s1") is None
    # 过期后可重新创建
    again = store.create("s1", npc="Andy Law", item_name="M4A1", quantity=1, turn_no=2)
    assert again.claim_id != claim.claim_id


def test_gift_detail_merges_same_turn():
    store = ItemClaimStore()
    store.record_gift("s1", turn_no=1, item_name="普通hp药剂", quantity=3)
    store.record_gift("s1", turn_no=1, item_name="普通hp药剂", quantity=2)
    store.record_gift("s1", turn_no=3, item_name="M4A1", quantity=1)
    block = store.gift_detail_block("s1")
    assert block == "【本组已赠送】\n第1轮：5个普通hp药剂\n第3轮：1个M4A1"
    store.reset("s1")
    assert store.gift_detail_block("s1") == ""


def test_recent_blocks_keep_three():
    store = ItemClaimStore()
    for index in range(5):
        store.record_success("s1", f"成功{index}")
        store.record_failure("s1", f"失败{index}")
    block = store.recent_block("s1")
    assert "成功0" not in block
    assert "成功4" in block
    assert "失败2" in block
    assert "失败4" in block


def test_ledger_format_and_persistence(memory_store):
    rows = [
        {"direction": "give", "item_name": "普通hp药剂", "quantity": 3},
        {"direction": "receive", "item_name": "M4A1", "quantity": 1},
    ]
    block = format_ledger_block(rows)
    assert "你曾经送给玩家以下物品：3个普通hp药剂" in block
    assert "玩家曾送给你以下物品：1个M4A1" in block
    empty = format_ledger_block([])
    assert "你曾经送给玩家以下物品：无" in empty
    assert "玩家曾送给你以下物品：无" in empty

    asyncio.run(record_item_ledger(
        memory_store, slot_key_value="slot-a", npc="Andy Law",
        direction="give", item_name="普通hp药剂", quantity=3,
    ))
    asyncio.run(record_item_ledger(
        memory_store, slot_key_value="slot-a", npc="Andy Law",
        direction="give", item_name="普通hp药剂", quantity=2,
    ))
    block = asyncio.run(load_item_ledger_block(
        memory_store, slot_key_value="slot-a", npc="Andy Law",
    ))
    assert "5个普通hp药剂" in block
    assert "玩家曾送给你以下物品：无" in block


def test_dialogue_action_first_result_wins(memory_store):
    assert memory_store.dialogue_action_save_sync(
        "a1", "s1", "item_grant", {"success": True},
    ) is True
    assert memory_store.dialogue_action_save_sync(
        "a1", "s1", "item_grant", {"success": False},
    ) is False
    row = memory_store.dialogue_action_get_sync("a1")
    assert row is not None
    assert row["result"] == {"success": True}

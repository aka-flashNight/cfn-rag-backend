"""session 单测：创建/复用、played 不含本轮 $PC、finalize 全量对账、幂等、未播不写。"""

from __future__ import annotations

from services.dialogue_interject.schemas import FinalizeRequest
from services.dialogue_interject.sessions import ensure_session, finalize, map_played_line
from tests.dialogue_interject import fixtures as fx
from tests.dialogue_interject.helpers import make_request


def _finalize(payload: dict) -> FinalizeRequest:
    return FinalizeRequest.model_validate(payload)


async def test_first_interject_creates_session_and_seeds_played(memory_store):
    binding = await ensure_session(make_request(fx.daily_request()), store=memory_store)

    assert binding.created is True
    assert binding.session_id
    assert binding.title == "日常插话 · Andy Law · 2026-10-18 14:03"

    ledger = memory_store.dialogue_ledger_read_sync(binding.session_id)
    assert [(row.seq, row.role, row.content) for row in ledger] == [
        (0, "assistant", "又是你。")
    ]
    # 本轮 $PC 由 player_input 表示，不在请求阶段写库
    history = memory_store.get_history_sync(binding.session_id)
    assert [message.content for message in history] == ["又是你。"]


async def test_second_turn_reuses_session_without_duplicate_seq(memory_store):
    first = await ensure_session(make_request(fx.daily_request()), store=memory_store)
    second = await ensure_session(make_request(fx.daily_turn_b_request()), store=memory_store)

    assert second.created is False
    assert second.session_id == first.session_id

    ledger = memory_store.dialogue_ledger_read_sync(first.session_id)
    assert [row.seq for row in ledger] == [0, 1, 2]
    assert [row.role for row in ledger] == ["assistant", "user", "assistant"]
    assert ledger[1].content == "最近怎么样？"
    assert ledger[2].content == "还能怎么样，老样子。"


async def test_finalize_f01_writes_full_played_transcript_once(memory_store):
    binding = await ensure_session(make_request(fx.daily_request()), store=memory_store)
    result = await finalize(_finalize(fx.F01_FINALIZE), store=memory_store)

    assert result.ok is True
    assert result.dialogue_instance_id == fx.DAILY_DIALOGUE_ID
    assert result.session_id == binding.session_id
    assert result.appended == 4  # seq 1..4（seq0 请求阶段已种入）
    assert result.total == 5

    ledger = memory_store.dialogue_ledger_read_sync(binding.session_id)
    assert [row.seq for row in ledger] == [0, 1, 2, 3, 4]
    assert [row.role for row in ledger] == [
        "assistant", "user", "assistant", "assistant", "assistant"
    ]
    assert ledger[1].content == "最近怎么样？"

    again = await finalize(_finalize(fx.F01_FINALIZE), store=memory_store)
    assert again.appended == 0
    assert again.total == 5
    assert memory_store.count_messages_sync(binding.session_id) == 5


async def test_unplayed_lines_are_not_written(memory_store):
    """finalize 只按 Host 已播现场写：未出现的 add/closing 不落库。"""
    binding = await ensure_session(make_request(fx.daily_request()), store=memory_store)
    partial = {
        "dialogue_instance_id": fx.DAILY_DIALOGUE_ID,
        "turn_id": fx.DAILY_TURN_A,
        "transcript": fx.F01_FINALIZE["transcript"][:3],
    }
    result = await finalize(_finalize(partial), store=memory_store)
    assert result.appended == 2
    assert result.total == 3

    contents = [row.content for row in memory_store.dialogue_ledger_read_sync(binding.session_id)]
    assert "你倒是难得过来一趟。" not in contents
    assert "说吧，想聊什么？" not in contents


async def test_other_npc_maps_to_user_with_bracketed_name(memory_store):
    payload = fx.daily_request()
    payload["dialogue"]["played"] = [
        {"seq": 0, "role": "other_npc", "name": "老赵", "text": "别挡道。"},
    ]
    binding = await ensure_session(make_request(payload), store=memory_store)
    ledger = memory_store.dialogue_ledger_read_sync(binding.session_id)
    assert [(row.role, row.content) for row in ledger] == [("user", "【老赵：别挡道。】")]


def test_map_played_line_roles():
    base = {"seq": 1, "name": fx.NPC, "text": "台词"}
    assert map_played_line(make_request(fx.daily_request()).dialogue.played[0])["role"] == "assistant"
    assert map_played_line(
        FinalizeRequest.model_validate({
            "dialogue_instance_id": fx.DAILY_DIALOGUE_ID,
            "transcript": [{**base, "role": "player", "name": "$PC"}],
        }).transcript[0]
    ) == {"seq": 1, "role": "user", "content": "台词"}
    assert map_played_line(
        FinalizeRequest.model_validate({
            "dialogue_instance_id": fx.DAILY_DIALOGUE_ID,
            "transcript": [{**base, "role": "ai"}],
        }).transcript[0]
    ) == {"seq": 1, "role": "assistant", "content": "台词"}
    assert map_played_line(
        FinalizeRequest.model_validate({
            "dialogue_instance_id": fx.DAILY_DIALOGUE_ID,
            "transcript": [{**base, "role": "other_npc", "name": "老赵"}],
        }).transcript[0]
    ) == {"seq": 1, "role": "user", "content": "【老赵：台词】"}


async def test_finalize_unknown_session_is_rebuilt_from_transcript(memory_store):
    request = _finalize({
        "dialogue_instance_id": "99999999-9999-4999-8999-999999999999",
        "transcript": [
            {"seq": 0, "role": "npc", "name": fx.NPC, "text": "你来了。", "emotion": "普通"},
        ],
    })
    result = await finalize(request, store=memory_store)

    assert result.ok is True
    assert result.appended == 1
    info = memory_store.dialogue_session_get_sync("99999999-9999-4999-8999-999999999999")
    assert info is not None
    assert info.npc_name == fx.NPC
    assert "对话插话" in info.title


async def test_finalize_unknown_session_without_transcript_is_not_created(memory_store):
    request = _finalize({
        "dialogue_instance_id": "88888888-8888-4888-8888-888888888888",
        "transcript": [],
    })
    result = await finalize(request, store=memory_store)
    assert result.ok is False
    assert result.session_id == ""
    assert memory_store.dialogue_session_get_sync(
        "88888888-8888-4888-8888-888888888888"
    ) is None


async def test_finalize_corrects_content_without_duplicating(memory_store):
    binding = await ensure_session(make_request(fx.daily_request()), store=memory_store)
    first = _finalize({
        "dialogue_instance_id": fx.DAILY_DIALOGUE_ID,
        "transcript": [
            {"seq": 0, "role": "npc", "name": fx.NPC, "text": "又是你。"},
            {"seq": 1, "role": "player", "name": "$PC", "text": "最近怎么样？"},
        ],
    })
    await finalize(first, store=memory_store)

    corrected = _finalize({
        "dialogue_instance_id": fx.DAILY_DIALOGUE_ID,
        "transcript": [
            {"seq": 0, "role": "npc", "name": fx.NPC, "text": "又是你。"},
            {"seq": 1, "role": "player", "name": "$PC", "text": "最近还好吗？"},
        ],
    })
    result = await finalize(corrected, store=memory_store)

    assert result.appended == 0
    assert result.total == 2
    history = memory_store.get_history_sync(binding.session_id)
    assert [message.content for message in history] == ["又是你。", "最近还好吗？"]

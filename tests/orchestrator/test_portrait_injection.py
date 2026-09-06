"""P7 编排层接入测试（07 §4/§8.3/§8.4）：

- 仅 purpose=chat 且模型 Profile 允许视觉时带图（vision=True/探测型）；
- vision=False 全程无图不失败；被标记图像不支持的模型不再带图；
- 立绘情绪跟随 current_emotion（上回合 meta emo）；
- appearance 有则恒入聊天 prompt（与是否发图无关），无则不报错。
"""

from __future__ import annotations

import pytest

from services.llm import LLMConfig
from services.llm.client import (
    _IMAGE_UNSUPPORTED_MODELS,
    _IMAGE_UNSUPPORTED_LOCK,
    mark_image_unsupported,
)
from services.orchestrator.context import should_send_image
from services.orchestrator.turn import TurnOrchestrator
from services.orchestrator.prompts import build_static_system
from services.npc.manager import NPCState
from tests.orchestrator.conftest import NPC_NAME, collect

FAKE_IMAGE_URL = "data:image/webp;base64,QUJD"


@pytest.fixture(autouse=True)
def _no_settings_merge(monkeypatch):
    """测试显式给 model_name，避免回落到本机 .env 的默认模型。"""
    monkeypatch.setattr(LLMConfig, "merged_with_settings", lambda self: self)


@pytest.fixture
def inject_fake_portrait(monkeypatch):
    """让装配层取到假图；断言里可读出实际请求的情绪。"""
    requested: list[tuple[str, str]] = []

    def fake_load(npc_name: str, emotion):
        requested.append((npc_name, emotion or "普通"))
        return FAKE_IMAGE_URL

    monkeypatch.setattr(
        "services.orchestrator.context.load_portrait_image_url", fake_load
    )
    return requested


def _image_urls(msg) -> list[str]:
    content = msg["content"]
    if not isinstance(content, list):
        return []
    return [p["image_url"]["url"] for p in content if p.get("type") == "image_url"]


def _vision_orch(env, query: str, *, current_emotion=None) -> TurnOrchestrator:
    return TurnOrchestrator(
        session_id=env.session_id,
        npc_name=NPC_NAME,
        query=query,
        current_emotion=current_emotion,
        llm_config=LLMConfig(model_name="deepseek-v4-flash-vision-exp"),
        deps=env.deps,
    )


# ---------------------------------------------------------------------------
# should_send_image（07 §4 判定）
# ---------------------------------------------------------------------------

def test_should_send_image_matrix():
    assert should_send_image(LLMConfig(model_name="deepseek-v4-flash-vision-exp")) is True
    assert should_send_image(LLMConfig(model_name="deepseek-v4-flash")) is False  # 纯文本 Profile
    # 探测型（DEFAULT_PROFILE）允许尝试带图，失败由 client 降级链兜底
    assert should_send_image(LLMConfig(model_name="totally-new-model")) is True


def test_should_send_image_false_after_unsupported_mark():
    model = "vision-marked-test-model"
    assert should_send_image(LLMConfig(model_name=model)) is True
    mark_image_unsupported(model)
    try:
        assert should_send_image(LLMConfig(model_name=model)) is False
    finally:
        with _IMAGE_UNSUPPORTED_LOCK:
            _IMAGE_UNSUPPORTED_MODELS.discard(model)


# ---------------------------------------------------------------------------
# 调用 #1 的图片注入
# ---------------------------------------------------------------------------

async def test_vision_model_request_carries_image(env, inject_fake_portrait):
    env.fake.add_stream(meta='{"emo":"微笑","fav":0,"act":null}', text="好呀。")
    orch = _vision_orch(env, "今天干什么", current_emotion="生气")
    await collect(orch)

    req = env.fake.stream_requests[0]
    assert req.purpose == "chat"
    msg = req.messages[-1]
    assert _image_urls(msg) == [FAKE_IMAGE_URL]
    # 文本 part 在前、图片 part 在后（02 §3.3）
    content = msg["content"]
    assert content[0]["type"] == "text" and "玩家：今天干什么" in content[0]["text"]
    assert content[-1]["type"] == "image_url"
    # 立绘情绪 = current_emotion（meta 未出前的请求侧来源，07 §3）
    assert inject_fake_portrait == [(NPC_NAME, "生气")]


async def test_default_emotion_is_normal_when_no_current(env, inject_fake_portrait):
    env.fake.add_stream(meta='{"emo":"普通","fav":0,"act":null}', text="嗯。")
    orch = _vision_orch(env, "在吗", current_emotion=None)
    await collect(orch)
    # 首轮 meta 未出 → 用「普通」
    assert inject_fake_portrait == [(NPC_NAME, "普通")]


async def test_non_vision_model_never_has_image(env):
    env.fake.add_stream(meta='{"emo":"普通","fav":0,"act":null}', text="哦。")
    orch = TurnOrchestrator(
        session_id=env.session_id,
        npc_name=NPC_NAME,
        query="你好",
        current_emotion="微笑",
        llm_config=LLMConfig(model_name="deepseek-v4-flash"),  # vision=False
        deps=env.deps,
    )
    events = await collect(orch)
    assert not [e for e in events if e.event == "error"]
    msg = env.fake.stream_requests[0].messages[-1]
    assert _image_urls(msg) == []
    assert isinstance(msg["content"], str)  # 纯文本消息


async def test_marked_unsupported_model_has_no_image(env, inject_fake_portrait):
    model = "vision-marked-orch-test"
    mark_image_unsupported(model)
    try:
        env.fake.add_stream(meta='{"emo":"普通","fav":0,"act":null}', text="哦。")
        orch = TurnOrchestrator(
            session_id=env.session_id,
            npc_name=NPC_NAME,
            query="你好",
            current_emotion="微笑",
            llm_config=LLMConfig(model_name=model),
            deps=env.deps,
        )
        events = await collect(orch)
        assert not [e for e in events if e.event == "error"]
        assert inject_fake_portrait == []  # 装配层直接跳过取图
        assert _image_urls(env.fake.stream_requests[0].messages[-1]) == []
    finally:
        with _IMAGE_UNSUPPORTED_LOCK:
            _IMAGE_UNSUPPORTED_MODELS.discard(model)


async def test_portrait_failure_does_not_break_turn(env, monkeypatch):
    """provider 取图抛异常 → 本回合无图，聊天主流程不受影响（07 §1）。"""

    def boom(*_a, **_k):
        raise RuntimeError("PIL boom")

    monkeypatch.setattr("services.portraits.get_portrait_data_url", boom)
    env.fake.add_stream(meta='{"emo":"普通","fav":0,"act":null}', text="没事。")
    orch = _vision_orch(env, "你好")
    events = await collect(orch)
    assert not [e for e in events if e.event == "error"]
    assert _image_urls(env.fake.stream_requests[0].messages[-1]) == []


async def test_subagent_and_merge_calls_never_carry_image(env, inject_fake_portrait):
    """任何子 Agent / 汇合调用永不带图（07 §4）：merge 用的 base_messages 是纯文本。"""
    env.fake.add_stream(
        meta='{"emo":"普通","fav":0,"act":null,"kind":"search","query":"废城"}', text="我查查。"
    )
    env.fake.add_chat(content='{"conclusion":"废城很危险"}')
    env.fake.add_stream(text="废城到处是天网残兵，别单独去。")
    orch = _vision_orch(env, "查一下废城")
    events = await collect(orch)
    assert not [e for e in events if e.event == "error"]
    # 调用 #1 带图；后续所有流式调用（merge #2）均无图
    assert _image_urls(env.fake.stream_requests[0].messages[-1]) == [FAKE_IMAGE_URL]
    for req in env.fake.stream_requests[1:]:
        assert _image_urls(req.messages[-1]) == []


# ---------------------------------------------------------------------------
# appearance 注入（07 §5）
# ---------------------------------------------------------------------------

def test_appearance_block_in_prompt_when_present():
    state = NPCState(favorability=10, appearance="银白色短发，赤瞳")
    prompt = build_static_system(npc_name="凯特", state=state)
    assert "【你的形象】" in prompt and "银白色短发，赤瞳" in prompt


def test_appearance_absent_ok():
    prompt = build_static_system(npc_name="铁匠", state=NPCState(favorability=10))
    assert "【你的形象】" not in prompt

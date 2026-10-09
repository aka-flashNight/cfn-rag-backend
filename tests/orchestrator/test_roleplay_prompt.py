"""聊天主 Agent 的角色扮演提示词回归（v2 口径还原）。

v3 重构时丢了一批 v2 业务验证过的扮演提示词，同时把问答模式的「陈述语气」误抄
进了扮演规则。本文件钉住还原后的措辞，防止再次被当成冗余删掉：
- system：情绪立绘选择句、以免出戏逃生口、发布动机自然融入；
- user shared core：检索为空时的自由发挥兜底、历史块引导语（按有无摘要二选一）；
- 末条 user：立绘文字说明 + 上一轮情绪连贯提示（生成前最后读到的文字）；
- 输出规则里不得再出现「陈述语气」这类会压死口语的硬要求。
"""

from __future__ import annotations

from services.orchestrator.prompts import (
    RAG_EMPTY_FALLBACK,
    TASK_MERGE_INSTRUCTION,
    build_emotion_hint,
    build_player_message,
    build_portrait_note,
)
from services.orchestrator.turn import TurnOrchestrator
from tests.orchestrator.conftest import NPC_NAME, collect

SUMMARY_TEXT = "玩家早先替铁匠跑过一趟腿，还问过废城的路。"
CONTINUITY = "在保持人物性格与情节连贯的前提下继续对话："


def _orch(env, query: str, **kwargs) -> TurnOrchestrator:
    return TurnOrchestrator(
        session_id=env.session_id,
        npc_name=NPC_NAME,
        query=query,
        deps=env.deps,
        **kwargs,
    )


def _system_prompt(env) -> str:
    return env.fake.stream_requests[0].messages[0]["content"]


def _shared_core(env) -> str:
    return env.fake.stream_requests[0].messages[1]["content"]


def _last_user(env) -> str:
    return env.fake.stream_requests[0].messages[-1]["content"]


async def _seed_history(env, count: int) -> None:
    for i in range(1, count + 1):
        role = "user" if i % 2 == 1 else "assistant"
        await env.store.add_message(env.session_id, role, f"历史消息{i:02d}")


async def _one_turn(env, query: str = "你好", **kwargs) -> None:
    env.fake.add_stream(meta='{"emo":"普通","fav":0}', text="嗯。")
    await collect(_orch(env, query, **kwargs))


# ---------------------------------------------------------------------------
# system：v2 扮演措辞
# ---------------------------------------------------------------------------

async def test_system_prompt_keeps_v2_roleplay_lines(env):
    await _one_turn(env)
    system = _system_prompt(env)
    # 情绪标签不是 JSON 填空，而是一次表演选择
    assert "请选择其中最合适的一种作为你当前的情绪立绘。" in system
    # 答不上来时的逃生口：少了它模型只能硬编，硬编就是伪人感的来源
    assert "无法把握的模糊地带可以略过或转移话题，不要自己乱加设定，以免出戏。" in system


async def test_task_constraint_requires_motivation_woven_into_dialogue(env):
    await _one_turn(env, "有什么活给我干吗")
    assert "发布动机要自然融入对话；" in _system_prompt(env)


async def test_dialogue_rules_do_not_force_declarative_tone(env):
    """「陈述语气」是 v2 问答助手模式的措辞，误抄进扮演规则会压死疑问/感叹/口语。"""
    await _one_turn(env)
    system = _system_prompt(env)
    assert "陈述语气" not in system
    assert "用简体中文；" in system


# ---------------------------------------------------------------------------
# user shared core：检索兜底与历史引导语
# ---------------------------------------------------------------------------

async def test_empty_retrieval_falls_back_to_free_play_hint(env):
    """env 的 engine=None → Tier-1 检索恒空，此时仍要给「自由发挥」的授权。"""
    await _one_turn(env)
    assert RAG_EMPTY_FALLBACK in _shared_core(env)


async def test_history_block_carries_continuity_instruction(env):
    await _seed_history(env, 4)
    await _one_turn(env)
    core = _shared_core(env)
    assert "下面是你与玩家之间的对话历史（按时间从早到晚排列），" in core
    assert "请在保持人物性格与情节连贯的前提下继续对话：\n" in core
    assert CONTINUITY in core
    assert "历史消息01" in core


async def test_history_header_switches_when_summary_present(env):
    """有摘要时引导语换成 v2 的「结合上述摘要与近期记录」版本。"""
    await env.store.save_summary(env.session_id, SUMMARY_TEXT, 30)
    await _seed_history(env, 4)
    await _one_turn(env)
    core = _shared_core(env)
    assert f"当前对话历史较长，早期对话已整理为以下摘要：\n{SUMMARY_TEXT}" in core
    assert "以下是最近的对话记录（按时间从早到晚排列），" in core
    assert "请结合上述摘要与近期记录，" in core
    assert CONTINUITY in core
    # 摘要块排在历史块之前（前缀缓存对齐：摘要比对话稳定）
    assert core.index(SUMMARY_TEXT) < core.index("以下是最近的对话记录")


async def test_history_block_omitted_when_no_history(env):
    """首轮无历史时整块不注入，不给模型一个空对话框。"""
    await _one_turn(env)
    core = _shared_core(env)
    assert CONTINUITY not in core
    assert "（暂无对话记录）" not in core


# ---------------------------------------------------------------------------
# 末条 user：立绘说明与情绪连贯提示
# ---------------------------------------------------------------------------

def test_portrait_note_names_the_character():
    assert build_portrait_note("铁匠") == (
        "现在传入了你所扮演的角色「铁匠」的肖像（不需要强行通过肖像内容开启话题）。"
    )
    assert "「当前 NPC」" in build_portrait_note("  ")


def test_emotion_hint_skips_default_and_blank():
    assert build_emotion_hint("微笑") == "你之前的情绪是「微笑」。"
    assert build_emotion_hint("普通") == ""
    assert build_emotion_hint(None) == ""
    assert build_emotion_hint("   ") == ""


def test_player_message_tail_order_and_omission():
    both = build_player_message("在吗", npc_name="铁匠", has_image=True, current_emotion="微笑")
    assert both == (
        "玩家：在吗\n\n"
        "现在传入了你所扮演的角色「铁匠」的肖像（不需要强行通过肖像内容开启话题）。"
        "你之前的情绪是「微笑」。"
    )
    # 无图无历史情绪 = 与还原前逐字一致，不凭空加尾巴
    assert build_player_message("在吗", npc_name="铁匠") == "玩家：在吗"
    assert build_player_message("在吗", npc_name="铁匠", current_emotion="普通") == "玩家：在吗"


async def test_emotion_hint_reaches_last_user_message(env):
    """上一轮情绪要进 prompt：v3 一度只用它挑立绘，模型每轮情绪从零起跳。"""
    await _one_turn(env, "还在生气吗", current_emotion="生气")
    assert "你之前的情绪是「生气」。" in _last_user(env)


async def test_no_emotion_hint_on_first_turn(env):
    await _one_turn(env, "你好")
    assert _last_user(env) == "玩家：你好"


# ---------------------------------------------------------------------------
# 汇合指令：候选句降为参考，但「征询是否接取」这句硬要求保留
# ---------------------------------------------------------------------------

def test_merge_instruction_marks_candidates_as_reference_only():
    assert "仅供参考，不要照抄" in TASK_MERGE_INSTRUCTION
    assert "用自己的话问" in TASK_MERGE_INSTRUCTION
    # 结尾征询接取是流程硬要求，不能因为反模板就一起删掉
    assert "结尾必须征询玩家是否接取" in TASK_MERGE_INSTRUCTION
    assert "不能替玩家拍板" in TASK_MERGE_INSTRUCTION

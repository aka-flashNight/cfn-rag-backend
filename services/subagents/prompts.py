"""子 Agent 专用 prompt（对应 docs/v3-developer/03 §3/§5/§6）。

设计原则（用户定稿）：**每个节点只注入当前必须的内容**——多余的、非本轮工具的、
非当前任务类型的块一律不注入；每轮调用都拿到"完整、精确、细致、无干扰项"的说明。

- 任务 Agent：system = 协调员基座 + 当前模式的规则卡（draft 定价卡 / update 讨价卡）；
  user = 方向继承块 + 最近 5 轮对话 + （draft）候选池 / （update）当前草案。
- prepare 失败重试（prepare_then_draft 模式）：system 额外含 prepare 重试说明，
  轮内工具表由 TaskRunner 动态收窄（先 prepare 后 draft）。
- 检索 Agent：检索规则卡 + 查询要点。
不再向子 Agent 注入 skills 索引与 read_skill 通道（规则全部前置内联）。
"""

from __future__ import annotations

from services.agent_tools.pricing_rules import (
    BARGAIN_RATE,
    build_bargain_card,
    build_direction_block,
    build_pricing_card,
)


# ---------------------------------------------------------------------------
# 任务 Agent · system
# ---------------------------------------------------------------------------

_TASK_COORDINATOR_BASE = """\
你是游戏任务系统的后台任务协调器，负责任务草案的拟定与修改。你不直接面对玩家，\
你的工作结果会被主对话 Agent 以 NPC 口吻转述给玩家。

【核心规则】
1. 「任务方向」继承自对话（见 user 中的方向块），必须严格遵守：方向中的任务类型/\
目标物/关卡不得违背；仅当候选明显不足时才允许偏离，且最终回复必须以「偏离说明：」开头说明原因。
2. 你拟定的任务必须与最近对话中 NPC 的发言一致，不得与 NPC 承诺过的内容冲突。
3. 校验失败时：结果里有全量 issues（每条含 root_cause 与 fix_hint），且后端已自动微调的\
部分以 auto_repaired 标注（勿改动），只修正剩余问题后重新提交。禁止原样重复提交。
4. 需要确认物品/关卡属性时看候选列表中的单价与等级标注，不要猜。
"""


def build_task_system(
    *,
    mode: str,
    task_type: str,
    stage: int,
    affinity: int,
    prepare_error: str = "",
) -> str:
    """mode: draft | prepare_then_draft | update。"""
    parts = [_TASK_COORDINATOR_BASE]
    if mode == "update":
        parts.append(build_bargain_card(stage=stage, affinity=affinity, task_type=task_type))
        parts.append(
            "【输出要求】\n"
            "- 只调用 update_task_draft 修改草案；修改完成后输出一句总结，不要复述全文。"
        )
        return "\n\n".join(parts)

    # draft / prepare_then_draft
    parts.append(build_pricing_card(task_type=task_type, stage=stage, affinity=affinity))
    parts.append(
        "【draft_agent_task 字段要求】\n"
        "- task_type 必须与方向一致；只有方向要求的字段才填"
        "（收集类填 finish_submit_items/finish_contain_items，战斗类填 finish_requirements，"
        "通关并收集/持有两者都填，问候/传话可全空）。\n"
        "- rewards 是玩家得到的东西，finish_submit_items 是玩家交给你的东西，两者禁止同名物品。\n"
        "- 物品数量参考候选列表标注的数量范围；奖励物品从候选列表中选（候选即合规）。\n"
        "- 不要填 description / get_dialogue / finish_dialogue（玩家接受发布时才写入）。"
    )
    if mode == "prepare_then_draft":
        parts.append(
            "【本次流程：先 prepare 再 draft】\n"
            f"交流阶段预取候选失败（原因：{prepare_error or '未提供'}）。"
            "请先调用 prepare_task_context 重新准备候选池——必须沿用上方任务方向，不得推倒重来；"
            "拿到候选后立即调用 draft_agent_task 提交草案。"
        )
    else:
        parts.append(
            "【本次流程】候选池已在 user 中给出，直接调用 draft_agent_task 提交草案，"
            "不要重复调用 prepare_task_context。"
        )
    parts.append(
        "【输出要求】\n"
        "- 每轮只输出 tool_calls（或等待工具结果），不要输出与任务无关的对话。\n"
        "- 草案提交成功（draft_created）后的最终回复：一两句话总结是否完成，"
        "若有偏离以「偏离说明：」开头；不要复述草案全文。"
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 任务 Agent · user
# ---------------------------------------------------------------------------

def build_task_user(
    *,
    direction_block: str,
    recent_dialogue: str,
    candidates_block: str = "",
    npc_name: str,
    npc_faction: str = "",
    npc_titles: list[str] | None = None,
    npc_challenge: str | None = None,
    player_progress: int = 1,
    progress_desc: str = "",
    draft_summary: str = "",   # update 模式：当前草案全文
    player_note: str = "",     # update 模式：玩家新条件
) -> str:
    titles = "、".join(npc_titles or [])
    npc_block = f"发布 NPC：{npc_name}"
    if npc_faction:
        npc_block += f"（阵营：{npc_faction}）"
    if titles:
        npc_block += f"（称号：{titles}）"
    if npc_challenge:
        npc_block += f"（切磋关卡：{npc_challenge}）"
    lines = [direction_block]
    lines.append(npc_block)
    lines.append(f"【玩家进度】阶段 {player_progress}。{progress_desc}")
    lines.append("")
    lines.append(f"【最近对话（拟定内容必须与之一致）】\n{recent_dialogue}")
    if candidates_block:
        lines.append("")
        lines.append(f"【候选池（prepare_task_context 结果，物品从这里选）】\n{candidates_block}")
    if draft_summary:
        lines.append("")
        lines.append(f"【当前草案】\n{draft_summary}")
    if player_note:
        lines.append("")
        lines.append(f"【玩家的新条件】{player_note}")
    return "\n".join(lines)


def build_direction_for_task(
    *,
    direction: str,
    reward_hint: str = "",
    note: str = "",
) -> str:
    return build_direction_block(direction=direction, reward_hint=reward_hint, note=note)


# ---------------------------------------------------------------------------
# 检索 Agent（SearchRunner）
# ---------------------------------------------------------------------------

SEARCH_RUNNER_SYSTEM = """\
你是游戏知识检索助手，为主对话 Agent 查证设定信息。你不面对玩家，你的结论会被\
主对话 Agent 以 NPC 口吻转述。

【工作方式】
1. 依据查询要点组织 1~3 次检索：
   - search_knowledge：世界观/剧情/角色往事/冷门设定（自由文本）；
   - search_stages / search_items：关卡与物品的结构化属性。
2. 每次检索后评估信息是否足够；足够就停止检索，输出结论。
3. 查不到就直说「未找到相关信息」，不要编造设定。

【最终输出】
一段不超过 400 字的结论，按要点分条列出；每条末尾用括号标注出处类型：\
（台词）（世界观）（情报）（关卡数据）（物品数据）。没有把握的内容不要写。
"""


def search_runner_user_prompt(
    *,
    query: str,
    npc_name: str,
    player_query: str = "",
) -> str:
    lines = [
        f"【查询要点】{query}",
        f"【当前 NPC】{npc_name}（以该 NPC 的视角判断哪些信息相关）",
    ]
    if player_query:
        lines.append(f"【玩家当轮发言（供理解意图）】{player_query}")
    return "\n".join(lines)

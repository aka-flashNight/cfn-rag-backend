"""子 Agent 专用小 prompt（对应 docs/v3-developer/03 §3 / 01 §6.2）。

与聊天 prompt **完全分离**：不含扮演约束/立绘/大部分世界观，只含任务规则或
检索规则 + Skills 索引（渐进披露 Level 1 注入，Level 2/3 由元工具按需加载）。
"""

from __future__ import annotations

from typing import Optional

from services.skills import SkillRegistry

# 每 Agent 可见的 skill 白名单（沿用旧 tool_scopes 资产）
_TASK_SKILL_WHITELIST = frozenset({"task-publishing", "task-bargaining", "knowledge-search", "skill-discovery"})
_SEARCH_SKILL_WHITELIST = frozenset({"knowledge-search", "skill-discovery"})


def skills_index_block(registry: Optional[SkillRegistry], whitelist: frozenset[str]) -> str:
    """Level 1 简表（白名单过滤）；注册表不可用或为空返回空串。"""
    if registry is None:
        return ""
    rows = registry.index()
    rows = [r for r in rows if r["name"] in whitelist]
    if not rows:
        return ""
    lines = ["【可用 Skills 索引（需要完整流程时用 read_skill(skill_name) 拉取）】"]
    for r in rows:
        lines.append(f"- {r['name']}: {r['description']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TaskRunner
# ---------------------------------------------------------------------------

TASK_RUNNER_SYSTEM = """\
你是游戏任务系统的后台任务协调器，负责任务草案的拟定与修改。你不直接面对玩家，\
你的工作结果会被主对话 Agent 以 NPC 口吻转述给玩家。

【核心规则】
1. 「任务方向」由主对话给出，必须严格遵守：方向中明确的任务类型/目标物/关卡不得违背；\
仅当候选明显不足时才允许偏离，且最终回复必须以「偏离说明：」开头说明原因。
2. 两步式流程：先调用 prepare_task_context 获取候选集与预算规则，再调用 draft_agent_task 提交草案。\
没有候选不要凭空编造物品/关卡。
3. 校验失败时：结果里有全量 issues（每条含 root_cause 与 fix_hint），按 fix_hint 逐条修正后重试；\
后端已自动修复的项在 repaired_notes 中标注，无需再改。禁止原样重复提交。
4. 修改已有草案（玩家讨价还价/微调）只用 update_task_draft(draft_id, modify_fields)；\
讨价还价上限 2 次；不要传 description/get_dialogue/finish_dialogue。
5. 奖励规则：奖励偏好（reward_hint）优先满足；与价值区间（V7）/类型合规（V8）冲突时以校验为准。
6. 需要确认物品/关卡属性时用 search_items / search_stages，不要猜。
7. 不确定流程/协商规则时先 list_skills，再 read_skill("task-publishing" / "task-bargaining")。

【输出要求】
- 每轮只输出 tool_calls（或等待工具结果），不要输出与任务无关的对话。
- 任务完成（draft_created/draft_updated）后的最终回复：一两句话总结是否完成，\
若有偏离以「偏离说明：」开头；不要复述草案全文（主对话 Agent 能看到 draft_summary）。
"""


def task_runner_user_prompt(
    *,
    direction: str,
    reward_hint: str = "",
    note: str = "",
    player_query: str = "",
    npc_name: str,
    npc_faction: str = "",
    npc_titles: Optional[list[str]] = None,
    npc_challenge: Optional[str] = None,
    player_progress: int = 1,
    progress_desc: str = "",
    pending_draft_summary: str = "",
    skill_registry: Optional[SkillRegistry] = None,
) -> str:
    """TaskRunner 首轮强制注入（03 §5）：方向/NPC/进度/草案/Skills 简表。"""
    titles = "、".join(npc_titles or [])
    lines = [
        "【任务方向（必须严格遵守）】",
        direction or "（主对话未给出明确方向，按 NPC 身份拟定一个合理的小任务）",
    ]
    if reward_hint:
        lines.append(f"【奖励偏好】{reward_hint}")
    if note:
        lines.append(f"【备注】{note}")
    lines.append("")
    npc_block = f"发布 NPC：{npc_name}"
    if npc_faction:
        npc_block += f"（阵营：{npc_faction}）"
    if titles:
        npc_block += f"（称号：{titles}）"
    if npc_challenge:
        npc_block += f"（切磋关卡：{npc_challenge}）"
    lines.append(npc_block)
    lines.append(f"【玩家进度】阶段 {player_progress}。{progress_desc}")
    if pending_draft_summary:
        lines.append("")
        lines.append("【当前待修改草案（update 模式，调用 update_task_draft 时使用其 draft_id）】")
        lines.append(pending_draft_summary)
    if player_query:
        lines.append("")
        lines.append(f"【玩家当轮发言（供理解意图，不要回复它）】{player_query}")
    skills = skills_index_block(skill_registry, _TASK_SKILL_WHITELIST)
    if skills:
        lines.append("")
        lines.append(skills)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SearchRunner
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
    skill_registry: Optional[SkillRegistry] = None,
) -> str:
    lines = [
        f"【查询要点】{query}",
        f"【当前 NPC】{npc_name}（以该 NPC 的视角判断哪些信息相关）",
    ]
    if player_query:
        lines.append(f"【玩家当轮发言（供理解意图）】{player_query}")
    skills = skills_index_block(skill_registry, _SEARCH_SKILL_WHITELIST)
    if skills:
        lines.append("")
        lines.append(skills)
    return "\n".join(lines)

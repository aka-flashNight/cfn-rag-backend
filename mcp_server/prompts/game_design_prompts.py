"""
游戏设计辅助 Prompt 模板。
"""

from __future__ import annotations

from fastmcp import FastMCP


def register_prompts(mcp: FastMCP) -> None:

    @mcp.prompt
    def design_task(
        task_type: str = "",
        npc_name: str = "",
        area: str = "",
    ) -> str:
        """生成任务设计 Prompt，用于协助策划创建与游戏设定一致的任务。

        task_type: 任务类型（greeting/message/clearance/cleanup/challenge/sparring/
                  resource-collection/equipment-submission/special-item-acquisition/
                  item-holding/clearance+collect/clearance+hold）
        npc_name: 发布任务的 NPC 名称
        area: 任务所属区域
        """
        return f"""你是一位 Crazy Flash Night 游戏的资深任务策划。请帮我设计一个任务。

任务类型: {task_type or '（待定）'}
发布 NPC: {npc_name or '（待定）'}
所属区域: {area or '（待定）'}

请先使用 search_items、search_stages 等工具查找相关数据，然后设计任务，包括：
1. 任务标题
2. 任务描述
3. 完成条件（关卡/物品提交/物品持有）
4. 奖励方案（确保奖励物品在游戏中存在）
5. NPC 接取和交付对话要点

设计原则：
- 奖励应符合任务难度和类型
- 物品名称和关卡名称必须来自游戏数据（通过 tools 验证）
- 物品数量要合理（参考同类型任务）
- 对话要符合 NPC 性格"""

    @mcp.prompt
    def write_npc_dialogue(npc_name: str = "", emotion: str = "") -> str:
        """生成 NPC 对话创作 Prompt。

        npc_name: NPC 名称
        emotion: 情绪标签（如 开心、生气、悲伤、惊讶）
        """
        return f"""你是一位游戏剧情写手，请为 Crazy Flash Night 游戏中的 NPC "{npc_name or '（待定）'}" 创作对话。

情绪基调: {emotion or '（待定，请根据上下文判断）'}

创作要求：
1. 先使用 search_tasks_by_npc 查看该 NPC 已有的任务对话，确保风格一致
2. 使用 search_knowledge 查看相关世界观设定
3. 对话要符合该 NPC 的身份和个性
4. 保持游戏的废土/科幻风格"""

    @mcp.prompt
    def explore_game_data(query: str = "") -> str:
        """游戏数据探索 Prompt，引导 AI 使用 MCP Tools 查询游戏数据。

        query: 用户的查询需求
        """
        return f"""请使用可用的工具查询 Crazy Flash Night 游戏数据来回答以下问题。

用户问题: {query or '（请帮我了解游戏中有哪些内容）'}

建议的查询路径：
1. 如果涉及物品，使用 search_items 或 get_item_detail
2. 如果涉及关卡，使用 search_stages 或 list_stages_in_area
3. 如果涉及 NPC，使用 list_npcs_with_shops
4. 如果涉及合成，使用 search_crafting
5. 如果涉及任务，使用 search_tasks_by_npc 或 list_agent_tasks

请综合查询结果，给出清晰、有条理的回答。"""

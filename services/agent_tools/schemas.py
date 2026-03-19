from __future__ import annotations

from typing import Any, List, TypedDict

from services.npc_mood_agent import UPDATE_NPC_MOOD_TOOL

# ---------------------------------------------------------------------------
# Enums / constants（来自 data_files_overview.md 的 6.3 / 6.4 章节）
# ---------------------------------------------------------------------------

TASK_TYPES: List[str] = [
    "问候",
    "传话",
    "通关",
    "清理",
    "挑战",
    "切磋",
    "资源收集",
    "装备缴纳",
    "特殊物品获取",
    "物品持有",
    "通关并收集",
    "通关并持有",
]

DIFFICULTIES: List[str] = ["简单", "冒险", "修罗", "地狱"]

REWARD_REGULAR: List[str] = ["金币", "经验"]
REWARD_OPTIONAL: List[str] = [
    "药剂",
    "弹夹",
    "K点",
    "技能点",
    "强化石",
    "战宠灵石",
    "材料",
    "食品",
    "武器",
    "防具",
    "插件",
]


# ---------------------------------------------------------------------------
# TypedDict（仅用于类型提示；运行时以 dict 结构为准）
# ---------------------------------------------------------------------------

class RewardTypes(TypedDict):
    regular: List[str]
    optional: List[str]


class RewardItem(TypedDict):
    item_name: str
    count: int


class StageRequirement(TypedDict, total=False):
    stage_name: str
    difficulty: str


class TaskDraft(TypedDict, total=False):
    task_type: str
    title: str
    description: str
    get_requirements: List[int]
    finish_requirements: List[StageRequirement]
    finish_submit_items: List[RewardItem]
    finish_contain_items: List[RewardItem]
    rewards: List[RewardItem]
    # 发布/完成 NPC：可选；为空时由后端默认当前 NPC
    get_npc: str
    finish_npc: str
    # 接取/完成对话：数组，每项为纯对话文本（不要包含【动作/神态/旁白】）
    get_dialogue: List[dict[str, Any]]
    finish_dialogue: List[dict[str, Any]]


# ---------------------------------------------------------------------------
# OpenAI Function Calling tools
# ---------------------------------------------------------------------------

PREPARE_TASK_CONTEXT_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_type": {"type": "string", "enum": TASK_TYPES},
        "reward_types": {
            "type": "object",
            "properties": {
                "regular": {
                    "type": "array",
                    "items": {"type": "string", "enum": REWARD_REGULAR},
                },
                "optional": {
                    "type": "array",
                    "items": {"type": "string", "enum": REWARD_OPTIONAL},
                },
            },
            "required": ["regular", "optional"],
            "additionalProperties": False,
        },
        # 用于 SSE/前端显示：非常短的“正在进行中”提示
        "ui_hint": {"type": "string", "maxLength": 12, "description": "前端显示的超短提示（<=12字），为空则后端使用默认提示。"},
    },
    "required": ["task_type", "reward_types"],
    "additionalProperties": False,
}

PREPARE_TASK_CONTEXT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "prepare_task_context",
        "description": "根据意向任务类型与奖励偏好筛选数据，返回该类型的完整上下文与规则说明。",
        "parameters": PREPARE_TASK_CONTEXT_PARAMETERS_SCHEMA,
    },
}


SEARCH_KNOWLEDGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": "复用现有 RAG 检索，获取设定/情报的摘要文本。",
        "parameters": {
            "type": "object",
            "properties": {"keyword": {"type": "string"}},
            "required": ["keyword"],
            "additionalProperties": False,
        },
    },
}


SEARCH_KNOWLEDGE_TOOL_PARAMETERS_SCHEMA: dict[str, Any] = SEARCH_KNOWLEDGE_TOOL["function"]["parameters"]


REWARD_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "item_name": {"type": "string"},
        "count": {"type": "integer", "minimum": 1},
    },
    "required": ["item_name", "count"],
    "additionalProperties": False,
}


STAGE_REQUIREMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "stage_name": {"type": "string"},
        "difficulty": {"type": "string", "enum": DIFFICULTIES},
    },
    "required": ["stage_name", "difficulty"],
    "additionalProperties": False,
}

TOP_NPC_EMOTION_HINT: str = "emotion：可选；用于拼接 char（NPC名#情绪 或 $PC_CHAR#情绪）。"

DIALOGUE_ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "NPC 名称或固定值 '$PC'（玩家）"},
        "title": {"type": "string", "description": "称号（NPC 称号或 '$PC_TITLE'）"},
        "emotion": {
            "type": "string",
            "description": TOP_NPC_EMOTION_HINT + "；允许空字符串表示默认情绪。",
        },
        "text": {
            "type": "string",
            "description": "纯对话内容（不要包含动作/神态/旁白/【...】）。"
        },
    },
    "required": ["name", "title", "text"],
    "additionalProperties": False,
}


DRAFT_AGENT_TASK_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_type": {"type": "string", "enum": TASK_TYPES},
        "title": {"type": "string", "description": "任务标题，简洁明了"},
        "description": {"type": "string", "description": "任务描述，简要说明目标"},
        "get_requirements": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "前置主线任务 ID 数组，空数组表示无前置。禁止使用 -1。",
        },
        "finish_requirements": {
            "type": "array",
            "items": STAGE_REQUIREMENT_SCHEMA,
        },
        "finish_submit_items": {
            "type": "array",
            "items": REWARD_ITEM_SCHEMA,
        },
        "finish_contain_items": {
            "type": "array",
            "items": REWARD_ITEM_SCHEMA,
        },
        "rewards": {
            "type": "array",
            "items": REWARD_ITEM_SCHEMA,
        },
        "get_npc": {"type": "string", "description": "接取时由谁发布（可为空，后端默认当前 NPC）"},
        "finish_npc": {"type": "string", "description": "完成时由谁发言（可为空，后端默认当前 NPC）"},
        "get_dialogue": {
            "type": "array",
            "items": DIALOGUE_ENTRY_SCHEMA,
            "description": "接取对话数组；可以包含 NPC 与玩家($PC)多条。",
        },
        "finish_dialogue": {
            "type": "array",
            "items": DIALOGUE_ENTRY_SCHEMA,
            "description": "完成对话数组；可以包含 NPC 与玩家($PC)多条。",
        },
        # 用于 SSE/前端显示：非常短的“正在进行中”提示
        "ui_hint": {"type": "string", "maxLength": 12, "description": "前端显示的超短提示（<=12字），为空则后端使用默认提示。"},
    },
    "required": [
        "task_type",
        "title",
        "description",
        "get_requirements",
        "rewards",
        "get_dialogue",
        "finish_dialogue",
    ],
    "additionalProperties": False,
}

DRAFT_AGENT_TASK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "draft_agent_task",
        "description": "生成并校验任务草案，暂存到 DB。",
        "parameters": DRAFT_AGENT_TASK_PARAMETERS_SCHEMA,
    },
}


UPDATE_TASK_DRAFT_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "draft_id": {
            "type": "string",
            "description": "要修改的草案 ID（从 draft_agent_task 返回值获取）",
        },
        # 用于 SSE/前端显示：非常短的“正在进行中”提示
        "ui_hint": {"type": "string", "maxLength": 12, "description": "前端显示的超短提示（<=12字），为空则后端使用默认提示。"},
        "modify_fields": {
            "type": "object",
            "description": "要修改的字段集合，仅包含需要变更的字段，未包含的字段保持原值不变",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "finish_requirements": {"type": "array", "items": STAGE_REQUIREMENT_SCHEMA},
                "finish_submit_items": {"type": "array", "items": REWARD_ITEM_SCHEMA},
                "finish_contain_items": {"type": "array", "items": REWARD_ITEM_SCHEMA},
                "rewards": {"type": "array", "items": REWARD_ITEM_SCHEMA},
                "get_npc": {"type": "string"},
                "finish_npc": {"type": "string"},
                "get_dialogue": {"type": "array", "items": DIALOGUE_ENTRY_SCHEMA},
                "finish_dialogue": {"type": "array", "items": DIALOGUE_ENTRY_SCHEMA},
            },
            "additionalProperties": False,
        },
    },
    "required": ["draft_id", "modify_fields"],
    "additionalProperties": False,
}

UPDATE_TASK_DRAFT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "update_task_draft",
        "description": "局部修改已有草案并触发增量校验（仅校验变更字段）。",
        "parameters": UPDATE_TASK_DRAFT_PARAMETERS_SCHEMA,
    },
}


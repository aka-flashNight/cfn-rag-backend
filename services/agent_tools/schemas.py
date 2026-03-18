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
    "装备获取",
    "特殊物品获取",
    "物品持有",
    "通关并收集",
    "通关并持有",
]

DIFFICULTIES: List[str] = ["普通", "冒险", "修罗", "地狱"]

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


class StageRequirement(TypedDict):
    stage_area: str
    stage_name: str
    difficulty: str


class TaskDraft(TypedDict, total=False):
    # V1-V6 用到的字段（其余字段由后续 Phase 实现补齐/计算）
    get_requirements: List[int]
    finish_requirements: List[StageRequirement]
    finish_submit_items: List[RewardItem]
    finish_contain_items: List[RewardItem]
    rewards: List[RewardItem]


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
        "stage_area": {"type": "string"},
        "stage_name": {"type": "string"},
        "difficulty": {"type": "string", "enum": DIFFICULTIES},
    },
    "required": ["stage_area", "stage_name", "difficulty"],
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
        "get_conversation_text": {"type": "string", "description": "接取时 NPC 的对话文本"},
        "finish_conversation_text": {"type": "string", "description": "完成时 NPC 的对话文本"},
    },
    "required": [
        "task_type",
        "title",
        "description",
        "get_requirements",
        "rewards",
        "get_conversation_text",
        "finish_conversation_text",
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
                "get_conversation_text": {"type": "string"},
                "finish_conversation_text": {"type": "string"},
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


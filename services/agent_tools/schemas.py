from __future__ import annotations

from typing import Any, List, Optional, TypedDict

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

REWARD_REGULAR: List[str] = ["金币", "经验值"]
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

# prepare_task_context：LLM 漏传或传空 reward_types 时的兜底默认（与 context_builder 单例集合一致）
REWARD_TYPES_DEFAULT_SINGLETON_REGULAR: List[str] = ["金币", "经验值"]
REWARD_TYPES_DEFAULT_SINGLETON_OPTIONAL: List[str] = ["K点", "技能点", "强化石"]


def normalize_reward_types_for_prepare_context(
    reward_types: Any,
    reward_keywords: Optional[list[str]],
) -> dict[str, list[str]]:
    """
    规范化 prepare_task_context 的 reward_types。

    - 若 regular/optional 合并后非空（且枚举合法），原样采用（非法项丢弃）。
    - 若合并后为空：从 reward_keywords 中抽取与 REWARD_REGULAR / REWARD_OPTIONAL **完全一致** 的字符串，
      分别归入 regular / optional（顺序为关键词出现顺序，同项去重）。
    - 若仍为空：使用默认单例组合（金币/经验值 + K点/技能点/强化石），避免 reward_item_candidates 整表为空。
    """
    reg_set = frozenset(REWARD_REGULAR)
    opt_set = frozenset(REWARD_OPTIONAL)

    def _canonical_reward_type_label(s: str) -> str:
        # 历史/口语简称「经验」与物品库正式名「经验值」对齐
        if s == "经验":
            return "经验值"
        return s

    def _coerce_list(v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        out: list[str] = []
        for x in v:
            if isinstance(x, str):
                s = _canonical_reward_type_label(x.strip())
                if s and (s in reg_set or s in opt_set):
                    out.append(s)
        return out

    regular: list[str] = []
    optional: list[str] = []
    if isinstance(reward_types, dict):
        regular = _coerce_list(reward_types.get("regular"))
        optional = _coerce_list(reward_types.get("optional"))

    if regular or optional:
        return {"regular": regular, "optional": optional}

    inferred_r: list[str] = []
    inferred_o: list[str] = []
    seen_r: set[str] = set()
    seen_o: set[str] = set()
    if isinstance(reward_keywords, list):
        for k in reward_keywords:
            if not isinstance(k, str):
                continue
            s = _canonical_reward_type_label(k.strip())
            if not s:
                continue
            if s in reg_set and s not in seen_r:
                inferred_r.append(s)
                seen_r.add(s)
            elif s in opt_set and s not in seen_o:
                inferred_o.append(s)
                seen_o.add(s)

    if inferred_r or inferred_o:
        return {"regular": inferred_r, "optional": inferred_o}

    return {
        "regular": list(REWARD_TYPES_DEFAULT_SINGLETON_REGULAR),
        "optional": list(REWARD_TYPES_DEFAULT_SINGLETON_OPTIONAL),
    }


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
    # 以下三项仅在 confirm_agent_task 时写入草案并落库；拟定/更新阶段不包含
    description: str
    get_requirements: List[int]
    finish_requirements: List[StageRequirement]
    finish_submit_items: List[RewardItem]
    finish_contain_items: List[RewardItem]
    rewards: List[RewardItem]
    # 发布/完成 NPC：可选；为空时由后端默认当前 NPC
    get_npc: str
    finish_npc: str
    # 接取/完成对话：数组，每项为纯对话文本（不要包含【动作/神态/旁白】）；仅 confirm 时写入
    get_dialogue: List[dict[str, Any]]
    finish_dialogue: List[dict[str, Any]]


# ---------------------------------------------------------------------------
# OpenAI Function Calling tools
# ---------------------------------------------------------------------------

PREPARE_TASK_CONTEXT_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_type": {
            "type": "string",
            "enum": TASK_TYPES,
            "description": (
                "NPC让玩家去做的事，例如让玩家打某关、让玩家提交某物给NPC等，满足NPC的需求。"
            ),
        },
        "reward_types": {
            "type": "object",
            "description": (
                "NPC给玩家的东西，也是玩家得到的东西，满足玩家的需求。"
            ),
            "properties": {
                "regular": {
                    "type": "array",
                    "items": {"type": "string", "enum": REWARD_REGULAR},
                    "description": "常规奖励类型（如金币、经验值）。",
                },
                "optional": {
                    "type": "array",
                    "items": {"type": "string", "enum": REWARD_OPTIONAL},
                    "description": "可选/附加奖励类型，可按玩家的需要勾选。",
                },
            },
            "required": ["regular", "optional"],
            "additionalProperties": False,
        },
        # 用于 SSE/前端显示：非常短的“正在进行中”提示
        "ui_hint": {"type": "string", "maxLength": 12, "description": "前端显示的超短提示（<=12字），为空则后端使用默认提示。"},
        "requirement_keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "可选。任务要求的关键词（模糊搜索）：用于将关卡名/区域说明/需提交或持有的物品名/物品类型等相关的候选项排到前面；"
                "如果你需要玩家给你更具体的某个/某类物品，或者让玩家去某关卡/某区域，请在此填写关键词，可以填多个。示例：「食品」「食材」「抗生素」「矛」「废城」等。"
            ),
        },
        "reward_keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "可选。奖励物品的关键词（模糊搜索），作用于 reward_item_candidates；"
                "如果你想给玩家更具体的某类/某个奖励，请在此填写关键词，可以填多个。示例：「头部装备」「上装装备」「手枪」「增效剂」「罐头」「食材」「霰弹」等。"
            ),
        },
    },
    "required": ["task_type", "reward_types"],
    "additionalProperties": False,
}

PREPARE_TASK_CONTEXT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "prepare_task_context",
        "description": (
            "根据意向任务类型与奖励偏好筛选数据，返回该类型的完整上下文与规则说明。"
            "可选用 requirement_keywords / reward_keywords 优先展示与当前情境更相关的关卡与物品。"
        ),
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
        # 用于 SSE/前端显示：非常短的“正在进行中”提示
        "ui_hint": {"type": "string", "maxLength": 12, "description": "前端显示的超短提示（<=12字），为空则后端使用默认提示。"},
    },
    "required": [
        "task_type",
        "title",
        "get_requirements",
        "rewards",
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
                "finish_requirements": {"type": "array", "items": STAGE_REQUIREMENT_SCHEMA},
                "finish_submit_items": {"type": "array", "items": REWARD_ITEM_SCHEMA},
                "finish_contain_items": {"type": "array", "items": REWARD_ITEM_SCHEMA},
                "rewards": {"type": "array", "items": REWARD_ITEM_SCHEMA},
                "get_npc": {"type": "string"},
                "finish_npc": {"type": "string"},
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


CONFIRM_AGENT_TASK_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "draft_id": {
            "type": "string",
            "description": "待确认的草案 ID（与 draft_summary 中一致）。",
        },
        "description": {
            "type": "string",
            "description": "任务说明/描述，须与最终确定的关卡、物品与奖励一致。",
        },
        "get_dialogue": {
            "type": "array",
            "items": DIALOGUE_ENTRY_SCHEMA,
            "description": "接取对话数组；可含 接取NPC 与玩家($PC)多条。对话发生在任务开始前， 接取NPC 向 玩家 发布任务",
        },
        "finish_dialogue": {
            "type": "array",
            "items": DIALOGUE_ENTRY_SCHEMA,
            "description": "完成对话数组；可含 完成NPC 与玩家($PC)多条。对话发生在任务完成后， 玩家 向 完成NPC 交付任务。",
        },
        "ui_hint": {
            "type": "string",
            "maxLength": 12,
            "description": "前端显示的超短提示（<=12字），为空则后端使用默认提示。",
        },
    },
    "required": ["draft_id", "description", "get_dialogue", "finish_dialogue"],
    "additionalProperties": False,
}

CONFIRM_AGENT_TASK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "confirm_agent_task",
        "description": (
            "玩家认可/接受/同意任务后调用（含此前已口头答应承接、本轮内刚拟定草案的情况）："
            "传入任务说明与接取/完成对话，与当前草案合并后校验并写入任务系统。不要和已有的任务对话高度重复。"
        ),
        "parameters": CONFIRM_AGENT_TASK_PARAMETERS_SCHEMA,
    },
}


---
name: task-publishing
description: NPC 发布任务给玩家的两步式完整流程（prepare_task_context → draft_agent_task，玩家接受后 → confirm_agent_task）。涵盖任务类型选择、奖励设计、物品/关卡候选抽取与任务发布原则。触发：玩家主动索要任务/奖励/关卡、对话自然延伸到 NPC 需要委托玩家某事时、好感度刚跨越关系等级可考虑初次委托。不触发：玩家正在倾诉烦恼、对话氛围不合适发布任务、当前已有无需修改的任务草案、你的身份不适合给玩家发任务、玩家近期刚拒绝过任务。
---

# 任务发布流程（两步式 + 确认）

## 总览

```
prepare_task_context  ──►  draft_agent_task  ──►  （玩家回复）  ──►  confirm_agent_task
       Step 1                   Step 2              协商/讨价                 写入
```

- **Step 1 & Step 2 在同一用户消息内完成**。你产出草案后，以自然语言向玩家描述任务内容，等玩家回复接受、拒绝、讨价还价或要求修改。
- **Step 3 (`confirm_agent_task`) 必须等玩家明确接受后才调用**。只有调用了 confirm_agent_task，任务才会真正写入。
- 如果玩家在同一条消息里"我接你任务"但你还没产出草案，可以在同一轮里顺序调用：`prepare_task_context → draft_agent_task → confirm_agent_task`。

## Step 1: prepare_task_context

调用入参：`task_type` + `reward_types` + 可选 `requirement_keywords` + 可选 `reward_keywords`。

### task_type（让玩家付出的劳动/物品，即 NPC 的需求）

12 种合法值：`问候 / 传话 / 通关 / 清理 / 挑战 / 切磋 / 资源收集 / 装备缴纳 / 特殊物品获取 / 物品持有 / 通关并收集 / 通关并持有`。

完整分类与触发条件见 [references/TASK_TYPES.md](references/TASK_TYPES.md)。

**核心原则**：

- `task_type` = 玩家要做的事；`reward_types` = 玩家得到的东西。二者不要混淆。
- 玩家索要资源时，**禁止**选 `资源收集`（玩家想要资源却让他交资源，荒谬）。
- 玩家索要装备时，**禁止**选 `装备缴纳`（玩家想要装备却让他交装备，同样荒谬）。
- 玩家要求某种奖励 → 改 `reward_types` 而非 `task_type`。

### reward_types（玩家得到的东西，即玩家的需求）

结构：`{"regular": [...], "optional": [...]}`。

- `regular` 合法值：`金币 / 经验值`。金币最常规；经验值仅挑战类任务可大量给。
- `optional` 合法值：`药剂 / 弹夹 / K点 / 技能点 / 强化石 / 战宠灵石 / 材料 / 食品 / 武器 / 防具 / 插件`。
- K 点仅玩家进度阶段 4+ 可选。
- 武器 / 防具 / 插件仅当前 NPC 有商店且商店覆盖对应类型时可选（layer2 会告诉你 `shop_reward_types` 可选集合）。

详细奖励预算、提交品/持有品价值加成、讨价还价调幅，见 [references/REWARD_RULES.md](references/REWARD_RULES.md)。

### requirement_keywords / reward_keywords（可选）

- `requirement_keywords`：任务要求的关键词（模糊搜索），用于把相关关卡/区域/需提交或持有的物品排到候选前面。例如玩家说"去废城找点东西" → 填 `["废城"]`。
- `reward_keywords`：奖励物品的关键词。例如玩家说"我要上装" → 填 `["上装装备"]`。

返回值重点字段：

- `stage_list` / `stage_requirement_candidates`：任务要求可选的关卡清单
- `submit_item_candidates` / `contain_item_candidates`：可供"提交/持有"任务选择的物品
- `reward_item_candidates`：可供奖励列表选择的物品（结构化，已过滤 reward_types）
- `reward_budget`：`{final_min, final_max}` 总价值上下限
- `npc_list`（仅问候/传话）：候选完成 NPC 及其情绪/称号列表
- `rule_hint`：该任务类型的人类可读规则摘要

## Step 2: draft_agent_task

根据 Step 1 返回数据，调用 `draft_agent_task` 产出结构化草案：

```json
{
  "task_type": "<与 Step 1 一致>",
  "title": "<简洁任务标题>",
  "finish_requirements": [{"stage_name": "...", "difficulty": "冒险"}],
  "finish_submit_items": [{"item_name": "...", "count": 3}],
  "finish_contain_items": [],
  "rewards": [{"item_name": "金币", "count": 1200}, {"item_name": "强化石", "count": 2}],
  "get_npc": "",
  "finish_npc": ""
}
```

关键约束：

- `rewards` 总价值（金币 + 各物品 `count × price`）必须落在 `reward_budget.final_min~final_max`。好感度 ≥ 50 可稍慷慨。
- 含提交品时 `final_min/final_max` 会提高（见 [references/REWARD_RULES.md](references/REWARD_RULES.md) 的"提交/持有品加成"）。
- **不要**在 draft_agent_task 里写 `description / get_dialogue / finish_dialogue`，这三项只在 confirm 时填。
- `get_npc` / `finish_npc` 可为空，留空后端默认当前 NPC；若希望任务由其他 NPC 完成，填入完成 NPC 的名字（该 NPC 必须在候选 `npc_list` 中）。
- 草案校验不过会返回 `validation_errors`，你需要按 errors 修正后重新调用 `draft_agent_task`，或者调用 `cancel_agent_task` 取消。

## Step 3: 协商 & confirm_agent_task

玩家回复后有 4 种分支（详见 [task-bargaining](../task-bargaining/SKILL.md)）：

1. **接受** → `confirm_agent_task(draft_id, title, description, get_dialogue, finish_dialogue)`。
2. **拒绝** → `cancel_agent_task(draft_id)` + 以角色身份自然回应。
3. **讨价还价奖励/要求** → `update_task_draft` 在允许幅度内调整；上限 2 次。
4. **整体重拟**（大幅改任务方向）→ 重新 `prepare_task_context` + `draft_agent_task`，替换当前草案。

### confirm_agent_task 入参约束

- `draft_id` 必须与当前 pending_draft 一致。
- `title`：最终任务标题，会覆盖草案阶段的标题。
- `description`：任务说明，须与最终关卡/物品/奖励一致。
- `get_dialogue` / `finish_dialogue`：对话数组，每条 `{name, title, emotion?, text}`。
  - `text` 必须是**纯对话**，**不要**包含动作/神态/旁白/`【...】`。
  - 神态/情绪放 `emotion` 字段（从可用情绪选，可为空）。
  - 玩家条目：`name="$PC"`, `title="$PC_TITLE"`，`emotion` 可填玩家情绪。
  - 数组顺序按对话发生时间排列。
  - **不要**和已有任务对话高度雷同。

## 发布原则（硬约束）

1. 只在对话氛围合适时考虑发布任务，不是每次对话都发任务。
2. 任务必须符合你的角色定位和能力范围。高级 / 商人 / 普通成员 / 科技学术等 NPC 的推荐任务类型见 NPC 层 prompt。
3. 发布动机自然融入对话，不要生硬插入。
4. 同一时间只保留一个待确认草案；换任务时重新 prepare + draft 会替换。
5. 玩家看不到你拟定任务的过程——你必须用对话把任务内容告诉玩家。
6. **只有调用了 `confirm_agent_task` 且成功返回 `status=confirmed`，任务才正式发布**。未调用前不要在对话里宣告任务已发布。

## 不应发布任务的场景

- 玩家正在倾诉烦恼 / 进行情感交流
- 玩家近期刚拒绝过任务
- 当前已有一个待确认且无需修改/取消的任务草案
- 你和玩家关系恶劣 / 身份不合适派任务给玩家

## UI hint

调用 task 系列工具时可以填 `ui_hint`（≤12 字）作为前端短提示。
为空则后端使用默认：`正在构思任务……` / `正在拟派清单……` / `正在调整草案……` / `正在提交任务……` / `正在取消任务……`。

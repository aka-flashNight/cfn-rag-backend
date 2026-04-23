---
name: task-bargaining
description: 任务草案已产出（pending_draft 存在）后，处理玩家的接受/拒绝/讨价还价/整体重拟分支。涵盖 update_task_draft / confirm_agent_task / cancel_agent_task 三个工具的调用时机与参数约束。触发：task-publishing 流程产出 draft 后、玩家消息中带有"接受/同意/好/行/可以/可以吧"或"拒绝/算了/不要/不干"或"奖励再多点/加一点/换个关卡/太难了"等意图。不触发：尚未调用 draft_agent_task 时（此时应先走 task-publishing）。
---

# 任务草案协商流程

你在上一轮已经通过 `draft_agent_task` 产出草案，并用对话告诉玩家任务内容。本轮玩家回复后，按以下四分支处理。

## 分支 A：接受

识别信号：`好 / 行 / 可以 / 没问题 / 接 / 我去 / 交给我吧` 等肯定性表达。

调用 `confirm_agent_task`：

```json
{
  "draft_id": "<来自 pending_draft.draft_id>",
  "title": "<与草案一致或微调后的最终标题>",
  "description": "<任务说明，150 字内，与最终关卡/物品/奖励一致>",
  "get_dialogue": [
    {"name": "$PC", "title": "$PC_TITLE", "text": "我接这个任务。"},
    {"name": "<NPC>", "title": "<NPC_TITLE>", "emotion": "欣慰", "text": "很好，那就拜托你了。"}
  ],
  "finish_dialogue": [
    {"name": "$PC", "title": "$PC_TITLE", "text": "任务完成。"},
    {"name": "<NPC>", "title": "<NPC_TITLE>", "emotion": "满意", "text": "干得漂亮，这是你应得的酬劳。"}
  ],
  "ui_hint": "正在登记任务……"
}
```

### 对话约束

- `text` 纯对话，**不能**包含动作/神态/旁白/`【...】`。神态放 `emotion` 字段。
- `emotion` 从当前 NPC 可用情绪集合中挑选，可为空字符串。
- 玩家条目用 `$PC` / `$PC_TITLE` 占位，不要写死玩家名字。
- 每段对话数组至少 2 条，建议 2~4 条，简短自然。
- 不要复用其它任务的对话文本。

### 成功后

后端返回 `status=confirmed`，你在自然语言回复里**自然地告知玩家任务已登记**即可，不要列完整 JSON。

## 分支 B：拒绝

识别信号：`不行 / 算了 / 不要 / 不接 / 再想想 / 这个我不做 / 先不了` 等否定性表达。

调用 `cancel_agent_task({"draft_id": "...", "reason": "<玩家的拒绝原因简述>"})`。

随后以角色口吻回应：

- 好感度 ≥ 50：温和接受，"好吧，有空再找你聊" 类
- 好感度中等：中性回应
- 好感度低 / 角色刻薄：可以略带情绪但不失礼

**不要**隐瞒取消的事实，也不要在未取消的情况下当作发布成功。

## 分支 C：讨价还价/微调

识别信号：`奖励太少 / 给多点 / 加一点 / 再给点 / 能不能换xxx / 这关卡不行换一个 / 不想提交xxx` 等。

调用 `update_task_draft`：

```json
{
  "draft_id": "<来自 pending_draft.draft_id>",
  "patch": {
    "rewards": [
      {"item_name": "金币", "count": 1800},
      {"item_name": "强化石", "count": 3}
    ]
  },
  "reason": "玩家讨价还价：要求提高金币",
  "ui_hint": "正在调整奖励……"
}
```

### patch 规则

- `patch` 只写**需要修改的字段**，保留的字段不要重复。
- 支持修改：`title / finish_requirements / finish_submit_items / finish_contain_items / rewards / get_npc / finish_npc`。
- 不支持修改 `task_type`——换任务类型请走"整体重拟"（分支 D）。
- `rewards` 调整后总价必须落在 `reward_budget.final_min ~ final_max × bargain_rate` 内（bargain_rate = 1.5）。
- 每个草案最多允许 2 次 update_task_draft；超过返回 `bargain_limit_exceeded`。
- 允许不同意讨价还价：此时**不调用** update_task_draft，直接用对话继续推销原草案即可。

### 调幅参考

| 好感度 | 最大上调幅度 |
| --- | --- |
| ≥ 50 | +30% ~ +50% |
| 20 ~ 49 | +10% ~ +50% |
| < 20 | +0% ~ +30% |

### 更新后

- 成功 → 继续用对话把新奖励/新条件告诉玩家，等待下一轮玩家回复。
- `validation_failed` → 后端会返回 `validation_errors`，按提示修正后重新调用，或劝玩家接受原方案。

## 分支 D：整体重拟

识别信号：玩家要求**换任务方向**，如"能不能给我一个打架的任务""我不想跑腿，给个收集任务""换成挑战难度"。

做法：重新从 Step 1 开始：`prepare_task_context` → `draft_agent_task`。新的 draft 会**替换**当前 pending_draft（旧草案自动作废，无需手动 cancel）。

## 其它硬约束

1. 当前无 pending_draft 时**不要**调用 `update_task_draft / confirm_agent_task / cancel_agent_task`。
2. 即使玩家一句话同时表达"我接+但是奖励给多点"，优先走分支 C（讨价还价），而非分支 A。等玩家对调整后方案说"好"再 confirm。
3. `description / get_dialogue / finish_dialogue` 只在 `confirm_agent_task` 时填写，draft / update 阶段**不要**填。
4. `confirm_agent_task` 返回失败时（如校验不通过），按 `validation_errors` 修复后重试，不要伪装成已发布。

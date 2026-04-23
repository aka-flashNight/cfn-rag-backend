# 任务奖励预算与加成规则

## reward_budget

`prepare_task_context` 返回值中的 `reward_budget` 字段决定了 `rewards` 总价值上下限：

```
reward_budget.final_min  ≤  Σ(rewards[i].count × price_of(rewards[i].item_name))  ≤  reward_budget.final_max
```

- `price_of("金币") = 1`
- `price_of("经验值") = 1`（内部折算，经验值给量通常是金币的 3~5 倍才等价）
- 其他物品以 `items.price` 为准

## 挑战任务的经验值占比

选 `挑战` 任务时，奖励中**经验值的金额占比 ≥ 50%**。例如 final_max=10000，经验值至少占 5000。

## 好感度调节

- 好感度 ≥ 50：奖励可**略慷慨**，向上靠 final_max。
- 好感度 < 0：奖励不要溢出到 final_max，取偏下限。

## 提交品/持有品加成

当任务含 `finish_submit_items`（提交品）或 `finish_contain_items`（持有品）时，`final_min/final_max` 会在基础上叠加：

| 任务类型 | 加成类型 | 加成倍率 |
| --- | --- | --- |
| 资源收集 / 装备缴纳 / 特殊物品获取 | 提交品价值加成 | +1×~2× 提交品总价 |
| 通关并收集 | 提交品价值加成 | +1×~2× 提交品总价 |
| 物品持有 | 持有品价值加成 | +0.5× 持有品总价 |
| 通关并持有 | 持有品价值加成 | +0.5× 持有品总价 |

实际后端会把这些加成折算到 `reward_budget.final_min / final_max`，你按返回的数值设计 `rewards` 即可。

## 讨价还价调幅

当玩家讨价还价（通常提出"奖励再多点"），你可视情况接受或拒绝：

| 好感度 | 可调幅 |
| --- | --- |
| ≥ 50 | +30% ~ +50% |
| 20 ~ 49 | +10% ~ +50% |
| < 20 | +0% ~ +30% |

- 上调后的 `rewards` 总价**仍然要落在新的 final_min/final_max 内**（update_task_draft 会以 `bargain_rate=1.5` 放宽上限校验）。
- 同一草案最多讨价还价 2 次，超过返回 `status=error`。
- 如果你不想让步也可以直接拒绝讨价还价，继续推销原任务，不调用 update_task_draft。

## 常见错误

- 把玩家索要的物品写进 `finish_submit_items`（应该写进 `rewards`）
- 总价超出 `final_max` → validation_failed
- 奖励物品 `count × price` 与任务难度明显不匹配（如高难度只给 100 金币）

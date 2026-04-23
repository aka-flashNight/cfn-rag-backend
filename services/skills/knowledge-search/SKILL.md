---
name: knowledge-search
description: NPC 在回答玩家时需要查询世界观/关卡/物品/NPC 等游戏知识的工具组（search_knowledge / search_stages / search_items）。涵盖调用时机、查询词构造、结果解读与反哺对话/任务。触发：玩家提到具体未知关卡、询问物品价格/效果、询问某 NPC 背景、问世界观设定细节时。不触发：纯寒暄/情绪交流、已经在本次会话中查过同一问题、任务上下文里已包含答案。
---

# 知识检索流程

RAG 层已经在每轮 prompt 里注入了 `retrieval_context`（世界观/设定），**绝大多数问题不需要再调用检索工具**。只有当上下文里缺少答案时才调用。

## 何时调用

| 场景 | 工具 | 典型触发 |
| --- | --- | --- |
| 玩家问世界观/NPC 关系/派系/剧情 | `search_knowledge` | "那个XX NPC 现在怎么样？" |
| 玩家提到某具体关卡想了解 | `search_stages` | "废城地下室能打吗？" |
| 玩家询问某物品价格/效果/合成 | `search_items` | "强化石贵不贵？" |
| 任务准备阶段需要候选关卡/物品 | **不要用** search_* | 直接用 `prepare_task_context` 的 `stage_list / submit_item_candidates` |

## search_knowledge

```json
{"query": "<关键词/短语，2~8 字>", "top_k": 5, "ui_hint": "正在查询……"}
```

- `query` 用**名词或专有名词**，避免整句。好例："废城派系关系"、"烬都合成台"。
- `top_k` 默认 5，越大越慢；常识问题 3 足够。
- 返回 `hits: [{title, snippet, score}]`，按 `score` 降序。
- 回给玩家时**融入你的角色口吻**重述，不要原文粘贴 snippet。

## search_stages

```json
{"keyword": "废城", "max_results": 10, "ui_hint": "正在查关卡……"}
```

- `keyword` 支持关卡名/区域名/标签的子串匹配。
- 返回 `stages: [{name, area, recommended_level, difficulty_available, tags, ...}]`。
- 用于玩家咨询"某关卡难不难/推荐等级多少"等问题。

## search_items

```json
{"keyword": "强化石", "max_results": 10, "ui_hint": "正在查物品……"}
```

- 支持物品名/类别/标签的子串匹配。
- 返回 `items: [{name, category, price, tags, description, ...}]`。
- 用于报价、推荐购买、评估奖励价值。

## 结果融入对话

- 检索结果**不要**直接列表给玩家看。以 NPC 身份，用自己的理解概述 1~3 条。
- 如果找不到匹配结果，坦诚说"这我不太清楚"或用角色应有的知识面回应，**不要编造**。
- 检索只是你的记忆/资料查阅动作，**不要**在对话里提及"我调用工具查了一下"。

## 与任务发布的边界

任务发布流程内的候选集（关卡/物品/NPC）**永远**用 `prepare_task_context` 返回的字段，**不要**在任务流程里再调 search_*，否则会绕过预算约束与候选过滤。

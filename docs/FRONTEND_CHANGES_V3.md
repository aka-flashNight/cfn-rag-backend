# 前端对接变更说明（路线三 · 多 Agent 编排，2026-04）

> 本次后端升级到 LangGraph Supervisor 多 Agent 架构。**前端几乎不需要改动**——SSE 事件类型、事件字段、`{...}` 系统消息解析、done 字段等都与原流式接口保持兼容。
>
> 唯一建议调整的两点：
> 1. `mood_update` 现在可能**提前到 `content` 之前**到达（supervisor 路由阶段就解析好了），前端收到后请**立刻应用**，不必等 `done`。
> 2. 一次回复中 `content` 之后可能**继续出现新的 `tool_status`**（例如"正在思考……" → "正在准备切磋任务……" → 中间插入 content → 还会有更多 tool_status），前端应允许 tool_status UI 在 content 开始后继续更新。

---

## 1. TL;DR

- **事件类型不变**，只在 SSE 里出现这几种：`content` / `tool_status` / `mood_update` / `agent_status` / `done` / `error`。
- **没有新事件类型需要处理**（旧的 `interim_content` / `interim_done` / `pending_confirmation` / `system` 都已删除，不会再出现）。
- `done.reply` 仍然是**完整文本**（含 supervisor 的过渡话 + `{任务发布成功}` 这种系统提示 + NPC 正文对白），历史接口读回来一致，前端的 `{...}` 解析逻辑照常用。

---

## 2. SSE 事件清单

| 事件 | 触发时机 | `data` 示例 | 前端处理 |
|------|---------|------------|---------|
| `tool_status` | supervisor 思考中、worker 调工具 | `{"tool_name":"supervisor","text":"正在思考……"}` / `{"tool_name":"draft_agent_task","text":"正在拟任务清单……"}` | 已接，**继续在 content 开始后允许更新**即可 |
| `agent_status` | supervisor 决定路由到哪个 worker | `{"agent":"task","text":"路由到 task：玩家索要任务"}` | 可选（前端目前不展示，保留字段以便日后做 debug 面板） |
| `mood_update` | 路由阶段解析好情绪 / 好感变化 | `{"emotion":"高兴","favorability_change":1}` | **建议改为：收到即应用**（立绘、数值动画、变色等）；`done` 里也仍会带同样的值，已经应用过就无需再应用 |
| `content` | supervisor 过渡话 / `{任务…}` 系统提示 / NPC 正文对白 | `{"delta":"…"}` | 已接，按原累加逻辑；内部 `{...}` 段继续按你现有解析逻辑渲染成系统消息 |
| `done` | 结尾 | 见 §4 | 已接，字段基本不变，多了可选 `awaiting_confirmation` / `confirmation_draft_id` |
| `error` | 异常 | `{"error":"…"}` | 已接 |

---

## 3. `content` 的分段顺序

一次回复中 `content` 可能分**多段到达**，前端把 delta 按顺序拼接就是 `done.reply`。典型顺序：

1. supervisor 的过渡话（**仅**在"玩家首次索要任务"这种场景才会有；普通聊天没有这一段）——后端模拟流式分段发送，约 40ms / 6 字一块。
2. 一个或多个 `{任务草案拟定完成}` / `{任务发布成功}` 等系统提示（每次一段，前后带空行）。
3. NPC 正文对白（dialogue agent 产出，一次性一段）。

> 之间可能穿插 `tool_status`（灰色状态条）更新，前端只要允许状态条在 content 进行中被覆盖成新文案即可；现在的实现如果在收到第一个 content 后就锁死 tool_status 不再更新，请解锁。

---

## 4. `done` 事件字段

```json
{
  "reply": "好，让我想想……\n\n{任务草案拟定完成}\n\n【Andy Law 看着你】看来你是闲得发慌了……",
  "npc_name": "Andy Law",
  "favorability": 26,
  "relationship_level": "熟悉",
  "favorability_change": 0,
  "emotion": "普通",
  "awaiting_confirmation": true,               // 可选，仅在"任务草案已生成，等玩家确认"时出现
  "confirmation_draft_id": "draft_1234"        // 可选，同上
}
```

- `reply` 与 `content` 累加出来的文本**一致**，前端渲染已确认时可以用它做一次最终校准（你的代码已经在做了）。
- `awaiting_confirmation=true`：建议展示"等待玩家确认"的视觉提示（可选）。玩家只要正常发下一条消息（"好"/"不要"/"加点金币"）即可继续流程，后端会自动路由到 confirm / cancel / update。

---

## 5. 历史接口

历史接口的 `assistant.content` 字段已经包含 supervisor 过渡话 + `{...}` 系统提示 + 正文对白的完整拼接。切会话、刷新、重新打开应用都能看到一致内容。前端沿用现有逻辑即可。

---

## 6. 需要做的前端改动（总共两处）

### 6.1 `mood_update` 尽早应用

现在 `mood_update` 可能在任何 `content` 之前就到达（supervisor 在路由阶段就决定了情绪/好感）：

```js
// 伪代码：
onMoodUpdate(data) {
  // 立即应用立绘 / 数值动画，不要等 done
  this.emotion = data.emotion;
  this.favorability_change = data.favorability_change;
}

onDone(data) {
  // done 里仍会带相同的 emotion / favorability_change；
  // 如果已经在 onMoodUpdate 应用过，这里直接跳过或只做一次冗余校准即可
}
```

如果没有 `mood_update` 事件（关闭 Agent 开关的 legacy 分支可能只在 `done` 里给）——保留现有"只读 done"的兼容逻辑。

### 6.2 `content` 开始后仍要更新 `tool_status`

多 Agent 路径下，SSE 的顺序可能是：

```
tool_status "正在思考……"
mood_update {...}
content  (supervisor 过渡话)
tool_status "正在准备切磋任务"
tool_status "正在拟任务清单……"
content  ({任务草案拟定完成})
content  (NPC 正文对白)
done
```

请确认前端的 `tool_status` UI 不会在第一个 `content` 之后停止更新。**如果以前是"收到 content 就把 tool_status 状态条隐藏"，改成"done 事件才隐藏"**。

---

## 7. 不需要做的事情

- **不用**新增 `event: system` / `event: interim_content` / `event: interim_done` / `event: pending_confirmation` 的监听（后端不会再发）。
- **不用**改 `{...}` 的解析逻辑。
- **不用**单独对接 `/api/ask/confirm`（自然语言回复 + `done.awaiting_confirmation` 已足够）；如果你想做"接受 / 拒绝 / 议价"快捷按钮 UI，再看这个接口即可。
- **不用**特别处理 `update_npc_mood` 工具——它是内部工具，后端不会再下发它的 `tool_status`。

---

## 8. 一次完整的事件流示例

### 8.1 普通闲聊（`"你好吗？"`）

```
event: tool_status   data: {"tool_name":"supervisor","text":"正在思考……"}
event: agent_status  data: {"agent":"dialogue","text":"路由到 dialogue：普通寒暄"}
event: mood_update   data: {"emotion":"普通","favorability_change":0}
event: content       data: {"delta":"我还不错，你呢？"}
event: done          data: {"reply":"我还不错，你呢？","npc_name":"Andy Law","favorability":25,"relationship_level":"熟悉","favorability_change":0,"emotion":"普通"}
```

### 8.2 首次索要任务（`"给我个任务吧"`）

```
event: tool_status   data: {"tool_name":"supervisor","text":"正在思考……"}
event: agent_status  data: {"agent":"task","text":"路由到 task：玩家索要任务"}
event: mood_update   data: {"emotion":"玩味","favorability_change":1}
event: content       data: {"delta":"好，让我"}
event: content       data: {"delta":"想想最近"}
event: content       data: {"delta":"有什么合适的……"}
event: tool_status   data: {"tool_name":"prepare_task_context","text":"正在准备切磋任务"}
event: tool_status   data: {"tool_name":"draft_agent_task","text":"挑战我的手段吧"}
event: content       data: {"delta":"\n\n"}
event: content       data: {"delta":"{任务草案拟定完成}"}
event: content       data: {"delta":"\n\n"}
event: content       data: {"delta":"【Andy Law 看着你】看来你是闲得发慌了……"}
event: done          data: {"reply":"好，让我想想最近有什么合适的……\n\n{任务草案拟定完成}\n\n【Andy Law 看着你】看来你是闲得发慌了……","awaiting_confirmation":true,"confirmation_draft_id":"draft_1234","emotion":"玩味","favorability":26,"favorability_change":1, ...}
```

### 8.3 玩家确认（`"好，就这样"`）

```
event: tool_status   data: {"tool_name":"supervisor","text":"正在思考……"}
event: agent_status  data: {"agent":"task","text":"路由到 task：玩家接受草案"}
event: mood_update   data: {"emotion":"高兴","favorability_change":2}
event: tool_status   data: {"tool_name":"confirm_agent_task","text":"正在提交任务"}
event: content       data: {"delta":"{任务发布成功}"}
event: content       data: {"delta":"\n\n"}
event: content       data: {"delta":"成交，去吧。"}
event: done          data: {"reply":"{任务发布成功}\n\n成交，去吧。","emotion":"高兴","favorability":28,"favorability_change":2,"awaiting_confirmation":false, ...}
```

### 8.4 玩家议价（`"奖励能多点吗"`）

```
event: tool_status   data: {"tool_name":"supervisor","text":"正在思考……"}
event: agent_status  data: {"agent":"task","text":"路由到 task：玩家讨价还价"}
event: mood_update   data: {"emotion":"玩味","favorability_change":0}
event: tool_status   data: {"tool_name":"update_task_draft","text":"正在调整草案"}
event: content       data: {"delta":"{任务草案已更新}"}
event: content       data: {"delta":"\n\n"}
event: content       data: {"delta":"那就再加 500 金币，这样可以吗？"}
event: done          data: {"reply":"{任务草案已更新}\n\n那就再加 500 金币，这样可以吗？","awaiting_confirmation":true, ...}
```

---

## 9. 迁移 Checklist（前端）

- [ ] `onMoodUpdate` 立即应用（立绘 / 数值 / 变色），不等 `done`。
- [ ] `tool_status` UI 不因第一个 `content` 到达而停止更新，改为 `done` 事件时再隐藏。
- [ ] （可选）读 `done.awaiting_confirmation` / `done.confirmation_draft_id`，在对话框显示"等待玩家确认"的小提示。
- [ ] （可选）`agent_status` 走 debug 面板，玩家 UI 直接忽略。

**无需处理**：`interim_content` / `interim_done` / `pending_confirmation` / `system` 事件——它们已被后端彻底移除。

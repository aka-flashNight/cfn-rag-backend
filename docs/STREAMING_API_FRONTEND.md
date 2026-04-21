# Ask 流式接口与前端配合说明

## 1. 接口约定

- **同一 POST 路径**：`/ask`，通过查询参数区分是否流式。
- **非流式（默认）**：`POST /ask` 或不带 `stream` / `stream=false`
  - 响应：JSON 体，即原来的 `NPCChatResponse`。
- **流式**：`POST /ask?stream=true`
  - 响应：`Content-Type: text/event-stream`，Server-Sent Events (SSE)。
  - 事件类型：`content`（正文片段）、`mood_update`（可选，提前下发情绪/好感变化）、`done`（结尾元数据）、`error`（错误时）。

## 2. 流式响应格式（SSE）

每个事件为多行文本，以两个换行 `\n\n` 结束：

- **正文片段**（打字机用）  
  ```
  event: content
  data: {"delta": "你"}
  ```
  - `data` 为 JSON：`{ "delta": "本次增量文本" }`，可能多次出现。
  - 前端应顺序拼接所有 `delta`，并实时渲染（如逐字/逐句显示）。

- **情绪/好感提前更新**（可选，在 `done` 之前可能出现 0 次或多次）  
  ```
  event: mood_update
  data: {"emotion":"高兴","favorability_change":1}
  ```
  - 当模型在流式阶段通过 `update_npc_mood` 工具给出可解析参数时，后端会尽早推送该事件，便于立绘/数值与打字机同步。
  - 字段与工具约定一致，至少包含 `emotion`（字符串）与 `favorability_change`（整数，可为 0）；最终以 `done` 中的 `emotion`、`favorability_change` 为准（与存库一致）。

- **结束元数据**（与 tool_calls 等价的结果）  
  ```
  event: done
  data: {"reply":"完整清理后回复","npc_name":"Andy Law","favorability":85,"relationship_level":"熟悉","favorability_change":0,"emotion":"普通"}
  ```
  - `data` 字段与 `NPCChatResponse` 同构：`reply`、`npc_name`、`favorability`、`relationship_level`、`favorability_change`、`emotion`。
  - 前端应在此事件中更新：当前回复最终版、情绪（立绘）、好感度与关系等级。

- **错误**  
  ```
  event: error
  data: {"error": "错误信息"}
  ```
  - 仅在流式过程中发生异常时发送；前端应提示用户并停止等待更多事件。

## 3. 前端如何调用（流式）

- 使用 **fetch + ReadableStream** 或 **EventSource** 均可；SSE 标准是事件流，推荐按「事件类型 + data 行」解析。
- 请求方式示例（保留原有 body，仅加查询参数）：

  ```ts
  const response = await fetch(`${baseUrl}/ask?stream=true`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: userInput,
      npc_name: npcName,
      session_id: sessionId,
      current_emotion: currentEmotion ?? null,
      player_identity: playerIdentity ?? null,
      // 其余字段同原 ask 请求
    }),
  });

  if (!response.ok) {
    // 处理 4xx/5xx
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  // 按行解析 SSE：遇到 "event: xxx" 记类型，遇到 "data: {...}" 解析 JSON 并分发
  // 若使用 EventSource：注意 EventSource 只支持 GET，故流式 ask 必须用 fetch + 手动解析 SSE
  ```

- 解析逻辑要点：
  - 每次读到一块字符串，追加到 `buffer`，按 `\n\n` 拆成多个事件。
  - 每个事件内按 `\n` 拆行，识别 `event: content` / `event: mood_update` / `event: done` / `event: error`，再取 `data: {...}` 行做 `JSON.parse`。
  - `content` → 把 `data.delta` 拼到当前回复字符串并更新 UI（打字机效果）。
  - `mood_update` → 可提前更新立绘/好感预览；收到 `done` 后应以 `done` 为准。
  - `done` → 用 `data.reply` 覆盖最终展示（后端已做末尾 JSON 剥离），并用 `data.emotion`、`data.favorability`、`data.favorability_change`、`data.relationship_level` 更新状态/立绘/好感度。
  - `error` → 展示 `data.error` 并结束。

## 4. 前端状态与展示建议

- **流式过程中**：
  - 用「当前累积的 content」做打字机展示；可每收到一个 `content` 就 append 并 re-render。
- **收到 `done` 后**：
  - 用 `data.reply` 作为该条消息的**最终内容**（与历史、存库一致），避免流式时若曾带出末尾 JSON 仍被展示。
  - 同时更新：
    - 情绪 / 立绘：`data.emotion`
    - 好感度：`data.favorability`、`data.favorability_change`
    - 关系等级：`data.relationship_level`
- **tool_calls 的等价信息**：全部在 `done` 的 `data` 里，无需再解析 content 中的 JSON；后端已从 tool_calls 或末尾 mood JSON 解析并写库，前端只需消费 `done` 即可。

## 5. 与非流式并存

- 同一 `POST /ask`：
  - `stream=false` 或不传 → 返回 JSON，前端按原逻辑解析一次拿到 `NPCChatResponse`。
  - `stream=true` → 返回 SSE，前端按上面方式解析 `content` / `done` / `error`。
- 前端可根据能力或配置选择是否传 `stream=true`（例如移动端或弱网可暂用非流式）。

## 6. 错误与超时

- 若请求未发或未返回 2xx：按原有错误处理（如 toast、重试）。
- 若已 2xx 且开始读流，中途出错：后端会发 `event: error`，前端应在解析到 `error` 时停止流并提示；未收到 `done` 即视为未完成，可提示「回复未完成」。

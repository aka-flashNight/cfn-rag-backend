---
name: skill-discovery
description: 使用 list_skills / read_skill / read_skill_file 动态发现与按需加载其他 skill 文档的元流程（渐进式披露 / progressive disclosure）。触发：Agent 遇到不熟悉场景需要判断该加载哪个 skill、需要查阅某 skill 的 references 子文件、Supervisor 想知道某 worker agent 可用 skill 清单。不触发：对应 skill 已被当前 agent 静态预载且内容足够。
---

# 技能发现与渐进式加载

**Anthropic 2026 Agent Skills 规范**把 skill 内容分为三级：

| 级别 | 内容 | 默认加载方式 |
| --- | --- | --- |
| L1 Metadata | `name + description`（来自 SKILL.md frontmatter） | Agent 启动时**全部加载** |
| L2 Body | SKILL.md 的 Markdown 正文 | 按需 `read_skill` |
| L3 References | `references/*.md` 等子文件 | 按需 `read_skill_file` |

大多数情况，Worker Agent 的系统提示已经**静态预载了**其工作相关的 L2 Body（例如 TaskAgent 预载 task-publishing + task-bargaining）。只有当 Agent 遇到不熟悉场景或需要查细节（L3）时才动态调用本 skill 下的工具。

## list_skills

```json
{"category": "", "ui_hint": "正在浏览技能……"}
```

返回：

```json
{"skills": [
  {"name": "task-publishing", "description": "...", "path": "services/skills/task-publishing"},
  ...
]}
```

- `category` 可选：`task / query / mood / system / ""`（空串=全部）。
- 结果按字典序排。
- 适合 Supervisor 路由前快速了解全局能力地图。

## read_skill

```json
{"name": "task-publishing", "ui_hint": "正在阅读技能……"}
```

返回：

```json
{
  "name": "task-publishing",
  "description": "...",
  "body": "<完整 Markdown 正文>",
  "references": ["TASK_TYPES.md", "REWARD_RULES.md"]
}
```

- 返回 L2 Body。如果 Body 很长，Agent 仅引用必要段落，**不要**把整段 body 复述给玩家。
- `references` 列出该 skill 目录下的 L3 文件相对名（供下一步 `read_skill_file` 使用）。

## read_skill_file

```json
{"skill": "task-publishing", "relative_path": "references/TASK_TYPES.md", "ui_hint": "正在查阅细则……"}
```

- `skill` = skill 名（如 `task-publishing`）。
- `relative_path` = 相对于该 skill 目录的路径，必须以 `references/` 开头。
- 后端会做**严格路径校验**：禁止 `..`、绝对路径、跨 skill 访问。
- 返回 `{"skill": "...", "path": "...", "content": "<文本>"}`。

## 使用原则

1. **优先使用静态预载**。如果 skill 已经出现在系统提示里，不要再调用 read_skill 重复加载。
2. **不要遍历**。避免"list_skills → read_skill 全部 → 逐个 read_skill_file"的无目的扫描；这会浪费 token。
3. **按路径精确取**。读 L3 文件前先看 `read_skill` 返回的 `references` 清单，避免瞎猜路径。
4. **读后即用**。读出来的内容只用于本轮推理，不要把它写进最终对话输出给玩家。

## 典型场景

- Supervisor 遇到"模糊请求"想快速判断走哪个 worker → `list_skills`（非必须，通常 Supervisor 自己的系统提示已含路由规则，本调用可省）。
- TaskAgent 忘了某任务类型规则 → `read_skill_file("task-publishing", "references/TASK_TYPES.md")`。
- 新增 skill 后验证生效 → 先 `list_skills` 看是否出现在清单中。

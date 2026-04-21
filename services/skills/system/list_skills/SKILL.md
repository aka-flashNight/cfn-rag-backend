列出当前可用工具清单；传入 skill_name 时返回该 skill 的完整 SKILL.md 正文（按需加载详细用法）。

触发：不确定还能调用哪些工具、需要自检能力边界、或觉得某个工具的简介不足以决定如何使用时。不触发：常规对话与任务流程中已经明确要用哪个工具的情况。

## 使用场景

1. 自省能力边界：
   - 不传参数或只传 categories → 返回 `[{name, category, description}]` 简表，每条 description 就是该 skill 的 OpenAI tool description。
2. 按需加载详细文档（渐进式披露）：
   - 传 `skill_name="draft_agent_task"` → 返回该 skill 目录下 `SKILL.md` 的完整正文。
   - 常用于：工具描述中提到的"详见 SKILL.md"、触发条件不明、参数组合复杂时。

## 返回示例

`list_skills()`：

```json
{
  "status": "ok",
  "skills": [
    {"name": "search_knowledge", "category": "query", "description": "..."},
    {"name": "draft_agent_task", "category": "task", "description": "..."}
  ],
  "hint": "如需某个 skill 的详细用法，再次调用 list_skills 时传入 skill_name='<name>' 即可获取完整文档。"
}
```

`list_skills(skill_name="draft_agent_task")`：

```json
{
  "status": "ok",
  "skill": {
    "name": "draft_agent_task",
    "category": "task",
    "description": "...",
    "doc": "创建任务草案并做服务端校验。\n\n触发：..."
  }
}
```

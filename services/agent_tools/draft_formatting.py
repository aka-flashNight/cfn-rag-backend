"""
任务草案摘要与实体行格式化（纯函数，供 skill 执行层与 prepare_context 注入使用）。
"""

from __future__ import annotations

from typing import Any, Optional

from services.game_data.registry import GameDataRegistry


def _draft_item_role_notes(draft: dict[str, Any], item_name: str) -> list[str]:
    """草案中该物品出现的角色与数量（奖励/提交/持有）。"""
    notes: list[str] = []
    for r in draft.get("rewards") or []:
        if not isinstance(r, dict):
            continue
        if (r.get("item_name") or "").strip() != item_name:
            continue
        notes.append(f"奖励×{r.get('count', '?')}")
    for r in draft.get("finish_submit_items") or []:
        if not isinstance(r, dict):
            continue
        if (r.get("item_name") or "").strip() != item_name:
            continue
        notes.append(f"提交×{r.get('count', '?')}")
    for r in draft.get("finish_contain_items") or []:
        if not isinstance(r, dict):
            continue
        if (r.get("item_name") or "").strip() != item_name:
            continue
        notes.append(f"持有×{r.get('count', '?')}")
    return notes


def _format_items_with_price(
    items: list[dict[str, Any]],
    game_data: Optional[GameDataRegistry] = None,
) -> list[str]:
    parts: list[str] = []
    for it in items[:10]:
        name = it.get("item_name", "?")
        count = it.get("count", "?")
        price = 0
        if game_data:
            try:
                price = game_data.items.get_price(name)
            except Exception:
                pass
        if price and name != "金币":
            parts.append(f"{name}x{count}(单价{price})")
        else:
            parts.append(f"{name}x{count}")
    return parts


def _build_draft_entity_context_lines(
    draft: dict[str, Any],
    game_data: Optional[GameDataRegistry],
    *,
    rag_context_text: Optional[str] = None,
) -> list[str]:
    """
    与本轮检索上下文中的【玩家可能提到的物品类型】【玩家可能提到的关卡】按名称去重：
    已在 RAG 块出现过的实体不再重复；未出现的实体给出与 RAG 相同格式的完整说明（另附草案数量/难度等）。
    """
    if game_data is None:
        return []
    from services.game_entity_prompts import (
        compute_reward_tags,
        format_item_prompt_line,
        format_stage_detail_line,
        get_stage_info_for_name,
        parse_rag_game_entity_mentions,
    )

    already_items, already_stages = parse_rag_game_entity_mentions(rag_context_text)

    lines: list[str] = []
    seen_stage: set[str] = set()
    for fr in draft.get("finish_requirements") or []:
        if not isinstance(fr, dict):
            continue
        sn = (fr.get("stage_name") or "").strip()
        if not sn or sn in seen_stage:
            continue
        seen_stage.add(sn)
        if sn in already_stages:
            continue
        diff = (fr.get("difficulty") or "").strip() or None
        si = get_stage_info_for_name(game_data.stages, sn)
        if si is not None:
            seg = format_stage_detail_line(si)
            if diff:
                seg += f"；草案要求难度：{diff}"
            lines.append(f"[关卡] {seg}")
        else:
            seg = f"关卡名称：{sn}"
            if diff:
                seg += f"；草案要求难度：{diff}"
            lines.append(f"[关卡] {seg}（未在库中解析到详情）")

    item_names: list[str] = []

    def _collect_from_items(arr: Any) -> None:
        if not isinstance(arr, list):
            return
        for it in arr:
            if not isinstance(it, dict):
                continue
            nm = (it.get("item_name") or "").strip()
            if nm and nm not in item_names:
                item_names.append(nm)

    _collect_from_items(draft.get("finish_submit_items"))
    _collect_from_items(draft.get("finish_contain_items"))
    _collect_from_items(draft.get("rewards"))

    for nm in item_names:
        if nm in already_items:
            continue
        it = game_data.items.get_by_name(nm)
        notes = _draft_item_role_notes(draft, nm)
        if it is not None:
            tags = compute_reward_tags(it, game_data.equipment_mods)
            seg = format_item_prompt_line(it, reward_tags=tags, price=it.price)
            if notes:
                seg += f"；草案涉及：{'；'.join(notes)}"
            lines.append(f"[物品] {seg}")
        else:
            extra = f"；{'；'.join(notes)}" if notes else ""
            lines.append(f"[物品] 名称：{nm}{extra}（未在库中解析到详情）")

    return lines


def _detailed_draft_summary(
    draft: dict[str, Any],
    game_data: Optional[GameDataRegistry] = None,
    *,
    rag_context_text: Optional[str] = None,
) -> str:
    """
    生成包含完整字段和物品单价的草案摘要，
    用于注入 prompt 让 LLM 了解当前任务的全部关键信息。
    """
    lines: list[str] = []
    did = draft.get("draft_id", "?")
    lines.append(f"草案ID: {did}")
    lines.append(f"发布NPC: {draft.get('npc_name', '?')}")
    lines.append(f"类型: {draft.get('task_type', '?')}")
    lines.append(f"标题: {draft.get('title', '?')}")
    _desc = draft.get("description")
    if isinstance(_desc, str) and _desc.strip():
        lines.append(f"描述: {_desc}")
    else:
        lines.append("描述: （尚未写入；玩家接受时在 confirm_agent_task 中填写）")
    npc_name_fallback = draft.get("npc_name") or "?"
    lines.append(f"接取NPC: {draft.get('get_npc') or npc_name_fallback}")
    lines.append(f"完成NPC: {draft.get('finish_npc') or npc_name_fallback}")

    finish_reqs = draft.get("finish_requirements") or []
    if finish_reqs:
        fr_strs = [
            f"{fr.get('stage_name', '?')}({fr.get('difficulty', '?')})"
            for fr in finish_reqs
        ]
        lines.append(f"通关要求: {', '.join(fr_strs)}")

    finish_submit = draft.get("finish_submit_items") or []
    if finish_submit:
        fs_strs = _format_items_with_price(finish_submit, game_data)
        lines.append(f"提交物品: {', '.join(fs_strs)}")

    finish_contain = draft.get("finish_contain_items") or []
    if finish_contain:
        fc_strs = _format_items_with_price(finish_contain, game_data)
        lines.append(f"持有物品: {', '.join(fc_strs)}")

    rewards = draft.get("rewards") or []
    if rewards:
        rw_strs = _format_items_with_price(rewards, game_data)
        lines.append(f"奖励: {', '.join(rw_strs)}")

    def _summarize_dialogue(dialogue: Any) -> str:
        if isinstance(dialogue, list):
            parts: list[str] = []
            for it in dialogue[:8]:
                if not isinstance(it, dict):
                    continue
                n = str(it.get("name") or "").strip()
                emo = str(it.get("emotion") or "").strip()
                t = str(it.get("text") or "").strip()
                if not n and not t:
                    continue
                label = n or "?"
                if emo:
                    label = f"{label}#{emo}"
                t_short = t[:60] + ("…" if len(t) > 60 else "")
                parts.append(f"{label}:{t_short}")
            return "；".join(parts)

        if isinstance(dialogue, str):
            s = dialogue.strip()
            if not s:
                return ""
            return s[:120] + ("…" if len(s) > 120 else "")
        return ""

    get_dialogue = draft.get("get_dialogue")
    get_summary = _summarize_dialogue(get_dialogue)
    if get_summary:
        lines.append(f"接取对话: {get_summary}")
    else:
        get_text = draft.get("get_conversation_text", "")
        if isinstance(get_text, str) and get_text.strip():
            lines.append(f"接取对话(旧字段): {get_text.strip()[:120]}{'…' if len(get_text.strip()) > 120 else ''}")
        else:
            lines.append("接取对话: （尚未写入；玩家接受时在 confirm_agent_task 中填写）")

    finish_dialogue = draft.get("finish_dialogue")
    finish_summary = _summarize_dialogue(finish_dialogue)
    if finish_summary:
        lines.append(f"完成对话: {finish_summary}")
    else:
        finish_text = draft.get("finish_conversation_text", "")
        if isinstance(finish_text, str) and finish_text.strip():
            lines.append(f"完成对话(旧字段): {finish_text.strip()[:120]}{'…' if len(finish_text.strip()) > 120 else ''}")
        else:
            lines.append("完成对话: （尚未写入；玩家接受时在 confirm_agent_task 中填写）")

    entity_lines = _build_draft_entity_context_lines(
        draft, game_data, rag_context_text=rag_context_text,
    )
    if entity_lines:
        lines.append("---")
        lines.append(
            "涉及关卡与物品补充说明（前文已有的内容此处不再重复）"
        )
        lines.extend(entity_lines)

    return "\n".join(lines)

"""把子 Agent 提交的整组修改落到两个覆盖层（阶段四的写入侧唯一入口）。

模型只提供「编号 + 每行的 name/text/emotion」，其余全由后端补齐：

- ``Title`` 不由模型写：既有行沿用原 Title，新增行按四级回落查（本组同说话人的
  Title → 基础对话数据 → NPC 设定的首个称号 → 说话人名，见 ``resolve_speaker_title``）；
- ``Char`` 由后端按情绪拼（``名#情绪``，没情绪就是纯名字）：模型给了合法情绪标签
  就用它，没给则沿用原行的 Char，组尾新增行取同说话人在原组里最后一次出现的 Char；
  玩家行缺省走 AS2 占位符（``$PC_TITLE`` / ``$PC_CHAR``，写成 ``$PC`` 会让称号槽
  显示角色名、立绘查不到），情绪挂在 ``$PC_CHAR`` 上；
- ``imageurl`` 等附加字段一律沿用原行；
- **按行序对齐**：第 i 行改第 i 行，只允许在组尾加行；
- **不能删除原有信息**：提交行数少于原组行数即打回（任务对话与日常对话同一口径）。

说话人白名单＝该 NPC / ``$PC`` / 原组里出现过的说话人，越界整组打回——防止模型
让某个 NPC 说出别人的台词（覆盖层按 NPC 块名落盘，串档是静默的）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.dialogue import overrides as overlay
from services.dialogue import task_text as text_overlay
from services.dialogue.models import (
    MAX_LINES_PER_GROUP,
    MAX_LINE_TEXT_CHARS,
    PLAYER_NAME,
    DialogueGroup,
    DialogueLine,
    compose_char,
    normalize_emotion,
    player_defaults,
    sanitize_text,
)
from services.dialogue.sources import resolve_speaker_title, speaker_emotions
from services.dialogue.view import DialogueView

logger = logging.getLogger(__name__)


@dataclass
class EditOutcome:
    """一次修改的结果：生效项、打回项、是否真的写了盘。"""

    applied: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    wrote_dialogue: bool = False
    wrote_task_text: bool = False

    @property
    def changed(self) -> bool:
        return self.wrote_dialogue or self.wrote_task_text


class DialogueEditor:
    """按编号映射表把模型提交应用到覆盖层。"""

    def __init__(self, view: DialogueView, index: list[dict[str, Any]]) -> None:
        self.view = view
        self.index = {int(entry["number"]): entry for entry in index if entry.get("number")}

    # ------------------------------------------------------------------

    def apply(
        self,
        *,
        npc: str,
        edits: Any,
        new_groups: Any = None,
        reason: str = "",
        source_task_id: int | None = None,
        source_trigger: str = "",
    ) -> EditOutcome:
        outcome = EditOutcome()
        name = (npc or "").strip()
        if not name:
            outcome.rejected.append("缺少 NPC 名，未做任何修改。")
            return outcome

        dialogue_entries: list[overlay.DialogueOverride] | None = None
        text_mapping: dict[str, list[dict[str, Any]]] | None = None

        for edit in edits if isinstance(edits, list) else []:
            number, lines = _parse_edit(edit)
            if number is None:
                outcome.rejected.append(f"修改项缺编号或编号非法，已跳过：{edit!r}")
                continue
            entry = self.index.get(number)
            if entry is None:
                outcome.rejected.append(f"编号 {number} 不在本次查询结果里，已跳过。")
                continue
            group = self._current_group(entry, name)
            if group is None:
                outcome.rejected.append(f"编号 {number}（{entry.get('key')}）在基础数据里已不存在，已跳过。")
                continue
            merged, error = self._merge_lines(group, lines, name)
            if error:
                outcome.rejected.append(f"编号 {number}：{error}")
                continue

            if entry["kind"] == "task":
                if text_mapping is None:
                    text_mapping = self.view.task_text_overrides()
                text_mapping = text_overlay.set_key(text_mapping, entry["key"], merged)
                outcome.wrote_task_text = True
                outcome.applied.append(f"编号 {number}（任务文本 {entry['key']}）：{len(merged)} 行")
            else:
                if dialogue_entries is None:
                    dialogue_entries = self.view.daily_overrides()
                group_id = str(entry["key"])
                is_agent_group = group_id.startswith(overlay.AGENT_GROUP_PREFIX)
                dialogue_entries = overlay.upsert_override(
                    dialogue_entries,
                    npc=name,
                    group_id=group_id,
                    action="append" if is_agent_group else "replace",
                    lines=merged,
                    base_dialogue_id=None if is_agent_group else group_id,
                    reason=reason,
                    source_task_id=source_task_id,
                    source_trigger=source_trigger,
                )
                outcome.wrote_dialogue = True
                outcome.applied.append(f"编号 {number}（日常对话组 {group_id}）：{len(merged)} 行")

        for raw_group in new_groups if isinstance(new_groups, list) else []:
            lines = _parse_lines(raw_group.get("lines") if isinstance(raw_group, dict) else raw_group)
            if not lines:
                outcome.rejected.append(f"新增组没有有效台词，已跳过：{raw_group!r}")
                continue
            merged, error = self._merge_lines(None, lines, name)
            if error:
                outcome.rejected.append(f"新增组：{error}")
                continue
            if dialogue_entries is None:
                dialogue_entries = self.view.daily_overrides()
            group_id = overlay.allocate_group_id(name, self.view.sources.groups(name), dialogue_entries)
            dialogue_entries = overlay.upsert_override(
                dialogue_entries,
                npc=name,
                group_id=group_id,
                action="append",
                lines=merged,
                reason=reason,
                source_task_id=source_task_id,
                source_trigger=source_trigger,
            )
            outcome.wrote_dialogue = True
            outcome.applied.append(f"新增日常对话组 {group_id}：{len(merged)} 行")

        self._flush(outcome, dialogue_entries, text_mapping)
        return outcome

    # ------------------------------------------------------------------

    def _current_group(self, entry: dict[str, Any], npc: str) -> DialogueGroup | None:
        """按编号映射取该组的**当前**内容（合并视图，含此前已落盘的覆盖）。"""
        if entry["kind"] == "task":
            for cand in self.view.task_candidates(npc):
                if cand.text_key == entry["key"]:
                    return cand.group
            return None
        for group in self.view.daily_groups(npc):
            if group.group_id == str(entry["key"]):
                return group
        return None

    def _merge_lines(
        self,
        group: DialogueGroup | None,
        submitted: list[tuple[str, str, str]],
        npc: str,
    ) -> tuple[list[DialogueLine], str]:
        """按行序对齐合并；返回 (结果行, 打回原因)——原因非空即整组不生效。"""
        if not submitted:
            return [], "没有提交任何台词。"
        if len(submitted) > MAX_LINES_PER_GROUP:
            return [], f"提交 {len(submitted)} 行，超过单组上限 {MAX_LINES_PER_GROUP} 行。"

        old = list(group.lines) if group is not None else []
        if len(submitted) < len(old):
            return [], (
                f"提交 {len(submitted)} 行少于原有 {len(old)} 行——只能改台词或在组尾加台词，"
                "不需要改的行请原样抄回，不能删除原有信息。"
            )

        allowed = {npc, PLAYER_NAME}
        allowed.update(line.name for line in old if line.name)

        merged: list[DialogueLine] = []
        for index, (raw_name, raw_text, raw_emotion) in enumerate(submitted):
            speaker = sanitize_text(raw_name) or npc
            # 模型爱把玩家写成中文「玩家」，而落盘必须是 $PC——AS2 按这个占位符换玩家
            # 角色名与立绘，写成「玩家」会当成陌生 NPC（称号槽与立绘都取不到）。
            # 另外把含 $PC 的变体一起收敛（$PC_TITLE、（$PC）、$pc）：模型会把落盘用的
            # 称号／立绘占位符当成说话人名，多写了前后缀。
            if speaker == "玩家" or PLAYER_NAME in speaker.upper():
                speaker = PLAYER_NAME
            text = sanitize_text(raw_text)
            if not text:
                return [], f"第 {index + 1} 行台词为空。"
            if len(text) > MAX_LINE_TEXT_CHARS:
                return [], f"第 {index + 1} 行台词 {len(text)} 字，超过上限 {MAX_LINE_TEXT_CHARS} 字。"
            if speaker not in allowed:
                return [], (
                    f"第 {index + 1} 行说话人「{speaker}」不在允许范围内"
                    f"（只能是 {npc}、{PLAYER_NAME} 或本组原有说话人）。"
                )
            emotion = normalize_emotion(raw_emotion, speaker_emotions(speaker))
            title, char = self._line_fields(speaker, old, index, emotion)
            origin = old[index] if index < len(old) else None
            merged.append(DialogueLine(
                name=speaker,
                title=title,
                char=char,
                text=text,
                extra=dict(origin.extra) if origin is not None else {},
            ))
        return merged, ""

    def _line_fields(
        self, speaker: str, old: list[DialogueLine], index: int, emotion: str,
    ) -> tuple[str, str]:
        """一行的 (Title, Char)：两样都由后端定，模型只给说话人、情绪与台词。

        Title 沿用本行原值（改台词不动铭牌），新增行按四级回落查
        （见 ``resolve_speaker_title``）。Char 只在模型给了合法情绪时才拼表情后缀；
        没给就沿用本行原值，新增行取同说话人在原组里最后一次出现的 Char——
        于是「原样抄回」的行既不会被削掉既有表情，也不会凭空多出表情。
        """
        is_player = speaker == PLAYER_NAME
        player_title, player_char = player_defaults()
        base_title = player_title if is_player else resolve_speaker_title(
            speaker, self.view.sources, own=_trailing_title(old, speaker),
        )
        # 玩家行的表情挂在 $PC_CHAR 上（与任务文本侧同一拼法）
        char_stem = player_char if is_player else speaker
        base_char = player_char if is_player else _trailing_char(old, speaker)
        origin = old[index] if index < len(old) else None
        title = (origin.title if origin is not None else "") or base_title
        char = compose_char(char_stem, emotion) if emotion else (
            (origin.char if origin is not None else "") or base_char
        )
        return title, char

    def _flush(
        self,
        outcome: EditOutcome,
        dialogue_entries: list[overlay.DialogueOverride] | None,
        text_mapping: dict[str, list[dict[str, Any]]] | None,
    ) -> None:
        if not outcome.changed:
            return
        try:
            dialogue_path, text_path = self.view.override_paths()
        except ValueError as exc:
            logger.warning("对话覆盖层无处落盘，放弃本次修改: %s", exc)
            outcome.rejected.append(f"{exc}，本次修改未生效。")
            outcome.wrote_dialogue = False
            outcome.wrote_task_text = False
            return

        try:
            if dialogue_entries is not None and outcome.wrote_dialogue:
                overlay.save_overrides(dialogue_path, dialogue_entries)
            if text_mapping is not None and outcome.wrote_task_text:
                text_overlay.save_task_text_overrides(text_path, text_mapping)
        except OSError as exc:
            logger.exception("对话覆盖层写盘失败")
            outcome.rejected.append(f"写盘失败：{exc}")
            outcome.wrote_dialogue = False
            outcome.wrote_task_text = False
            return

        from services.agent_tools.task_tools import bump_dialogue_version

        try:
            bump_dialogue_version(self.view.data_root)
        except OSError as exc:
            # 覆盖层已经写成了，只是通知游戏侧热失效的版本号没 bump 上去：
            # 这里绝不能报错，否则模型会重交同一批修改、把追加行写重。
            logger.warning("dialogue_version 更新失败（覆盖层已落盘，游戏侧下次载入生效）: %s", exc)


def _trailing_char(old: list[DialogueLine], speaker: str) -> str:
    """同说话人在原组里最后一次出现的 Char（带表情后缀），没有就用说话人名。"""
    for line in reversed(old):
        if line.name == speaker and line.char:
            return line.char
    return speaker


def _trailing_title(old: list[DialogueLine], speaker: str) -> str:
    """同说话人在原组里最后一次出现的 Title，没有就空串。"""
    for line in reversed(old):
        if line.name == speaker and line.title:
            return line.title
    return ""


def _parse_edit(raw: Any) -> tuple[int | None, list[tuple[str, str, str]]]:
    if not isinstance(raw, dict):
        return None, []
    try:
        number = int(raw.get("number"))
    except (TypeError, ValueError):
        return None, []
    return number, _parse_lines(raw.get("lines"))


def _parse_lines(raw: Any) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        text = sanitize_text(item.get("text"))
        if not text:
            continue
        out.append((
            str(item.get("name") or "").strip(),
            text,
            str(item.get("emotion") or "").strip(),
        ))
    return out

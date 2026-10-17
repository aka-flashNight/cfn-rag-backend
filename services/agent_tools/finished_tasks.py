"""「本 NPC 发布的 Agent 委托已被玩家完成」的实时拼接（阶段三·任务发布→完成）。

只能直读磁盘：TaskRegistry 看不到按存档的 data/rag/agent_saves/<slotKey>/ 覆盖层，
而玩家台词不进检索语料，接取与完成对话原文只能从 agent_text.json 取。
读取范围＝共享文件 ∪ 本存档文件，按 id 合并（存档优先），与语料加载口径一致。

无状态：每轮按当前存档投影里的 tasks_finished 实时算，不做「已通知」记账——
投影只在打开面板/读档/存档落盘/任务完成四处刷新，任务完成后这一轮就能看到。

阶段四追加两件事（同一份磁盘读取里一起给出，不再读第二遍）：
① 每条委托明确带上任务 id（§4.0.2-3：委派子 Agent 时「必须给任务 id」）；
② 发布时挂了对话重写标记的委托，常驻提醒模型「已自动触发、勿重复调用」——
   任务完成后把完成信息给模型时这句也一直保留，未标记的委托不加。
   这句只进聊天主 Agent 的已完成委托块；触发一装配给对话修改子 Agent 的任务详情不带它
   （子 Agent 正是被派来改对话的那一个，提醒对它是自我指涉的干扰）。

另外两块针对**未完成**的委托（同源读取、同源渲染，只多完成 NPC 场景与关卡怪物）：
③ ``build_published_task_block``：刚发布的那一份——发布轮追加到本轮回合结果块，
   后续若干轮由会话库的记录驱动、进 user 侧共享上下文；
④ ``build_active_tasks_block``：绑定投影 ``progress.tasks_active`` 里仍在途的委托，
   每轮实时算、只读、不记账。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.integration.save_paths import agent_save_task_paths, shared_task_paths

logger = logging.getLogger(__name__)

#: 一轮里最多拼接几条已完成委托（按 id 从新到旧取，控制 token）
MAX_FINISHED_TASKS_PER_TURN = 3

#: 已挂对话重写标记的委托的常驻提醒（§4.0.2-3 兜底提醒，逐字口径）
MARKED_TASK_HINT = "该任务已自动触发对话修改的子 Agent，请勿因该任务而重复调用。"


@dataclass
class FinishedTaskBlock:
    """已完成委托块：拼好的文本＋「是否存在未挂重写标记的委托」。

    后者是触发三的条件闸口径（§4.0.2-3）：仅当当轮真的拼了已完成委托、
    且其中有未标记的重要任务时，委派工具的提示词才保留任务兜底表述；
    否则任务相关表述整段不给（与注册闸同一精神：给了才提）。
    """

    text: str = ""
    has_unmarked: bool = False


@dataclass
class TaskRewriteMaterial:
    """触发一自动委派要的材料：任务详情与对话＋发布时挂的标记（一次读取拿全）。"""

    task_id: int
    npc: str
    marked: bool
    instruction: str
    publish_session_id: str
    detail_block: str


def _read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        text = path.read_text("utf-8")
        return json.loads(text) if text.strip() else default
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取 %s 失败，按空处理: %s", path, exc)
        return default


def _merged_docs(data_root: Path, slot_key: str | None) -> tuple[dict[int, dict], dict[str, Any]]:
    """(任务记录 by id, 文本表)；共享文件先读、本存档文件后读并覆盖同 id。"""
    tasks: dict[int, dict] = {}
    texts: dict[str, Any] = {}
    pairs = [shared_task_paths(data_root)]
    if slot_key:
        pairs.append(agent_save_task_paths(data_root, slot_key))
    for tasks_path, text_path in pairs:
        doc = _read_json(tasks_path, {"tasks": []})
        records = doc.get("tasks") if isinstance(doc, dict) else None
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict):
                continue
            try:
                task_id = int(record.get("id"))
            except (TypeError, ValueError):
                continue
            tasks[task_id] = record
        text_doc = _read_json(text_path, {})
        if isinstance(text_doc, dict):
            texts.update(text_doc)
    return tasks, texts


def _dialogue_lines(items: Any) -> list[str]:
    """完成对话数组 → 「说话人：台词」行；玩家占位符 $PC 显示为「玩家」。"""
    lines: list[str] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        name = str(item.get("name") or "").strip()
        speaker = "玩家" if name == "$PC" else (name or "对方")
        lines.append(f"{speaker}：{text}")
    return lines


def _item_text(values: Any) -> str:
    """["物品名#数量"] → 「物品名×数量、…」；无有效项返回空串。"""
    parts: list[str] = []
    for raw in values if isinstance(values, list) else []:
        name, _, count = str(raw or "").strip().partition("#")
        if not name.strip():
            continue
        parts.append(f"{name.strip()}×{count.strip()}" if count.strip() else name.strip())
    return "、".join(parts)


def _item_phrase(values: Any, prefix: str) -> str:
    """物品短语带前缀；只呈现，不参与判定。"""
    joined = _item_text(values)
    return f"{prefix}：{joined}" if joined else ""


def _stage_text(values: Any) -> str:
    """["关卡名#难度"] → 「关卡名（难度）、…」——难度套上 × 会被读成数量。"""
    parts: list[str] = []
    for raw in values if isinstance(values, list) else []:
        name, _, difficulty = str(raw or "").strip().partition("#")
        if not name.strip():
            continue
        tail = difficulty.strip()
        parts.append(f"{name.strip()}（{tail}）" if tail else name.strip())
    return "、".join(parts)


def _completion_phrase(record: dict) -> str:
    """任务达成内容一行：提交给谁／获取（不提交）／通关；落盘无需求字段时写「无」。"""
    parts: list[str] = []
    submit = _item_text(record.get("finish_submit_items"))
    contain = _item_text(record.get("finish_contain_items"))
    stages = _stage_text(record.get("finish_requirements"))
    if submit:
        # 落盘记录的 finish_npc 恒非空（缺省时回退为发布 NPC），不必再防「给谁」缺失
        parts.append(f"提交{submit}给{str(record.get('finish_npc') or '').strip()}")
    if contain:
        parts.append(f"获取{contain}（不提交）")
    if stages:
        parts.append(f"通关{stages}")
    return "任务达成内容（玩家达成）：" + ("；".join(parts) if parts else "无")


def _resolve_title(task_id: int, texts: dict[str, Any]) -> str:
    """任务展示标题（$AGENT_TITLE_<id>）；缺失＝空串（调用方跳过该条并已记警告）。"""
    title = str(texts.get(f"$AGENT_TITLE_{task_id}") or "").strip()
    if not title:
        logger.warning("委托 %d 缺少标题文本，跳过拼接", task_id)
    return title


def _description_reward_lines(
    task_id: int, record: dict, texts: dict[str, Any],
) -> list[str]:
    """任务说明行与报酬行（有才出现）；已完成块与未完成块共用同一份呈现口径。"""
    lines: list[str] = []
    description = str(texts.get(f"$AGENT_DESCRIPTION_{task_id}") or "").strip()
    if description:
        lines.append(f"任务说明：{description}")
    reward = _item_phrase(record.get("rewards"), "报酬")
    if reward:
        lines.append(reward)
    return lines


def _task_lines(
    task_id: int, record: dict, texts: dict[str, Any], header: str,
    *, include_marked_hint: bool = True,
) -> list[str]:
    """一条委托的呈现行（标题行由调用方给）；缺标题文本＝返回空列表（跳过该条）。

    ``include_marked_hint=False`` 给对话修改子 Agent 用：那句提醒是拦聊天主 Agent
    重复派发的，被派来改对话的那个 Agent 读到它只会自我怀疑。
    """
    title = _resolve_title(task_id, texts)
    if not title:
        return []
    lines = [header.format(title=title, task_id=task_id)]
    lines.extend(_description_reward_lines(task_id, record, texts))
    get_dialogue = _dialogue_lines(texts.get(f"$AGENT_GET_{task_id}"))
    if get_dialogue:
        lines.append("任务前：")
        lines.extend(get_dialogue)
    lines.append(_completion_phrase(record))
    finish_dialogue = _dialogue_lines(texts.get(f"$AGENT_FINISH_{task_id}"))
    if finish_dialogue:
        lines.append("任务后：")
        lines.extend(finish_dialogue)
    if include_marked_hint and record.get("dialogue_rewrite"):
        lines.append(MARKED_TASK_HINT)
    return lines


_FINISHED_HEADER = "此前你发布的委托「{title}」（任务 id {task_id}）已被玩家完成。"


def build_finished_task_block(
    *,
    data_root: Path,
    slot_key: str | None,
    npc_name: str,
    finished_ids: Any,
) -> FinishedTaskBlock:
    """该 NPC 发布、且当前存档已完成的委托块；没有匹配项＝空块（整块不注入）。"""
    npc = (npc_name or "").strip()
    if not npc:
        return FinishedTaskBlock()
    try:
        finished = {int(v) for v in (finished_ids or [])}
    except (TypeError, ValueError):
        logger.warning("存档投影 tasks_finished 含非整数项，跳过完成委托拼接")
        return FinishedTaskBlock()
    if not finished:
        return FinishedTaskBlock()

    tasks, texts = _merged_docs(Path(data_root), slot_key)
    matched = [
        (task_id, record)
        for task_id, record in tasks.items()
        if task_id in finished and str(record.get("get_npc") or "").strip() == npc
    ]
    if not matched:
        return FinishedTaskBlock()
    matched.sort(key=lambda pair: pair[0], reverse=True)

    sections: list[str] = []
    has_unmarked = False
    for task_id, record in matched[:MAX_FINISHED_TASKS_PER_TURN]:
        lines = _task_lines(task_id, record, texts, _FINISHED_HEADER)
        if not lines:
            continue
        sections.append("\n".join(lines))
        # 条件闸只看真的拼进这一轮的那些委托：没拼出来的模型也指认不了 id
        if not record.get("dialogue_rewrite"):
            has_unmarked = True

    if not sections:
        return FinishedTaskBlock()
    return FinishedTaskBlock(
        text="【你发布的委托已被玩家完成】\n" + "\n\n".join(sections),
        has_unmarked=has_unmarked,
    )


_DETAIL_HEADER = "你发布的委托「{title}」（任务 id {task_id}），玩家刚刚完成。"


def load_task_rewrite_material(
    *, data_root: Path, slot_key: str | None, task_id: int,
) -> TaskRewriteMaterial | None:
    """触发一：按任务 id 取该委托的详情与对话，以及发布时挂的重写标记。

    任务不存在（或该存档看不到它）＝None，调用方据此记 dismissed。
    这是给对话修改子 Agent 的材料，不拼「勿重复调用」那句聊天侧提醒。
    """
    tasks, texts = _merged_docs(Path(data_root), slot_key)
    record = tasks.get(int(task_id))
    if not isinstance(record, dict):
        return None
    lines = _task_lines(
        int(task_id), record, texts, _DETAIL_HEADER, include_marked_hint=False,
    )
    return TaskRewriteMaterial(
        task_id=int(task_id),
        npc=str(record.get("get_npc") or "").strip(),
        marked=bool(record.get("dialogue_rewrite")),
        instruction=str(record.get("dialogue_rewrite_instruction") or "").strip(),
        publish_session_id=str(record.get("publish_session_id") or "").strip(),
        detail_block="\n".join(lines),
    )


# ---------------------------------------------------------------------------
# 未完成委托：刚发布（工作项 1）与尚在途中（工作项 2）
# ---------------------------------------------------------------------------
#
# 两块与已完成块同源：同一份磁盘读取（共享 ∪ 本存档，存档覆盖共享）、
# 同一套呈现口径（标题/说明/报酬/达成内容），只多两段「完成 NPC 场景」与
# 「关卡与怪物」。差别只在开头一句与块尾的行为约束。
#
# 两块都**不做任何写盘**：读磁盘只读；rounds 计数由会话库
# （session_published_task_highlights）自行维护，不碰 npc_state_db / status / progress。

#: 一轮里最多拼接几条未完成委托（与已完成块同量级）
MAX_ACTIVE_TASKS_PER_TURN = 3

_PUBLISHED_HEADER = "此前你发布的委托「{title}」（任务 id {task_id}）刚刚交给玩家，目前尚未完成。"
_ACTIVE_HEADER = "此前你发布的委托「{title}」（任务 id {task_id}）仍在进行中，玩家尚未完成。"

_PUBLISHED_TAIL = (
    "这份委托由你刚发布，玩家尚未完成；玩家问起时按上述要点说明，不要重复发布同类委托。"
)
_ACTIVE_TAIL = (
    "以上委托由你发布，玩家均未完成；玩家问起时按当前进度说明，"
    "不要声称已经完成，也不要重复发布同类委托。"
)


def _int_ids(values: Any) -> list[int]:
    """投影里的 id 数组 → int 列表（非法项跳过，不抛）。"""
    out: list[int] = []
    for raw in values if isinstance(values, (list, tuple, set)) else []:
        try:
            out.append(int(raw))
        except (TypeError, ValueError):
            continue
    return out


def _stage_names(record: dict) -> list[str]:
    """finish_requirements 的关卡名（去难度、去重、保序）。"""
    names: list[str] = []
    for raw in record.get("finish_requirements") or []:
        name = str(raw or "").strip().partition("#")[0].strip()
        if name and name not in names:
            names.append(name)
    return names


def _finish_npc_scene(game_data: Any, finish_npc: str) -> str:
    """完成 NPC 的驻场景（多驻地「、」连接）；查不到/无注册表＝空串（不写括号）。"""
    if game_data is None or not finish_npc:
        return ""
    try:
        from services.npc.scene_context import npc_scene_text

        return npc_scene_text(getattr(game_data, "map_registry", None), finish_npc)
    except Exception:
        return ""


def _finish_extra_lines(record: dict, game_data: Any) -> list[str]:
    """完成 NPC（含场景）与逐关卡怪物块——查不到就整行省略，不写「未知」。"""
    lines: list[str] = []
    finish_npc = str(record.get("finish_npc") or "").strip()
    if finish_npc:
        scene = _finish_npc_scene(game_data, finish_npc)
        lines.append(f"完成 NPC：{finish_npc}（{scene}）" if scene else f"完成 NPC：{finish_npc}")
    stages = _stage_names(record)
    if stages and game_data is not None:
        from services.agent_tools.draft_formatting import _stage_monster_text

        rows: list[str] = []
        for stage_name in stages:
            block = _stage_monster_text(game_data, stage_name)
            if block:
                rows.append(f"- {stage_name}：{block}")
        if rows:
            lines.append("关卡与怪物：")
            lines.extend(rows)
    return lines


def _unfinished_task_lines(
    task_id: int, record: dict, texts: dict[str, Any], header: str, game_data: Any,
) -> list[str]:
    """一条未完成委托的呈现行；缺标题文本＝空列表（跳过该条）。

    未完成任务还没有完成对话，所以不出「任务前/任务后」两组——只给说明、报酬、
    达成内容与完成 NPC/关卡怪物。
    """
    title = _resolve_title(task_id, texts)
    if not title:
        return []
    lines = [header.format(title=title, task_id=task_id)]
    lines.extend(_description_reward_lines(task_id, record, texts))
    lines.append(_completion_phrase(record))
    lines.extend(_finish_extra_lines(record, game_data))
    return lines


def build_published_task_block(
    *,
    data_root: Path,
    slot_key: str | None,
    npc_name: str,
    task_id: Any,
    game_data: Any = None,
) -> str:
    """「你刚发布的委托」块（发布轮追加到 result_block，后续几轮进 user core）。

    读不到任务记录 / 标题缺失 / 不是该 NPC 发布 → 空串：调用方据此既不注入、
    也**不**写供给期记录（prompt 里不许出现半截内容）。

    ``game_data`` 只用于补完成 NPC 场景与关卡怪物；缺省时这两段整段省略。
    """
    npc = (npc_name or "").strip()
    if not npc:
        return ""
    try:
        tid = int(task_id)
    except (TypeError, ValueError):
        logger.warning("发布轮缺有效任务 id（%r），跳过达成要点拼接", task_id)
        return ""
    tasks, texts = _merged_docs(Path(data_root), slot_key)
    record = tasks.get(tid)
    if not isinstance(record, dict):
        logger.warning("发布轮读不到委托 %d 的记录，跳过达成要点拼接", tid)
        return ""
    if str(record.get("get_npc") or "").strip() != npc:
        logger.warning("发布轮委托 %d 的发布 NPC 不是「%s」，跳过达成要点拼接", tid, npc)
        return ""
    lines = _unfinished_task_lines(tid, record, texts, _PUBLISHED_HEADER, game_data)
    if not lines:
        return ""
    return "\n".join(["【你刚发布的委托（玩家尚未完成）】", *lines, _PUBLISHED_TAIL])


def build_active_tasks_block(
    *,
    data_root: Path,
    slot_key: str | None,
    npc_name: str,
    active_ids: Any,
    finished_ids: Any,
    exclude_ids: Any = None,
    game_data: Any = None,
) -> str:
    """「你发布的委托尚在途中」块（工作项 2）：按绑定时投影的 tasks_active 实时拼。

    **纯读**：只读共享 ∪ 本存档的 Agent 任务文件，不写数据库、不写 npc_state_db、
    不改 progress、不写 status；每轮实时算、不记账。

    过滤链：合法 id → 只留 ``get_npc`` 等于当前 NPC → 去掉已完成（防投影短暂不同步）
    → 去掉正在被「刚发布」块展示的那条 → id 倒序取最近若干条。
    ``active_ids`` 缺失/为空/无匹配＝空串（整块不注入）。
    """
    npc = (npc_name or "").strip()
    if not npc:
        return ""
    active = set(_int_ids(active_ids))
    if not active:
        return ""
    finished = set(_int_ids(finished_ids))
    excluded = set(_int_ids(exclude_ids))
    tasks, texts = _merged_docs(Path(data_root), slot_key)
    matched = [
        (task_id, record)
        for task_id, record in tasks.items()
        if task_id in active
        and task_id not in finished
        and task_id not in excluded
        and str(record.get("get_npc") or "").strip() == npc
    ]
    if not matched:
        return ""
    matched.sort(key=lambda pair: pair[0], reverse=True)

    sections: list[str] = []
    for task_id, record in matched[:MAX_ACTIVE_TASKS_PER_TURN]:
        lines = _unfinished_task_lines(task_id, record, texts, _ACTIVE_HEADER, game_data)
        if lines:
            sections.append("\n".join(lines))
    if not sections:
        return ""
    block = (
        "【你发布的委托尚在途中（玩家尚未完成）】\n"
        "以下委托由你发布，玩家仍未完成：\n\n"
        + "\n\n".join(sections)
    )
    remainder = len(matched) - len(sections)
    if remainder > 0:
        block += f"\n\n另有 {remainder} 条进行中的委托未展开。"
    return f"{block}\n\n{_ACTIVE_TAIL}"



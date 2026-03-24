from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.game_data.registry import GameDataRegistry


_WRITE_LOCK = threading.Lock()


def _atomic_write_text(path: Path, text: str) -> None:
    """
    原子写入：先写临时文件再 os.replace。
    """
    tmp_path = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    try:
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(str(tmp_path), str(path))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            # 清理失败不影响主流程
            pass


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return default
    return json.loads(raw)


def _dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _to_roman(num: int) -> str:
    """
    1..3999 转罗马数字；超范围时回退为十进制字符串。
    """
    if num <= 0 or num >= 4000:
        return str(num)
    vals = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    n = num
    out: list[str] = []
    for v, sym in vals:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def resolve_task_title_for_display(raw_title: Any, game_data: GameDataRegistry) -> str:
    """
    统一任务标题展示值：
    - "$KEY" 且在 task_texts 有映射 -> 映射文本
    - 其他情况 -> 原样字符串
    """
    s = str(raw_title or "").strip()
    if not s:
        return ""
    if s.startswith("$"):
        resolved = game_data.task_texts.resolve_str(s)
        return str(resolved or s).strip()
    return s


def collect_existing_task_titles(game_data: GameDataRegistry) -> set[str]:
    """
    收集所有已有任务的“展示标题”（已解析文本 key），用于标题去重。
    """
    titles: set[str] = set()
    for t in game_data.tasks.list_all_tasks():
        title = resolve_task_title_for_display(getattr(t, "title", ""), game_data)
        if title:
            titles.add(title)
    return titles


def make_unique_task_title(
    title: str,
    *,
    game_data: GameDataRegistry,
) -> tuple[str, bool]:
    """
    若标题与已有任务重复，则自动追加简洁后缀直到不重复：
    - 第一次冲突："{title} II"
    - 后续："{title} III" / "{title} IV" ...
    返回 (最终标题, 是否发生自动改名)。
    """
    base = (title or "").strip()
    if not base:
        return "", False
    existing = collect_existing_task_titles(game_data)
    if base not in existing:
        return base, False
    idx = 2
    while True:
        candidate = f"{base} {_to_roman(idx)}"
        if candidate not in existing:
            return candidate, True
        idx += 1


def _sync_state_path(data_root: Path) -> Path:
    """RAG 同步状态：与 npc_state_db 等并列于 <data>/rag/。"""
    return (Path(data_root).resolve() / "rag" / "sync_state.json").resolve()


def _bump_task_publish_version(sync_path: Path) -> None:
    """
    任务成功落盘后更新 task_publish_version：
    - 文件不存在：写入 {"task_publish_version": 1}
    - 已存在：若有该键则 +1；若无该键则置为 1；其它键原样保留。
    """
    sync_path.parent.mkdir(parents=True, exist_ok=True)
    if not sync_path.exists():
        _atomic_write_text(sync_path, _dump_json({"task_publish_version": 1}))
        return
    try:
        raw = sync_path.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    if "task_publish_version" not in data:
        data["task_publish_version"] = 1
    else:
        try:
            cur = int(data["task_publish_version"])
        except (TypeError, ValueError):
            cur = 0
        data["task_publish_version"] = cur + 1
    _atomic_write_text(sync_path, _dump_json(data))


def _reward_items_to_expr(items: Any) -> List[str]:
    """
    [{item_name, count}] -> ["物品名#数量", ...]
    """
    out: List[str] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("item_name")
        count = it.get("count")
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            n = int(count)
        except Exception:
            continue
        if n <= 0:
            continue
        out.append(f"{name}#{n}")
    return out


def _stage_reqs_to_strings(stage_reqs: Any) -> List[str]:
    """
    [{stage_name, difficulty}] -> ["关卡名#难度", ...]
    """
    out: List[str] = []
    if not isinstance(stage_reqs, list):
        return out
    for sr in stage_reqs:
        if not isinstance(sr, dict):
            continue
        stage_name = sr.get("stage_name")
        difficulty = sr.get("difficulty")
        if not isinstance(stage_name, str) or not stage_name.strip():
            continue
        if not isinstance(difficulty, str) or not difficulty.strip():
            continue
        out.append(f"{stage_name}#{difficulty}")
    return out


def _ensure_int_list(v: Any) -> List[int]:
    if not isinstance(v, list):
        return []
    out: List[int] = []
    for x in v:
        try:
            out.append(int(x))
        except Exception:
            continue
    return out


def _stage_reqs_unlock_ids(
    *,
    finish_requirements: Any,
    game_data: GameDataRegistry,
) -> List[int]:
    """
    从 finish_requirements 反推对应关卡（stage_name）的主线解锁 id（unlock_condition）。

    规则：
    - unlock_condition 缺失/为非正数：认为是副本等无主线解锁需求，不写入 get_requirements。
    - 若同一 stage_name 在多个大区存在多个 unlock_condition：取最小值（更符合“最早解锁前置”）。
    """
    stage_registry = getattr(game_data, "stages", None)
    stage_infos_raw = getattr(stage_registry, "_stage_infos", None) if stage_registry else None
    if not isinstance(stage_infos_raw, dict):
        return []

    out: set[int] = set()
    if not isinstance(finish_requirements, list):
        return []

    for sr in finish_requirements:
        if not isinstance(sr, dict):
            continue
        stage_name = sr.get("stage_name")
        if not isinstance(stage_name, str) or not stage_name.strip():
            continue

        unlocks: list[int] = []
        for (_area, name), si in stage_infos_raw.items():
            if name != stage_name:
                continue
            unlock = getattr(si, "unlock_condition", None)
            if isinstance(unlock, int) and unlock > 0:
                unlocks.append(unlock)

        if unlocks:
            out.add(min(unlocks))

    return sorted(out)


def _strip_bracket_expressions(text: str) -> str:
    """
    去掉形如 `【...】` / `（...）` 的动作/神态/旁白标记，确保 text 是纯对话。
    """
    import re

    s = (text or "").strip()
    # 常见格式：把开头的【...】/（...）当作动作/旁白前缀直接剥离
    s = re.sub(r"^(?:【[^】]*】|（[^）]*）|\s)+", "", s)
    # 其余位置的【...】也直接删掉
    cleaned = re.sub(r"【[^】]*】", "", s)
    # 若文本仍然是纯旁白段落，也把（...）删掉（尽量严格）
    cleaned = re.sub(r"（[^）]*）", "", cleaned)
    return cleaned.strip()


def _extract_leading_emotion_from_brackets(text: str) -> tuple[str, str]:
    """
    从字符串开头的 `【情绪】` 前缀提取 emotion，并返回去前缀后的纯文本。
    """
    import re

    s = (text or "").strip()
    m = re.match(r"^【([^】]+)】(.*)$", s)
    if not m:
        return "", _strip_bracket_expressions(s)
    emotion = (m.group(1) or "").strip()
    rest = (m.group(2) or "").strip()
    return emotion, _strip_bracket_expressions(rest)


def _dialogue_entry_to_agent_text_item(
    entry: Dict[str, Any],
) -> Dict[str, str]:
    """
    将草案 dialogue entry 映射到 agent_text.json 的单条：
    { name, title, char, text }
    """
    name = str(entry.get("name") or "").strip()
    title = str(entry.get("title") or "").strip()
    emotion = str(entry.get("emotion") or "").strip()
    text = str(entry.get("text") or "").strip()
    text = _strip_bracket_expressions(text)

    if name == "$PC":
        title = "$PC_TITLE" if not title else title
        char_base = "$PC_CHAR"
        char = f"{char_base}#{emotion}" if emotion else char_base
        return {"name": "$PC", "title": title, "char": char, "text": text}

    # NPC：char 用 NPC名#emotion（可选）
    if not title:
        title = name
    char = f"{name}#{emotion}" if emotion else name
    return {"name": name, "title": title, "char": char, "text": text}


def write_confirmed_agent_task_files(
    *,
    draft: Dict[str, Any],
    npc_name_fallback: str,
    game_data: GameDataRegistry,
) -> tuple[str, int]:
    """
    将已确认的任务草案写入：
    - resources/data/task/agent_tasks.json
    - resources/data/task/text/agent_text.json

    在锁内根据磁盘上已有 agent 任务的最大 ID 分配新 ID（避免内存 TaskRegistry 未重载时重复 ID 覆盖旧任务）。

    返回 (写入描述字符串, 分配的任务 ID)。
    """
    with _WRITE_LOCK:
        data_root = game_data.data_root
        agent_tasks_path = (data_root / "task" / "agent_tasks.json").resolve()
        agent_text_path = (data_root / "task" / "text" / "agent_text.json").resolve()

        tasks_doc = _read_json(agent_tasks_path, default={"tasks": []})
        if not isinstance(tasks_doc, dict):
            tasks_doc = {"tasks": []}
        tasks = tasks_doc.get("tasks", [])
        if not isinstance(tasks, list):
            tasks = []

        max_agent = 200000
        for t in tasks:
            if not isinstance(t, dict):
                continue
            try:
                tid = int(t.get("id"))
            except Exception:
                continue
            if 200001 <= tid <= 300000 and tid > max_agent:
                max_agent = tid

        task_id_int = max_agent + 1
        draft["id"] = task_id_int

        get_npc = (
            draft.get("get_npc")
            if isinstance(draft.get("get_npc"), str) and draft.get("get_npc").strip()
            else draft.get("npc_name")
        )
        finish_npc = (
            draft.get("finish_npc")
            if isinstance(draft.get("finish_npc"), str) and draft.get("finish_npc").strip()
            else draft.get("npc_name")
        )
        if not isinstance(get_npc, str) or not get_npc.strip():
            get_npc = npc_name_fallback
        if not isinstance(finish_npc, str) or not finish_npc.strip():
            finish_npc = npc_name_fallback

        title = draft.get("title") if isinstance(draft.get("title"), str) else ""
        # 最终落盘前统一做标题去重，避免与已有任务重名。
        title, title_auto_renamed = make_unique_task_title(title, game_data=game_data)
        draft["title"] = title
        description = (
            draft.get("description") if isinstance(draft.get("description"), str) else ""
        )

        # 新结构：对话数组（优先）
        get_dialogue = draft.get("get_dialogue")
        finish_dialogue = draft.get("finish_dialogue")

        # 旧结构兼容：字符串对话
        get_text_old = draft.get("get_conversation_text")
        finish_text_old = draft.get("finish_conversation_text")
        if not isinstance(get_text_old, str):
            get_text_old = ""
        if not isinstance(finish_text_old, str):
            finish_text_old = ""

        # 生成 agent_text.json 的数组（纯对话）
        get_items: List[Dict[str, str]] = []
        finish_items: List[Dict[str, str]] = []

        if isinstance(get_dialogue, list) and get_dialogue:
            for it in get_dialogue:
                if isinstance(it, dict):
                    get_items.append(_dialogue_entry_to_agent_text_item(it))
        elif get_text_old:
            # 只有旧字段：为当前 get_npc 生成一条
            emo, text_clean = _extract_leading_emotion_from_brackets(get_text_old)
            get_items = [
                {
                    "name": get_npc,
                    "title": get_npc,
                    "char": f"{get_npc}#{emo}" if emo else get_npc,
                    "text": text_clean,
                }
            ]

        if isinstance(finish_dialogue, list) and finish_dialogue:
            for it in finish_dialogue:
                if isinstance(it, dict):
                    finish_items.append(
                        _dialogue_entry_to_agent_text_item(it)
                    )
        elif finish_text_old:
            emo, text_clean = _extract_leading_emotion_from_brackets(finish_text_old)
            finish_items = [
                {
                    "name": finish_npc,
                    "title": finish_npc,
                    "char": f"{finish_npc}#{emo}" if emo else finish_npc,
                    "text": text_clean,
                }
            ]

        # 如果 LLM 没给对话数组，但有字符串字段为空，至少要给一个空对话占位
        # 避免游戏端/前端解析空数组出错
        if not get_items:
            get_items = [{"name": get_npc, "title": get_npc, "char": get_npc, "text": ""}]
        if not finish_items:
            finish_items = [{"name": finish_npc, "title": finish_npc, "char": finish_npc, "text": ""}]

        get_req_list = _ensure_int_list(draft.get("get_requirements"))
        # 若大模型没填 get_requirements，则由后端根据 finish_requirements 的关卡解锁条件补全。
        # 该逻辑不依赖任务类型：只要 finish_requirements 中的关卡有 unlock_condition，就应写入前置主线。
        if not get_req_list:
            get_req_list = _stage_reqs_unlock_ids(
                finish_requirements=draft.get("finish_requirements"),
                game_data=game_data,
            )
        finish_reqs = _stage_reqs_to_strings(draft.get("finish_requirements"))
        finish_submit = _reward_items_to_expr(draft.get("finish_submit_items"))
        finish_contain = _reward_items_to_expr(draft.get("finish_contain_items"))
        rewards = _reward_items_to_expr(draft.get("rewards"))

        # -------- agent_tasks.json（tasks_doc 已在上方读取）--------
        # 防止重复写入同 id
        tasks = [t for t in tasks if not (isinstance(t, dict) and str(t.get("id")) == str(task_id_int))]

        tasks.append(
            {
                "id": task_id_int,
                "title": f"$AGENT_TITLE_{task_id_int}",
                "description": f"$AGENT_DESCRIPTION_{task_id_int}",
                "get_requirements": get_req_list,
                "get_conversation": f"$AGENT_GET_{task_id_int}",
                "get_npc": get_npc,
                "finish_requirements": finish_reqs,
                "finish_submit_items": finish_submit,
                "finish_contain_items": finish_contain,
                "finish_conversation": f"$AGENT_FINISH_{task_id_int}",
                "finish_npc": finish_npc,
                "rewards": rewards,
                "announcement": "",
                "chain": "委托",
            }
        )
        new_tasks_doc = {"tasks": tasks}

        # -------- agent_text.json --------
        text_doc = _read_json(agent_text_path, default={})
        if not isinstance(text_doc, dict):
            text_doc = {}

        text_doc[f"$AGENT_TITLE_{task_id_int}"] = title
        text_doc[f"$AGENT_DESCRIPTION_{task_id_int}"] = description
        # 接取/完成：写入 dialogue array 映射后的多条对话
        text_doc[f"$AGENT_GET_{task_id_int}"] = get_items
        text_doc[f"$AGENT_FINISH_{task_id_int}"] = finish_items

        _atomic_write_text(agent_tasks_path, _dump_json(new_tasks_doc))
        _atomic_write_text(agent_text_path, _dump_json(text_doc))

        _bump_task_publish_version(_sync_state_path(data_root))

        if title_auto_renamed:
            msg = (
                f"任务 {task_id_int} 已写入 agent_tasks.json 与 agent_text.json。"
                f"标题重复，已自动调整为「{title}」。"
            )
        else:
            msg = f"任务 {task_id_int} 已写入 agent_tasks.json 与 agent_text.json。"
        return msg, task_id_int


from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.dialogue.models import (
    PLAYER_NAME,
    compose_char,
    normalize_emotion,
    player_defaults,
)
from services.dialogue.sources import DialogueSources, resolve_speaker_title, speaker_emotions
from services.dialogue.view import shared_sources
from services.game_data.registry import GameDataRegistry
from services.integration.save_paths import resolve_task_write_paths, shared_task_paths
from services.integration.state import get_integration_state

# 当前最大任务进度常量：写入 get_requirements 时，若计算值大于此值则使用此值
MAX_MAINLINE_TASK_PROGRESS: int = 77
# Agent 任务 id 区间（上游 200000 为基准，超出 300000 视为越界）
AGENT_TASK_ID_BASE: int = 200000
AGENT_TASK_ID_MAX: int = 300000

logger = logging.getLogger(__name__)


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


def _bump_sync_version(sync_path: Path, key: str) -> None:
    """
    sync_state.json 里某个版本号 +1（调用方须持 _WRITE_LOCK）：
    - 文件不存在：写入 {key: 1}
    - 已存在：若有该键则 +1；若无该键则置为 1；其它键原样保留。
    """
    sync_path.parent.mkdir(parents=True, exist_ok=True)
    if not sync_path.exists():
        _atomic_write_text(sync_path, _dump_json({key: 1}))
        return
    try:
        raw = sync_path.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    if key not in data:
        data[key] = 1
    else:
        try:
            cur = int(data[key])
        except (TypeError, ValueError):
            cur = 0
        data[key] = cur + 1
    _atomic_write_text(sync_path, _dump_json(data))


def _bump_task_publish_version(sync_path: Path) -> None:
    """任务成功落盘后更新 task_publish_version。"""
    _bump_sync_version(sync_path, "task_publish_version")


def bump_dialogue_version(data_root: Path) -> None:
    """对话覆盖层落盘后更新 dialogue_version（阶段四 B4-2-4）。

    与任务版本号同一文件、同一把 _WRITE_LOCK，防两条写链撞车；
    游戏侧轮询该键变化后使对话覆盖层缓存失效（G4-5）。
    """
    with _WRITE_LOCK:
        _bump_sync_version(_sync_state_path(data_root), "dialogue_version")


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


def _agent_text_item(
    name: Any, emotion: Any, text: str, sources: DialogueSources | None,
) -> Dict[str, str]:
    """任务文本的一条：模型只给 name/emotion/text，Title 与 Char 都由后端补齐。

    Title 玩家行走 ``$PC_TITLE`` 占位符，NPC 行按四级回落查（见 ``resolve_speaker_title``）；
    Char 只在情绪合法时拼表情后缀（玩家行挂在 ``$PC_CHAR`` 上），没情绪就是说话人名。
    """
    speaker = str(name or "").strip()
    if speaker == PLAYER_NAME:
        title, char_stem = player_defaults()
    else:
        title, char_stem = resolve_speaker_title(speaker, sources), speaker
    return {
        "name": speaker,
        "title": title,
        "char": compose_char(char_stem, normalize_emotion(emotion, speaker_emotions(speaker))),
        "text": text,
    }


def _dialogue_entry_to_agent_text_item(
    entry: Dict[str, Any],
    sources: DialogueSources | None,
) -> Dict[str, str]:
    """
    将草案 dialogue entry 映射到 agent_text.json 的单条：
    { name, title, char, text }
    """
    text = _strip_bracket_expressions(str(entry.get("text") or "").strip())
    return _agent_text_item(entry.get("name"), entry.get("emotion"), text, sources)


def resolve_task_npcs(
    draft: Dict[str, Any],
    npc_name_fallback: str,
) -> tuple[str, str]:
    """解析任务记录的接取 NPC 与完成 NPC（结果已 strip）。

    优先级：草稿的 get_npc / finish_npc → 草稿的 npc_name → 调用方兜底。
    confirm 生成与写盘共用本函数，避免两处 fallback 不一致（跨 NPC 时会说错人）。
    """
    fallback = (npc_name_fallback or "").strip()
    raw_draft_npc = draft.get("npc_name")
    draft_npc = raw_draft_npc.strip() if isinstance(raw_draft_npc, str) else ""

    def _pick(key: str) -> str:
        value = draft.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return draft_npc or fallback

    return _pick("get_npc"), _pick("finish_npc")


def _collect_agent_task_ids(scan_paths: List[Path]) -> set[int]:
    """扫描给定任务文件中 200001–300000 段的现有 id（B2-3：共享 ∪ 本存档）。"""
    ids: set[int] = set()
    for path in scan_paths:
        doc = _read_json(path, default={"tasks": []})
        tasks = doc.get("tasks", []) if isinstance(doc, dict) else []
        if not isinstance(tasks, list):
            continue
        for t in tasks:
            if not isinstance(t, dict):
                continue
            try:
                tid = int(t.get("id"))
            except Exception:
                continue
            if AGENT_TASK_ID_BASE < tid <= AGENT_TASK_ID_MAX:
                ids.add(tid)
    return ids


def _history_agent_task_ids(progress: Any) -> set[int]:
    """存档投影里已完成/已接取的 Agent 任务 id。

    完成任务的任务文件记录可能早已被清理；只扫任务文件会把旧 id 重新分配出去，
    新任务一落盘就被游戏按 tasks_finished 判成“已完成”，永远无法接取。
    """
    ids: set[int] = set()
    if not isinstance(progress, dict):
        return ids
    for key in ("tasks_finished", "tasks_active"):
        values = progress.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            try:
                tid = int(value)
            except (TypeError, ValueError):
                continue
            if AGENT_TASK_ID_BASE < tid <= AGENT_TASK_ID_MAX:
                ids.add(tid)
    return ids


def _next_free_agent_task_id(used: set[int], start: int) -> int:
    """从 start 起按顺序取第一个未使用的 Agent 任务 id；越界抛 ValueError。"""
    tid = max(start, AGENT_TASK_ID_BASE + 1)
    while tid in used:
        tid += 1
    if tid > AGENT_TASK_ID_MAX:
        raise ValueError(
            f"Agent 任务 id 已超出 {AGENT_TASK_ID_BASE + 1}-{AGENT_TASK_ID_MAX} 区间"
            f"（当前应分配 {tid}），拒绝写入越界任务。"
        )
    return tid


def write_confirmed_agent_task_files(
    *,
    draft: Dict[str, Any],
    npc_name_fallback: str,
    game_data: GameDataRegistry,
    dialogue_rewrite: bool = False,
    dialogue_rewrite_instruction: str = "",
    publish_session_id: str = "",
    return_for_reward: bool = False,
    return_description: str = "",
    return_get_dialogue: Optional[List[Dict[str, Any]]] = None,
    return_finish_dialogue: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, int]:
    """
    将已确认的任务草案写入（B2-2 按绑定分流）：
    - bind → data/rag/agent_saves/<slotKey>/agent_tasks.json + agent_text.json
      （文件级隔离：每存档一份完整文件，只含该存档自己发布的任务）；
    - 未 bind → 共享 data/task/agent_tasks.json + data/task/text/agent_text.json（现状）。

    在锁内根据磁盘上已有 agent 任务的最大 ID 分配新 ID（扫描范围＝共享 ∪ 本存档，B2-3，
    避免内存 TaskRegistry 未重载时重复 ID 覆盖旧任务）。

    对话重写标记（阶段四触发一，§4.0.2-1）：模型在发布时判定「这个任务很重要」即挂
    dialogue_rewrite 与它的重写指示，外加发布那场对话的会话 id——三者一起落进任务记录，
    任务完成事件推来时据此自动委派对话修改子 Agent。未标记的任务记录形状与现状逐字一致。

    ``return_for_reward``＝「回接取 NPC 领奖」两段式：任务一把话带到（完成 NPC＝B，
    只给固定 1000 经验），任务二回到接取 NPC（A）领真实奖励。两条任务一次写入、只 bump
    一次版本号；不启用时与单任务路径逐字一致。返回 (写入描述, 任务一的 ID)。
    """
    with _WRITE_LOCK:
        data_root = game_data.data_root
        agent_tasks_path, agent_text_path, _slot_key = resolve_task_write_paths(data_root)

        tasks_doc = _read_json(agent_tasks_path, default={"tasks": []})
        if not isinstance(tasks_doc, dict):
            tasks_doc = {"tasks": []}
        tasks = tasks_doc.get("tasks", [])
        if not isinstance(tasks, list):
            tasks = []

        shared_tasks_path, _shared_text_path = shared_task_paths(data_root)
        scan_paths = [shared_tasks_path]
        if agent_tasks_path != shared_tasks_path:
            scan_paths.append(agent_tasks_path)
        used_ids = _collect_agent_task_ids(scan_paths)
        start_id = max(used_ids, default=AGENT_TASK_ID_BASE) + 1
        # 已绑定存档时，完成/接取历史同样占号：任务文件可能已被清理，但游戏仍按
        # tasks_finished 判重，复用旧 id 会让新任务落盘即“已完成”。
        if _slot_key is not None:
            snapshot = get_integration_state().snapshot()
            if snapshot.get("bound_slot") == _slot_key:
                used_ids |= _history_agent_task_ids(snapshot.get("progress"))
        task_id_int = _next_free_agent_task_id(used_ids, start_id)
        used_ids.add(task_id_int)
        draft["id"] = task_id_int

        get_npc, finish_npc = resolve_task_npcs(draft, npc_name_fallback)

        # -------- 两段式配对的入参防御（不静默降级：降级只发生在 confirm 生成阶段与 id 越界）--------
        paired = bool(return_for_reward)
        paired_get_rows: List[Dict[str, Any]] = []
        paired_finish_rows: List[Dict[str, Any]] = []
        if paired:
            if get_npc == finish_npc:
                raise ValueError(
                    f"两段式配对要求接取 NPC 与完成 NPC 不同（都是「{get_npc}」）。"
                )
            paired_get_rows = [
                r for r in (return_get_dialogue or [])
                if isinstance(r, dict) and str(r.get("text") or "").strip()
            ]
            paired_finish_rows = [
                r for r in (return_finish_dialogue or [])
                if isinstance(r, dict) and str(r.get("text") or "").strip()
            ]
            if not paired_get_rows:
                raise ValueError("两段式配对缺少回访接取对话（return_get_dialogue 为空）。")
            if not paired_finish_rows:
                raise ValueError("两段式配对缺少回访完成对话（return_finish_dialogue 为空）。")

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
        # 基础日常对话数据：写盘行的 Title 从这里查（玩家看到的铭牌取自它）
        sources = shared_sources(Path(data_root))

        if isinstance(get_dialogue, list) and get_dialogue:
            for it in get_dialogue:
                if isinstance(it, dict):
                    get_items.append(_dialogue_entry_to_agent_text_item(it, sources))
        elif get_text_old:
            # 只有旧字段：为当前 get_npc 生成一条
            emo, text_clean = _extract_leading_emotion_from_brackets(get_text_old)
            get_items = [_agent_text_item(get_npc, emo, text_clean, sources)]

        if isinstance(finish_dialogue, list) and finish_dialogue:
            for it in finish_dialogue:
                if isinstance(it, dict):
                    finish_items.append(
                        _dialogue_entry_to_agent_text_item(it, sources)
                    )
        elif finish_text_old:
            emo, text_clean = _extract_leading_emotion_from_brackets(finish_text_old)
            finish_items = [_agent_text_item(finish_npc, emo, text_clean, sources)]

        # 如果 LLM 没给对话数组，但有字符串字段为空，至少要给一个空对话占位
        # 避免游戏端/前端解析空数组出错
        if not get_items:
            get_items = [_agent_text_item(get_npc, "", "", sources)]
        if not finish_items:
            finish_items = [_agent_text_item(finish_npc, "", "", sources)]

        # get_req_list = _ensure_int_list(draft.get("get_requirements"))
        # # 若大模型没填 get_requirements，则由后端根据 finish_requirements 的关卡解锁条件补全。
        # # 该逻辑不依赖任务类型：只要 finish_requirements 中的关卡有 unlock_condition，就应写入前置主线。
        # if not get_req_list:
        #     get_req_list = _stage_reqs_unlock_ids(
        #         finish_requirements=draft.get("finish_requirements"),
        #         game_data=game_data,
        #     )

        # 现在大模型不再传入 finish_requirements
        get_req_list = _stage_reqs_unlock_ids(
            finish_requirements=draft.get("finish_requirements"),
            game_data=game_data,
        )
        # 限制 get_requirements 不超过当前最大任务进度
        get_req_list = [
            min(req_id, MAX_MAINLINE_TASK_PROGRESS) for req_id in get_req_list
        ]
        finish_reqs = _stage_reqs_to_strings(draft.get("finish_requirements"))
        finish_submit = _reward_items_to_expr(draft.get("finish_submit_items"))
        finish_contain = _reward_items_to_expr(draft.get("finish_contain_items"))
        draft_rewards = _reward_items_to_expr(draft.get("rewards"))

        # -------- agent_tasks.json（tasks_doc 已在上方读取）--------
        # 防止重复写入同 id
        tasks = [t for t in tasks if not (isinstance(t, dict) and str(t.get("id")) == str(task_id_int))]

        # id2 越界时按单任务写：真实奖励必须全部挂在唯一那条任务上，绝不留下固定的 1000 经验。
        id_range_downgraded = False
        next_id = 0
        if paired:
            try:
                next_id = _next_free_agent_task_id(used_ids, task_id_int + 1)
            except ValueError:
                id_range_downgraded = True
                paired = False
                logger.warning(
                    "回访任务 id 越界或区间耗尽（起始 %s），本次按单任务写入，奖励取草稿真实奖励",
                    task_id_int + 1,
                )

        if paired:
            record = {
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
                "rewards": _reward_items_to_expr([{"item_name": "经验值", "count": 1000}]),
                "announcement": "",
                "chain": "委托",
                "next_task_id": next_id,
            }
        else:
            record = {
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
                "rewards": draft_rewards,
                "announcement": "",
                "chain": "委托",
            }
        # 委派按任务记录的 get_npc 解析权威发布 NPC：任务二的 get_npc 是 B，标记挂任务二
        # 会在最终领奖时去改 B 的对话；标记挂任务一（A）才对，代价是委派在任务一完成
        # （送达 B）时触发，而不是最终领奖时。
        if dialogue_rewrite:
            record["dialogue_rewrite"] = True
            record["dialogue_rewrite_instruction"] = (dialogue_rewrite_instruction or "").strip()
            record["publish_session_id"] = (publish_session_id or "").strip()
        tasks.append(record)

        return_record: Optional[Dict[str, Any]] = None
        if paired:
            # 任务二只是回去说话领奖：完成条件、提交品、持有品全部留空（全留在任务一）。
            # get_requirements 必须在 clamp 之后单独赋 [任务一 id]——把它送进
            # min(id, 77) 会被压成 77，前置条件直接消失，玩家不做任务一也能接。
            return_record = {
                "id": next_id,
                "title": f"$AGENT_TITLE_{next_id}",
                "description": f"$AGENT_DESCRIPTION_{next_id}",
                "get_requirements": [task_id_int],
                "get_conversation": f"$AGENT_GET_{next_id}",
                "get_npc": finish_npc,
                "finish_requirements": [],
                "finish_submit_items": [],
                "finish_contain_items": [],
                "finish_conversation": f"$AGENT_FINISH_{next_id}",
                "finish_npc": get_npc,
                "rewards": draft_rewards,
                "announcement": "",
                "chain": "委托",
            }
            tasks.append(return_record)

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

        if paired:
            return_title, return_title_renamed = make_unique_task_title(
                f"{title}·回访", game_data=game_data,
            )
            title_auto_renamed = title_auto_renamed or return_title_renamed
            return_description_text = (return_description or "").strip() or (
                f"回到{get_npc}处复命，领取这份委托的报酬。"
            )
            return_get_items = [
                _dialogue_entry_to_agent_text_item(
                    {**it, "name": str(it.get("name") or "").strip() or finish_npc},
                    sources,
                )
                for it in paired_get_rows
            ]
            return_finish_items = [
                _dialogue_entry_to_agent_text_item(
                    {**it, "name": str(it.get("name") or "").strip() or get_npc},
                    sources,
                )
                for it in paired_finish_rows
            ]
            text_doc[f"$AGENT_TITLE_{next_id}"] = return_title
            text_doc[f"$AGENT_DESCRIPTION_{next_id}"] = return_description_text
            text_doc[f"$AGENT_GET_{next_id}"] = return_get_items
            text_doc[f"$AGENT_FINISH_{next_id}"] = return_finish_items

        # -------- 原子写盘：先快照，text → tasks → 单次 bump；任一步失败恢复两个数据文件 --------
        snapshot_text = agent_text_path.read_bytes() if agent_text_path.exists() else None
        snapshot_tasks = agent_tasks_path.read_bytes() if agent_tasks_path.exists() else None

        def _restore(path: Path, snapshot: Optional[bytes]) -> None:
            try:
                if snapshot is None:
                    path.unlink(missing_ok=True)  # type: ignore[arg-type]
                else:
                    _atomic_write_text(path, snapshot.decode("utf-8"))
            except Exception:
                logger.exception("回滚任务文件失败：%s", path)

        try:
            _atomic_write_text(agent_text_path, _dump_json(text_doc))
            _atomic_write_text(agent_tasks_path, _dump_json(new_tasks_doc))
            _bump_task_publish_version(_sync_state_path(data_root))
        except Exception:
            _restore(agent_text_path, snapshot_text)
            _restore(agent_tasks_path, snapshot_tasks)
            raise

        if paired:
            msg = (
                f"任务 {task_id_int} 与回访任务 {next_id} 已写入 agent_tasks.json 与 agent_text.json。"
            )
        elif id_range_downgraded:
            msg = (
                f"任务 {task_id_int} 已写入 agent_tasks.json 与 agent_text.json。"
                f"回访任务因 id 区间不足未创建，奖励已全部挂在本次任务上。"
            )
        else:
            msg = f"任务 {task_id_int} 已写入 agent_tasks.json 与 agent_text.json。"
        if title_auto_renamed:
            msg += f"标题重复，已自动调整为「{title}」。"

    return msg, task_id_int


"""dump 一次真实聊天回合的完整 prompt 输入与 SSE 输出（/api/ask 同链路）。

用法（项目根目录下）：
    ./venv/Scripts/python.exe scripts/dump_chat_prompt.py "Andy Law" "最近有什么活？"

说明：
- 完整复用生产链路：assemble_context（含 Tier-1 RAG 检索）→ TurnOrchestrator → LLM 真实调用；
- 使用临时 memory.db 与 npc_state_db.json 副本，不污染生产数据；
- LLM 连接配置读 .env 全局默认；
- 结果写入 debug/prompt_dump_<NPC>_<时间戳>.md。
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

DEBUG_DIR = PROJECT_ROOT / "debug"
TMP_DIR = PROJECT_ROOT / ".workbuddy" / "tmp_prompt_dump"


class EngineRecorder:
    """包装 RetrievalEngine，记录每轮 retrieve 的输入输出。"""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[dict] = []

    @property
    def ready(self):
        return self._inner.ready

    def retrieve(self, inp):
        out = self._inner.retrieve(inp)
        self.calls.append({"input": inp, "bundle": out})
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _mask_image(content):
    """末条 user 消息里的图片 part 用占位符替换 base64。"""
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                masked = (url[:60] + f"...<base64 共 {len(url)} 字符，已省略>") if url else ""
                out.append({"type": "image_url", "image_url": {"url": masked}})
            else:
                out.append(part)
        return out
    return content


def _fmt_events(events: list[dict]) -> str:
    lines = []
    for ev in events:
        kind, data = ev["event"], ev["data"]
        if kind == "content":
            lines.append(data.get("delta", ""))
        elif kind == "meta":
            lines.append(f"\n[meta] {json.dumps(data, ensure_ascii=False)}\n")
        elif kind in ("system_notice", "tool_status", "agent_status", "error"):
            lines.append(f"\n[{kind}] {json.dumps(data, ensure_ascii=False)}\n")
        elif kind == "done":
            lines.append(f"\n[done] {json.dumps(data, ensure_ascii=False)}\n")
    return "".join(lines)


async def main() -> None:
    npc_name = sys.argv[1] if len(sys.argv) > 1 else "Andy Law"
    query = sys.argv[2] if len(sys.argv) > 2 else "最近有什么活？"

    from core.startup import setup_logging
    setup_logging()

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- NPC 状态（用副本，避免好感度写回生产库）----
    from services.npc.manager import NPCManager

    state_src = Path(__file__).resolve().parents[1]
    from services.game_data.paths import find_resources_directory

    real_state = find_resources_directory() / "data" / "rag" / "npc_state_db.json"
    state_tmp = TMP_DIR / "npc_state_db.json"
    shutil.copy2(real_state, state_tmp)
    npc_manager = await NPCManager.load(state_tmp)

    # ---- 游戏数据 ----
    from services.game_data.registry import init_game_data_registry

    init_game_data_registry()
    game_data = get_game_data_registry = None
    from services.game_data.registry import get_game_data_registry as _gdr

    game_data = _gdr()

    # ---- 检索引擎（真实索引，未命中则重建）----
    from services.retrieval import get_retrieval_engine
    from services.retrieval.loader import compute_corpus_fingerprint, load_corpus

    engine = get_retrieval_engine()
    t0 = time.perf_counter()
    fingerprint = compute_corpus_fingerprint()
    if not engine.try_load(fingerprint):
        print("[engine] 向量索引未命中，开始重建（首次/语料变更后正常）...")
        nodes = load_corpus()
        engine.build_store(nodes, fingerprint)
        print(f"[engine] 重建完成 {len(nodes)} 条，耗时 {time.perf_counter() - t0:.1f}s")
    else:
        print(f"[engine] 向量索引命中，{time.perf_counter() - t0:.2f}s 加载")
    recorder = EngineRecorder(engine)

    # ---- 临时会话存储 ----
    from services.memory.store import MemoryStore

    memory = MemoryStore(db_path=TMP_DIR / "memory_debug.db")

    from services.tools.base import get_tool_registry

    registry = get_tool_registry()

    # ---- LLM 工厂：记录每次真实调用的完整 messages ----
    llm_calls: list[dict] = []

    def llm_factory(cfg):
        from services.llm import LLMClient

        client = LLMClient.for_config(cfg)
        orig_stream = client.chat_stream

        def wrapped_stream(req):
            llm_calls.append({
                "messages": json.loads(json.dumps(req.messages, ensure_ascii=False)),
                "tools": req.tools,
                "purpose": req.purpose,
                "send_image": req.send_image,
            })
            return orig_stream(req)

        client.chat_stream = wrapped_stream
        return client

    from services.orchestrator.turn import OrchestratorDeps, TurnOrchestrator

    deps = OrchestratorDeps(
        memory=memory,
        npc_manager=npc_manager,
        registry=registry,
        game_data=game_data,
        engine=recorder,
        llm_factory=llm_factory,
    )

    session_id = f"debug-prompt-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    orch = TurnOrchestrator(
        session_id=session_id,
        npc_name=npc_name,
        query=query,
        player_identity="",
        progress_stage=1,
        deps=deps,
    )

    events: list[dict] = []
    t1 = time.perf_counter()
    async for ev in orch.run():
        events.append({"event": ev.event, "data": ev.data})
    elapsed = time.perf_counter() - t1

    # ---- dump ----
    npc_slug = npc_name.replace(" ", "")
    out_path = DEBUG_DIR / f"prompt_dump_{npc_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    parts: list[str] = []
    parts.append(f"# 聊天 Prompt 完整 dump：{npc_name}\n")
    parts.append(f"- 时间：{datetime.now().isoformat(timespec='seconds')}")
    parts.append(f"- session_id：`{session_id}`（临时库，不污染生产）")
    parts.append(f"- 玩家发言：`{query}`")
    parts.append(f"- progress_stage：1 / player_identity：（空）")
    parts.append(f"- 回合耗时：{elapsed:.1f}s\n")

    parts.append("## 一、Tier-1 RAG 检索结果（assemble_context）\n")
    if not recorder.calls:
        parts.append("（无检索调用或引擎未就绪）\n")
    for ci, call in enumerate(recorder.calls, 1):
        inp = call["input"]
        bundle = call["bundle"]
        parts.append(f"### 检索输入 #{ci}")
        parts.append(f"- user_query：{inp.user_query}")
        parts.append(f"- npc_name：{inp.npc_name}；titles：{inp.npc_titles}；faction：{inp.npc_faction}")
        parts.append(f"- npc_last_message：{inp.npc_last_message!r}\n")
        parts.append("### 检索输出（各池命中）")
        for pool in ("npc_dialogue", "world_lore", "loading", "npc_task",
                     "supp_intel", "other_npc", "entity_items", "entity_stages"):
            nodes = getattr(bundle, pool)
            if not nodes:
                parts.append(f"- `{pool}`：0 条")
                continue
            parts.append(f"- `{pool}`：{len(nodes)} 条")
            for sn in nodes:
                text = (sn.node.text or "").replace("\n", " ")[:80]
                parts.append(
                    f"    - [{sn.node.type}|{sn.node.source_file or sn.node.character or ''}] "
                    f"dense={sn.dense_score:.4f} fused={sn.fused_score:.4f} | {text}…"
                )
        parts.append("")

    parts.append("## 二、LLM 调用（调用 #1：聊天主 Agent）\n")
    for ci, call in enumerate(llm_calls, 1):
        parts.append(f"### 调用 #{ci}（purpose={call['purpose']}，send_image={call['send_image']}）\n")
        for mi, msg in enumerate(call["messages"], 1):
            role = msg.get("role")
            content = _mask_image(msg.get("content"))
            parts.append(f"#### message[{mi}] role={role}")
            parts.append("```")
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.append(json.dumps(content, ensure_ascii=False, indent=2))
            parts.append("```\n")
        if call["tools"]:
            parts.append("#### tools（随请求携带）")
            parts.append("```json")
            parts.append(json.dumps(call["tools"], ensure_ascii=False, indent=2))
            parts.append("```\n")

    parts.append("## 三、SSE 事件流（真实输出）\n")
    parts.append("```")
    parts.append(_fmt_events(events))
    parts.append("```\n")
    parts.append("## 四、完整事件 JSON\n")
    parts.append("```json")
    parts.append(json.dumps(events, ensure_ascii=False, indent=2))
    parts.append("```\n")

    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"\n=== dump 已写入: {out_path} ===")


if __name__ == "__main__":
    asyncio.run(main())

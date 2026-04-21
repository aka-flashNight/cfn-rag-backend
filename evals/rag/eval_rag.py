"""
Ragas 端到端评估：调用 GameRAGService.ask（非流式）并计算 faithfulness 等指标。

需要：`.env` 中可用的 LLM_API_KEY；本地 bge 嵌入模型（与主项目一致）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import get_settings
from schemas.knowledge_schema import NPCChatRequest
from services.game_rag_service import GameRAGService
from services.llm_client import call_llm
from services.memory_manager import MemoryManager
from services.npc_manager import NPCManager
from services.npc_mood_agent import UPDATE_NPC_MOOD_TOOL, strip_trailing_mood_json


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _git_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "nogit"


def _contexts_from_retrieved(blob: str, max_chunks: int = 12) -> list[str]:
    if not (blob or "").strip():
        return []
    parts = [p.strip() for p in blob.split("\n\n") if p.strip()]
    if not parts:
        return [blob.strip()]
    return parts[:max_chunks]


def _ground_truth_row(row: dict) -> str:
    parts = row.get("expected_answer_contains") or []
    if isinstance(parts, list) and parts:
        return "回答应体现或涉及：" + "、".join(str(p) for p in parts[:5])
    return "回答应符合游戏世界观与角色设定。"


async def _run_one(
    service: GameRAGService,
    row: dict,
    memory: MemoryManager,
    npc_manager: NPCManager,
) -> dict:
    npc = (row.get("npc_name") or row.get("filter_character") or "Andy Law").strip()
    if not npc:
        npc = "Andy Law"
    sess = await memory.create_session(npc, "eval-rag")
    session_id = sess["session_id"]
    q = (row.get("question") or "").strip()
    payload = NPCChatRequest(
        query=q,
        npc_name=npc,
        session_id=session_id,
        agent_enabled=False,
        progress_stage=None,
    )
    ctx = await service._prepare_ask_context(payload, npc_manager, memory)
    contexts = _contexts_from_retrieved(ctx.retrieved_context)
    # 与 _ask_legacy 一致的一次 LLM 调用，避免 ask 内重复检索
    reply_text, _tool_calls = await call_llm(
        api_key=ctx.effective_api_key,
        api_base=ctx.effective_api_base,
        model_name=ctx.effective_model,
        system_prompt=ctx.system_prompt,
        user_prompt=ctx.user_prompt,
        image_path=ctx.image_path,
        image_description=ctx.image_description,
        emotion_hint=ctx.emotion_hint or None,
        tools=[UPDATE_NPC_MOOD_TOOL],
    )
    cleaned, _fb_d, _fb_e = strip_trailing_mood_json(
        reply_text or "", allowed_emotions=ctx.emotions
    )
    answer = ((cleaned or reply_text) or "").strip() or "【无回复】"
    return {
        "question": q,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": _ground_truth_row(row),
    }


async def amain() -> None:
    parser = argparse.ArgumentParser(description="Ragas RAG evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evals/datasets/tiny_golden.jsonl",
    )
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        print(
            "[eval_rag] 缺少依赖，请先: pip install ragas datasets langchain-community langchain-openai",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        sys.exit(1)

    settings = get_settings()
    if not settings.llm_api_key:
        print("[eval_rag] 未配置 LLM_API_KEY，无法运行 Ragas。", file=sys.stderr)
        sys.exit(1)

    ds_path = _ROOT / args.dataset
    if not ds_path.is_file():
        print(f"[eval_rag] 数据集不存在: {ds_path}", file=sys.stderr)
        sys.exit(1)

    rows = _load_jsonl(ds_path)
    if args.sample and args.sample > 0:
        rows = rows[: args.sample]

    from ai_engine.game_data_loader import LOCAL_MODEL_DIR, ensure_embed_model

    ensure_embed_model(offline=True)

    llm = ChatOpenAI(
        model=settings.llm_model_name,
        api_key=settings.llm_api_key,
        base_url=settings.llm_api_base,
        temperature=0.2,
    )
    ragas_llm = LangchainLLMWrapper(llm)
    hf_emb = HuggingFaceEmbeddings(
        model_name=str(LOCAL_MODEL_DIR),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    ragas_emb = LangchainEmbeddingsWrapper(embeddings=hf_emb)

    service = GameRAGService()
    memory = await MemoryManager.create()
    npc_manager = await NPCManager.load()

    samples: list[dict] = []
    for row in rows:
        samples.append(await _run_one(service, row, memory, npc_manager))

    ds = Dataset.from_dict(
        {
            "question": [s["question"] for s in samples],
            "answer": [s["answer"] for s in samples],
            "contexts": [s["contexts"] for s in samples],
            "ground_truth": [s["ground_truth"] for s in samples],
        }
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    result = evaluate(
        ds,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    gh = _git_short()
    out_md = _ROOT / "evals" / "reports" / f"rag_{ts}_{gh}.md"
    out_json = _ROOT / "evals" / "reports" / f"rag_{ts}_{gh}.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    # result 可能是 Dataset 或 dict，兼容处理
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        scores = {c: float(df[c].mean()) for c in df.columns if c not in ("question", "answer", "contexts", "ground_truth")}
        per_row = df.to_dict(orient="records")
    else:
        scores = dict(result) if isinstance(result, dict) else {"raw": str(result)}
        per_row = samples

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {"aggregate": scores, "per_sample": per_row},
            f,
            ensure_ascii=False,
            indent=2,
        )

    lines = [
        "# Ragas 评估报告",
        "",
        f"- 数据集: `{args.dataset}`",
        f"- 样本数: {len(samples)}",
        "",
        "## 聚合指标",
        "",
    ]
    for k, v in sorted(scores.items()):
        lines.append(f"- **{k}**: {v:.4f}")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval_rag] wrote {out_md}")
    print(f"[eval_rag] wrote {out_json}")


def main() -> None:
    import asyncio

    asyncio.run(amain())


if __name__ == "__main__":
    main()

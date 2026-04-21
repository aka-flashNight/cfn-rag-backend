"""
Ragas 端到端评估：复用 ``GameRAGService`` 的检索与 prompt 拼装，但 **生成答案** 使用
``call_llm_eval_text_only``（仅文本），不经过 ``call_llm`` 的立绘多模态与 tools，与线路上下文隔离。

需要：本地 bge 嵌入模型（与主项目一致）。Judge LLM：需 ``LLM_API_KEY``；
本脚本在检测到本地 Ollama 风格 ``LLM_API_BASE`` 且 Key 为空时，会临时使用占位符 ``ollama``
（与正式 ``/ask`` 链路无关；正式对话请在 .env 中显式设置 ``LLM_API_KEY``，例如 ``ollama``）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import get_settings
from schemas.knowledge_schema import NPCChatRequest
from services.game_rag_service import GameRAGService
from services.llm_client import call_llm_eval_text_only
from services.memory_manager import MemoryManager
from services.npc_manager import NPCManager
from services.npc_mood_agent import strip_trailing_mood_json


def _eval_local_ollama_openai_base(api_base: str) -> bool:
    """仅用于本评估脚本：判断是否为本机 Ollama 等 OpenAI 兼容端点（长超时、空 key 占位）。"""
    b = (api_base or "").lower()
    return any(x in b for x in ("11434", "ollama", "localhost", "127.0.0.1"))


# 部分推理模型会在正文里输出思考块；Ragas 的 faithfulness / answer_relevancy 会把它当成答案一部分，导致分数失真或 nan。
_THINK_BLOCKS = re.compile(
    r"<(?:redacted_)?thinking\b[^>]*>[\s\S]*?</(?:redacted_)?thinking\s*>",
    re.IGNORECASE,
)


def _strip_thinking_for_eval(text: str) -> str:
    if not (text or "").strip():
        return (text or "").strip()
    t = _THINK_BLOCKS.sub("", text)
    return t.strip()


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
    resolved_api_key: str,
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
        api_key=resolved_api_key,
    )
    ctx = await service._prepare_ask_context(payload, npc_manager, memory)
    contexts = _contexts_from_retrieved(ctx.retrieved_context)
    # 生成：评估专用纯文本 API，不传立绘、不传 tools（与正式 call_llm 隔离）
    user_eval = ctx.user_prompt
    if ctx.emotion_hint and str(ctx.emotion_hint).strip():
        user_eval = user_eval + "\n\n" + str(ctx.emotion_hint).strip()
    ek = (ctx.effective_api_key or "").strip()
    if not ek:
        raise RuntimeError(
            "eval_rag: 缺少 LLM API Key。请在 .env 设置 LLM_API_KEY（本机 Ollama OpenAI 兼容可设为 ollama）。"
        )
    reply_text = await call_llm_eval_text_only(
        api_key=ek,
        api_base=ctx.effective_api_base,
        model_name=ctx.effective_model,
        system_prompt=ctx.system_prompt,
        user_prompt=user_eval,
    )
    cleaned, _fb_d, _fb_e = strip_trailing_mood_json(
        reply_text or "", allowed_emotions=ctx.emotions
    )
    answer = ((cleaned or reply_text) or "").strip() or "【无回复】"
    answer = _strip_thinking_for_eval(answer) or "【无回复】"
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
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Ragas 单次 LLM 调用超时（秒），慢速兼容端点可调大",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Ragas 并行 worker 数；本地 Ollama 建议 1，云端可适当增大",
    )
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from ragas import RunConfig, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        # answer_relevancy 默认 strictness=3 会并行多路生成；Gemini OpenAI 兼容端点常报
        # “Multiple candidates is not enabled”，故用 strictness=1。
        from ragas.metrics import context_precision, context_recall
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._faithfulness import Faithfulness
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
    api_key = (settings.llm_api_key or "").strip()
    api_base = (settings.llm_api_base or "").strip().lower()
    if not api_key:
        # Ollama OpenAI 兼容接口通常不要求真实密钥，ChatOpenAI 仍需非空字符串
        if _eval_local_ollama_openai_base(api_base):
            api_key = "ollama"
        else:
            print(
                "[eval_rag] 未配置 LLM_API_KEY（非 Ollama 本地端点时必须设置）。",
                file=sys.stderr,
            )
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

    base_norm = (settings.llm_api_base or "").strip().rstrip("/")
    llm_kw: dict = {
        "model": settings.llm_model_name,
        "api_key": api_key,
        "base_url": base_norm or settings.llm_api_base,
        "temperature": 0.2,
    }
    # Ragas 通过 LangChain 调 Ollama 时，默认超时偏短，长推理易失败
    if _eval_local_ollama_openai_base(settings.llm_api_base or ""):
        llm_kw["request_timeout"] = 900
    llm = ChatOpenAI(**llm_kw)
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
        samples.append(await _run_one(service, row, memory, npc_manager, api_key))

    ds = Dataset.from_dict(
        {
            "question": [s["question"] for s in samples],
            "answer": [s["answer"] for s in samples],
            "contexts": [s["contexts"] for s in samples],
            "ground_truth": [s["ground_truth"] for s in samples],
        }
    )

    metrics = [
        Faithfulness(),
        AnswerRelevancy(strictness=1),
        context_precision,
        context_recall,
    ]

    run_cfg = RunConfig(
        timeout=int(args.timeout),
        max_workers=max(1, int(args.workers)),
    )
    result = evaluate(
        ds,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_cfg,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    gh = _git_short()
    out_md = _ROOT / "evals" / "reports" / f"rag_{ts}_{gh}.md"
    out_json = _ROOT / "evals" / "reports" / f"rag_{ts}_{gh}.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    # result 可能是 Dataset 或 dict，兼容处理
    _INPUT_COLS = frozenset({"question", "answer", "contexts", "ground_truth"})
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        scores: dict[str, float] = {}
        for c in df.columns:
            if c in _INPUT_COLS:
                continue
            ser = df[c]
            if pd.api.types.is_numeric_dtype(ser):
                m = ser.mean(skipna=True)
                scores[c] = float(m) if pd.notna(m) else float("nan")
            else:
                num = pd.to_numeric(ser, errors="coerce")
                if num.notna().any():
                    m = num.mean(skipna=True)
                    scores[c] = float(m) if pd.notna(m) else float("nan")
        per_row = df.to_dict(orient="records")
    else:
        scores = dict(result) if isinstance(result, dict) else {"raw": str(result)}
        per_row = samples

    def _json_sanitize(obj: object) -> object:
        if isinstance(obj, dict):
            return {k: _json_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_sanitize(v) for v in obj]
        try:
            if isinstance(obj, float) and (obj != obj):  # NaN
                return None
        except TypeError:
            pass
        return obj

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            _json_sanitize({"aggregate": scores, "per_sample": per_row}),
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
        if isinstance(v, float) and v != v:
            lines.append(f"- **{k}**: （无有效分数）")
        elif isinstance(v, (int, float)):
            lines.append(f"- **{k}**: {v:.4f}")
        else:
            lines.append(f"- **{k}**: {v}")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval_rag] wrote {out_md}")
    print(f"[eval_rag] wrote {out_json}")


def main() -> None:
    import asyncio

    asyncio.run(amain())


if __name__ == "__main__":
    main()

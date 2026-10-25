"""int8 量化档位对比实验：全量 int8（余弦 ~0.964，曾否决）vs MatMul/Gemm int8（余弦 ~0.995，现役）vs fp32。

动机：0.964 与 0.995 的差距在真实游戏检索里到底造成多大排序影响？
方法：三种编码器各自「自洽建库」（生产形态：索引与 query 同模型），对同一批
真实场景查询对比：目标文档排名 / top-10 重合率 / 相似度向量相关性 / 池阈值穿越。

用法：python evals/bench_int8_compare.py   （需 dev 依赖 torch/transformers/onnx）
"""

from __future__ import annotations

import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_HF_DIR = PROJECT_ROOT / "models" / "bge-small-zh-v1.5"
SHIPPED_INT8_DIR = PROJECT_ROOT / "models" / "bge-small-zh-v1.5-onnx-int8"
WORK = PROJECT_ROOT / "models" / "_export_onnx_tmp" / "int8_cmp"

FP32_DIR = WORK / "fp32"
FULL_DIR = WORK / "int8_full"

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]

# 现役各池的 dense 阈值（services/retrieval/config.py）
POOL_THRESHOLDS = {"world_lore": 0.22, "loading": 0.28, "npc_task": 0.28, "supp_intel": 0.30, "other_npc": 0.36, "entity": 0.48, "npc_dialogue": 0.0}


# ---------------------------------------------------------------------------
# 模型准备
# ---------------------------------------------------------------------------

def _ensure_tokenizer(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in TOKENIZER_FILES:
        src = SHIPPED_INT8_DIR / f
        if not (dst / f).exists() and src.exists():
            shutil.copy2(src, dst / f)


def prepare_models() -> None:
    """fp32 导出 + 全量 int8 量化（现役 int8 已在磁盘，跳过）。"""
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel

    _ensure_tokenizer(FP32_DIR)
    _ensure_tokenizer(FULL_DIR)

    if not (FP32_DIR / "model.onnx").exists():
        print("[准备] 导出 fp32 ONNX ...")
        model = AutoModel.from_pretrained(str(SRC_HF_DIR))
        model.eval()

        class _Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask, token_type_ids):
                return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state,

        dummy = (torch.ones(1, 8, dtype=torch.int64), torch.ones(1, 8, dtype=torch.int64), torch.zeros(1, 8, dtype=torch.int64))
        with torch.no_grad():
            torch.onnx.export(
                _Wrap(model), dummy, str(FP32_DIR / "model.onnx"),
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["last_hidden_state"],
                dynamic_axes={k: {0: "batch", 1: "seq"} for k in ("input_ids", "attention_mask", "token_type_ids", "last_hidden_state")},
                opset_version=17, dynamo=False,
            )

    if not (FULL_DIR / "model.onnx").exists():
        print("[准备] 全量 int8 量化（quantize_dynamic 默认参数，即被否决的 0.964 档）...")
        quantize_dynamic(
            model_input=str(FP32_DIR / "model.onnx"),
            model_output=str(FULL_DIR / "model.onnx"),
            weight_type=QuantType.QInt8,
        )


# ---------------------------------------------------------------------------
# 场景（真实语料 + 贴近玩家口吻的捏造 query）
# ---------------------------------------------------------------------------

def _resolve(meta: list[dict], **kw) -> dict | None:
    """按字段/子串在语料里定位目标节点。"""
    for m in meta:
        if kw.get("type") and m["type"] != kw["type"]:
            continue
        if kw.get("item_name") and m.get("item_name") != kw["item_name"]:
            continue
        if kw.get("stage_name") and m.get("stage_name") != kw["stage_name"]:
            continue
        if kw.get("char") and (m.get("character") or "") != kw["char"]:
            continue
        if kw.get("contains") and kw["contains"] not in m["text"]:
            continue
        return m
    return None


def _pool_filter(meta: list[dict], pool: str, npc_char: str | None = None):
    if pool == "entity":
        return [m for m in meta if m["type"] in ("game_item", "game_stage")]
    if pool == "npc_task":
        return [m for m in meta if m["type"] == "task" and (m.get("character") or "") == npc_char]
    if pool == "npc_dialogue":
        return [m for m in meta if m["type"] == "dialogue" and (m.get("character") or "") == npc_char]
    type_map = {"world_lore": {"world_lore"}, "loading": {"loading_lore"},
                "supp_intel": {"supplementary_lore", "intelligence"}, "other_npc": {"dialogue", "task"}}
    return [m for m in meta if m["type"] in type_map[pool]]


def main() -> int:
    prepare_models()

    from services.retrieval.embedder import OnnxEmbedder
    from services.retrieval.loader import load_corpus

    fp32 = OnnxEmbedder(FP32_DIR)
    i8full = OnnxEmbedder(FULL_DIR)
    i8lin = OnnxEmbedder(SHIPPED_INT8_DIR)
    variants = {"fp32": fp32, "int8_full": i8full, "int8_linear": i8lin}

    # ---------- 0. 量化保真度复现（50 句 vs fp32） ----------
    print("=" * 78)
    print("0. 量化保真度（10 句，int8 输出 vs fp32 输出，CLS+L2）")
    probe = ["今天铁匠铺开门吗？", "废城北边的情报说了什么", "这把步枪能装消音器吗",
             "大崩坏之前的世界是什么样的", "有个任务要收集三个猫爪", "摇滚公园的军阀是谁",
             "医疗包在哪可以买到", "雪山的基地怎么打", "克隆人还有活着的吗", "潜行的时候别惊动守卫"]
    v_fp32 = fp32.encode(probe)
    for name in ("int8_full", "int8_linear"):
        v = variants[name].encode(probe)
        cos = np.sum(v * v_fp32, axis=1)
        print(f"   {name:12s} 余弦 mean={cos.mean():.4f} min={cos.min():.4f}")

    # ---------- 1. 语料与自洽索引 ----------
    print("[加载] 语料 ...")
    nodes = load_corpus()
    meta = [{"id": n.id, "text": n.text, "type": n.type, "character": n.character,
             "item_name": n.item_name, "stage_name": n.stage_name} for n in nodes]
    row_of = {m["id"]: i for i, m in enumerate(meta)}

    matrices: dict[str, np.ndarray] = {}
    print("[建库] 三方各自编码全库（自洽形态，与生产一致）...")
    for name, enc in variants.items():
        t0 = time.perf_counter()
        matrices[name] = enc.encode([m["text"] for m in meta])
        print(f"   {name:12s} {time.perf_counter() - t0:5.1f}s")

    def rank_of(name: str, query: str, pool_meta: list[dict], target_id: str) -> tuple[int, float]:
        q = variants[name].encode_query(query)
        rows = [row_of[m["id"]] for m in pool_meta]
        sims = matrices[name][rows] @ q
        target_local = next((k for k, m in enumerate(pool_meta) if m["id"] == target_id), None)
        if target_local is None:
            return -1, 0.0
        rank = int((np.argsort(-sims) == target_local).nonzero()[0][0]) + 1
        return rank, float(sims[target_local])

    # ---------- 2. 场景对比 ----------
    scenarios = [
        # (场景, 查询, 目标节点, 池名, 目标解析)
        # 物品目标按语料中真实存在的名字解析（多候选回退）
        ("物品询问", "上色涂料是干嘛用的，装备上色有什么讲究？", "entity",
         lambda: next((m for nm in ("上色涂料", "精炼石") if (m := _resolve(meta, type="game_item", item_name=nm))), None)),
        ("关卡询问", "诺亚雪山那个秘密基地多少级能去打？", "entity", lambda: _resolve(meta, type="game_stage", stage_name="诺亚雪山部队")),
        ("NPC 往事", "之前你说过有人目睹世界末日像神明的审判，后来呢？", "npc_dialogue", lambda: _resolve(meta, type="dialogue", char="bard", contains="末日审判")),
        ("任务委托", "帮我打听下牛仔帮跟摇滚公园是不是一伙的", "npc_task", lambda: _resolve(meta, type="task", char="酒保", contains="摇滚公园的影响力有限")),
        ("情报检索", "系统日志显示响应时间0.3秒后来TAG没信号是什么情况", "supp_intel", lambda: _resolve(meta, type="intelligence", contains="INACTIVE TAG")),
        ("世界观", "3xf那个巨型企业覆灭之后，剩下的克隆人怎么样了", "world_lore", lambda: _resolve(meta, type="world_lore", contains="巨型企业 3xf")),
        ("loading提示", "废城郊区是不是有盗贼出没？要小心吗", "loading", lambda: _resolve(meta, type="loading_lore", contains="盗贼")),
    ]

    print("=" * 78)
    print("1. 真实场景：目标文档在池内排名（池过滤同生产规则，阈值见括号）")
    header = f"   {'场景':<8} {'池':<13} {'fp32':>10} {'int8_full':>16} {'int8_linear':>16}"
    print(header)
    thr_summary = {v: {} for v in ("fp32", "int8_full", "int8_linear")}
    for name, query, pool, resolver in scenarios:
        target = resolver()
        if target is None:
            print(f"   {name:<8} 目标节点未找到，跳过")
            continue
        pool_meta = _pool_filter(meta, pool, target.get("character") or "")
        thr = POOL_THRESHOLDS[pool]
        cells = []
        for v in ("fp32", "int8_full", "int8_linear"):
            rank, cos = rank_of(v, query, pool_meta, target["id"])
            thr_summary[v][pool] = thr_summary[v].get(pool, []) + [(cos, cos >= thr)]
            mark = "" if cos >= thr else "×不过阈值"
            cells.append(f"#{rank} cos={cos:.3f}{mark}")
        print(f"   {name:<8} {pool:<13} {cells[0]:>10} {cells[1]:>16} {cells[2]:>16}")

    # ---------- 3. 池阈值穿越统计 ----------
    print("=" * 78)
    print("2. 池阈值穿越（目标文档是否仍能过阈值进入池子；×即生产中会被滤掉）")
    for v in ("fp32", "int8_full", "int8_linear"):
        fails = [f"{pool}({cos:.3f}<{thr})" for pool, lst in thr_summary[v].items()
                 for cos, ok in lst if not ok for thr in [POOL_THRESHOLDS[pool]]]
        print(f"   {v:12s} 未过阈值: {fails if fails else '无'}")

    # ---------- 4. 统计性对比（随机 60 条自检索 + 全库相似度向量相关） ----------
    print("=" * 78)
    print("3. 统计对比（60 条随机语料自检索；括号为 int8_full / int8_linear 相对 fp32）")
    id_list = [m["id"] for m in meta]
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(meta), size=60, replace=False)
    overlaps10 = {"int8_full": [], "int8_linear": []}
    for k in sample_idx:
        q_text = meta[int(k)]["text"][:80]
        base = np.argsort(-(matrices["fp32"] @ fp32.encode_query(q_text)))[:10]
        base_ids = {id_list[int(i)] for i in base}
        for v in ("int8_full", "int8_linear"):
            q = variants[v].encode_query(q_text)
            top = np.argsort(-matrices[v] @ q)[:10]
            top_ids = {id_list[int(i)] for i in top}
            overlaps10[v].append(len(base_ids & top_ids) / 10)
    for v in ("int8_full", "int8_linear"):
        o = overlaps10[v]
        print(f"   {v:12s} top-10 重合率: mean={statistics.mean(o):.3f} min={min(o):.1f} 完全一致比例={sum(1 for x in o if x == 1.0) / len(o):.2f}")

    for v in ("int8_full", "int8_linear"):
        rs = []
        for k in sample_idx[::6]:
            q_fp = fp32.encode_query(meta[int(k)]["text"][:80])
            q_v = variants[v].encode_query(meta[int(k)]["text"][:80])
            s_fp = matrices["fp32"] @ q_fp
            s_v = matrices[v] @ q_v
            rs.append(float(np.corrcoef(s_fp, s_v)[0, 1]))
        print(f"   {v:12s} 全库相似度向量 Pearson r（vs fp32）: mean={statistics.mean(rs):.4f}")

    print("=" * 78)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())

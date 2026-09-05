"""把 models/bge-small-zh-v1.5 导出为 ONNX 并动态 int8 量化（开发机一次性执行）。

流程（对应 docs/v3-developer/04-检索与向量模型.md §1.2）：
 1. optimum 导出 fp32 ONNX（task=feature-extraction，输出 last_hidden_state）
 2. onnxruntime.quantization.quant_dynamic（weight_type=QInt8）→ model.onnx（int8）
 3. 拷贝 tokenizer 文件（运行时只需要 tokenizers 库可读的文件）
 4. 校验：随机 50 句，int8 与 fp32 输出（CLS pooling + L2 归一化）余弦相似度均值 ≥ 0.99
    才允许落盘；同时生成 tests/fixtures/embedder_snapshot.json 供 pytest 回归。

用法：
    .\\venv\\Scripts\\Activate.ps1
    python scripts/export_onnx_int8.py

依赖（仅 requirements-dev.txt，不进运行时）：torch(cpu)、transformers、optimum[onnxruntime]、onnxruntime。
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SRC_MODEL_DIR = PROJECT_ROOT / "models" / "bge-small-zh-v1.5"
DST_MODEL_DIR = PROJECT_ROOT / "models" / "bge-small-zh-v1.5-onnx-int8"
WORK_DIR = PROJECT_ROOT / "models" / "_export_onnx_tmp"
SNAPSHOT_PATH = PROJECT_ROOT / "tests" / "fixtures" / "embedder_snapshot.json"

# 运行时 embedder（tokenizers 库）需要的最小文件集
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]

VALIDATION_SENTENCES = [
    "你好，今天天气怎么样？",
    "这把武器伤害很高，适合打副本。",
    "我想发布一个收集材料的任务。",
    "世界曾经经历过大崩坏，文明几乎毁灭。",
    "铁匠铺可以强化装备。",
    "今晚的月色真好，适合出去走走。",
    "任务奖励是金币和一把步枪。",
    "副本任务的推荐等级是多少？",
    "很高兴见到你，冒险者。",
    "情报显示北边出现了新的变异体。",
]

# 运行时一致的池化方式：bge-small-zh-v1.5 的 1_Pooling/config.json 为 CLS pooling
# （pooling_mode_cls_token=true）。手册 04 §1.3 写 mean pooling 与事实不符，以模型配置为准。
MAX_SEQ_LENGTH = 512


def _cls_pool_and_normalize(last_hidden_state, attention_mask) -> list[list[float]]:
    """CLS pooling：取每个序列首个 token 的隐状态，再做 L2 归一化。"""
    import numpy as np

    hidden = np.asarray(last_hidden_state, dtype="float32")
    mask = np.asarray(attention_mask, dtype="int64")
    # CLS token 在 BERT 输入序列的位置 0（padding 在后面），直接取 hidden[:, 0]
    vecs = hidden[:, 0, :]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (vecs / norms).tolist()


def _encode_with_session(session, tokenizer, texts: list[str]) -> list[list[float]]:
    """用 onnxruntime InferenceSession + transformers tokenizer 编码（导出侧校验用）。"""
    import numpy as np

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="np",
    )
    feed = {}
    for inp in session.get_inputs():
        name = inp.name
        if name in enc:
            feed[name] = enc[name].astype(np.int64)
        elif name.endswith("_ids") and name.replace("_ids", "") in enc:
            feed[name] = enc[name.replace("_ids", "")].astype(np.int64)
    if "input_ids" not in feed:
        raise RuntimeError(f"ONNX 模型缺少 input_ids 输入：{[i.name for i in session.get_inputs()]}")
    if "attention_mask" not in feed and "attention_mask" in enc:
        feed["attention_mask"] = enc["attention_mask"].astype(np.int64)
    out_names = [o.name for o in session.get_outputs()]
    target_out = "last_hidden_state" if "last_hidden_state" in out_names else out_names[0]
    result = session.run([target_out], feed)
    return _cls_pool_and_normalize(result[0], enc["attention_mask"])


def _cosine_mean(a: list[list[float]], b: list[list[float]]) -> float:
    import numpy as np

    va = np.asarray(a, dtype="float32")
    vb = np.asarray(b, dtype="float32")
    dots = np.sum(va * vb, axis=1)
    return float(np.mean(dots))  # 两侧均已 L2 归一化，点积即余弦


def main() -> int:
    if not SRC_MODEL_DIR.exists():
        print(f"[导出] 源模型目录不存在: {SRC_MODEL_DIR}")
        print("[导出] 请先运行 python scripts/download_model.py 下载模型。")
        return 1

    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    print(f"[导出] 源模型: {SRC_MODEL_DIR}")
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    # 1. optimum 导出 fp32 ONNX
    print("[导出] 步骤 1/4: 导出 fp32 ONNX（optimum, task=feature-extraction）...")
    fp32_model = ORTModelForFeatureExtraction.from_pretrained(
        str(SRC_MODEL_DIR), export=True
    )
    fp32_dir = WORK_DIR / "fp32"
    fp32_model.save_pretrained(str(fp32_dir))
    fp32_onnx = fp32_dir / "model.onnx"
    if not fp32_onnx.exists():
        print(f"[导出] 失败：未找到导出的 fp32 ONNX: {fp32_onnx}")
        return 1
    print(f"[导出] fp32 ONNX 大小: {fp32_onnx.stat().st_size / 1024 / 1024:.1f} MB")

    # 2. 动态 int8 量化
    print("[导出] 步骤 2/4: 动态 int8 量化（quantize_dynamic, QInt8）...")
    int8_dir = WORK_DIR / "int8"
    int8_dir.mkdir(parents=True)
    int8_onnx = int8_dir / "model.onnx"
    quantize_dynamic(
        model_input=str(fp32_onnx),
        model_output=str(int8_onnx),
        weight_type=QuantType.QInt8,
    )
    print(f"[导出] int8 ONNX 大小: {int8_onnx.stat().st_size / 1024 / 1024:.1f} MB")

    # 3. 校验（余弦相似度 ≥ 0.99 才落盘）
    print("[导出] 步骤 3/4: 校验 int8 与 fp32 余弦相似度（50 句，CLS pooling + L2）...")
    random.seed(20260905)
    extra_sentences = [
        f"随机校验句子{i}: 商店里有药剂、弹夹和材料出售。" for i in range(20)
    ] + [
        f"随机校验句子{i}: 玩家询问关于关卡和副本的情报。" for i in range(20, 50)
    ]
    texts = VALIDATION_SENTENCES + extra_sentences

    tokenizer = AutoTokenizer.from_pretrained(str(SRC_MODEL_DIR))
    fp32_session = ort.InferenceSession(
        str(fp32_onnx), providers=["CPUExecutionProvider"]
    )
    int8_session = ort.InferenceSession(
        str(int8_onnx), providers=["CPUExecutionProvider"]
    )

    fp32_vecs = _encode_with_session(fp32_session, tokenizer, texts)
    int8_vecs = _encode_with_session(int8_session, tokenizer, texts)
    mean_cos = _cosine_mean(fp32_vecs, int8_vecs)
    min_cos = float(
        np.min(np.sum(np.asarray(fp32_vecs) * np.asarray(int8_vecs), axis=1))
    )
    print(f"[导出] 余弦相似度均值: {mean_cos:.6f}（最低 {min_cos:.6f}）")
    if mean_cos < 0.99:
        print("[导出] 校验未达标（要求 ≥0.99），不落盘，保留临时目录供排查。")
        return 1

    # 4. 落盘：int8 模型 + tokenizer
    print("[导出] 步骤 4/4: 落盘到 models/bge-small-zh-v1.5-onnx-int8/")
    if DST_MODEL_DIR.exists():
        shutil.rmtree(DST_MODEL_DIR)
    DST_MODEL_DIR.mkdir(parents=True)
    shutil.copy2(int8_onnx, DST_MODEL_DIR / "model.onnx")
    for fname in TOKENIZER_FILES:
        src = SRC_MODEL_DIR / fname
        if src.exists():
            shutil.copy2(src, DST_MODEL_DIR / fname)

    # 生成测试快照（fp32 参考向量），供 pytest 校验运行时 OnnxEmbedder 与导出侧一致
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "model": "bge-small-zh-v1.5-onnx-int8",
        "pooling": "cls",
        "note": "fp32 导出侧参考向量（CLS pooling + L2 归一化），运行时 int8 输出与之余弦应 ≥0.99",
        "texts": texts[: len(VALIDATION_SENTENCES)],
        "vectors_fp32": fp32_vecs[: len(VALIDATION_SENTENCES)],
        "vectors_int8": int8_vecs[: len(VALIDATION_SENTENCES)],
        "mean_cosine": mean_cos,
    }
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[导出] 测试快照已写入: {SNAPSHOT_PATH}")

    shutil.rmtree(WORK_DIR)
    total_mb = sum(f.stat().st_size for f in DST_MODEL_DIR.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"[导出] 完成 ✓ 目录总大小 {total_mb:.1f} MB: {DST_MODEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

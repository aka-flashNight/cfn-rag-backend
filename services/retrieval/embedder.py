"""OnnxEmbedder：tokenizers.Tokenizer + onnxruntime.InferenceSession 的本地嵌入器。

对应 docs/v3-developer/04-检索与向量模型.md §1.3。模型不换（bge-small-zh-v1.5），
推理链路换 ONNX Runtime + int8（torch/transformers 永不进运行时依赖，修 F1）。

注意：bge-small-zh-v1.5 的 1_Pooling/config.json 为 CLS pooling
（pooling_mode_cls_token=true），手册 04 §1.3 写 mean pooling 与事实不符，
以模型配置为准：CLS 向量 + L2 归一化，输出 float32[512]。
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 与 sentence_bert_config.json 的 max_seq_length 一致
MAX_SEQ_LENGTH = 512

DEFAULT_MODEL_SUBDIR = "models/bge-small-zh-v1.5-onnx-int8"


def get_model_dir() -> Path:
    """定位 int8 ONNX 模型目录（开发环境 / PyInstaller 打包环境通用）。"""
    if getattr(sys, "frozen", False):
        base_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        # services/retrieval/embedder.py → 项目根
        base_dir = Path(__file__).resolve().parent.parent.parent
    return base_dir / DEFAULT_MODEL_SUBDIR


class OnnxEmbedder:
    """会话全局单例；启动时预载并预热一句空文本。请求级缓存避免同回合重复编码（修 C1）。"""

    def __init__(self, model_dir: Path | None = None) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir) if model_dir else get_model_dir()
        model_path = self.model_dir / "model.onnx"
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not model_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError(
                f"ONNX 嵌入模型不完整: {self.model_dir}\n"
                "请先运行 python scripts/export_onnx_int8.py 导出 int8 模型。"
            )

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        self._tokenizer.no_padding()  # padding 在 _forward 前手工按批内最长做，避免整批被长文档拖到 512

        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_names = [i.name for i in self._session.get_inputs()]
        out_shape = self._session.get_outputs()[0].shape
        self.dim = int(out_shape[-1]) if isinstance(out_shape[-1], int) else 512

        self._cache: dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 编码
    # ------------------------------------------------------------------

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """批量编码 → float32[N, dim]，L2 归一化。顺序与输入严格一致。

        缺失项按 token 长度降序分批（长度相近的文本同批，padding 浪费最小化），
        结果按原顺序回填；顺序稳定性由 pytest 保证。
        """
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        cache_hits: dict[int, np.ndarray] = {}
        misses: list[int] = []
        with self._cache_lock:
            for i, t in enumerate(texts):
                cached = self._cache.get(t)
                if cached is not None:
                    cache_hits[i] = cached
                else:
                    misses.append(i)
            # 请求级缓存上限（防长会话累积）
            if len(self._cache) > 256:
                self._cache.clear()
        for i, v in cache_hits.items():
            vecs[i] = v

        if not misses:
            return vecs

        encodings = self._tokenizer.encode_batch([texts[i] for i in misses])
        order = sorted(range(len(misses)), key=lambda k: len(encodings[k].ids), reverse=True)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            batch_enc = [encodings[k] for k in batch]
            max_len = max(len(e.ids) for e in batch_enc)
            n = len(batch_enc)
            input_ids = np.zeros((n, max_len), dtype=np.int64)
            attention_mask = np.zeros((n, max_len), dtype=np.int64)
            for r, e in enumerate(batch_enc):
                ids = e.ids
                input_ids[r, : len(ids)] = ids
                attention_mask[r, : len(ids)] = e.attention_mask
            result = self._forward(input_ids, attention_mask)
            for r, k in enumerate(batch):
                vecs[misses[k]] = result[r]
                with self._cache_lock:
                    self._cache[texts[misses[k]]] = result[r]
        return vecs

    def encode_query(self, text: str) -> np.ndarray:
        """单条查询编码 → float32[dim]。"""
        return self.encode([text])[0]

    def clear_cache(self) -> None:
        """清空请求级缓存（编排器可在回合开始时调用）。"""
        with self._cache_lock:
            self._cache.clear()

    def _forward(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """ONNX 前向 → CLS pooling + L2 归一化（与模型 1_Pooling 配置一致）。"""
        feed: dict[str, np.ndarray] = {}
        for name in self._input_names:
            if name == "input_ids":
                feed[name] = input_ids
            elif name == "attention_mask":
                feed[name] = attention_mask
            elif name == "token_type_ids":
                feed[name] = np.zeros_like(input_ids)

        output_name = self._session.get_outputs()[0].name
        result = self._session.run([output_name], feed)[0]
        vecs = np.asarray(result, dtype=np.float32)[:, 0, :]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)

    def warmup(self) -> None:
        """预热一句空文本，触发权重/DLL 加载。"""
        self.encode([""])


_DEFAULT_EMBEDDER: OnnxEmbedder | None = None
_EMBEDDER_LOCK = threading.Lock()


def get_default_embedder() -> OnnxEmbedder:
    """进程级单例 embedder（startup 预载；测试可 set_default_embedder 注入）。"""
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is not None:
        return _DEFAULT_EMBEDDER
    with _EMBEDDER_LOCK:
        if _DEFAULT_EMBEDDER is None:
            _DEFAULT_EMBEDDER = OnnxEmbedder()
        return _DEFAULT_EMBEDDER


def set_default_embedder(embedder: OnnxEmbedder | None) -> None:
    global _DEFAULT_EMBEDDER
    with _EMBEDDER_LOCK:
        _DEFAULT_EMBEDDER = embedder

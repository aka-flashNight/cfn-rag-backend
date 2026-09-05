"""二进制向量存储：vectors.npy(fp16) + meta.json + fingerprint.json。

对应 docs/v3-developer/04-检索与向量模型.md §2。替代 LlamaIndex JSON 向量库
（50MB → ~6.5MB），指纹校验不一致才重建（替代旧「手动删目录」）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from services.game_data.paths import find_resources_directory

logger = logging.getLogger(__name__)

INDEX_DIR_NAME = "vector_index_v3"
VECTORS_FILE = "vectors.npy"
META_FILE = "meta.json"
FINGERPRINT_FILE = "fingerprint.json"


class IndexStaleError(RuntimeError):
    """指纹不匹配 / 索引文件缺失或损坏，需要重建。"""


@dataclass
class Node:
    """检索语料节点（与 matrix 行对齐）。type 为 metadata 白名单键。"""

    id: str
    text: str
    type: str
    character: str | None = None
    source_file: str | None = None
    task_source: str | None = None  # "guide" 或 None
    region: str | None = None
    unlock: str | None = None
    item_name: str | None = None
    stage_area: str | None = None
    stage_name: str | None = None
    entity_key: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def to_meta(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_meta(cls, raw: dict[str, object]) -> "Node":
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        extra = {k: str(v) for k, v in raw.items() if k not in known and k != "id" and k != "text"}
        return cls(
            id=str(raw.get("id") or ""),
            text=str(raw.get("text") or ""),
            type=str(raw.get("type") or ""),
            character=(str(raw["character"]) if raw.get("character") else None),
            source_file=(str(raw["source_file"]) if raw.get("source_file") else None),
            task_source=(str(raw["task_source"]) if raw.get("task_source") else None),
            region=(str(raw["region"]) if raw.get("region") else None),
            unlock=(str(raw["unlock"]) if raw.get("unlock") else None),
            item_name=(str(raw["item_name"]) if raw.get("item_name") else None),
            stage_area=(str(raw["stage_area"]) if raw.get("stage_area") else None),
            stage_name=(str(raw["stage_name"]) if raw.get("stage_name") else None),
            entity_key=(str(raw["entity_key"]) if raw.get("entity_key") else None),
            extra=extra,
        )


def get_index_dir() -> Path:
    """向量库落盘目录：<resources>/tools/vector_index_v3。"""
    index_dir = find_resources_directory() / "tools" / INDEX_DIR_NAME
    index_dir.mkdir(parents=True, exist_ok=True)
    return index_dir


def corpus_fingerprint(file_states: list[tuple[str, int, int]]) -> str:
    """由 (相对路径, mtime_ns, size) 集合计算语料指纹。"""
    payload = json.dumps(sorted(file_states), ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class VectorStore:
    """matrix 与 nodes 行对齐；内存 float32（磁盘 float16）。"""

    matrix: np.ndarray  # [N, dim] float32
    nodes: list[Node]
    dim: int
    fingerprint: str

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, index_dir: Path, expected_fingerprint: str) -> "VectorStore":
        """存在且指纹匹配 → 加载（<1s）；否则抛 IndexStaleError 由调用方重建。"""
        fp_path = index_dir / FINGERPRINT_FILE
        vec_path = index_dir / VECTORS_FILE
        meta_path = index_dir / META_FILE
        if not (fp_path.exists() and vec_path.exists() and meta_path.exists()):
            raise IndexStaleError(f"索引文件缺失: {index_dir}")

        try:
            with fp_path.open("r", encoding="utf-8") as f:
                fp = json.load(f)
            if fp.get("fingerprint") != expected_fingerprint:
                raise IndexStaleError("语料指纹不匹配，需要重建索引")
            # 存在即 mmap 加载（校验通过后转内存 float32）
            matrix16 = np.load(vec_path, mmap_mode="r")
            if matrix16.ndim != 2:
                raise IndexStaleError("vectors.npy 形状异常")
            with meta_path.open("r", encoding="utf-8") as f:
                raw_nodes = json.load(f)
            if not isinstance(raw_nodes, list) or len(raw_nodes) != matrix16.shape[0]:
                raise IndexStaleError("meta.json 与向量矩阵行数不一致")
            nodes = [Node.from_meta(r) for r in raw_nodes]
        except IndexStaleError:
            raise
        except Exception as exc:  # npy 损坏 / JSON 损坏等一律触发重建
            raise IndexStaleError(f"索引文件损坏: {exc}") from exc

        logger.info("向量库加载完成: %d 条, dim=%d", len(nodes), matrix16.shape[1])
        return cls(
            matrix=np.asarray(matrix16, dtype=np.float32),
            nodes=nodes,
            dim=int(matrix16.shape[1]),
            fingerprint=expected_fingerprint,
        )

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        nodes: list[Node],
        embedder,  # services.retrieval.embedder.OnnxEmbedder（鸭子类型，便于测试注入）
        expected_fingerprint: str,
        batch_size: int = 64,
    ) -> "VectorStore":
        """全量构建（4426 条 ONNX int8 预计 ≤25s）。"""
        if not nodes:
            raise ValueError("没有可用语料节点，无法构建向量库")
        texts = [n.text for n in nodes]
        vecs = embedder.encode(texts, batch_size=batch_size)
        logger.info("向量库构建完成: %d 条, dim=%d", len(nodes), vecs.shape[1])
        return cls(
            matrix=vecs.astype(np.float16).astype(np.float32),
            nodes=nodes,
            dim=int(vecs.shape[1]),
            fingerprint=expected_fingerprint,
        )

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, index_dir: Path) -> None:
        """fp16 落盘（4426×512 ≈ 4.5MB）；tmp+replace 原子写。"""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        vec_path = index_dir / VECTORS_FILE
        meta_path = index_dir / META_FILE
        fp_path = index_dir / FINGERPRINT_FILE

        matrix16 = self.matrix.astype(np.float16)
        _atomic_save_npy(vec_path, matrix16)
        _atomic_write_json(meta_path, [n.to_meta() for n in self.nodes])
        _atomic_write_json(
            fp_path,
            {
                "fingerprint": self.fingerprint,
                "count": len(self.nodes),
                "dim": self.dim,
                "format": "vector_index_v3",
            },
        )
        logger.info("向量库已保存: %s（%d 条）", index_dir, len(self.nodes))


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        np.save(tmp_name, arr)  # np.save 会补 .npy 后缀
        os.replace(tmp_name + ".npy", path)
    except Exception:
        _silent_remove(tmp_name + ".npy")
        _silent_remove(tmp_name)
        raise


def _atomic_write_json(path: Path, payload: object) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp_name, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_name, path)
    except Exception:
        _silent_remove(tmp_name)
        raise


def _silent_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass

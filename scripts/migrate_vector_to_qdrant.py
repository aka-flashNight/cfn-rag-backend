#!/usr/bin/env python
"""
LlamaIndex 本地索引 → Qdrant 向量库迁移脚本。

用法::

    python scripts/migrate_vector_to_qdrant.py \
        --qdrant-url http://localhost:6333 \
        --collection cfn_game \
        --batch-size 100

功能：
1. 从 ``resources/tools/vector_index/`` 读取所有节点
2. 对每个节点生成 embedding
3. 批量 upsert 到 Qdrant collection
"""

from __future__ import annotations

import asyncio
import argparse
import os
import sys
import time
from pathlib import Path


# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="LlamaIndex 本地索引 → Qdrant 迁移")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--collection", default="cfn_game")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--dry-run", action="store_true", help="仅打印节点数，不上传")
    return p.parse_args()


async def main():
    args = parse_args()

    # 设置离线模式（避免从 HuggingFace 下载）
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print("[迁移] 加载嵌入模型...")
    from ai_engine.game_data_loader import ensure_embed_model

    ensure_embed_model(offline=True)

    print("[迁移] 加载本地向量索引...")
    from ai_engine.game_data_loader import get_cached_index, iter_docstore_nodes

    index = get_cached_index()
    nodes = list(iter_docstore_nodes(index))
    print(f"[迁移] 节点总数: {len(nodes)}")

    if args.dry_run:
        print("[迁移] --dry-run，跳过上传。")
        return

    print(f"[迁移] 连接 Qdrant: {args.qdrant_url}")
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = AsyncQdrantClient(url=args.qdrant_url)

    # 创建 collection（如不存在）
    exists = await client.collection_exists(args.collection)
    if not exists:
        await client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"[迁移] 创建 collection: {args.collection}")
    else:
        print(f"[迁移] Collection 已存在: {args.collection}")

    # 批量获取 embedding 并上传
    from llama_index.core import Settings

    embed_model = Settings.embed_model

    total = len(nodes)
    batch_size = args.batch_size
    uploaded = 0
    t0 = time.monotonic()

    for i in range(0, total, batch_size):
        batch = nodes[i : i + batch_size]
        texts = [n.get_content(metadata_mode="none") for n in batch]

        # embedding（离线模式，直接算）
        embeddings = embed_model.get_text_embedding_batch(texts)

        # upsert points
        from qdrant_client.models import PointStruct

        points = []
        for j, node in enumerate(batch):
            meta = dict(node.metadata or {})
            meta["_node_content"] = node.get_content(metadata_mode="none")
            meta["_node_id"] = node.node_id

            points.append(
                PointStruct(
                    id=node.node_id if node.node_id else f"{i + j}",
                    vector=embeddings[j],
                    payload=meta,
                )
            )

        await client.upsert(
            collection_name=args.collection,
            points=points,
            wait=True,
        )
        uploaded += len(batch)
        elapsed = time.monotonic() - t0
        pct = uploaded / total * 100
        speed = uploaded / elapsed if elapsed > 0 else 0
        print(
            f"\r[迁移] {uploaded}/{total} ({pct:.1f}%) — {speed:.1f} nodes/s",
            end="",
            flush=True,
        )

    print(f"\n[迁移] 完成 ✓ — 共 {uploaded} 个节点，耗时 {time.monotonic() - t0:.1f}s")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())

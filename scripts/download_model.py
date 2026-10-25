#!/usr/bin/env python3
"""下载 HuggingFace 模型到本地项目目录（开发机一次性执行，导出 ONNX int8 的前置）。

用法:
    python scripts/download_model.py [--proxy http://127.0.0.1:10809] [--mirror] [--modelscope]

v3 说明：本脚本只在开发机下载 bge-small-zh-v1.5 原始 HF 权重（供
scripts/export_onnx_int8.py 转换 int8 ONNX）；运行时只依赖
models/bge-small-zh-v1.5-onnx-int8/，torch/transformers 不进运行时依赖。
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "bge-small-zh-v1.5"
MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODELSCOPE_ID = "AI-ModelScope/bge-small-zh-v1.5"


def _model_valid() -> bool:
    has_config = (MODEL_DIR / "config.json").exists()
    has_model = any(
        (MODEL_DIR / f).exists() for f in ("model.safetensors", "pytorch_model.bin")
    )
    return has_config and has_model


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 HuggingFace 模型到本地")
    parser.add_argument("--proxy", type=str, default=None, help="代理地址，例如 http://127.0.0.1:10809")
    parser.add_argument("--mirror", action="store_true", help="使用 HuggingFace 镜像站 (hf-mirror.com)，无需代理")
    parser.add_argument("--modelscope", action="store_true", help="使用 ModelScope 国内镜像（推荐国内用户使用）")
    args = parser.parse_args()

    if MODEL_DIR.exists() and _model_valid():
        print(f"[提示] 本地模型已存在且完整: {MODEL_DIR}")
        print("无需重复下载。")
        return

    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[配置] 使用 HuggingFace 镜像站: https://hf-mirror.com")
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        print(f"[配置] 使用代理: {args.proxy}")

    try:
        if args.modelscope:
            from modelscope import snapshot_download

            print(f"[下载] 使用 ModelScope 镜像: {MODELSCOPE_ID}")
            cache_dir = snapshot_download(MODELSCOPE_ID)
            import shutil

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            for item in Path(cache_dir).iterdir():
                dest = MODEL_DIR / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            from transformers import AutoModel, AutoTokenizer

            print(f"[下载] 正在下载模型: {MODEL_ID}")
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            AutoModel.from_pretrained(MODEL_ID).save_pretrained(str(MODEL_DIR))
            AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(str(MODEL_DIR))
    except Exception as e:
        print(f"\n[错误] 下载失败: {e}")
        print("\n可能的原因:")
        print("1. 网络连接问题 - 请检查是否能访问 HuggingFace / ModelScope")
        print("2. 需要代理 - 请使用 --proxy 参数指定代理地址")
        print("3. 磁盘空间不足 - 模型需要约 100MB 空间")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print(f"\n[成功] 模型下载完成！保存位置: {MODEL_DIR}")
    print("下一步：python scripts/export_onnx_int8.py 导出运行时 int8 模型。")


if __name__ == "__main__":
    main()

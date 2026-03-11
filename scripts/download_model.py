#!/usr/bin/env python3
"""
下载 HuggingFace 模型到本地项目目录
用法: python scripts/download_model.py [--proxy http://127.0.0.1:10809] [--mirror]
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    # 确定项目根目录（脚本所在目录的父目录）
    project_root = Path(__file__).resolve().parent.parent
    
    # 🔧 修复：将项目根目录添加到 Python 路径
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    parser = argparse.ArgumentParser(description="下载 HuggingFace 模型到本地")
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="代理地址，例如 http://127.0.0.1:10809"
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="使用 HuggingFace 镜像站 (hf-mirror.com)，无需代理"
    )
    parser.add_argument(
        "--modelscope",
        action="store_true",
        help="使用 ModelScope 国内镜像（推荐国内用户使用）"
    )
    args = parser.parse_args()

    # 检查模型是否已存在
    models_dir = project_root / "models" / "bge-small-zh-v1.5"
    required_files = ["config.json", "pytorch_model.bin"]
    is_valid = all((models_dir / f).exists() for f in required_files)

    if models_dir.exists() and is_valid:
        print(f"[提示] 本地模型已存在且完整: {models_dir}")
        print("无需重复下载，可直接运行程序。")
        return

    # 设置镜像站
    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[配置] 使用 HuggingFace 镜像站: https://hf-mirror.com")

    # 设置代理
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        print(f"[配置] 使用代理: {args.proxy}")

    # 导入并执行下载
    try:
        from ai_engine.game_data_loader import download_model_to_local, LOCAL_MODEL_DIR
        download_model_to_local(use_modelscope=args.modelscope)
        print("\n[成功] 模型下载完成！")
        print(f"模型保存位置: {LOCAL_MODEL_DIR}")
        print("现在可以直接运行 CFN-RAG.exe，无需联网下载模型。")
    except Exception as e:
        print(f"\n[错误] 下载失败: {e}")
        print("\n可能的原因:")
        print("1. 网络连接问题 - 请检查是否能访问 HuggingFace")
        print("2. 需要代理 - 请使用 --proxy 参数指定代理地址")
        print("3. 磁盘空间不足 - 模型需要约 100MB 空间")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

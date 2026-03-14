"""
立绘裁剪与压缩，供传给 AI 前使用。仅依赖 Pillow，可被轻量脚本单独调用。
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

from PIL import Image

# 立绘有效区域（绝大部分立绘不占满 4096*2304）：(left, top, right, bottom) 像素
PORTRAIT_CROP_LEFT = 200
PORTRAIT_CROP_TOP = 100
PORTRAIT_CROP_RIGHT = 3500
PORTRAIT_CROP_BOTTOM = 1700
# 传给 AI 前缩放比例（长宽各除以 2，减少 token）
PORTRAIT_SCALE_DENOM = 2
# WebP 质量（0-100），兼顾清晰度与体积
PORTRAIT_WEBP_QUALITY = 85


def prepare_portrait_for_ai(
    image_path: Path,
    output_path: Path | None = None,
) -> Tuple[bytes, str]:
    """
    对立绘图像做裁剪与缩放，得到适合传给 AI 的 WebP 字节与 MIME 类型。

    - 裁剪：优先尝试按图像实际占用像素区域（getbbox，通常很快），
      若无有效 bbox 或图像无透明通道则使用固定区域 (200,100)-(3500,1700)，并钳位到图像尺寸。
    - 缩放：裁剪后长宽各除以 PORTRAIT_SCALE_DENOM（默认 2）。
    - 输出：编码为 WebP，返回 (bytes, "image/webp")；若提供 output_path 则同时写入该文件。

    兼容 PNG/WebP 输入；输出统一为 WebP 以减小 Base64 体积。
    """
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    # 1. 裁剪区域：优先内容框，否则固定区域并钳位
    bbox = img.getbbox()
    if bbox and (bbox[2] > bbox[0] and bbox[3] > bbox[1]):
        left, top, right, bottom = bbox
        # 可选：稍微留一点边距，避免贴边
        margin = min(20, (right - left) // 20, (bottom - top) // 20)
        left = max(0, left - margin)
        top = max(0, top - margin)
        right = min(w, right + margin)
        bottom = min(h, bottom + margin)
    else:
        left = max(0, min(PORTRAIT_CROP_LEFT, w - 1))
        top = max(0, min(PORTRAIT_CROP_TOP, h - 1))
        right = max(left + 1, min(PORTRAIT_CROP_RIGHT, w))
        bottom = max(top + 1, min(PORTRAIT_CROP_BOTTOM, h))

    img = img.crop((left, top, right, bottom))

    # 2. 缩放：长宽各 /2
    nw, nh = max(1, img.width // PORTRAIT_SCALE_DENOM), max(1, img.height // PORTRAIT_SCALE_DENOM)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)

    # 3. 编码为 WebP
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=PORTRAIT_WEBP_QUALITY)
    buf.seek(0)
    data = buf.read()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)

    return data, "image/webp"

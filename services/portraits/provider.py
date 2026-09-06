"""立绘图片处理与供给（对应 docs/v3-developer/07 §3/§6）。

- 多模态模型用：manifest 查表 → bounds 裁剪（人物本体）→ 长边 ≤512 → WebP q80
  → base64 data URL（OpenAI image_url 格式；图片 part 放在文本 part 之后）。
- 独立前端展示用（assets_api）：manifest 查表 → bounds 裁剪 → 原分辨率 PNG。
- 缓存 key = (npc_key, 命中情绪, 源文件 mtime)（cache.py），修 F2「每请求重新 PIL 处理」。
- 主角（heroKeys）/查不到角色/文件缺失/任何处理异常 → 返回 None（无图模式，不报错）。
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

from services.portraits.cache import get_cache, portrait_cache_key
from services.portraits.manifest_lookup import (
    DEFAULT_EXPRESSION,
    DialoguePortraitLookup,
    get_portrait_lookup,
)

logger = logging.getLogger(__name__)

MAX_LONG_EDGE = 480  # 长边上限（≈480p，控制多模态 token 消耗与传输体积，07 §3）
WEBP_QUALITY = 80

# OpenAI 兼容多模态消息中的图片 part（图片 part 在文本 part 之后，02 §3.3）
_WEBP_CACHE_PREFIX = "webp:"
_PNG_CACHE_PREFIX = "png:"


def is_portrait_available() -> bool:
    """manifest 是否可用（不可用 = 无图模式）。"""
    return get_portrait_lookup() is not None


def _resolve_asset(
    lookup: DialoguePortraitLookup, npc_name: str, emotion: str
) -> Optional[dict]:
    """查表 + 主角/缺失处理；返回协议 resolve 的 dict 形态，否则 None。"""
    result = lookup.resolve(npc_name, emotion or DEFAULT_EXPRESSION)
    if not isinstance(result, dict):  # "hero"（主角无静态图）或 None（查不到角色）
        return None
    return result


def _load_processed(
    npc_name: str, emotion: str, *, prefix: str, resize: bool
) -> Optional[tuple[bytes, str]]:
    """查表 → 读源文件 →（缓存命中直接返回）→ bounds 裁剪 [→ 缩放] → 编码。

    返回 (图片字节, mime)；任何失败返回 None（不抛错，无图模式兜底）。
    """
    lookup = get_portrait_lookup()
    if lookup is None:
        return None
    try:
        asset = _resolve_asset(lookup, npc_name, emotion)
        if asset is None:
            return None
        png_path = asset["png_path"]
        mtime = png_path.stat().st_mtime
    except OSError as exc:
        logger.warning("立绘源文件读取失败（无图兜底）: %s", exc)
        return None

    cache = get_cache()
    key = (prefix, *portrait_cache_key(asset["key"], asset["expression_used"], mtime))
    hit = cache.get(key)
    if hit is not None:
        return hit

    try:
        processed = _process_image(asset, resize=resize)
    except Exception as exc:
        logger.warning("立绘处理失败（无图兜底）: %s", exc)
        return None
    if processed is None:
        return None
    cache.put(key, processed)
    return processed


def _process_image(asset: dict, *, resize: bool) -> Optional[tuple[bytes, str]]:
    """PIL 处理：bounds 裁剪（两类 source 通用）→ [长边 ≤512] → WebP q80。"""
    from PIL import Image

    bounds = asset.get("bounds")
    with Image.open(asset["png_path"]) as im:
        im = im.convert("RGBA")
        if bounds:
            canvas_w, canvas_h = im.size
            x0 = max(0, int(bounds.get("x") or 0))
            y0 = max(0, int(bounds.get("y") or 0))
            x1 = min(canvas_w, x0 + max(0, int(bounds.get("width") or 0)))
            y1 = min(canvas_h, y0 + max(0, int(bounds.get("height") or 0)))
            if x1 > x0 and y1 > y0:
                im = im.crop((x0, y0, x1, y1))
        if resize:
            width, height = im.size
            long_edge = max(width, height)
            if long_edge > MAX_LONG_EDGE:
                scale = MAX_LONG_EDGE / long_edge
                im = im.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    Image.LANCZOS,
                )
        buf = io.BytesIO()
        im.save(buf, "WEBP", quality=WEBP_QUALITY)
    return buf.getvalue(), "image/webp"


def get_portrait_data_url(npc_name: str, emotion: str = DEFAULT_EXPRESSION) -> Optional[str]:
    """多模态模型用：裁剪+缩放后的 WebP base64 data URL；无图返回 None。"""
    result = _load_processed(npc_name, emotion, prefix=_WEBP_CACHE_PREFIX, resize=True)
    if result is None:
        return None
    data, mime = result
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def get_portrait_png(npc_name: str, emotion: str = DEFAULT_EXPRESSION) -> Optional[bytes]:
    """assets 展示接口用：bounds 裁剪后的原分辨率 PNG（重编码，去画布杂质）。"""
    result = _load_processed(npc_name, emotion, prefix=_PNG_CACHE_PREFIX, resize=False)
    if result is None:
        return None

    # WebP bytes → PNG bytes（缓存层存统一形态；PNG 转换廉价且展示接口低频）
    from PIL import Image

    data, _mime = result
    try:
        with Image.open(io.BytesIO(data)) as im:
            out = io.BytesIO()
            im.save(out, "PNG")
            return out.getvalue()
    except Exception as exc:
        logger.warning("立绘 PNG 转码失败（无图兜底）: %s", exc)
        return None


def build_image_message_content(text: str, image_url: Optional[str]) -> str | list[dict]:
    """把可选图片拼进 user 消息 content：有图 [文本 part, 图片 part]，无图纯文本。"""
    if not image_url:
        return text
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]

"""立绘与多模态（docs/v3-developer/07-立绘与多模态.md）。

- manifest_lookup：游戏项目对话立绘 manifest 查表（协议权威见游戏项目 docs）；
- cache：进程内 LRU（key=(npc_key, 命中情绪, 源文件 mtime)）；
- provider：bounds 裁剪 + 长边 ≤512 + WebP q80 → base64 data URL（模型用）；
  bounds 裁剪 PNG（assets 展示用）。

manifest 缺失/损坏 → 无图模式（查表器返回 None），聊天主功能不受影响。
"""

from services.portraits.manifest_lookup import (
    DialoguePortraitLookup,
    get_portrait_lookup,
    reset_portrait_lookup,
)
from services.portraits.provider import (
    build_image_message_content,
    get_portrait_data_url,
    get_portrait_png,
    is_portrait_available,
)

__all__ = [
    "DialoguePortraitLookup",
    "build_image_message_content",
    "get_portrait_data_url",
    "get_portrait_lookup",
    "get_portrait_png",
    "is_portrait_available",
    "reset_portrait_lookup",
]

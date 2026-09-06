"""对话立绘 manifest 查表（对应 docs/v3-developer/07-立绘与多模态.md §2）。

查表协议权威：游戏项目 docs/对话立绘查表映射-外部Python读取指南-2026-09-05.md
（实现逐条对齐其 §5 Python 参考实现，标准库即可）：

1. 角色名归一化三注册：原始名 / lower / 去全部空白（[\\s\\ufeff]+）后 lower；
   注册来源三处（entries key、entry.aliases、顶层 aliases），三处都要注册。
2. 情绪回退链：请求情绪 → 普通 → defaultExpression → 首个表情（JSON 顺序）。
3. 路径解析：portrait_dir / asset.uri（uri 相对 manifest 所在目录）。
4. 主角特例：heroKeys（$PC_CHAR/主角模板/玩家）无静态图，resolve 返回 "hero"。
5. 查不到角色 → None（不用「无头像」条目占位）。

manifest 缺失/损坏 → 记 warning，立绘功能整体降级为无图模式，绝不影响聊天主功能。
manifest 每次进程启动现读（单例惰性建一次），不缓存跨版本路径（协议 §6.7）。
禁止扫目录猜测 p_*/e_* hash 名；禁止硬编码情绪全集。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_EXPRESSION = "普通"
_WS_RE = re.compile(r"[\s\ufeff]+")  # 对齐 JS 的 /\s+/g（\s 含 \ufeff）
MANIFEST_REL_PATH = Path("launcher") / "web" / "assets" / "dialogue-portraits" / "manifest.json"


def _strip_ws(s: str) -> str:
    return _WS_RE.sub("", s)


class DialoguePortraitLookup:
    """manifest 查表器：角色名 + 情绪 → PNG 路径与本体包围盒。"""

    def __init__(self, portrait_dir: Path) -> None:
        self.portrait_dir = Path(portrait_dir)
        manifest_path = self.portrait_dir / "manifest.json"
        self.manifest: dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        schema = str(self.manifest.get("schema") or "")
        if schema and schema != "cf7-dialogue-portraits-v2":
            logger.warning("立绘 manifest schema 异常（继续按现有结构查表）: %s", schema)
        self.entries: dict[str, dict[str, Any]] = self.manifest.get("entries") or {}
        self._index: dict[str, dict[str, Any]] = {}
        self._build_index()

    def _build_index(self) -> None:
        def register(label: Any, entry: dict[str, Any]) -> None:
            if not isinstance(label, str) or not label:
                return
            self._index[label] = entry
            self._index[label.lower()] = entry
            self._index[_strip_ws(label).lower()] = entry

        for key, entry in self.entries.items():
            register(key, entry)
            for alias in entry.get("aliases") or []:
                register(alias, entry)
        for alias, target in (self.manifest.get("aliases") or {}).items():
            if target in self.entries:
                register(alias, self.entries[target])

    def is_hero(self, name: str) -> bool:
        """主角无静态立绘（Web 端由 dressup rig 实时渲染），应跳过查表。"""
        return str(name).strip() in (self.manifest.get("heroKeys") or [])

    def find_entry(self, name: str) -> Optional[dict[str, Any]]:
        name = str(name).strip()
        for probe in (name, name.lower(), _strip_ws(name).lower()):
            entry = self._index.get(probe)
            if entry:
                return entry
        return None

    def find_expression(self, entry: dict[str, Any], expression: str) -> Optional[dict[str, Any]]:
        exprs = entry.get("expressions") or {}
        expression = (str(expression).strip() if expression else "") or DEFAULT_EXPRESSION
        for probe in (expression, DEFAULT_EXPRESSION, entry.get("defaultExpression")):
            if probe and probe in exprs:
                return exprs[probe]
        return next(iter(exprs.values()), None)  # 首个表情（JSON 顺序）

    def resolve(
        self, name: str, expression: str = DEFAULT_EXPRESSION
    ) -> Optional[dict[str, Any]] | str:
        """查表：角色名 + 情绪 → dict(png_path, canvas_size, bounds, source, key,
        expression_used)；查不到角色返回 None；主角返回 "hero"（无静态图）。"""
        if self.is_hero(name):
            return "hero"
        entry = self.find_entry(name)
        if entry is None:
            return None
        asset = self.find_expression(entry, expression)
        if asset is None:
            return None
        return {
            "png_path": self.portrait_dir / asset["uri"],
            "canvas_size": (asset.get("width"), asset.get("height")),
            "bounds": asset.get("bounds"),  # 人物本体包围盒（两类 source 通用）
            "source": entry.get("source"),  # external-swf / dialogue-ui-sprite
            "key": entry.get("key"),  # 实际命中的规范角色 key
            "expression_used": next(
                (k for k, v in entry.get("expressions", {}).items() if v is asset), None
            ),
        }


# ---------------------------------------------------------------------------
# manifest 位置与进程内单例
# ---------------------------------------------------------------------------

def discover_portrait_dir() -> Optional[Path]:
    """定位含 manifest.json 的 dialogue-portraits 目录（07 §1）。

    优先级：配置 CFN_GAME_PROJECT_DIR（显式指定游戏项目根）→
    自动探测（resources 目录的同级/上级目录中，寻找含
    launcher/web/assets/dialogue-portraits/manifest.json 的路径）。
    """
    from core.config import get_settings

    explicit = (get_settings().cfn_game_project_dir or "").strip()
    if explicit:
        candidate = Path(explicit) / MANIFEST_REL_PATH
        if candidate.is_file():
            return candidate.parent
        logger.warning("CFN_GAME_PROJECT_DIR 指定路径下未找到立绘 manifest: %s", candidate)
        return None

    try:
        from services.game_data.paths import find_resources_directory

        resources_dir = find_resources_directory()
    except Exception as exc:
        logger.warning("resources 目录未找到，立绘功能降级为无图模式: %s", exc)
        return None

    # resources 的同级与上级各扫一层（游戏项目根常与 resources 同级，或在上级下作兄弟目录）
    for base in (resources_dir.parent, resources_dir.parent.parent):
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            manifest = child / MANIFEST_REL_PATH
            if manifest.is_file():
                return manifest.parent
    return None


_LOOKUP: Optional[DialoguePortraitLookup] = None
_LOOKUP_LOCK = threading.Lock()
_LOOKUP_INITIALIZED = False


def get_portrait_lookup() -> Optional[DialoguePortraitLookup]:
    """进程内单例（惰性建一次）。manifest 缺失/损坏时返回 None（无图模式，不报错）。"""
    global _LOOKUP, _LOOKUP_INITIALIZED
    if _LOOKUP_INITIALIZED:
        return _LOOKUP
    with _LOOKUP_LOCK:
        if _LOOKUP_INITIALIZED:
            return _LOOKUP
        try:
            portrait_dir = discover_portrait_dir()
            if portrait_dir is None:
                logger.warning(
                    "未找到对话立绘 manifest（launcher/web/assets/dialogue-portraits/manifest.json），"
                    "立绘功能降级为无图模式（appearance 文本仍在 prompt 中）"
                )
            else:
                _LOOKUP = DialoguePortraitLookup(portrait_dir)
                logger.info("对话立绘 manifest 已加载: %s", portrait_dir / "manifest.json")
        except Exception as exc:
            logger.warning("立绘 manifest 读取失败，降级为无图模式: %s", exc)
            _LOOKUP = None
        _LOOKUP_INITIALIZED = True
        return _LOOKUP


def reset_portrait_lookup() -> None:
    """测试用：重置单例（下次调用现读 manifest）。"""
    global _LOOKUP, _LOOKUP_INITIALIZED
    with _LOOKUP_LOCK:
        _LOOKUP = None
        _LOOKUP_INITIALIZED = False

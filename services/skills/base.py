"""
Skill data model + YAML frontmatter parser + registry.

与 Anthropic 2026 Agent Skills 规范对齐::

    ---
    name: task-publishing
    description: NPC 发布任务给玩家的完整流程（prepare → draft → confirm）。触发：玩家索要任务……
    ---

    # 任务发布流程
    …… markdown body ……

本模块只负责**解析与索引**：

- 启动时扫描所有 SKILL.md，校验 frontmatter 合法性；
- 提供 ``index()`` 返回 Level 1 简表（name + 截断后的 description）；
- 提供 ``get_body(name)`` / ``read_reference(name, file)`` 供渐进式披露使用。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

try:
    import yaml  # PyYAML（LlamaIndex 已依赖）
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "需要 PyYAML 解析 SKILL.md frontmatter，请运行 pip install pyyaml"
    ) from e


_FRONTMATTER_PATTERN = re.compile(
    r"\A---\s*\r?\n(?P<frontmatter>.*?)\r?\n---\s*\r?\n?(?P<body>.*)\Z",
    flags=re.DOTALL,
)

_VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")

DEFAULT_DESCRIPTION_PREVIEW_LEN = 300


@dataclass(frozen=True)
class Skill:
    """一个 skill 的不可变视图（启动时一次性解析，后续只读）。"""

    name: str
    description: str
    body: str
    md_path: Path
    refs_dir: Optional[Path] = None
    raw_frontmatter: dict[str, Any] = field(default_factory=dict)


def parse_skill_md(md_path: Path) -> Skill:
    """
    解析一个 SKILL.md 文件，返回 Skill。失败抛 ValueError（启动快速失败）。
    """
    text = md_path.read_text(encoding="utf-8")
    m = _FRONTMATTER_PATTERN.match(text)
    if m is None:
        raise ValueError(
            f"SKILL.md 缺少合法的 YAML frontmatter: {md_path}\n"
            "正确格式：文件开头 --- 块内用 YAML 提供 name / description，然后 --- 关闭。"
        )
    fm_text = m.group("frontmatter") or ""
    body_text = (m.group("body") or "").strip()
    try:
        fm_raw = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"SKILL.md frontmatter YAML 解析失败 {md_path}: {e}") from e

    if not isinstance(fm_raw, dict):
        raise ValueError(
            f"SKILL.md frontmatter 必须是对象 {md_path}，实际为 {type(fm_raw).__name__}"
        )

    name = str(fm_raw.get("name") or "").strip()
    description = str(fm_raw.get("description") or "").strip()
    if not name:
        raise ValueError(f"SKILL.md 缺少 'name' 字段: {md_path}")
    if not _VALID_NAME_RE.match(name):
        raise ValueError(
            f"SKILL.md name 不合法 {md_path}: {name!r}；"
            "只允许小写字母/数字/短横，首尾不能是短横，长度 3-64。"
        )
    if not description:
        raise ValueError(f"SKILL.md 缺少 'description' 字段: {md_path}")

    refs_dir = md_path.parent / "references"
    refs_dir_opt = refs_dir if refs_dir.is_dir() else None

    return Skill(
        name=name,
        description=description,
        body=body_text,
        md_path=md_path,
        refs_dir=refs_dir_opt,
        raw_frontmatter=fm_raw,
    )


class SkillRegistry:
    """Skill 注册表：按 name 索引。"""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"duplicate skill name: {skill.name}")
        self._skills[skill.name] = skill

    def discover(self, root: Optional[Path] = None) -> None:
        """
        扫描 ``services/skills/<skill-name>/SKILL.md`` 自动注册。
        """
        if root is None:
            root = Path(__file__).resolve().parent
        if not root.is_dir():
            return

        found: list[Skill] = []
        for skill_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if skill_dir.name.startswith("_"):
                continue
            md_path = skill_dir / "SKILL.md"
            if not md_path.is_file():
                continue
            try:
                skill = parse_skill_md(md_path)
            except ValueError:
                # 启动期快速失败：让使用者立刻看到 SKILL.md 写错了
                raise
            found.append(skill)

        for s in found:
            if s.name in self._skills:
                continue
            self._skills[s.name] = s

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def all_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def index(
        self,
        *,
        categories: Optional[Iterable[str]] = None,
        description_preview: int = DEFAULT_DESCRIPTION_PREVIEW_LEN,
    ) -> list[dict[str, str]]:
        """
        Level 1 简表：[{name, description}]。

        ``categories`` 目前按 **skill name 精确匹配 OR 前缀匹配** 过滤，例如传
        ``["task-publishing", "mood-tracking"]`` 只返回这两个；
        传 ``["task-"]`` 返回所有 task- 前缀的 skill。
        """
        names_filter: Optional[set[str]] = None
        prefixes: list[str] = []
        if categories is not None:
            cats = [str(c).strip() for c in categories if str(c).strip()]
            names_filter = {c for c in cats if not c.endswith("-")}
            prefixes = [c for c in cats if c.endswith("-")]

        rows: list[dict[str, str]] = []
        for s in self._skills.values():
            if names_filter is not None:
                in_names = s.name in names_filter
                in_prefix = any(s.name.startswith(p) for p in prefixes) if prefixes else False
                if not (in_names or in_prefix):
                    continue
            desc = s.description
            if description_preview and len(desc) > description_preview:
                desc = desc[:description_preview].rstrip() + "…"
            rows.append({"name": s.name, "description": desc})
        return rows

    def get_body(self, name: str) -> Optional[str]:
        """Level 2：返回 SKILL.md 的 Markdown body。"""
        s = self._skills.get(name)
        if s is None:
            return None
        return s.body

    def list_reference_files(self, name: str) -> list[str]:
        """返回 skill 目录下 references/ 内的相对路径列表（按字典序）。"""
        s = self._skills.get(name)
        if s is None or s.refs_dir is None:
            return []
        out: list[str] = []
        for p in sorted(s.refs_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(s.md_path.parent).as_posix()
            out.append(rel)
        return out

    def read_reference(self, name: str, rel_path: str) -> Optional[str]:
        """
        Level 3：读取 ``references/`` 下的附件文本。

        严格校验：路径必须以 'references/' 开头 + 禁止 ``..`` / 绝对路径 / 跳出 skill 目录。
        """
        s = self._skills.get(name)
        if s is None:
            return None
        cleaned = (rel_path or "").strip().lstrip("./")
        if not cleaned.startswith("references/"):
            return None
        if ".." in cleaned.split("/"):
            return None
        candidate = (s.md_path.parent / cleaned).resolve()
        try:
            candidate.relative_to(s.md_path.parent.resolve())
        except ValueError:
            return None
        if not candidate.is_file():
            return None
        try:
            return candidate.read_text(encoding="utf-8")
        except Exception:  # pragma: no cover
            return None


_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        r = SkillRegistry()
        r.discover()
        _registry = r
    return _registry

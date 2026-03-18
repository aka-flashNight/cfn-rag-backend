from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles


def _get_resources_dir() -> Path:
    """
    获取resources目录路径。
    resources是外部项目文件夹，和本项目放在同一目录下。

    目录结构：
        父目录/
        ├── resources/          # 外部游戏数据
        └── cfn-rag-backend/    # 本项目（开发环境）
            └── ...

        或打包后：
        部署目录/
        ├── resources/          # 外部游戏数据
        └── CFN-RAG.exe         # 打包后的exe
    """
    # 1. 检查环境变量（由launcher.py设置）
    env_path = os.environ.get('CFN_RESOURCES_DIR')
    if env_path:
        return Path(env_path)

    # 2. 检查是否在PyInstaller打包环境
    if getattr(sys, 'frozen', False):
        # 打包环境：exe和resources在同一目录
        exe_dir = Path(sys.executable).parent
        resources_path = exe_dir / "resources"
        if resources_path.exists():
            return resources_path
        raise FileNotFoundError(
            f"打包环境未找到resources目录。\n"
            f"已查找: {resources_path}\n"
            f"请确保CFN-RAG.exe和resources文件夹在同一目录"
        )

    # 3. 开发环境：resources在父目录
    # 当前文件位置: cfn-rag-backend/services/npc_manager.py
    # resources位置: cfn-rag-backend/../resources
    project_dir = Path(__file__).resolve().parent.parent  # cfn-rag-backend
    parent_dir = project_dir.parent
    resources_path = parent_dir / "resources"

    if resources_path.exists():
        return resources_path

    # 如果父目录没有，再检查同级目录（兼容其他部署方式）
    sibling_path = project_dir / "resources"
    if sibling_path.exists():
        return sibling_path

    raise FileNotFoundError(
        f"开发环境未找到resources目录。\n"
        f"已查找: {resources_path} 和 {sibling_path}\n"
        f"请确保resources文件夹在项目父目录或同级目录"
    )


def _get_npc_state_path() -> Path:
    """动态获取NPC状态文件路径"""
    resources_dir = _get_resources_dir()
    rag_dir = resources_dir / "data" / "rag"
    return rag_dir / "npc_state_db.json"


# 保持向后兼容，但不在模块加载时计算路径
RESOURCES_DIR: Path = _get_resources_dir()
RAG_DIR: Path = RESOURCES_DIR / "data" / "rag"
NPC_STATE_PATH: Path = RAG_DIR / "npc_state_db.json"


@dataclass
class NPCState:
    favorability: int
    relationship_level: str
    sex: Optional[str] = None
    # 可选：当前 NPC 的切磋关卡名（npc_state_db.json 的 challenge 字段）
    challenge: Optional[str] = None
    emotions: list[str] = field(default_factory=list)
    faction: Optional[str] = None
    titles: list[str] = field(default_factory=list)


class NPCManager:
    """
    管理 NPC 好感度与关系等级的本地状态。

    使用示例（在异步上下文中）：

        manager = await NPCManager.load()
        manager.update_favorability("Andy Law", 5)
        await manager.save()
    """

    def __init__(self, state: Dict[str, NPCState]) -> None:
        self._state: Dict[str, NPCState] = state

    @classmethod
    async def load(cls) -> "NPCManager":
        """
        从本地 JSON 文件初始化 NPCManager。
        如果文件不存在，则创建空文件。
        """
        # 动态获取路径（支持打包环境）
        rag_dir = _get_resources_dir() / "data" / "rag"
        npc_state_path = rag_dir / "npc_state_db.json"

        rag_dir.mkdir(parents=True, exist_ok=True)

        if npc_state_path.exists():
            async with aiofiles.open(npc_state_path, "r", encoding="utf-8") as f:
                raw_text = await f.read()
            try:
                raw_data: Dict[str, Dict[str, Any]] = (
                    json.loads(raw_text) if raw_text.strip() else {}
                )
            except json.JSONDecodeError:
                raw_data = {}
        else:
            raw_data = {}

        state: Dict[str, NPCState] = {}
        for name, item in raw_data.items():
            if not isinstance(item, dict):
                continue
            favorability = int(item.get("favorability", 0))
            level = str(item.get("relationship_level") or "").strip()
            if not level:
                level = cls._compute_relationship_level(favorability)
            sex_raw = str(item.get("sex") or "").strip()
            sex: Optional[str] = sex_raw or None
            challenge_raw = str(item.get("challenge") or "").strip()
            challenge: Optional[str] = challenge_raw or None
            emotions_raw = item.get("emotions")
            emotions: list[str] = []
            if isinstance(emotions_raw, list):
                for e in emotions_raw:
                    if isinstance(e, str) and e.strip():
                        emotions.append(e.strip())
            if not emotions:
                emotions = ["普通"]
            faction_raw = str(item.get("faction") or "").strip()
            faction: Optional[str] = faction_raw or None
            titles_raw = item.get("titles")
            titles: list[str] = []
            if isinstance(titles_raw, list):
                for t in titles_raw:
                    if isinstance(t, str) and t.strip():
                        titles.append(t.strip())
            state[name] = NPCState(
                favorability=favorability,
                relationship_level=level,
                sex=sex,
                challenge=challenge,
                emotions=emotions,
                faction=faction,
                titles=titles,
            )

        manager = cls(state=state)

        # 如果文件不存在，尝试从对话数据中初始化NPC列表
        if not npc_state_path.exists():
            await manager._init_from_dialogues()
            await manager._save_to_path(npc_state_path)

        return manager

    async def _init_from_dialogues(self) -> None:
        """从对话数据中提取NPC名称并初始化状态"""
        try:
            resources_dir = _get_resources_dir()
            dialogues_dir = resources_dir / "data" / "dialogues"
            list_path = dialogues_dir / "list.xml"

            if not list_path.exists():
                print(f"[NPCManager] 未找到对话列表文件: {list_path}")
                return

            import xml.etree.ElementTree as ET

            tree = ET.parse(list_path)
            root = tree.getroot()

            # 从 list.xml 获取所有对话文件名
            filenames = [
                (elem.text or "").strip()
                for elem in root.findall(".//items")
                if (elem.text or "").strip()
            ]

            npc_names: set[str] = set()

            for name in filenames:
                file_path = dialogues_dir / name
                if not file_path.exists():
                    continue

                xml_tree = ET.parse(file_path)
                xml_root = xml_tree.getroot()

                # 尝试读取文件级别的角色名
                file_level_name_elem = xml_root.find(".//Dialogues/Name")
                file_level_name = (
                    file_level_name_elem.text.strip()
                    if file_level_name_elem is not None
                    else None
                )

                for sub in xml_root.findall(".//SubDialogue"):
                    sub_name_elem = sub.find("Name")
                    sub_char_elem = sub.find("Char")

                    # 优先使用 Char，其次使用 Name
                    char_name = None
                    if sub_char_elem is not None and (sub_char_elem.text or "").strip():
                        char_name = sub_char_elem.text.strip()
                    elif sub_name_elem is not None and (sub_name_elem.text or "").strip():
                        char_name = sub_name_elem.text.strip()
                    else:
                        char_name = file_level_name

                    if char_name:
                        npc_names.add(char_name)

            # 初始化NPC状态
            for npc_name in sorted(npc_names):
                self._state[npc_name] = NPCState(
                    favorability=0,
                    relationship_level="陌生",
                    sex=None,
                    emotions=["普通"],
                    faction=None,
                    titles=[],
                )

            print(f"[NPCManager] 从对话数据初始化了 {len(npc_names)} 个NPC")

        except Exception as e:
            print(f"[NPCManager] 从对话数据初始化失败: {e}")
            # 失败时不阻断，只是状态为空

    @property
    def state(self) -> Dict[str, NPCState]:
        """
        返回当前全部 NPC 状态（只读视图）。
        """

        return dict(self._state)

    def update_favorability(self, npc_name: str, change_value: int) -> NPCState:
        """
        更新指定 NPC 的好感度，并根据区间自动刷新关系等级。

        区间规则：
        - 0-20    -> 陌生
        - 20-50   -> 熟悉
        - 50-80   -> 朋友
        - 80-100  -> 生死之交

        注意：本方法只更新内存，不落盘；需要调用 save() 才会写入文件。
        """

        npc_name = npc_name.strip()
        if not npc_name:
            raise ValueError("npc_name 不能为空。")

        current: NPCState = self._state.get(
            npc_name,
            NPCState(favorability=0, relationship_level="陌生", emotions=["普通"], faction=None, titles=[]),
        )

        new_favorability: int = current.favorability + int(change_value)
        # 限制在 0-100 之间
        new_favorability = max(0, min(100, new_favorability))

        new_level: str = self._compute_relationship_level(new_favorability)

        # 使用 dataclasses.replace 只修改好感度相关字段，保留其他所有字段
        updated = replace(
            current,
            favorability=new_favorability,
            relationship_level=new_level,
        )
        self._state[npc_name] = updated
        return updated

    @staticmethod
    def _compute_relationship_level(favorability: int) -> str:
        """
        根据好感度数值计算关系等级。
        """

        if favorability < 20:
            return "陌生"
        if favorability < 50:
            return "熟悉"
        if favorability < 80:
            return "朋友"
        return "生死之交"

    async def save(self) -> None:
        """
        将当前 NPC 状态保存到本地 JSON 文件。
        """
        npc_state_path = _get_npc_state_path()
        await self._save_to_path(npc_state_path)

    async def _save_to_path(self, path: Path) -> None:
        """保存到指定路径的内部实现"""
        serializable: Dict[str, Dict[str, Any]] = {}
        for name, state in self._state.items():
            item: Dict[str, Any] = {
                "favorability": state.favorability,
                "relationship_level": state.relationship_level,
            }
            if state.sex:
                item["sex"] = state.sex
            if state.challenge:
                item["challenge"] = state.challenge
            if state.emotions:
                item["emotions"] = list(state.emotions)
            serializable[name] = item

            if state.faction:
                item["faction"] = state.faction
            if state.titles:
                item["titles"] = list(state.titles)

        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(serializable, ensure_ascii=False, indent=2))


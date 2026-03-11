from __future__ import annotations

import os
import sys
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

_EMBED_MODEL_CONFIGURED: bool = False

# 模型配置
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def _get_model_dir() -> Path:
    """
    获取模型目录路径。
    支持普通 Python 运行和 PyInstaller 打包环境。
    """
    # 检查是否在 PyInstaller 打包环境中
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包环境：数据文件被解压到 sys._MEIPASS 临时目录
        # 模型通过 --add-data 打包，在运行时位于 _MEIPASS/models/
        if hasattr(sys, '_MEIPASS'):
            base_dir = Path(sys._MEIPASS)
        else:
            # 备用方案：使用 exe 所在目录（某些PyInstaller版本可能没有_MEIPASS）
            base_dir = Path(sys.executable).parent
    else:
        # 普通 Python 运行：使用项目根目录
        base_dir = Path(__file__).resolve().parent.parent

    return base_dir / "models" / "bge-small-zh-v1.5"


LOCAL_MODEL_DIR = _get_model_dir()


def _is_local_model_valid() -> bool:
    """检查本地模型是否完整（通过检查关键文件是否存在）。"""
    required_files = ["config.json", "pytorch_model.bin"]
    return all((LOCAL_MODEL_DIR / f).exists() for f in required_files)


def ensure_embed_model(offline: bool = True) -> None:
    """
    懒加载配置本地向量模型，避免在模块 import 时触发 HuggingFace 联网请求。

    - 优先从本地项目目录加载模型 (models/bge-small-zh-v1.5)
    - 如果本地不存在，则尝试从 HuggingFace 下载（需要网络）
    - offline=True 时会禁用联网，仅从本地加载
    """

    global _EMBED_MODEL_CONFIGURED
    if _EMBED_MODEL_CONFIGURED:
        return

    # 强制离线环境变量（先设置，确保任何情况下都不联网）
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_LOCAL_DIR_USE_SYMLINKS"] = "0"

    # 检查本地模型是否完整
    if LOCAL_MODEL_DIR.exists() and _is_local_model_valid():
        print(f"[模型] 使用本地模型: {LOCAL_MODEL_DIR}")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=str(LOCAL_MODEL_DIR),
            device="cpu",
            trust_remote_code=False,
        )
    else:
        if offline:
            raise FileNotFoundError(
                f"本地模型不存在或不完整: {LOCAL_MODEL_DIR}\n"
                f"请先运行: python scripts/download_model.py 下载模型。"
            )
        print(f"[模型] 本地模型不存在，尝试从 HuggingFace 下载: {MODEL_NAME}")
        Settings.embed_model = HuggingFaceEmbedding(model_name=MODEL_NAME)

    _EMBED_MODEL_CONFIGURED = True


def download_model_to_local(use_modelscope: bool = False) -> None:
    """
    下载 HuggingFace 模型到本地项目目录，供离线使用。
    运行此函数需要网络连接和代理（如果需要）。

    Args:
        use_modelscope: 是否使用 ModelScope（国内镜像）下载
    """
    if use_modelscope:
        # 使用 ModelScope 国内镜像
        try:
            from modelscope import snapshot_download
            print(f"[下载] 使用 ModelScope 镜像下载: {MODEL_NAME}")
            print(f"[下载] 目标路径: {LOCAL_MODEL_DIR}")

            LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

            # ModelScope 的模型 ID 格式不同
            modelscope_id = "AI-ModelScope/bge-small-zh-v1.5"
            cache_dir = snapshot_download(modelscope_id)

            # 复制文件到本地目录
            import shutil
            for item in Path(cache_dir).iterdir():
                dest = LOCAL_MODEL_DIR / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)

            print(f"[下载] 模型下载完成！保存在: {LOCAL_MODEL_DIR}")
            return
        except ImportError:
            print("[警告] 未安装 modelscope，将尝试使用 HuggingFace 下载")
            print("       安装命令: pip install modelscope")

    # 使用 HuggingFace 下载
    from transformers import AutoModel, AutoTokenizer

    print(f"[下载] 正在下载模型到本地: {MODEL_NAME}")
    print(f"[下载] 目标路径: {LOCAL_MODEL_DIR}")

    # 创建模型目录
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 下载模型和 tokenizer
    print("[下载] 下载模型中...")
    model = AutoModel.from_pretrained(MODEL_NAME)
    print("[下载] 下载 tokenizer 中...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 保存到本地
    print("[下载] 保存模型到本地...")
    model.save_pretrained(str(LOCAL_MODEL_DIR))
    tokenizer.save_pretrained(str(LOCAL_MODEL_DIR))

    print(f"[下载] 模型下载完成！保存在: {LOCAL_MODEL_DIR}")
    print("[下载] 现在可以使用本地模型启动，无需联网。")

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
    # 当前文件位置: cfn-rag-backend/ai_engine/game_data_loader.py
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


def _ensure_resources_dir() -> None:
    resources_dir = _get_resources_dir()
    if not resources_dir.exists():
        raise FileNotFoundError(f"resources 目录不存在: {resources_dir}")


def load_dialogue_documents() -> List[Document]:
    """
    读取 ../resources/data/dialogues 下的 NPC 日常对话 XML，
    将每个角色的台词合并为一个 Document。

    XML 格式示例：
    <root>
      <Dialogues>
        <Name>冷兵器商人</Name>
        <Dialogue>
          <SubDialogue>
            <Name>冷兵器商人</Name>
            <Text>...</Text>
          </SubDialogue>
        </Dialogue>
      </Dialogues>
    </root>
    """

    _ensure_resources_dir()

    resources_dir = _get_resources_dir()
    dialogues_dir: Path = resources_dir / "data" / "dialogues"
    list_path: Path = dialogues_dir / "list.xml"
    if not list_path.exists():
        raise FileNotFoundError(f"未找到对话列表文件: {list_path}")

    tree = ET.parse(list_path)
    root = tree.getroot()

    # list.xml 中的 <items>npc_dialogue_shop.xml</items>
    filenames: List[str] = [
        (elem.text or "").strip()
        for elem in root.findall(".//items")
        if (elem.text or "").strip()
    ]

    # 角色 -> 台词列表
    character_lines: Dict[str, List[str]] = defaultdict(list)

    for name in filenames:
        file_path: Path = dialogues_dir / name
        if not file_path.exists():
            continue

        xml_tree = ET.parse(file_path)
        xml_root = xml_tree.getroot()

        # 尝试读取文件级别的角色名（如 <Dialogues><Name>冷兵器商人</Name>）
        file_level_name_elem = xml_root.find(".//Dialogues/Name")
        file_level_name: str | None = (
            file_level_name_elem.text.strip() if file_level_name_elem is not None else None
        )

        for sub in xml_root.findall(".//SubDialogue"):
            sub_name_elem = sub.find("Name")
            sub_char_elem = sub.find("Char")
            text_elem = sub.find("Text")

            text: str = (text_elem.text or "").strip() if text_elem is not None else ""
            if not text:
                continue

            # 优先 Char，其次 SubDialogue 内 Name，最后文件级 Name
            char_name: str | None = None
            if sub_char_elem is not None and (sub_char := (sub_char_elem.text or "").strip()):
                char_name = sub_char
            elif sub_name_elem is not None and (sub_name := (sub_name_elem.text or "").strip()):
                char_name = sub_name
            else:
                char_name = file_level_name

            if not char_name:
                char_name = "Unknown"

            character_lines[char_name].append(text)

    documents: List[Document] = []
    for character, lines in character_lines.items():
        merged_text: str = "\n".join(lines)
        metadata = {
            "character": character,
            "type": "dialogue",
        }
        documents.append(Document(text=merged_text, metadata=metadata))

    return documents


def _flatten_task_dialogues(dialogues: Iterable[dict]) -> str:
    """将 challenge_text.json 中的对话数组拼接为可阅读文本。"""

    parts: List[str] = []
    for item in dialogues:
        if not isinstance(item, dict):
            continue
        speaker: str = str(item.get("name") or "").strip()
        text: str = str(item.get("text") or "").strip()
        if not text:
            continue

        if speaker:
            parts.append(f"{speaker}: {text}")
        else:
            parts.append(text)

    return "\n".join(parts)


def load_task_documents() -> List[Document]:
    """
    读取挑战任务数据 challenge_tasks.json 和 challenge_text.json，
    按任务拼接剧情并构建 Document，metadata 中写入相关角色信息。
    """

    _ensure_resources_dir()

    resources_dir = _get_resources_dir()
    task_dir: Path = resources_dir / "data" / "task"
    text_dir: Path = task_dir / "text"

    tasks_path: Path = task_dir / "challenge_tasks.json"
    text_path: Path = text_dir / "challenge_text.json"

    if not tasks_path.exists():
        raise FileNotFoundError(f"未找到任务配置: {tasks_path}")
    if not text_path.exists():
        raise FileNotFoundError(f"未找到任务文本配置: {text_path}")

    with tasks_path.open("r", encoding="utf-8") as f:
        tasks_data = json.load(f)
    with text_path.open("r", encoding="utf-8") as f:
        text_data = json.load(f)

    tasks: List[dict] = tasks_data.get("tasks") or []

    documents: List[Document] = []

    for task in tasks:
        if not isinstance(task, dict):
            continue

        task_id = task.get("id")
        title_key: str | None = task.get("title")
        desc_key: str | None = task.get("description")

        get_conv_key: str | None = task.get("get_conversation")
        finish_conv_key: str | None = task.get("finish_conversation")

        get_npc: str | None = task.get("get_npc")
        finish_npc: str | None = task.get("finish_npc")

        # 解析标题和描述（从 challenge_text.json 中的字符串）
        title: str = str(text_data.get(title_key, title_key or "")).strip()
        description: str = str(text_data.get(desc_key, desc_key or "")).strip()

        # 解析获取/完成阶段对话
        get_dialogues_raw = text_data.get(get_conv_key) or []
        finish_dialogues_raw = text_data.get(finish_conv_key) or []

        get_dialogues_text = (
            _flatten_task_dialogues(get_dialogues_raw)
            if isinstance(get_dialogues_raw, list)
            else str(get_dialogues_raw)
        )
        finish_dialogues_text = (
            _flatten_task_dialogues(finish_dialogues_raw)
            if isinstance(finish_dialogues_raw, list)
            else str(finish_dialogues_raw)
        )

        # 收集涉及到的角色
        characters: Set[str] = set()
        for npc in (get_npc, finish_npc):
            if isinstance(npc, str) and npc.strip():
                characters.add(npc.strip())

        def _collect_characters_from_dialogues(items: Iterable[dict]) -> None:
            for item in items:
                if not isinstance(item, dict):
                    continue
                char = str(item.get("char") or "").strip()
                if not char:
                    continue
                # 排除玩家占位符
                if char.startswith("$PC"):
                    continue
                # 去掉表情后缀，如 "Andy Law#微笑"
                base_char = char.split("#", maxsplit=1)[0].strip()
                if base_char:
                    characters.add(base_char)

        if isinstance(get_dialogues_raw, list):
            _collect_characters_from_dialogues(get_dialogues_raw)
        if isinstance(finish_dialogues_raw, list):
            _collect_characters_from_dialogues(finish_dialogues_raw)

        # 组合成一个可阅读的任务剧情文本
        text_sections: List[str] = []
        if title:
            text_sections.append(f"任务标题: {title}")
        if description:
            text_sections.append(f"任务描述: {description}")
        if get_dialogues_text:
            text_sections.append("[获取任务阶段]\n" + get_dialogues_text)
        if finish_dialogues_text:
            text_sections.append("[完成任务阶段]\n" + finish_dialogues_text)

        if not text_sections:
            continue

        full_text: str = "\n\n".join(text_sections)

        metadata = {
            "type": "task",
            "task_id": task_id,
            "task_chain": task.get("chain"),
            "task_title_key": title_key,
            "task_description_key": desc_key,
            "get_conversation_key": get_conv_key,
            "finish_conversation_key": finish_conv_key,
            "characters": sorted(characters) if characters else [],
        }

        documents.append(Document(text=full_text, metadata=metadata))

    return documents


def load_lore_documents() -> List[Document]:
    """
    使用 SimpleDirectoryReader 读取 ../resources/docs 下的
    PDF / DOCX 世界观设定文档。
    """

    _ensure_resources_dir()

    resources_dir = _get_resources_dir()
    docs_dir: Path = resources_dir / "docs"
    if not docs_dir.exists():
        raise FileNotFoundError(f"未找到世界观设定目录: {docs_dir}")

    reader = SimpleDirectoryReader(
        input_dir=str(docs_dir),
        required_exts=[".pdf", ".docx"],
    )
    documents = reader.load_data()

    # 为了更好地区分类型，在 metadata 中统一标记 type
    for doc in documents:
        meta = dict(doc.metadata or {})
        meta.setdefault("type", "world_lore")
        doc.metadata = meta

    return documents


def build_index() -> VectorStoreIndex:
    """
    统一构建内存向量索引，将对话、任务、世界观设定全部加入。
    """

    ensure_embed_model(offline=True)

    # 设置简单的 node parser，避免 tiktoken 编码问题
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core import Settings

    # 使用简单的字符分割器，完全避免 tiktoken
    # 针对中英文混合优化：英文按单词，中文按字符
    def simple_tokenizer(text: str) -> list[str]:
        """简单tokenizer：空格分割英文，字符分割中文"""
        import re
        # 匹配英文单词或单个字符（包括中文）
        tokens = re.findall(r'[a-zA-Z]+|\d+|[^\s\w]', text)
        return tokens if tokens else text.split()

    node_parser = SentenceSplitter(
        chunk_size=512,  # 减小chunk大小，因为中文字符分割比tiktoken细
        chunk_overlap=50,
        tokenizer=simple_tokenizer,
    )
    Settings.node_parser = node_parser

    dialogue_docs = load_dialogue_documents()
    task_docs = load_task_documents()
    lore_docs = load_lore_documents()

    all_docs: List[Document] = []
    all_docs.extend(dialogue_docs)
    all_docs.extend(task_docs)
    all_docs.extend(lore_docs)

    if not all_docs:
        raise ValueError("没有加载到任何 Document，无法构建索引。")

    index: VectorStoreIndex = VectorStoreIndex.from_documents(all_docs)
    return index


# 索引缓存，避免每次请求都重建
_index_cache: VectorStoreIndex | None = None


def get_cached_index() -> VectorStoreIndex:
    """获取或构建并缓存索引。"""
    global _index_cache
    if _index_cache is None:
        ensure_embed_model(offline=True)
        _index_cache = build_index()
    return _index_cache


def get_query_engine():
    """
    获取游戏知识库 RAG 查询引擎，使用 .env 中的 GEMINI_API_KEY 作为 LLM。

    使用示例：

        from ai_engine.game_data_loader import get_query_engine
        qe = get_query_engine()
        resp = qe.query("冷兵器商人是怎样评价冷兵器的？")
    """
    from llama_index.llms.google_genai import GoogleGenAI

    from core.config import get_settings

    settings = get_settings()
    if not settings.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY 未配置，请在 .env 中设置。"
        )

    llm = GoogleGenAI(
        model=settings.gemini_model,
        api_key=settings.gemini_api_key,
    )
    index = get_cached_index()
    return index.as_query_engine(llm=llm)


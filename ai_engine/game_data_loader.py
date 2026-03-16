from __future__ import annotations

import os
import re
import sys
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Set

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from services.memory_manager import get_db_path

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
    """检查本地模型是否完整。"""
    required_files = ["config.json"]
    # 支持 safetensors 或 pytorch 格式
    model_files = ["model.safetensors", "pytorch_model.bin"]
    
    has_config = all((LOCAL_MODEL_DIR / f).exists() for f in required_files)
    has_model = any((LOCAL_MODEL_DIR / f).exists() for f in model_files)
    
    return has_config and has_model


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


def get_vector_index_dir() -> Path:
    """
    获取知识库向量索引的持久化目录，与 memory.db 同位于 resources/tools 下。
    开发环境与 exe 运行时均使用同一套路径解析逻辑（通过 get_db_path 复用）。
    """
    return get_db_path().parent / "vector_index"


def _is_vector_index_valid(persist_dir: Path) -> bool:
    """检查 persist_dir 下是否存在有效的 LlamaIndex 持久化索引（至少需有 index_store）。"""
    if not persist_dir.is_dir():
        return False
    # LlamaIndex 默认会持久化 docstore.json、index_store.json 等
    return (persist_dir / "index_store.json").exists() or (persist_dir / "docstore.json").exists()


def load_dialogue_documents() -> List[Document]:
    """
    读取 ../resources/data/dialogues 下的 NPC 日常对话 XML，
    将每个 <Dialogues>/<Dialogue> 作为一个 Document。

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
    documents: List[Document] = []

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

        # 按每个 <Dialogue> 生成一个 Document，以保持局部上下文又避免整角色文本过长稀释相似度
        for dlg in xml_root.findall(".//Dialogues/Dialogue"):
            lines: List[str] = []
            character_key: str | None = None

            for sub in dlg.findall("./SubDialogue"):
                sub_name_elem = sub.find("Name")
                sub_char_elem = sub.find("Char")
                text_elem = sub.find("Text")

                text: str = (text_elem.text or "").strip() if text_elem is not None else ""
                if not text:
                    continue

                # 跳过玩家视角的台词（$PC / $PC_TITLE / $PC_CHAR 等）
                sub_name = (sub_name_elem.text or "").strip() if sub_name_elem is not None else ""
                sub_char = (sub_char_elem.text or "").strip() if sub_char_elem is not None else ""
                if sub_name == "$PC" or sub_char.startswith("$PC"):
                    continue

                # 角色标注使用 Name（角色名），不用 Char（资源/表情标识）；缺省时再回退到文件级 Name 或 Char
                if character_key is None:
                    if sub_name:
                        character_key = sub_name
                    elif file_level_name:
                        character_key = file_level_name
                    elif sub_char:
                        character_key = sub_char.split("#")[0].strip() if "#" in sub_char else sub_char
                    else:
                        character_key = "Unknown"

                lines.append(text)

            if not lines or not character_key:
                continue

            merged_text: str = "\n".join(lines)
            # 使用小写规范化，与检索端 npc_name 过滤一致，避免 "King" vs "king" 导致命中为空
            character_normalized = (character_key or "").strip().lower()
            metadata = {
                "character": character_normalized,
                "type": "dialogue",
            }
            documents.append(Document(text=merged_text, metadata=metadata))

    return documents


def _is_player_dialogue_item(item: dict) -> bool:
    """判断对话条是否为玩家（$PC / $PC_TITLE / $PC_CHAR），此类不进入任务台词检索。"""
    if not isinstance(item, dict):
        return True
    name = str(item.get("name") or "").strip()
    title = str(item.get("title") or "").strip()
    char = str(item.get("char") or "").strip()
    if name == "$PC" or title == "$PC_TITLE":
        return True
    if char and (char == "$PC_CHAR" or char.startswith("$PC_CHAR#")):
        return True
    return False


def _task_character_from_item(item: dict) -> str | None:
    """从对话条取 NPC 角色名（Name 优先），规范化为小写；玩家条返回 None。"""
    if _is_player_dialogue_item(item):
        return None
    name = str(item.get("name") or "").strip()
    if not name:
        char = str(item.get("char") or "").strip()
        if char and not char.startswith("$PC"):
            name = char.split("#", maxsplit=1)[0].strip()
    if not name:
        return None
    return name.lower()


def load_task_documents() -> List[Document]:
    """
    读取所有任务配置（*tasks*.json）与剧情文本（text/*.json），
    将任务对话拆成「单条 NPC 台词」为一份 Document，仅保留 NPC 发言（排除 $PC/$PC_TITLE 等），
    并打上 character（小写角色名），便于检索时按当前 NPC 过滤且仅按对话内容相似度检索。
    """

    _ensure_resources_dir()

    resources_dir = _get_resources_dir()
    task_dir: Path = resources_dir / "data" / "task"
    text_dir: Path = task_dir / "text"

    if not task_dir.exists() or not text_dir.exists():
        return []

    # 合并所有 text/*.json 的 key，使 $SIDE_GET_50001 等可从 logistics_text 等任意文件解析
    # 注意：preview_text.json 仅用于任务预览，对正式对话与设定检索无意义，这里显式跳过。
    text_data: Dict[str, Any] = {}
    for jpath in sorted(text_dir.glob("*.json")):
        if jpath.stem == "preview_text":
            continue
        try:
            with jpath.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                text_data.update(data)
        except Exception as e:
            print(f"[知识库] 跳过任务文本 {jpath.name}: {e}")
            continue

    # 收集所有任务（来自所有 *tasks*.json），并标注来源以便检索时区分（如教学引导类提高分数要求）
    all_tasks: List[dict] = []
    for jpath in sorted(task_dir.glob("*tasks*.json")):
        try:
            with jpath.open("r", encoding="utf-8") as f:
                data = json.load(f)
            tasks = data.get("tasks") if isinstance(data, dict) else []
            if not isinstance(tasks, list):
                continue
            # 根据文件名打标：guide 类任务仅在高分时才采用，避免教学引导与 NPC 形象弱关联时混入
            task_source = "guide" if "guide" in jpath.name.lower() else None
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                t = dict(task)
                if task_source:
                    t["_task_source"] = task_source
                all_tasks.append(t)
        except Exception as e:
            print(f"[知识库] 跳过任务配置 {jpath.name}: {e}")
            continue

    documents: List[Document] = []

    for task in all_tasks:
        if not isinstance(task, dict):
            continue

        get_conv_key: str | None = task.get("get_conversation")
        finish_conv_key: str | None = task.get("finish_conversation")
        task_source: str | None = task.get("_task_source")  # "guide" 或 None

        for key in (get_conv_key, finish_conv_key):
            if not key:
                continue
            raw = text_data.get(key)
            if not isinstance(raw, list):
                continue
            for item in raw:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                character = _task_character_from_item(item)
                if not character:
                    continue
                metadata: Dict[str, Any] = {
                    "type": "task",
                    "character": character,
                }
                if task_source:
                    metadata["task_source"] = task_source
                documents.append(Document(text=text, metadata=metadata))

    return documents


def load_intelligence_documents() -> List[Document]:
    """
    读取 resources/data/intelligence/ 下的 TXT 情报文件。

    如果文件内含 @@@X_Y@@@ 分节标记，则按标记拆分为多个文档片段；
    否则整个文件作为一个 Document。
    """

    _ensure_resources_dir()

    resources_dir = _get_resources_dir()
    intel_dir: Path = resources_dir / "data" / "intelligence"
    if not intel_dir.exists():
        print("[知识库] 情报目录不存在，跳过: " + str(intel_dir))
        return []

    import re
    _SECTION_RE = re.compile(r'@@@\d+(?:_\d+)?@@@')

    documents: List[Document] = []
    for txt_file in sorted(intel_dir.glob("*.txt")):
        try:
            content = txt_file.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if len(content) < 10:
            continue

        source_name = txt_file.stem

        if _SECTION_RE.search(content):
            sections = _SECTION_RE.split(content)
            for sec in sections:
                sec = sec.strip()
                if len(sec) < 10:
                    continue
                documents.append(Document(
                    text=sec,
                    metadata={"type": "intelligence", "source_file": source_name},
                ))
        else:
            documents.append(Document(
                text=content,
                metadata={"type": "intelligence", "source_file": source_name},
            ))

    print(f"[知识库] 情报文件: {len(documents)} 个文档片段，"
          f"来自 {len(list(intel_dir.glob('*.txt')))} 个 txt 文件")
    return documents


def load_lore_documents() -> List[Document]:
    """
    使用 SimpleDirectoryReader 读取 ../resources/docs 下的
    PDF / DOCX 文档，并根据文件名区分核心世界观与补充设定。

    命名规则：
      - 文件名含 "核心设定与世界合理性补足" → type = "world_lore"（核心世界观）
      - 其余文件 → type = "supplementary_lore"（补充/局部设定，检索优先级较低）
    """

    _ensure_resources_dir()

    resources_dir = _get_resources_dir()
    docs_dir: Path = resources_dir / "docs"
    if not docs_dir.exists():
        print("[知识库] 世界观设定目录不存在，跳过: " + str(docs_dir))
        return []

    matching_files = [
        f for f in docs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".pdf", ".docx")
    ]
    if not matching_files:
        print("[知识库] docs 目录中无 PDF/DOCX 文件，跳过世界观文档加载")
        return []

    reader = SimpleDirectoryReader(
        input_dir=str(docs_dir),
        required_exts=[".pdf", ".docx"],
    )
    documents = reader.load_data()

    for doc in documents:
        meta = dict(doc.metadata or {})
        file_path = Path(meta.get("file_path", meta.get("file_name", "")))
        if "核心设定与世界合理性补足" in file_path.stem:
            meta["type"] = "world_lore"
        else:
            meta["type"] = "supplementary_lore"
        doc.metadata = meta

    loaded_types = {}
    for doc in documents:
        t = doc.metadata.get("type", "unknown")
        fn = Path(doc.metadata.get("file_path", doc.metadata.get("file_name", "?"))).name
        loaded_types.setdefault(t, set()).add(fn)

    for t, fnames in loaded_types.items():
        print(f"[知识库] 类型 {t}: {', '.join(sorted(fnames))}")

    return documents


# 设定文档按章节/段落切分：识别标题行（用于按章节切分）
_LORE_HEADING_PATTERN = re.compile(
    r"^\s*(?:"
    r"#+\s+.+|"  # Markdown: # ## ###
    r"[一二三四五六七八九十百千]+[、．.]\s*.+|"  # 一、 二、
    r"\d+[.、]\s*.+|"  # 1. 2. 1、
    r"[（(][一二三四五六七八九十\d]+[)）]\s*.+"  # （一） (1)
    r")\s*$",
    re.MULTILINE,
)


def _split_by_headings(text: str) -> List[str]:
    """
    按标题行将文本拆成多个章节。
    识别：Markdown #、一、二、、1. 2.、（一）等。
    """
    if not (text or "").strip():
        return []
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not blocks:
        return [text.strip()] if text.strip() else []

    sections: List[str] = []
    current: List[str] = []

    for i, block in enumerate(blocks):
        first_line = block.split("\n")[0] if "\n" in block else block
        is_heading = bool(_LORE_HEADING_PATTERN.match(first_line.strip()))
        if is_heading and current:
            sections.append("\n\n".join(current))
            current = [block]
        else:
            current.append(block)
    if current:
        sections.append("\n\n".join(current))
    return sections


def _split_sentences(text: str) -> List[str]:
    """按句号、问号、叹号、分号分句，保留边界完整（不 mid-sentence 截断）。"""
    if not text or not text.strip():
        return []
    # 在。！？； 后切分，保留分隔符在上一句末尾
    parts = re.split(r"([。！？；])", text)
    sentences: List[str] = []
    buf = ""
    for i, p in enumerate(parts):
        buf += p
        if p.strip() in "。！？；" and buf.strip():
            sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return sentences


# 用于判断「以标点结尾」的中英文标点（行尾为这些则不视为软换行，不拼接下一行）
_LINE_END_PUNCTUATION = set("。！？；，、．·.?!;:：,，！？；")


def _ends_with_punctuation(s: str) -> bool:
    """当前行是否以标点结尾（用于区分应保留的换行与 PDF 软换行）。"""
    t = (s or "").rstrip()
    if not t:
        return False
    return t[-1] in _LINE_END_PUNCTUATION


def _normalize_pdf_soft_line_breaks(text: str) -> str:
    """
    只把「不以标点结尾的换行」拼接到下一行，保留真正的段落大换行（\\n\\n）。
    用于 PDF 解析结果中因行宽产生的 mid-sentence 换行，避免误伤 Word 等已有正确段落的结构。
    """
    if not text or not text.strip():
        return text
    lines = text.split("\n")
    result: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 空行视为段落边界，原样保留
        if not line.strip():
            result.append("")
            i += 1
            continue
        buf = line
        j = i + 1
        # 若当前行不以标点结尾，则与后续非空行拼接，直到遇到以标点结尾的行或空行
        while j < len(lines) and lines[j].strip() and not _ends_with_punctuation(buf):
            buf = buf + lines[j]
            j += 1
        result.append(buf)
        i = j
    return "\n".join(result)


def chunk_lore_documents(
    lore_docs: List[Document],
    tokenizer: Callable[[str], List[str]],
) -> List[Document]:
    """
    对设定文档按章节/段落切分，并应用 256/512 token 规则，避免断句。

    规则概要：
    - 先按标题拆成章节；章节内若多段合计 <= 512 且 > 256 可保留整章，<= 256 则与后续短章合并到约 256。
    - 单段/单章 > 512 时按段落再拆；单段 256 < len <= 512 保留整段；单段 > 512 再按句号分句后按 256/512 成块。
    - 多段合并时以约 256 token 为目标；单块允许最大 512（单段或单章时）。
    """
    CHUNK_TARGET = 256
    CHUNK_MAX_SINGLE = 512

    def token_count(t: str) -> int:
        return len(tokenizer(t)) if t and t.strip() else 0

    result: List[Document] = []

    for doc in lore_docs:
        text = (doc.text or "").strip()
        if not text:
            continue
        meta = dict(doc.metadata or {})
        # 仅对 PDF 做软换行整合（不以标点结尾的换行拼成一行），Word 等保持原样
        file_path = meta.get("file_path") or meta.get("file_name") or ""
        if str(file_path).lower().endswith(".pdf"):
            text = _normalize_pdf_soft_line_breaks(text)

        sections = _split_by_headings(text)
        if not sections:
            sections = [text]

        # 第一轮：每个章节得到若干「候选块」（可能 <256，或 256~512，或 >512 再拆后的多块）
        candidates: List[str] = []

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            n = token_count(sec)
            if n <= CHUNK_MAX_SINGLE:
                candidates.append(sec)
                continue
            # 章节 > 512：按段落拆
            paras = [p.strip() for p in sec.split("\n\n") if p.strip()]
            for para in paras:
                if not para:
                    continue
                pn = token_count(para)
                if pn <= CHUNK_MAX_SINGLE:
                    candidates.append(para)
                    continue
                # 段落 > 512：按句号分句后成块
                sentences = _split_sentences(para)
                buf = ""
                for s in sentences:
                    sn = token_count(s)
                    if sn > CHUNK_MAX_SINGLE:
                        if buf.strip():
                            candidates.append(buf.strip())
                            buf = ""
                        candidates.append(s)
                        continue
                    if token_count(buf + "\n" + s if buf else s) <= CHUNK_TARGET:
                        buf = (buf + "\n" + s).strip() if buf else s
                        continue
                    if buf.strip():
                        candidates.append(buf.strip())
                    buf = s
                if buf.strip():
                    candidates.append(buf.strip())

        # 第二轮：合并过短的候选块到约 CHUNK_TARGET
        i = 0
        while i < len(candidates):
            chunk = candidates[i]
            n = token_count(chunk)
            if n >= CHUNK_TARGET:
                result.append(Document(text=chunk, metadata=meta))
                i += 1
                continue
            # 合并后续块直到 >= CHUNK_TARGET 或单块已 > CHUNK_MAX_SINGLE
            merged = chunk
            j = i + 1
            while j < len(candidates):
                next_block = candidates[j]
                merged_next = (merged + "\n\n" + next_block).strip()
                tn = token_count(merged_next)
                if tn > CHUNK_MAX_SINGLE:
                    break
                merged = merged_next
                j += 1
                if token_count(merged) >= CHUNK_TARGET:
                    break
            result.append(Document(text=merged, metadata=meta))
            i = j

    return result


def load_loading_documents() -> List[Document]:
    """
    读取 resources/data/stages/loading_data.xml 中的 loading 文本提示。

    仅向量化未被注释掉的 <Text> 内容；
    每条 <Text> 作为一个独立 Document，以便检索时直接返回短句。
    """

    _ensure_resources_dir()

    resources_dir = _get_resources_dir()
    stages_dir: Path = resources_dir / "data" / "stages"
    xml_path: Path = stages_dir / "loading_data.xml"

    if not xml_path.exists():
        print(f"[知识库] 未找到 loading 文本文件，跳过: {xml_path}")
        return []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as exc:
        print(f"[知识库] 解析 loading_data.xml 出错，跳过: {exc}")
        return []

    documents: List[Document] = []

    # 只读取未被注释掉的 <Text> 节点；xml 解析本身会自动忽略注释中的内容
    for group in root.findall(".//LoadingText/Group"):
        region_elem = group.find("Region")
        unlock_elem = group.find("Unlock")
        region = (region_elem.text or "").strip() if region_elem is not None else ""
        unlock_raw = (unlock_elem.text or "").strip() if unlock_elem is not None else ""

        for text_elem in group.findall("Text"):
            text = (text_elem.text or "").strip() if text_elem is not None else ""
            if not text:
                continue
            metadata: Dict[str, Any] = {
                "type": "loading_lore",
                "source_file": "loading_data.xml",
            }
            if region:
                metadata["region"] = region
            if unlock_raw:
                metadata["unlock"] = unlock_raw
            documents.append(Document(text=text, metadata=metadata))

    print(f"[知识库] loading 文本: {len(documents)} 条（来自 loading_data.xml）")
    return documents


def build_index(persist_dir: Path | None = None) -> VectorStoreIndex:
    """
    统一构建向量索引（对话、任务、世界观设定等）。
    若传入 persist_dir，则使用 StorageContext 构建并持久化到该目录。
    """

    ensure_embed_model(offline=True)

    from llama_index.core.text_splitter import TokenTextSplitter
    from llama_index.core import Settings

    import re
    _CJK_TOKEN_RE = re.compile(
        r'[a-zA-Z]+'       # 英文单词
        r'|\d+'             # 数字
        r'|[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]'  # CJK 汉字（每字一 token）
        r'|[^\s]'           # 其余非空白字符（标点等）
    )

    def cjk_tokenizer(text: str) -> list[str]:
        """中英文混合 tokenizer：英文按词、中文按字、标点逐个。"""
        tokens = _CJK_TOKEN_RE.findall(text)
        return tokens if tokens else ['']

    text_splitter = TokenTextSplitter(
        separator="\n",
        chunk_size=256,
        chunk_overlap=48,
        tokenizer=cjk_tokenizer,
        backup_separators=["。", "！", "？", ".", "!", "?", "；", "，", " "],
    )

    Settings.text_splitter = text_splitter

    dialogue_docs = load_dialogue_documents()
    task_docs = load_task_documents()

    try:
        lore_docs = load_lore_documents()
    except Exception as exc:
        print(f"[知识库] 加载世界观文档时出错，跳过: {exc}")
        lore_docs = []

    try:
        loading_docs = load_loading_documents()
    except Exception as exc:
        print(f"[知识库] 加载 loading 文本时出错，跳过: {exc}")
        loading_docs = []

    try:
        intel_docs = load_intelligence_documents()
    except Exception as exc:
        print(f"[知识库] 加载情报文件时出错，跳过: {exc}")
        intel_docs = []

    # 设定文档单独按章节/段落切分（256/512 token 规则），插入时用 TextNode 避免检索时报 "Node must be a TextNode to get text"
    lore_chunk_nodes: List[TextNode] = []
    if lore_docs:
        lore_chunk_docs = chunk_lore_documents(lore_docs, cjk_tokenizer)
        lore_chunk_nodes = [
            TextNode(text=(d.text or ""), metadata=dict(d.metadata or {}))
            for d in lore_chunk_docs
        ]
        print(f"[知识库] 设定文档切分: {len(lore_docs)} 个原文档 -> {len(lore_chunk_nodes)} 个块")

    all_docs: List[Document] = []
    all_docs.extend(dialogue_docs)
    all_docs.extend(task_docs)
    all_docs.extend(loading_docs)
    all_docs.extend(intel_docs)

    print(
        f"[知识库] 共加载 {len(all_docs)} 个文档 "
        f"(对话={len(dialogue_docs)}, 任务={len(task_docs)}, "
        f"设定块={len(lore_chunk_nodes)}, loading={len(loading_docs)}, 情报={len(intel_docs)})"
    )

    if not all_docs and not lore_chunk_nodes:
        raise ValueError("没有加载到任何 Document，无法构建索引。")

    # 设定文档已单独切块并以 TextNode 插入，不经过全局 256 token 切分；from_documents 仅接受 Document
    docs_for_index = all_docs if all_docs else [Document(text=" ", metadata={"type": "placeholder"})]
    if persist_dir is not None:
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex.from_documents(docs_for_index, storage_context=storage_context)
        if lore_chunk_nodes:
            index.insert_nodes(lore_chunk_nodes)
        storage_context.persist(persist_dir=str(persist_dir))
        print(f"[知识库] 索引已持久化到: {persist_dir}")
    else:
        index = VectorStoreIndex.from_documents(docs_for_index)
        if all_docs and lore_chunk_nodes:
            index.insert_nodes(lore_chunk_nodes)

    return index


# 索引缓存，避免每次请求都重建
_index_cache: VectorStoreIndex | None = None


def get_cached_index(force_rebuild: bool = False) -> VectorStoreIndex:
    """
    获取或构建并缓存索引。
    - 若 force_rebuild=True：清除缓存并强制重新构建，写入 resources/tools/vector_index。
    - 否则：若本地已存在有效索引则从 resources/tools/vector_index 加载；不存在则构建并持久化后再缓存。
    """
    global _index_cache

    persist_dir = get_vector_index_dir()

    if force_rebuild:
        _index_cache = None
        if persist_dir.exists():
            import shutil
            shutil.rmtree(persist_dir)
            print("[知识库] 已清除本地向量索引目录，准备重新构建")
        ensure_embed_model(offline=True)
        _index_cache = build_index(persist_dir=persist_dir)
        return _index_cache

    if _index_cache is not None:
        return _index_cache

    ensure_embed_model(offline=True)

    if _is_vector_index_valid(persist_dir):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
            _index_cache = load_index_from_storage(storage_context)
            print(f"[知识库] 已从本地加载向量索引: {persist_dir}")
            return _index_cache
        except Exception as e:
            print(f"[知识库] 从本地加载索引失败，将重新构建: {e}")
            _index_cache = None

    _index_cache = build_index(persist_dir=persist_dir)
    return _index_cache


def rebuild_vector_index() -> VectorStoreIndex:
    """
    强制重新构建知识库向量索引并持久化到 resources/tools/vector_index。
    供打包脚本或需要重建索引的场景调用。
    """
    return get_cached_index(force_rebuild=True)


# 核心设定文档文件名需包含的标识，用于判断是否允许重置知识库
CORE_LORE_DOC_MARKER = "核心设定与世界合理性补足"


def has_core_lore_document() -> bool:
    """
    检查 resources/docs 下是否存在文件名（不含扩展名）包含
    「核心设定与世界合理性补足」的文档（仅看 .pdf / .docx）。
    用于决定是否允许强制重置向量库。
    """
    try:
        _ensure_resources_dir()
    except FileNotFoundError:
        return False
    resources_dir = _get_resources_dir()
    docs_dir: Path = resources_dir / "docs"
    if not docs_dir.exists() or not docs_dir.is_dir():
        return False
    for f in docs_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (".pdf", ".docx"):
            if CORE_LORE_DOC_MARKER in f.stem:
                return True
    return False


def reset_knowledge_base() -> tuple[bool, str]:
    """
    按业务规则执行「重置知识库」：
    - 若 docs 中存在「核心设定与世界合理性补足」文档：强制重建向量库并覆盖，返回 (True, 成功说明)。
    - 若不存在该文档：
      - 若当前已有向量库：不允许重置，返回 (False, 数据文档不全错误说明)。
      - 若当前没有向量库：生成一次向量库，返回 (True, 成功说明)。

    Returns:
        (success, message) 供 API 返回给前端。
    """
    if has_core_lore_document():
        rebuild_vector_index()
        return True, "知识库已重置并重新生成。"
    persist_dir = get_vector_index_dir()
    if _is_vector_index_valid(persist_dir):
        return (
            False,
            "数据文档不全：未找到「核心设定与世界合理性补足」设定文档，且当前已有向量库，无法重置。"
            "请先放入该文档于 resources\docs 后再重置，或删除现有向量库目录后重新生成。",
        )
    # 无核心文档、且尚无向量库：生成一次
    ensure_embed_model(offline=True)
    get_cached_index()
    return True, "知识库已生成（未检测到核心设定文档，仅根据现有数据生成）。"




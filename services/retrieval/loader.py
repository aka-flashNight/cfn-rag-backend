"""索引构建语料加载：六类源 → Node 列表（对话/任务/世界观/loading/情报/实体）。

对应 docs/v3-developer/04-检索与向量模型.md §3.4。文档解析与切分逻辑照搬
ai_engine/game_data_loader.py 的现有实现（业务行为不变），仅把 LlamaIndex
Document 换成 retrieval.store.Node；PDF 用 pypdf、DOCX 用 python-docx 直接解析。
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, List

from services.game_data.paths import find_resources_directory
from services.retrieval.hybrid import cjk_tokenizer
from services.retrieval.store import Node, corpus_fingerprint

logger = logging.getLogger(__name__)

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

# 情报 TXT 的 @@@X_Y@@@ 分节标记
_INTEL_SECTION_RE = re.compile(r"@@@\d+(?:_\d+)?@@@")

# 任务配置文件黑名单：这些文件不在发行包内或为废弃数据，检索与向量化一律不命中。
# mercenary_tasks_old.json 在游戏打包配置中被 task/*_old.json 排除（其 25 条任务
# 已废弃、不在任何 list 引用链上），但本地 glob("*tasks*.json") 会误命中。
_TASK_FILE_DENYLIST: frozenset[str] = frozenset({"mercenary_tasks_old"})

# 情报正文分块参数：supp_intel 池对每条节点按 350 字符截断（SECTION_MAX_CHARS），
# 故块长必须压在阈值内，否则上下文会被切掉尾巴。
#   _CHUNK_SOFT  目标块长（尽量往这个长度靠）
#   _CHUNK_HARD  单块上限（超过即触发池内截断，必须 < 350）
_INTEL_CHUNK_SOFT = 280
_INTEL_CHUNK_HARD = 340

# h5 schema（intelligence_h5 / glossary）的纯展示字段，对大模型无信息量，解析时丢弃。
# 说明：文本内容散落在 text/content/title/label/note/entries/rows/items/fragments 等
# 字段中，由 _render_h5_blocks 按块类型还原，故这里只列出「确认无用」的字段。
_H5_DECORATIVE_BLOCK_TYPES: frozenset[str] = frozenset({"surfaceMark", "divider"})

# 核心设定文档文件名标识（「重置知识库」的业务判定）
CORE_LORE_DOC_MARKER = "核心设定与世界合理性补足"

# 设定文档扩展名：PDF/DOCX 需专门解析，MD/JSON 为纯文本直读
_LORE_DOC_SUFFIXES: tuple[str, ...] = (".pdf", ".docx", ".md", ".json")

# 参与语料指纹的源目录（相对 resources 根）；任一文件 mtime/size 变化即触发重建
_FINGERPRINT_SOURCE_DIRS: tuple[str, ...] = (
    "data/dialogues",
    "data/task",
    "docs/story",
    "data/intelligence",
    "data/intelligence_h5",
    "data/glossary",
    "data/stages",
    "data/items",
)


def _resources_dir() -> Path:
    return find_resources_directory()


# ---------------------------------------------------------------------------
# 语料指纹
# ---------------------------------------------------------------------------

def compute_corpus_fingerprint(resources_dir: Path | None = None) -> str:
    """
    扫描全部语料源文件，产出 (相对路径, mtime_ns, size) 集合哈希。

    设定文档的贡献随 resolve_lore_dir 的实际取值而定：docs/story 正常时按该目录
    全量计入；回退到 docs 顶层时只计入被采纳的核心设定文档，否则 docs 下 200+
    开发文档的任意改动都会误触发索引重建。
    """
    root = Path(resources_dir) if resources_dir else _resources_dir()
    states: list[tuple[str, int, int]] = []
    for rel_dir in _FINGERPRINT_SOURCE_DIRS:
        base = root / rel_dir
        if not base.exists():
            continue
        if rel_dir == "docs/story":
            resolved = resolve_lore_dir(root)
            if resolved is not None and resolved != base:
                # 回退到 docs 顶层：只纳入被采纳的核心设定文档
                for f in _list_lore_docs(resolved):
                    if CORE_LORE_DOC_MARKER not in f.stem:
                        continue
                    st = _stat_or_none(f)
                    if st is not None:
                        states.append((f.relative_to(root).as_posix(), st[0], st[1]))
                continue
        for f in base.rglob("*"):
            if f.is_file():
                st = _stat_or_none(f)
                if st is not None:
                    states.append((f.relative_to(root).as_posix(), st[0], st[1]))
    return corpus_fingerprint(states)


def _stat_or_none(f: Path) -> tuple[int, int] | None:
    """取 (mtime_ns, size)，stat 失败返回 None。"""
    try:
        stat = f.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


# ---------------------------------------------------------------------------
# 1. 日常对话 XML
# ---------------------------------------------------------------------------

def load_dialogue_nodes(resources_dir: Path | None = None) -> List[Node]:
    """读取 resources/data/dialogues 下的 NPC 日常对话 XML，每个 <Dialogue> 一条。"""
    root_dir = Path(resources_dir) if resources_dir else _resources_dir()
    dialogues_dir = root_dir / "data" / "dialogues"
    list_path = dialogues_dir / "list.xml"
    if not list_path.exists():
        logger.warning("未找到对话列表文件: %s", list_path)
        return []

    filenames = [
        (elem.text or "").strip()
        for elem in ET.parse(list_path).getroot().findall(".//items")
        if (elem.text or "").strip()
    ]

    nodes: List[Node] = []
    seq = 0
    for name in filenames:
        file_path = dialogues_dir / name
        if not file_path.exists():
            continue
        xml_root = ET.parse(file_path).getroot()
        file_level_name_elem = xml_root.find(".//Dialogues/Name")
        file_level_name = (
            file_level_name_elem.text.strip() if file_level_name_elem is not None else None
        )

        for dlg in xml_root.findall(".//Dialogues/Dialogue"):
            lines: List[str] = []
            character_key: str | None = None
            for sub in dlg.findall("./SubDialogue"):
                text_elem = sub.find("Text")
                text = (text_elem.text or "").strip() if text_elem is not None else ""
                if not text:
                    continue
                sub_name = (sub.find("Name").text or "").strip() if sub.find("Name") is not None else ""
                sub_char = (sub.find("Char").text or "").strip() if sub.find("Char") is not None else ""
                # 跳过玩家视角的台词（$PC / $PC_TITLE / $PC_CHAR 等）
                if sub_name == "$PC" or sub_char.startswith("$PC"):
                    continue
                # 角色标注使用 Name（角色名），缺省时回退文件级 Name 或 Char
                if character_key is None:
                    if sub_name:
                        character_key = sub_name
                    elif file_level_name:
                        character_key = file_level_name
                    elif sub_char:
                        character_key = sub_char.split("#")[0].strip()
                    else:
                        character_key = "Unknown"
                lines.append(text)

            if not lines or not character_key:
                continue
            seq += 1
            nodes.append(
                Node(
                    id=f"dialogue-{seq}",
                    text="\n".join(lines),
                    type="dialogue",
                    character=character_key.strip().lower(),  # 与检索端 npc_name 过滤一致
                    source_file=name,
                )
            )
    return nodes


# ---------------------------------------------------------------------------
# 2. 任务台词 JSON
# ---------------------------------------------------------------------------

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
    return name.lower() if name else None


def load_task_nodes(resources_dir: Path | None = None) -> List[Node]:
    """任务台词：*tasks*.json + text/*.json，单条 NPC 台词一节点（guide 类打 task_source）。"""
    root_dir = Path(resources_dir) if resources_dir else _resources_dir()
    task_dir = root_dir / "data" / "task"
    text_dir = task_dir / "text"
    if not task_dir.exists() or not text_dir.exists():
        return []

    # 合并所有 text/*.json 的 key；preview_text.json 仅用于任务预览，显式跳过
    text_data: dict[str, object] = {}
    for jpath in sorted(text_dir.glob("*.json")):
        if jpath.stem == "preview_text":
            continue
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                text_data.update(data)
        except Exception as exc:
            logger.warning("跳过任务文本 %s: %s", jpath.name, exc)

    all_tasks: List[dict] = []
    for jpath in sorted(task_dir.glob("*tasks*.json")):
        if jpath.stem in _TASK_FILE_DENYLIST:
            logger.info("跳过黑名单任务配置（不参与检索/向量化）: %s", jpath.name)
            continue
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
            tasks = data.get("tasks") if isinstance(data, dict) else []
            if not isinstance(tasks, list):
                continue
            # 教学引导类仅在高分时才采用，避免与 NPC 形象弱关联时混入
            task_source = "guide" if "guide" in jpath.name.lower() else None
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                t = dict(task)
                if task_source:
                    t["_task_source"] = task_source
                all_tasks.append(t)
        except Exception as exc:
            logger.warning("跳过任务配置 %s: %s", jpath.name, exc)

    nodes: List[Node] = []
    seq = 0
    for task in all_tasks:
        task_source: str | None = task.get("_task_source")
        for key in (task.get("get_conversation"), task.get("finish_conversation")):
            if not key:
                continue
            raw = text_data.get(str(key))
            if not isinstance(raw, list):
                continue
            for item in raw:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                character = _task_character_from_item(item)
                if not text or not character:
                    continue
                seq += 1
                nodes.append(
                    Node(
                        id=f"task-{seq}",
                        text=text,
                        type="task",
                        character=character,
                        task_source=task_source,
                    )
                )
    return nodes


# ---------------------------------------------------------------------------
# 3. 情报 / 术语（h5 JSON schema，缺失时回退 legacy TXT）
# ---------------------------------------------------------------------------

def _h5_inline_text(obj: object) -> str:
    """
    递归取出 h5 content 结构里的可见文本。

    content 项形如 {"type":"text","text":"..."}、
    {"type":"colorToken","token":"...","content":[{"type":"text","text":"..."}]}、
    {"type":"strong","content":[...]} —— 各种装饰类型（strong/underline/colorToken/
    damageText/decryptText/outburst）只是排版样式，文本都在 text 或嵌套 content 里，
    故统一递归取 text，样式标记（type/token/tone/…）一律丢弃。
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return "".join(_h5_inline_text(x) for x in obj)
    if isinstance(obj, dict):
        if isinstance(obj.get("text"), str):
            return obj["text"]
        if "content" in obj:
            return _h5_inline_text(obj["content"])
    return ""


def _render_h5_blocks(blocks: object) -> List[str]:
    """
    把 h5 blocks 还原为纯文本行，按块类型保留有语义的结构，丢弃排版信息。

    保留：段落/标题正文、表格（表头 + 行，用「 | 」分隔）、列表项、时间线条目
    （「标签｜内容」）、终端日志、图纸材料、解密块、标注、手写注记等。
    丢弃：x/y/w/h/rotate/opacity/layout/variant/tone/status/token/level 等纯展示字段，
    以及 surfaceMark / divider 这类无文字装饰块。
    """
    lines: List[str] = []
    if not isinstance(blocks, list):
        return lines
    for b in blocks:
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype in _H5_DECORATIVE_BLOCK_TYPES:
            continue

        if btype == "table":
            cols = b.get("columns") or []
            if cols:
                lines.append(" | ".join(str(c).strip() for c in cols))
            for row in b.get("rows") or []:
                cells = [_h5_inline_text(c).strip() for c in (row or [])]
                lines.append(" | ".join(cells))
            continue

        if btype == "list":
            for item in b.get("items") or []:
                s = _h5_inline_text(item).strip()
                if s:
                    lines.append(s)
            continue

        if btype == "timeline":
            for e in b.get("entries") or []:
                if not isinstance(e, dict):
                    continue
                s = _h5_inline_text(e.get("content")).strip()
                lab = str(e.get("label") or "").strip()
                if s:
                    lines.append(f"{lab}｜{s}" if lab else s)
            continue

        if btype == "terminalLog":
            title = _h5_inline_text(b.get("title")).strip()
            if title:
                lines.append(title)
            for e in b.get("entries") or []:
                s = _h5_inline_text(e.get("content") if isinstance(e, dict) else e).strip()
                if s:
                    lines.append(s)
            continue

        if btype == "blueprint":
            title = _h5_inline_text(b.get("title")).strip()
            if title:
                lines.append(title)
            for m in b.get("materials") or []:
                s = _h5_inline_text(m).strip()
                if s:
                    lines.append(s)
            continue

        if btype == "hardwareExtract":
            label = _h5_inline_text(b.get("label")).strip()
            if label:
                lines.append(label)
            for s in b.get("steps") or []:
                s = str(s).strip()
                if s:
                    lines.append(s)
            lines.extend(_render_h5_blocks(b.get("reveal")))
            continue

        if btype == "decryptBlock":
            label = _h5_inline_text(b.get("label")).strip()
            if label:
                lines.append(label)
            lines.extend(_render_h5_blocks(b.get("plain")))
            continue

        if btype == "paperStage":
            for frag in b.get("fragments") or []:
                s = _h5_inline_text((frag or {}).get("content") if isinstance(frag, dict) else frag).strip()
                if s:
                    lines.append(s)
            continue

        if btype == "annotation":
            s = _h5_inline_text(b.get("content")).strip()
            if s:
                lines.append(s)
            note = _h5_inline_text(b.get("note")).strip()
            if note:
                lines.append(note)
            continue

        # paragraph / heading / note / quote / stamp / handwritten 等：正文 + 兜底标题
        s = _h5_inline_text(b.get("content")).strip()
        if not s:
            s = _h5_inline_text(b.get("title")).strip()
        if s:
            lines.append(s)
    return lines


def _h5_document_text(doc: dict) -> str:
    """把一个 h5 文档（pages[].blocks[]）还原为整篇纯文本。"""
    lines: List[str] = []
    for page in doc.get("pages") or []:
        if not isinstance(page, dict):
            continue
        lines.extend(_render_h5_blocks(page.get("blocks")))
    return "\n".join(line for line in lines if line.strip())


def _split_long_sentence(sentence: str, limit: int) -> List[str]:
    """超长句按次级标点（；，、：）就近切分；实在无标点才硬切，尽量不断在句中。"""
    if len(sentence) <= limit:
        return [sentence]
    out: List[str] = []
    rest = sentence
    secondary = "；;，,、：:）)"
    while len(rest) > limit:
        cut = -1
        for i in range(limit, max(limit // 2, 1) - 1, -1):
            if rest[i - 1] in secondary:
                cut = i
                break
        if cut <= 0:
            cut = limit
        out.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    if rest:
        out.append(rest)
    return [x for x in out if x]


def _chunk_plain_text(text: str, soft: int, hard: int) -> List[str]:
    """
    把正文切成 soft~hard 字符的块，只在句末断开，避免断在句中。

    切分优先级：空行/换行 → 句末（。！？…）→ 次级标点（；，）→ 硬切。
    单块保证不超过 hard，从而不会被池的 max_chars 截断。
    """
    sentences: List[str] = []
    for para in (text or "").split("\n"):
        para = para.strip()
        if not para:
            continue
        for sent in re.findall(r"[^。！？!?…]+[。！？!?…]+|[^。！？!?…]+", para):
            sent = sent.strip()
            if sent:
                sentences.extend(_split_long_sentence(sent, hard))

    chunks: List[str] = []
    buf = ""
    for sent in sentences:
        if not buf:
            buf = sent
        elif len(buf) + len(sent) <= soft:
            buf += sent
        else:
            chunks.append(buf)
            buf = sent
    if buf:
        chunks.append(buf)

    # 尾块过短时向前合并（合并后不得超 hard），避免产生无意义的碎片
    if len(chunks) >= 2 and len(chunks[-1]) < hard - len(chunks[-2]) and len(chunks[-2]) + len(chunks[-1]) <= hard:
        chunks[-2] = chunks[-2] + chunks[-1]
        chunks.pop()
    return chunks


def _load_intelligence_from_h5(h5_dir: Path) -> List[Node]:
    """解析 data/intelligence_h5/*.json（游戏当前权威格式）。"""
    nodes: List[Node] = []
    for path in sorted(h5_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("跳过情报 h5 %s: %s", path.name, exc)
            continue
        if not isinstance(doc, dict):
            continue
        name = str(doc.get("itemName") or path.stem)
        text = _h5_document_text(doc)
        for chunk in _chunk_plain_text(text, _INTEL_CHUNK_SOFT, _INTEL_CHUNK_HARD):
            nodes.append(
                Node(id=f"intel-{len(nodes) + 1}", text=chunk, type="intelligence", source_file=name)
            )
    return nodes


def _load_intelligence_from_txt(txt_dir: Path) -> List[Node]:
    """解析 legacy data/intelligence/*.txt（@@@X_Y@@@ 分节），仅作 h5 缺失时的兜底。"""
    nodes: List[Node] = []
    for path in sorted(txt_dir.glob("*.txt")):
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            continue
        sections = _INTEL_SECTION_RE.split(content) if _INTEL_SECTION_RE.search(content) else [content]
        stem = path.stem
        for sec in sections:
            for chunk in _chunk_plain_text(sec.strip(), _INTEL_CHUNK_SOFT, _INTEL_CHUNK_HARD):
                nodes.append(
                    Node(id=f"intel-{len(nodes) + 1}", text=chunk, type="intelligence", source_file=stem)
                )
    return nodes


def load_intelligence_nodes(resources_dir: Path | None = None) -> List[Node]:
    """
    情报语料：优先 data/intelligence_h5（游戏当前权威格式，结构化正文），
    h5 目录缺失或无有效文档时回退 data/intelligence（legacy TXT）。

    两者内容有重叠（h5 由 txt 生成后又经增强），同时纳入会在 supp_intel 池内
    造成同主题重复召回，故按「二选一」处理。
    """
    root_dir = Path(resources_dir) if resources_dir else _resources_dir()
    h5_dir = root_dir / "data" / "intelligence_h5"
    txt_dir = root_dir / "data" / "intelligence"

    if h5_dir.is_dir() and any(h5_dir.glob("*.json")):
        nodes = _load_intelligence_from_h5(h5_dir)
        if nodes:
            logger.info("情报语料来源: intelligence_h5（h5 优先），%d 块", len(nodes))
            return nodes
        logger.warning("intelligence_h5 存在但未解析出内容，回退 legacy TXT")

    if txt_dir.is_dir():
        nodes = _load_intelligence_from_txt(txt_dir)
        logger.info("情报语料来源: intelligence（legacy TXT 兜底），%d 块", len(nodes))
        return nodes

    logger.warning("情报目录不存在（h5 与 legacy 均缺失），跳过")
    return []


def load_glossary_nodes(resources_dir: Path | None = None) -> List[Node]:
    """
    名词术语表 data/glossary/*.json → world_lore 节点（一条术语一个节点）。

    术语是权威世界观定义，与核心设定同属「世界观设定」，故走 world_lore 池
    （不截断，可保留完整多页定义）；若放 supp_intel 会因 350 字符逐条截断而
    切掉多页补录。glossary_index.json 只是目录，不含正文，跳过。
    """
    root_dir = Path(resources_dir) if resources_dir else _resources_dir()
    glossary_dir = root_dir / "data" / "glossary"
    if not glossary_dir.is_dir():
        logger.warning("术语表目录不存在，跳过: %s", glossary_dir)
        return []

    nodes: List[Node] = []
    for path in sorted(glossary_dir.glob("*.json")):
        if path.stem == "glossary_index":
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("跳过术语 %s: %s", path.name, exc)
            continue
        if not isinstance(doc, dict):
            continue
        term = str(doc.get("termName") or path.stem)
        display = str(doc.get("displayName") or term)
        body = _h5_document_text(doc)
        if not body.strip():
            continue
        # 首行带上术语名（含别名），便于检索命中与模型辨识。
        # displayName 通常已含 termName（如「军阀（莱昂利亚自由革命军）」），避免重复拼接。
        if term in display:
            header = display
        elif display and display != term:
            header = f"{term}（{display}）"
        else:
            header = term
        nodes.append(
            Node(
                id=f"glossary-{len(nodes) + 1}",
                text=f"{header}\n{body}",
                type="world_lore",
                source_file=term,
            )
        )
    if nodes:
        logger.info("术语表加载: %d 条（world_lore）", len(nodes))
    return nodes


# ---------------------------------------------------------------------------
# 4. 世界观 PDF/DOCX/MD/JSON（含按标题/段落/句子的 256/512 token 切分）
# ---------------------------------------------------------------------------

def _read_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx_text(path: Path) -> str:
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                parts.append("\t".join(cells))
    return "\n".join(parts)


def _read_plain_text(path: Path) -> str:
    """读取纯文本类文档（md/json）。按 utf-8 → utf-8-sig → gbk 依次尝试解码。

    游戏项目文档多为 UTF-8，但存在带 BOM 与少量 GBK 编码的文件，逐级回退避免乱码。
    """
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    # 全部失败时用 utf-8 忽略错误字节兜底，好过整体丢弃该文件
    return raw.decode("utf-8", errors="ignore")


def _read_lore_text(path: Path) -> str:
    """按扩展名分派设定文档解析。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf_text(path)
    if suffix == ".docx":
        return _read_docx_text(path)
    return _read_plain_text(path)


def _list_lore_docs(docs_dir: Path) -> List[Path]:
    """列出目录下可解析的设定文档，按文件名排序。"""
    if not docs_dir.is_dir():
        return []
    return sorted(
        f for f in docs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in _LORE_DOC_SUFFIXES
    )


def resolve_lore_dir(root_dir: Path) -> Path | None:
    """
    定位设定文档目录：优先 docs/story；其中无设定文档时回退到 docs 顶层。

    回退到 docs 顶层属容错设计——游戏项目 docs/ 下混有大量开发文档（ADR、迁移
    记录、技术备忘等），因此回退时只采纳命中核心设定命名的文档，不采纳 docs 顶层
    的其他文档，避免把开发文档灌进世界观池。
    """
    story_dir = root_dir / "docs" / "story"
    if _list_lore_docs(story_dir):
        return story_dir

    docs_dir = root_dir / "docs"
    fallback = [f for f in _list_lore_docs(docs_dir) if CORE_LORE_DOC_MARKER in f.stem]
    if fallback:
        logger.warning(
            "docs/story 下无设定文档，回退到 docs 顶层并仅采纳核心设定文档: %s",
            docs_dir,
        )
        return docs_dir
    return None


def load_lore_nodes(resources_dir: Path | None = None) -> List[Node]:
    """
    读取世界观设定文档，按文件名区分核心/补充设定。

    目录选择见 resolve_lore_dir：优先 docs/story；docs/story 不存在或其中没有
    设定文档时，回退到 docs 顶层且只取命中核心设定命名的文档。
    """
    root_dir = Path(resources_dir) if resources_dir else _resources_dir()
    docs_dir = resolve_lore_dir(root_dir)
    if docs_dir is None:
        logger.warning("未找到世界观设定文档（docs/story 与 docs 顶层均无），跳过")
        return []

    # 回退到 docs 顶层时必须只保留命中核心设定命名的文档
    in_fallback = docs_dir.name != "story"
    matching_files = _list_lore_docs(docs_dir)
    if in_fallback:
        matching_files = [f for f in matching_files if CORE_LORE_DOC_MARKER in f.stem]
    if not matching_files:
        logger.warning("设定文档目录中无可解析文件，跳过世界观文档加载: %s", docs_dir)
        return []

    nodes: List[Node] = []
    for f in matching_files:
        try:
            text = _read_lore_text(f)
        except Exception as exc:
            logger.warning("解析设定文档 %s 失败，跳过: %s", f.name, exc)
            continue
        if not (text or "").strip():
            continue
        doc_type = "world_lore" if CORE_LORE_DOC_MARKER in f.stem else "supplementary_lore"
        nodes.append(
            Node(id=f"lore-raw-{len(nodes) + 1}", text=text, type=doc_type, source_file=f.name)
        )
    return nodes
    return nodes


def _split_by_headings(text: str) -> List[str]:
    """按标题行将文本拆成多个章节（Markdown #、一、二、、1. 2.、（一）等）。"""
    if not (text or "").strip():
        return []
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not blocks:
        return [text.strip()] if text.strip() else []

    sections: List[str] = []
    current: List[str] = []
    for block in blocks:
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
    """按。！？；分句，保留边界完整（不 mid-sentence 截断）。"""
    if not text or not text.strip():
        return []
    parts = re.split(r"([。！？；])", text)
    sentences: List[str] = []
    buf = ""
    for p in parts:
        buf += p
        if p.strip() in "。！？；" and buf.strip():
            sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())
    return sentences


_LINE_END_PUNCTUATION = set("。！？；，、．·.?!;:：,，！？；")


def _ends_with_punctuation(s: str) -> bool:
    t = (s or "").rstrip()
    return bool(t) and t[-1] in _LINE_END_PUNCTUATION


def _normalize_pdf_soft_line_breaks(text: str) -> str:
    """只把「不以标点结尾的换行」拼接到下一行，保留真正的段落大换行（PDF 行宽换行修正）。"""
    if not text or not text.strip():
        return text
    lines = text.split("\n")
    result: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            result.append("")
            i += 1
            continue
        buf = line
        j = i + 1
        while j < len(lines) and lines[j].strip() and not _ends_with_punctuation(buf):
            buf = buf + lines[j]
            j += 1
        result.append(buf)
        i = j
    return "\n".join(result)


def chunk_lore_nodes(lore_nodes: List[Node], tokenizer: Callable[[str], List[str]]) -> List[Node]:
    """设定文档按章节/段落切分并应用 256/512 token 规则，避免断句（沿用旧实现）。"""
    CHUNK_TARGET = 256
    CHUNK_MAX_SINGLE = 512

    def token_count(t: str) -> int:
        return len(tokenizer(t)) if t and t.strip() else 0

    result: List[Node] = []
    for raw in lore_nodes:
        text = (raw.text or "").strip()
        if not text:
            continue
        # 仅对 PDF 做软换行整合；Word 等已有正确段落结构保持原样
        if (raw.source_file or "").lower().endswith(".pdf"):
            text = _normalize_pdf_soft_line_breaks(text)

        sections = _split_by_headings(text) or [text]
        candidates: List[str] = []

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            if token_count(sec) <= CHUNK_MAX_SINGLE:
                candidates.append(sec)
                continue
            # 章节 > 512：按段落拆
            for para in (p.strip() for p in sec.split("\n\n")):
                if not para:
                    continue
                if token_count(para) <= CHUNK_MAX_SINGLE:
                    candidates.append(para)
                    continue
                # 段落 > 512：按句号分句后成块
                buf = ""
                for s in _split_sentences(para):
                    if token_count(s) > CHUNK_MAX_SINGLE:
                        if buf.strip():
                            candidates.append(buf.strip())
                            buf = ""
                        candidates.append(s)
                        continue
                    merged = (buf + "\n" + s).strip() if buf else s
                    if token_count(merged) <= CHUNK_TARGET:
                        buf = merged
                        continue
                    if buf.strip():
                        candidates.append(buf.strip())
                    buf = s
                if buf.strip():
                    candidates.append(buf.strip())

        # 合并过短的候选块到约 CHUNK_TARGET
        i = 0
        while i < len(candidates):
            chunk = candidates[i]
            if token_count(chunk) >= CHUNK_TARGET:
                result.append(Node(id="", text=chunk, type=raw.type, source_file=raw.source_file))
                i += 1
                continue
            merged = chunk
            j = i + 1
            while j < len(candidates):
                merged_next = (merged + "\n\n" + candidates[j]).strip()
                if token_count(merged_next) > CHUNK_MAX_SINGLE:
                    break
                merged = merged_next
                j += 1
                if token_count(merged) >= CHUNK_TARGET:
                    break
            result.append(Node(id="", text=merged, type=raw.type, source_file=raw.source_file))
            i = j

    for k, node in enumerate(result):
        node.id = f"lore-{k + 1}"
    return result


# ---------------------------------------------------------------------------
# 5. loading 文案 XML
# ---------------------------------------------------------------------------

def load_loading_nodes(resources_dir: Path | None = None) -> List[Node]:
    root_dir = Path(resources_dir) if resources_dir else _resources_dir()
    xml_path = root_dir / "data" / "stages" / "loading_data.xml"
    if not xml_path.exists():
        logger.warning("未找到 loading 文本文件，跳过: %s", xml_path)
        return []
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as exc:
        logger.warning("解析 loading_data.xml 出错，跳过: %s", exc)
        return []

    nodes: List[Node] = []
    for group in root.findall(".//LoadingText/Group"):
        region_elem = group.find("Region")
        unlock_elem = group.find("Unlock")
        region = (region_elem.text or "").strip() if region_elem is not None else ""
        unlock_raw = (unlock_elem.text or "").strip() if unlock_elem is not None else ""
        for text_elem in group.findall("Text"):
            text = (text_elem.text or "").strip() if text_elem is not None else ""
            if not text:
                continue
            nodes.append(
                Node(
                    id=f"loading-{len(nodes) + 1}",
                    text=text,
                    type="loading_lore",
                    source_file="loading_data.xml",
                    region=region or None,
                    unlock=unlock_raw or None,
                )
            )
    return nodes


# ---------------------------------------------------------------------------
# 6. 物品/关卡结构化实体（整实体一条）
# ---------------------------------------------------------------------------

def load_game_entity_nodes(resources_dir: Path | None = None) -> List[Node]:
    """物品与关卡各建一条向量节点；构建索引时直接实例化 Registry 读盘。"""
    from services.game_data.item_registry import ItemRegistry
    from services.game_data.paths import get_game_data_root
    from services.game_data.stage_registry import StageRegistry
    from services.game_entity_prompts import format_item_embedding_text, format_stage_embedding_text

    root = get_game_data_root()
    items = ItemRegistry(data_root=root)
    items.load()
    stages = StageRegistry(data_root=root)
    stages.load()

    nodes: List[Node] = []
    for it in items.items:
        text = format_item_embedding_text(it)
        if not (text or "").strip():
            continue
        nodes.append(
            Node(id=f"game-item-{len(nodes) + 1}", text=text, type="game_item", item_name=it.name)
        )
    for si in stages._stage_infos.values():
        text = format_stage_embedding_text(si)
        nodes.append(
            Node(
                id=f"game-stage-{len(nodes) + 1}",
                text=text,
                type="game_stage",
                stage_area=si.area,
                stage_name=si.name,
                entity_key=f"{si.area}::{si.name}",
            )
        )
    return nodes


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def load_corpus(resources_dir: Path | None = None) -> List[Node]:
    """加载全部六类语料（世界观文档单独按 256/512 token 规则切块）。"""
    root_dir = Path(resources_dir) if resources_dir else _resources_dir()

    nodes: List[Node] = []
    nodes.extend(load_dialogue_nodes(root_dir))
    nodes.extend(load_task_nodes(root_dir))
    try:
        lore_raw = load_lore_nodes(root_dir)
    except Exception as exc:
        logger.warning("加载世界观文档时出错，跳过: %s", exc)
        lore_raw = []
    try:
        nodes.extend(load_loading_nodes(root_dir))
    except Exception as exc:
        logger.warning("加载 loading 文本时出错，跳过: %s", exc)
    try:
        nodes.extend(load_intelligence_nodes(root_dir))
    except Exception as exc:
        logger.warning("加载情报文件时出错，跳过: %s", exc)
    try:
        nodes.extend(load_glossary_nodes(root_dir))
    except Exception as exc:
        logger.warning("加载术语表时出错，跳过: %s", exc)
    try:
        nodes.extend(load_game_entity_nodes(root_dir))
    except Exception as exc:
        logger.warning("加载游戏实体向量文档时出错，跳过: %s", exc)

    if lore_raw:
        lore_chunks = chunk_lore_nodes(lore_raw, cjk_tokenizer)
        logger.info("设定文档切分: %d 个原文档 -> %d 个块", len(lore_raw), len(lore_chunks))
        nodes.extend(lore_chunks)

    logger.info(
        "语料加载完成: %d 条节点（对话=%d 任务=%d 世界观块=%d loading=%d 情报=%d 实体=%d）",
        len(nodes),
        sum(1 for n in nodes if n.type == "dialogue"),
        sum(1 for n in nodes if n.type == "task"),
        sum(1 for n in nodes if n.type in ("world_lore", "supplementary_lore")),
        sum(1 for n in nodes if n.type == "loading_lore"),
        sum(1 for n in nodes if n.type == "intelligence"),
        sum(1 for n in nodes if n.type in ("game_item", "game_stage")),
    )
    if not nodes:
        raise ValueError("没有加载到任何语料节点，无法构建检索索引。")
    return nodes


def has_core_lore_document(resources_dir: Path | None = None) -> bool:
    """是否存在「核心设定与世界合理性补足」文档（重置知识库的业务判定）。

    目录选择与 load_lore_nodes 保持一致（docs/story 优先，可回退 docs 顶层），
    否则回退场景下会误判为「无核心设定」而触发多余的知识库重置。
    """
    try:
        root_dir = Path(resources_dir) if resources_dir else _resources_dir()
    except FileNotFoundError:
        return False
    docs_dir = resolve_lore_dir(root_dir)
    if docs_dir is None:
        return False
    return any(CORE_LORE_DOC_MARKER in f.stem for f in _list_lore_docs(docs_dir))

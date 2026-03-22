import os
import sys
import json
import xml.etree.ElementTree as ET
import re
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.game_data.paths import RESOURCE_FOLDER_NAMES, pick_existing_or_default_resource_root

# ================= 路径获取函数 =================
def is_packaged_environment() -> bool:
    """检查是否在PyInstaller打包环境中运行"""
    return hasattr(sys, '_MEIPASS') or getattr(sys, 'frozen', False)


def get_resources_dir() -> Path:
    """
    获取 resources 目录的路径。
    """
    # 获取基础目录
    if is_packaged_environment():
        # 打包环境：exe 所在目录
        base_dir = Path(os.path.dirname(sys.executable))
    else:
        # 开发环境：当前文件所在目录的父目录的父目录
        script_dir = Path(__file__).resolve().parent
        base_dir = script_dir.parent.parent
    
    # 情况 1: base_dir/<resources|CrazyFlashNight>
    for name in RESOURCE_FOLDER_NAMES:
        resources_path = base_dir / name
        if resources_path.exists():
            return resources_path

    # 情况 2–3: 当前目录及其父目录
    cwd = Path(os.getcwd()).resolve()
    for name in RESOURCE_FOLDER_NAMES:
        p = cwd / name
        if p.exists():
            return p
        p = cwd.parent / name
        if p.exists():
            return p

    # 若均不存在：有 resources 用 resources；仅有 CrazyFlashNight 等别名时建在别名下；都没有则新建 resources
    resources_path = pick_existing_or_default_resource_root(base_dir)
    resources_path.mkdir(parents=True, exist_ok=True)
    return resources_path


def get_data_dir(resources_dir: Path) -> Path:
    """
    获取 data 目录路径
    """
    return resources_dir / "data"


def get_output_db_path(resources_dir: Path) -> Path:
    """
    获取 npc_state_db.json 的输出路径
    """
    db_dir = resources_dir / "data" / "rag"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "npc_state_db.json"

# ================= 配置区域 =================
def get_config_paths():
    """
    获取所有配置文件的路径
    """
    resources_dir = get_resources_dir()
    data_dir = get_data_dir(resources_dir)
    
    # 1. 日常对话配置
    dialogues_dir = data_dir / "dialogues"
    dialogue_list_file = dialogues_dir / "list.xml"
    
    # 2. 任务对话配置
    task_text_dir = data_dir / "task" / "text"
    task_list_file = task_text_dir / "list.xml"
    
    # 3. 输出数据库配置
    output_db_file = get_output_db_path(resources_dir)
    
    return {
        'resources_dir': resources_dir,
        'data_dir': data_dir,
        'dialogues_dir': dialogues_dir,
        'dialogue_list_file': dialogue_list_file,
        'task_text_dir': task_text_dir,
        'task_list_file': task_list_file,
        'output_db_file': output_db_file
    }
# ===========================================


def parse_name_and_emotion(char_string):
    """
    解析字符字段，提取名字和情绪。
    格式示例："Andy Law#微笑" 或 "冷兵器商人"
    """
    if not char_string:
        return None, None
    
    char_string = char_string.strip()
    if '#' in char_string:
        parts = char_string.split('#', 1)
        name = parts[0].strip()
        emotion = parts[1].strip()
    else:
        name = char_string
        emotion = "普通"
    
    return name, emotion


def extract_faction_from_filename(filename):
    """
    从对话 XML 文件名提取阵营信息
    例如：npc_dialogue_A 兵团.xml -> "A 兵团"
    """
    pattern = r'npc_dialogue_(.+)\.xml$'
    match = re.match(pattern, filename)
    if match:
        return match.group(1).strip()
    return ""


def is_a兵团_faction(faction):
    """
    判断是否为 A 兵团相关阵营
    """
    return faction in ["A 兵团", "A 兵团元老"]


def extract_from_dialogue_xml(file_path, npc_data, filename):
    """
    解析对话 XML 文件 (如 npc_dialogue_shop.xml)
    提取所有 <Char> 标签的内容和 <Title> 标签的内容
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # 从文件名提取阵营
        faction = extract_faction_from_filename(filename)
        
        # 遍历所有 SubDialogue 节点
        for sub_dialogue in root.iter('SubDialogue'):
            name = None
            title = None
            char_text = None
            
            # 提取 Name、Title、Char
            for child in sub_dialogue:
                if child.tag == 'Name' and child.text:
                    name = child.text.strip()
                elif child.tag == 'Title' and child.text:
                    title = child.text.strip()
                elif child.tag == 'Char' and child.text:
                    char_text = child.text.strip()
            
            # 优先使用 Char 中的名字（可能包含情绪）
            if char_text:
                char_name, emotion = parse_name_and_emotion(char_text)
                if char_name:
                    name = char_name
                    npc_data[name]['emotions'].add(emotion)
                    if faction:
                        npc_data[name]['faction'] = faction
                    if title:
                        npc_data[name]['titles'].add(title)
            elif name:
                # 如果没有 Char，使用 Name
                npc_data[name]['emotions'].add("普通")
                if faction:
                    npc_data[name]['faction'] = faction
                if title:
                    npc_data[name]['titles'].add(title)
                    
    except Exception as e:
        print(f"[Error] 解析对话 XML {file_path} 失败：{e}")


def extract_from_task_json(file_path, npc_data):
    """
    解析任务 JSON 文件 (如 challenge_text.json)
    遍历 JSON 结构，查找包含 'char' 和 'title' 键的字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 遍历 JSON 根目录下的所有值
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        # 提取 char 和 title
                        char_val = item.get('char', '')
                        title_val = item.get('title', '')
                        
                        if char_val:
                            name, emotion = parse_name_and_emotion(char_val)
                            if name:
                                npc_data[name]['emotions'].add(emotion)
                                if title_val:
                                    npc_data[name]['titles'].add(title_val)
    except Exception as e:
        print(f"[Error] 解析任务 JSON {file_path} 失败：{e}")


def get_file_list_from_xml(list_file_path, tag_name):
    """
    读取 list.xml 获取文件名列表
    """
    filenames = []
    try:
        tree = ET.parse(list_file_path)
        root = tree.getroot()
        for elem in root.iter(tag_name):
            if elem.text:
                filenames.append(elem.text.strip())
    except Exception as e:
        print(f"[Error] 解析列表文件 {list_file_path} 失败：{e}")
    return filenames


def determine_sex(name):
    """
    根据名字判断性别
    包含 'girl' 为女，否则为空字符串
    """
    if 'girl' in name.lower():
        return "女"
    return ""


def get_default_favorability_and_relationship(faction):
    """
    根据阵营获取默认好感度和关系等级
    """
    if is_a兵团_faction(faction):
        return 25, "熟悉"
    else:
        return 10, "陌生"


def load_existing_db(output_db_file):
    """
    加载已存在的数据库
    """
    if output_db_file.exists():
        try:
            with open(output_db_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] 加载现有数据库失败：{e}，将创建新数据库")
    return {}


def merge_npc_data(existing_db, new_npc_data):
    """
    合并现有数据库和新数据
    只补充缺失的属性，不覆盖已有值
    """
    merged_db = {}
    
    # 处理现有 NPC
    for name, existing_data in existing_db.items():
        merged_db[name] = existing_data.copy()
        
        # 如果该 NPC 也在新数据中，补充缺失属性
        if name in new_npc_data:
            new_data = new_npc_data[name]
            
            # 补充 emotions
            if 'emotions' in merged_db[name]:
                existing_emotions = set(merged_db[name]['emotions'])
                existing_emotions.update(new_data['emotions'])
                merged_db[name]['emotions'] = sorted(list(existing_emotions))
            else:
                merged_db[name]['emotions'] = sorted(list(new_data['emotions']))
            
            # 补充 titles（如果不存在则添加）
            if 'titles' not in merged_db[name] or not merged_db[name]['titles']:
                merged_db[name]['titles'] = sorted(list(new_data['titles']))
            else:
                existing_titles = set(merged_db[name]['titles'])
                existing_titles.update(new_data['titles'])
                merged_db[name]['titles'] = sorted(list(existing_titles))
            
            # 补充 faction（如果为空则添加）
            if new_data['faction']:
                merged_db[name]['faction'] = new_data['faction']
            
            # 补充 sex（如果为空则添加）
            if 'sex' not in merged_db[name] or not merged_db[name]['sex']:
                merged_db[name]['sex'] = new_data['sex']
            
            # 补充 favorability（如果是默认值 0 则更新）
            if 'favorability' not in merged_db[name] or merged_db[name]['favorability'] == 0:
                faction = merged_db[name].get('faction', '')
                default_fav, _ = get_default_favorability_and_relationship(faction)
                merged_db[name]['favorability'] = default_fav
            
            # 补充 relationship_level（如果是默认值"陌生"且阵营是 A 兵团则更新）
            if 'relationship_level' not in merged_db[name] or merged_db[name]['relationship_level'] == "陌生":
                faction = merged_db[name].get('faction', '')
                _, default_rel = get_default_favorability_and_relationship(faction)
                # 只有当阵营是 A 兵团时才更新为"熟悉"
                if is_a兵团_faction(faction):
                    merged_db[name]['relationship_level'] = default_rel
    
    # 处理新 NPC（现有数据库中不存在的）
    for name, new_data in new_npc_data.items():
        if name not in merged_db:
            faction = new_data['faction']
            default_fav, default_rel = get_default_favorability_and_relationship(faction)
            
            merged_db[name] = {
                "favorability": default_fav,
                "relationship_level": default_rel,
                "sex": new_data['sex'],
                "emotions": sorted(list(new_data['emotions'])),
                "titles": sorted(list(new_data['titles'])),
                "faction": faction
            }
    
    return merged_db


def main():
    print("--- 开始更新 NPC 状态数据库 ---")
    
    # 获取所有路径配置
    config = get_config_paths()
    print(f"Resources 目录：{config['resources_dir']}")
    print(f"输出数据库：{config['output_db_file']}")
    
    # 加载现有数据库（如果存在）
    existing_db = load_existing_db(config['output_db_file'])
    if existing_db:
        print(f"[Info] 发现现有数据库，包含 {len(existing_db)} 个 NPC，将进行增量更新")
    
    # 使用 defaultdict 积累数据，情绪和身份使用 set 去重
    # 结构：{ "NPC 名": { "emotions": set(), "titles": set(), "faction": str, "sex": str } }
    npc_data = defaultdict(lambda: {
        'emotions': set(),
        'titles': set(),
        'faction': "",
        'sex': ""
    })
    
    # 1. 处理日常对话文件
    print(f"[1/2] 扫描日常对话列表：{config['dialogue_list_file']}")
    if config['dialogue_list_file'].exists():
        dialogue_files = get_file_list_from_xml(config['dialogue_list_file'], "items")
        for filename in dialogue_files:
            file_path = config['dialogues_dir'] / filename
            if file_path.exists():
                extract_from_dialogue_xml(file_path, npc_data, filename)
                faction = extract_faction_from_filename(filename)
                print(f"      已扫描：{filename} (阵营：{faction if faction else '无'})")
            else:
                print(f"      [警告] 文件未找到：{file_path}")
    else:
        print(f"      [警告] 对话列表文件不存在")
    
    # 2. 处理任务对话文件
    print(f"[2/2] 扫描任务对话列表：{config['task_list_file']}")
    if config['task_list_file'].exists():
        task_files = get_file_list_from_xml(config['task_list_file'], "text")
        for filename in task_files:
            file_path = config['task_text_dir'] / filename
            if file_path.exists():
                extract_from_task_json(file_path, npc_data)
                print(f"      已扫描：{filename}")
            else:
                print(f"      [警告] 文件未找到：{file_path}")
    else:
        print(f"      [警告] 任务列表文件不存在")
    
    # 3. 合并现有数据和新数据
    final_db = merge_npc_data(existing_db, npc_data)
    
    # 4. 保存文件
    config['output_db_file'].parent.mkdir(parents=True, exist_ok=True)
    with open(config['output_db_file'], 'w', encoding='utf-8') as f:
        json.dump(final_db, f, ensure_ascii=False, indent=1)
    
    print("--- 更新完成 ---")
    print(f"共发现 NPC 数量：{len(final_db)}")
    print(f"新增 NPC 数量：{len(final_db) - len(existing_db)}")
    print(f"数据库已保存至：{config['output_db_file']}")


if __name__ == "__main__":
    main()
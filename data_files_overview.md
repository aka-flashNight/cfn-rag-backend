## 数据文件与任务系统总览

本说明用于给后端 / Agent 逻辑提供一个**稳定的数据格式参考**，后续实现时可以直接按此文档设计解析与校验逻辑，而无需频繁重新阅读原始数据文件。

- 在实现任意与“玩家进度”“推荐等级”“主线 id 范围”相关的逻辑前，**需要先参考 `services/game_progress.py` 中的阶段配置与工具函数**：
  - 阶段/进度 1–6 分别对应：
    - 阶段名称：废城 / 堕落城 / 荒漠军阀 / 黑铁会总堂 / 诺亚 / 雪山；
    - 推荐等级区间：`min_level`–`max_level`；
    - 主要地图区域名 `stage_name`（如 `基地门口`、`基地车库` 等，对应 `stages` 子目录）；
    - 主线任务 id 范围：`main_task_min_id`–`main_task_max_id`；
  - 提供的关键工具函数包括：
    - `get_progress_stage_level_range(stage)`：按阶段返回推荐等级区间（也允许小于该区间）；
    - `get_progress_stage_main_task_range(stage)`：按阶段返回主线 id 推荐区间（也允许小于该区间）；
    - `is_valid_stage_root(stage_root)`：判断某个 `stages` 子目录是否为“有效大区”；
  - 后文关于**关卡选择（解锁 id / 推荐等级）、副本关卡选择、装备类物品等级约束、任务前置主线 id**等规则，都默认基于这份阶段配置来做“等于（约等于）优先，其次小于，禁止大于”的筛选策略。
  
### 1. 目录与 list.xml 约定

- **通用规则**
  - 以下目录均为项目外部游戏目录 `resources\data` 内的子文件夹目录（而不是当前项目目录），用于查询游戏各项数据。
  - 若目录下不存在 `list.xml`，则采用固定文件，如`infrastructure/infrastructure.xml` ，为基建项目的xml。
  - 只要目录下存在 `list.xml`，**仅 `list.xml` 中登记的条目视为有效数据**；
  - 目录下存在 `list.xml`时，未在 `list.xml` 中出现的文件或子文件夹，一律视为“非关键 / 仅供游戏使用”，在 Agent 逻辑中默认不参与候选与校验；；
  - `list.xml` 中的元素名称不同，语义也不同，大致有三类：
    - **记录具体文件名**：例如 `task/list.xml`、`task/text/list.xml`、`items/list.xml`、`enemy_properties/list.xml`、`shops/list.xml`、`kshop/list.xml` 等；
    - **记录无后缀文件名**：例如 `crafting/list.xml`，需要拼接.json文件类型后缀。
    - **记录子目录名**：例如 `stages/list.xml`，需要进入对应子目录做二次解析。
  - 关卡大区（`stages` 根目录下子目录）的过滤规则：
    - 只有以下子目录视为“有效大区”，可参与 agent 任务生成与筛选：
      - `基地门口`、`基地车库`、`基地房顶`、`黑铁会总部`、`诺亚前线基地深处`、`雪山` (阶段对应关系参考 `services/game_progress.py` )；
      - 以及会穿插多个进度或仅支线的：`地下2层`、`副本任务`、`试炼场深处`；
    - 其他 `stages` 子目录一律视为“无效大区”，不参与 agent 任务候选关卡集合。

- **已确认的 list.xml 结构示例**

  - `data/task/list.xml`

    ```xml
    <?xml version="1.0" encoding="UTF-8"?>
    <root>
      <task>tasks1.json</task>              <!-- 主线任务 -->
      <task>tasks2.json</task>              <!-- 剧情支线任务 -->
      <task>general_tasks.json</task>       <!-- 一般支线（如将军支线） -->
      <task>guide_tasks.json</task>         <!-- 教学/引导类任务 -->
      <task>challenge_tasks.json</task>     <!-- 挑战类任务 -->
      <task>mercenary_tasks.json</task>     <!-- 副本类委托（副本任务专用） -->
      <task>preview_tasks.json</task>       <!-- 预览类任务（无需参考） -->
      <task>bonus_tasks.json</task>         <!-- 彩蛋类任务（无需参考） -->
      <task>school_tasks.json</task>        <!-- 阵营类支线（大学） -->
      <task>logistics_tasks.json</task>     <!-- 烹饪任务 -->
      <task>eastzone_tasks.json</task>      <!-- 废城支线 -->
      <task>agent_tasks.json</task>         <!-- 生成类委托（你只能修改这个tasks文件） -->
    </root>
    ```

  - `data/task/text/list.xml`

    ```xml
    <?xml version="1.0" encoding="UTF-8"?>
    <root>
      <text>text1.json</text>               <!-- 主线文本 1 段 -->
      <text>text2.json</text>               <!-- 剧情支线文本 -->
      <text>general_texts.json</text>       <!-- 一般支线文本 -->
      <text>guide_text.json</text>          <!-- 教学任务文本 -->
      <text>challenge_text.json</text>      <!-- 挑战类任务文本 -->
      <text>mercenary_text.json</text>      <!-- mercenary 副本委托任务文本 -->
      <text>preview_text.json</text>
      <text>bonus_text.json</text>
      <text>school_texts.json</text>
      <text>logistics_text.json</text>
      <text>eastzone_text.json</text>
      <task>agent_text.json</task>          <!-- 生成类委托任务文本（你只能修改这个text文件） -->
    </root>
    ```

  - `data/items/list.xml`

    ```xml
    <?xml version="1.0" encoding="UTF-8"?>
    <root>
      <items>消耗品_货币.xml</items>        <!-- 金币 / 经验值 / 技能点 / K点 等 -->
      <items>消耗品_弹夹.xml</items>
      <items>消耗品_药剂.xml</items>
      <items>消耗品_药剂_食品.xml</items>
      <items>收集品_材料.xml</items>
      <items>收集品_材料_插件.xml</items>
      <items>消耗品_手雷.xml</items>
      <items>消耗品_材料_食材.xml</items>
      <items>收集品_情报.xml</items>
      <items>防具_颈部装备.xml</items>
      <items>防具_0-19级.xml</items>
      <items>防具_20-39级.xml</items>
      <items>防具_40+级.xml</items>
      <items>武器_刀_默认.xml</items>
      <!-- 省略若干武器 / 防具分类 -->
    </root>
    ```

  - `data/stages/list.xml`

    ```xml
    <?xml version='1.0' encoding='utf-8'?>
    <root>
      <stages>基地门口</stages>             <!-- 子目录名：需进入 data/stages/基地门口 -->
      <stages>基地车库</stages>
      <stages>基地房顶</stages>
      <stages>地下2层</stages>
      <stages>副本任务</stages>
      <stages>黑铁会总部</stages>
      <stages>诺亚前线基地深处</stages>
      <stages>诺亚前线基地深处第二层</stages>
      <stages>沙漠虫洞</stages>
      <stages>试炼场深处</stages>
      <stages>亡灵沙漠</stages>
      <stages>雪山</stages>
      <stages>雪山第二层</stages>
      <stages>雪山内部</stages>
      <stages>雪山内部第二层</stages>
      <stages>异界战场</stages>
      <stages>坠毁战舰</stages>
    </root>
    ```

  - `data/shops/list.xml`

    ```xml
    <?xml version="1.0" encoding="UTF-8"?>
    <root>
      <shops>shops.json</shops>             <!-- NPC 金币商店 -->
    </root>
    ```

  - `data/kshop/list.xml`

    ```xml
    <?xml version="1.0" encoding="UTF-8"?>
    <root>
      <kshop>kshop.json</kshop>             <!-- K 点商城 -->
    </root>
    ```

  - `data/crafting/list.xml`

    ```xml
    <?xml version="1.0" encoding="UTF-8"?>
    <root>
      <list>铁枪会</list>                   <!-- 需要拼接.json -->
      <list>属性武器</list>
      <list>烹饪</list>
      <list>化学生产</list>
      <list>武器合成</list>
      <list>饰品合成</list>
      <list>进阶防具</list>
      <list>基础防具</list>
      <list>公社防具</list>
      <list>黑白契约</list>
      <list>插件合成</list>
      <list>大学装备</list>
    </root>
    ```

  - `data/enemy_properties/list.xml`

    ```xml
    <?xml version='1.0' encoding='utf-8'?>
    <root>
      <items>原版敌人 2011-2012.xml</items>
      <items>原版敌人 2013-2016.xml</items>
      <items>换皮敌人与战宠.xml</items>
      <items>彩蛋支线.xml</items>
      <items>诺亚新敌人.xml</items>
      <items>魔神.xml</items>
      <items>天网.xml</items>
      <items>boss重做.xml</items>
      <items>下水道.xml</items>
      <items>军阀新人物.xml</items>
      <items>大学.xml</items>
    </root>
    ```

  - `data/dialogues/list.xml`

    ```xml
    <?xml version='1.0' encoding='utf-8'?>
    <root>
      <items>npc_dialogue_商人.xml</items>
      <items>npc_dialogue_彩蛋.xml</items>
      <items>npc_dialogue_成员.xml</items>
      <items>npc_dialogue_摇滚公园.xml</items>
      <items>npc_dialogue_联合大学.xml</items>
      <items>npc_dialogue_军阀.xml</items>
      <items>npc_dialogue_黑铁会.xml</items>
      <items>npc_dialogue_A兵团.xml</items>
      <items>npc_dialogue_A兵团元老.xml</items>
      <items>npc_dialogue_禁区人员.xml</items>
      <items>npc_dialogue_闲杂人等.xml</items>
      <items>npc_dialogue_探索者.xml</items>
      <items>npc_dialogue_通缉犯.xml</items>
    </root>
    ```

### 2. 任务数据（task）与文本（task/text）

#### 2.1 主线与支线分类

- `tasks1.json`：**主线任务**（id 从 0 开始，严格递增，当前主线到 77，未来可能扩展到 ~150）；
- `tasks2.json` 及其它 `*_tasks.json`：**支线/系统任务**，包括：
  - `general_tasks.json`：一般支线；
  - `guide_tasks.json`：引导/教学任务；
  - `challenge_tasks.json`：挑战类任务（偏高难度、经验奖励为主）；
  - `mercenary_tasks.json`：**副本类委托任务**，即绑定副本的任务，特殊逻辑（详见下文）；
  - `preview_tasks.json` / `bonus_tasks.json` / `school_tasks.json` / `logistics_tasks.json` / `eastzone_tasks.json`：其它主题支线；
  - `agent_tasks.json`：**生成类委托任务**，不绑定副本，可以正常接取的委托任务，你负责生成、

#### 2.2 单个任务的 JSON 结构示例（以 `tasks1.json` 为例）

```json
{
  "tasks": [
    {
      "id": 0,
      "title": "$MAIN_TITLE_0",                 // 标题 key，映射到 task/text/text1.json
      "description": "$MAIN_DESCRIPTION_0",     // 描述 key

      "get_requirements": [],                   // 接取前置条件（任务 id 列表，空数组表示无前置；禁止使用 -1，-1 仅委托任务使用）
      "get_conversation": "$MAIN_GET_0",        // 接取对话段 key
      "get_npc": "Andy Law",                    // 发布 NPC 名

      "finish_requirements": [],                // 完成条件：关卡通关要求数组（"关卡名#难度"）
      "finish_submit_items": [],                // 完成条件：提交物品数组（"物品名#数量"）
      "finish_contain_items": [],               // 完成条件：背包持有物品数组（"物品名#数量"）
      "finish_conversation": "$MAIN_FINISH_0",  // 完成对话段 key
      "finish_npc": "Andy Law",                 // 完成 NPC 名

      "rewards": [
        "金币#500",
        "经验值#500",
        "普通hp药剂#3",
        "普通mp药剂#3"
      ],

      "announcement": "可选公告文本",            // 可选：任务接取时的公告，通常为空
      "chain": "主线#1"                         // 任务线标记（详见 chain 规范），委托类任务只填写"委托"，无需后缀
    },
    {
        // 其他任务……
    }
  ]
}
```

- **通关要求（finish_requirements）**
  - 结构：字符串数组，每项形如 `"关卡名#难度"`；
  - 四个标准难度名：`简单`、`冒险`、`修罗`、`地狱`；
  - 仅**地图关卡（非副本任务）**拥有四难度；
  - `data/stages/副本任务` 中的关卡通常只允许 `简单` 难度；
  - 若某副本在 `mercenary_tasks` 中配置了 `challenge` 额外难度（且该额外难度推荐等级满足玩家等级），则允许在任务要求中选择该额外难度；
  - 当你在任务要求中选择了副本的非 `简单` 难度时，LLM 必须在任务说明（title/description）与对话台词中明显提到玩家正在选择/挑战该高难度模式。

- **提交/持有物品要求**
  - `finish_submit_items`：表示完成时**消耗**指定物品；数组元素示例：`"抗生素#1"`；
  - `finish_contain_items`：表示完成时背包中**只需要拥有**指定物品，不强制消耗；数组元素示例：`"能量干扰盾#1"`。

- **奖励（rewards）**
  - 字符串数组，每项 `"物品名#数量"`；
  - 物品名需在 items 系统中存在，对应唯一价格，用于任务奖励总价值的量化与管控。

#### 2.3 文本文件结构（以 `task/text/text1.json` 为例）

- 任务文本使用 key → 数组 的形式，任务实例中只存储 `$MAIN_TITLE_n`、`$MAIN_DESCRIPTION_n`、`$MAIN_GET_n`、`$MAIN_FINISH_n` 等 key。
- `name` 必须为精确的name，`title` 可以是任意符合的称呼，`char` 可选情绪，如NPC名#情绪，但必须在该NPC给定的情绪中选择，也可以无情绪。

```json
{
  "$MAIN_TITLE_0": "AndyLaw的基地",
  "$MAIN_DESCRIPTION_0": "到地下室找Andy谈谈。走楼梯可到地下室。",
  "$MAIN_GET_0": [
    {
      "name": "$PC",
      "title": "$PC_TITLE",
      "char": "$PC_CHAR",
      "text": "不知不觉间我走进了这里。..."
    },
    {
      "name": "Andy Law",
      "title": "东区最强战士",
      "char": "Andy Law#微笑",
      "text": "独行者，在基地的感觉还不错吧？"
    },
    {
      "name": "Andy Law",
      "title": "东区最强战士",
      "char": "Andy Law",
      "text": "这里是A兵团，我是这里的负责人。"
    }
  ],
  "$MAIN_FINISH_0": [
    // 完成时对话数组，同上，但需要改为完成的NPC的对话
  ],
  "$MAIN_TITLE_1": "新手试炼",
  "$MAIN_DESCRIPTION_1": "通关简单难度“废城-新手练习场”后，找Andy谈谈。",
  "$MAIN_GET_1": [ /* 对话数组 */ ],
  "$MAIN_FINISH_1": [ /* 对话数组 */ ]
}
```

### 3. 现有特殊任务类型说明

#### 3.1 mercenary_tasks（委托任务）

- 文件：`data/task/mercenary_tasks.json`；
- 特点：
  - `get_requirements` 统一配置为 `[-1]`：
    - 含义：**只能在对应副本入口处接取**；
    - 正常与 NPC 对话时不可见/不可直接接取；
  - `chain` 统一为 `"委托"`，**不需要区分编号**（如 `委托#1` 等）；
  - 绑定 `data/stages/副本任务` 中的关卡；
  - 部分委托会扩展副本难度（如增加 `修罗` / `地狱`）：当玩家满足 mercenary 的推荐等级（含 challenge 的推荐）时，Agent 任务可在副本难度里选择额外模式；选择非 `简单` 时必须在任务说明与对话台词中明显提示玩家正在挑战高难度模式。

#### 3.2 我们计划新增的 agent 任务（agent_tasks）

- 单独任务文件：`data/task/agent_tasks.json`（在 `task/list.xml` 中对应 `<task>agent_tasks.json</task>`）；
- 对应文本：`data/task/text/agent_text.json`（在 `task/text/list.xml` 中对应 `<text>agent_text.json</text>`）；
- **强约束**：
  - Agent 允许读取所有现有任务与文本文件；
  - **只允许写入 / 修改 `agent_tasks.json` 与 `agent_text.json`** 两个文件，所有其它任务文档只能读取，不得写。

- **agent 任务 ID 规划**
  - 预留区间：`200001`–`300000`（共 100000 个 id）；
  - 要求：
    - 所有 agent 任务 `id` 必须落在该区间；
    - **不允许重复**，按递增方式分配即可；

- **agent 任务允许的 chain 标记**
  - 可以沿用 `"委托"` 作为 `chain` 值（与 mercenary 同名），无需后缀；
  - 但语义不同：agent 任务需要**能在 NPC 处正常接取**，不是入口限定委托。
  - 因此， `get_requirements` 不允许使用 `-1`。

- **agent 任务接取前置（get_requirements）规则**
  - 数组元素为任务 id（主线 / 支线均可），委托类任务通常采用主线id作为前置。可以是空数组，表示无前置。
  - **禁止使用 `-1`**。
  - 典型约束：
    - **通关类任务**：必须绑定当前阶段的**主线任务 id** （参考后文关卡解锁配置，从`stages/*/__list__.xml` 中读取）作为前置（至少一个 id），避免超进度任务；
    - **问候类 / 运输类 / 低级物品收集类**：
      - 可允许 `get_requirements: []` 或绑定较早的主线 id，方便低门槛互动；
    - **高级物品收集、通关 + 收集类**：
      - 应该配置合理的主线前置 id，确保玩家已经达到相关阶段，避免提前获取高价值资源。

### 4. 物品数据与价格系统（items）

#### 4.1 货币与基础值（消耗品_货币.xml）

示例：`data/items/消耗品_货币.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!-- 物品ID已自动生成，无需手动维护 -->
<root>
  <item>
    <name>金币</name>
    <displayname>金钱</displayname>
    <icon>金钱</icon>
    <type>消耗品</type>
    <use>货币</use>
    <price>1</price>
    <description>一个金币</description>
  </item>
  <item>
    <name>经验值</name>
    <displayname>经验值</displayname>
    <icon>经验值</icon>
    <type>消耗品</type>
    <use>货币</use>
    <price>1</price>
    <description>经验值*1</description>
  </item>
  <item>
    <name>技能点</name>
    <displayname>技能点</displayname>
    <icon>技能点</icon>
    <type>消耗品</type>
    <use>货币</use>
    <price>1500</price>
    <description>技能点（SP点）*1</description>
  </item>
  <item>
    <name>K点</name>
    <displayname>K点</displayname>
    <icon>K点</icon>
    <type>消耗品</type>
    <use>货币</use>
    <price>100</price>
    <description>一个K点</description>
  </item>
</root>
```

- **任务奖励价值量化思路**
  - 通过 `items` 中的 `price` 字段将所有可奖励资源统一换算为“金币等价”；
  - 任务奖励数组 `["物品名#数量", ...]` 可以在后端解析为 `sum(price(物品名) * 数量)`；
  - 不同主线阶段设定不同的**奖励价值区间**，并根据：
    - 玩家当前阶段（1–6）；
    - 当前 NPC 好感度；
    - 玩家在对话中是否讨价还价；
    - 任务类型（问候 / 通关 / 挑战 / 收集 / 装备获取委托等）；
    - 基础奖励区间计算公式：[阶段 * 1万金币, 阶段 * 2万金币] 的上下限区间。
    - 有通关需求的情况，奖励额度 * 2 。有收集需求的话，奖励需要额外增加收集品的 1.5 - 2 倍价值（上限不超过当前任务奖励额度的 200%，装备类不超过 300% ）。持有类需求，则额外增加物品的 0.5 倍价值（上限不超过当前任务奖励额度的 50% ）；
    - 来在区间内偏多或偏少地选择实际奖励。

#### 4.2 NPC 金币商店（shops）

- 文件：`data/shops/shops.json`；
- 逻辑含义：**NPC 金币商店清单**；
  - 结构：`NPC名` → `"商品索引（字符串）"` → `"物品名"`；
  - 示例（局部）：

```json
{
  "Andy Law": {
    "0": "浅灰蒙面",
    "1": "医用口罩",
    "2": "防毒面具2号"
    /* ... */
  },
  "小F": {
    "0": "A兵团制式套装改造图纸",
    "1": "镜之虎彻制作图纸"
  },
  "Boy": {
    "0": "TYPE79",
    "1": "汤姆逊冲锋枪",
    "56": "冲锋枪通用弹药"
    /* ... */
  }
}
```

- Agent 在构造**收集类任务**时，可以考虑：
  - 要求玩家从**非本阵营商人**处购买某些物品（物品如果有等级需求，则需要符合或小于玩家对应的等级区间）；
  - 将**本NPC商店**的物品作为奖励（物品如果是武器/防具且有等级需求，则需要符合玩家对应的等级区间）。

#### 4.3 K 点商城（kshop）

- 文件：`data/kshop/kshop.json`；
- 逻辑含义：**K 点（付费货币）商城列表**；

```json
[
  {
    "id": "S20130922000002",
    "item": "战宠灵石大盒",
    "type": "新品推荐",
    "price": "9000"
  },
  {
    "id": "S20130402000002",
    "item": "强化石",
    "type": "新品推荐",
    "price": "50"
  }
  /* ... */
]
```

- Agent 一般**不会直接使用 K 点商城**物品作为任务奖励（避免破坏经济系统），但可以用作为**任务物品要求**（物品如果有等级需求，则需要符合或小于玩家对应的等级区间）并给出匹配价值的奖励。（只有在非常合适的情景下才可以发布此任务，例如稀有武器/装备获取的委托，不作为常规任务类型）


#### 4.4 物品合成（crafting）

- **目录位置与 list.xml**
  - 目录：`resources/data/crafting/`
  - 存在 `list.xml`，只有其中登记的条目是有效的合成表：

  ```xml
  <?xml version="1.0" encoding="UTF-8"?>
  <root>
    <list>铁枪会</list>
    <list>属性武器</list>
    <list>烹饪</list>
    <list>化学生产</list>
    <list>武器合成</list>
    <list>饰品合成</list>
    <list>进阶防具</list>
    <list>基础防具</list>
    <list>公社防具</list>
    <list>黑白契约</list>
    <list>插件合成</list>
    <list>大学装备</list>
  </root>
  ```

  - 每个 `<list>` 元素对应一个同名 `.json` 文件，例如：
    - `铁枪会` → `crafting/铁枪会.json`
    - `烹饪` → `crafting/烹饪.json`
    - `武器合成` → `crafting/武器合成.json`
    - `插件合成` → `crafting/插件合成.json`
    - 等等。

- **统一的合成 JSON 结构**

  所有合成表使用统一的结构：**数组**，每个元素是一条“配方”记录：

  ```json
  [
    {
      "title": "展示用标题",
      "name": "合成产物物品名",
      "price": 1000,
      "kprice": 10,
      "value": 1,              // 可选字段：有的合成表用于批量合成或药剂增强时出现
      "materials": [
        "材料物品名#数量",
        "材料物品名#数量"
      ]
    }
  ]
  ```

  - **关键字段说明**
    - `title`：该配方在 UI 中展示的名称（通常带“改装图纸”、“制作图纸”等字样）；
    - `name`：**实际产出物品名**，必须与 `items` 系统中的物品名一致（或计划新增）；
    - `price`：合成产物在金币中的标价，用于价值衡量；
    - `kprice`：合成产物在 K 点中的标价（如有）；多数普通配方为 0；
    - `value`：部分配方存在的辅助数值（如批量合成时的数量），对任务系统通常可忽略；
    - `materials`：材料数组，元素格式为 `"物品名#数量"`，对应所需投入的物品与数量。

- **各子表示例（节选）**

  - `crafting/武器合成.json`（武器改装示例）

  ```json
  [
    {
      "title": "AK47火麒麟改装图纸",
      "name": "AK47火麒麟",
      "price": 50000,
      "kprice": 0,
      "materials": [
        "国庆纪念币#10"
      ]
    },
    {
      "title": "Colt Anaconda制作图纸",
      "name": "Colt Anaconda",
      "price": 55000,
      "kprice": 0,
      "materials": [
        "COLT PYTHON#1",
        "螺丝套件#2",
        "弹簧#2",
        "战术导轨#1",
        "动力液压杆#1"
      ]
    }
  ]
  ```

  - `crafting/插件合成.json`（装备插件产物示例，与 `items/equipment_mods` 呼应）

  ```json
  [
    {
      "title": "能量干扰盾改装图纸",
      "name": "能量干扰盾",
      "price": 1000,
      "kprice": 10,
      "materials": [
        "能量电池#5",
        "电脑芯片#1",
        "增效剂#1"
      ]
    },
    {
      "title": "绯红忆弦轮",
      "name": "绯红忆弦轮",
      "price": 10000,
      "kprice": 100,
      "materials": [
        "汲丝虹吸匣#1",
        "透镜#1",
        "复合式军用塑料#1",
        "高耐力橡胶#1"
      ]
    }
  ]
  ```

  - `crafting/烹饪.json`（料理类物品）

  ```json
  [
    {
      "title": "番茄炒蛋",
      "name": "番茄炒蛋",
      "price": 50,
      "kprice": 0,
      "materials": [
        "基础家常菜配方#1",
        "番茄#1",
        "鸡蛋#1",
        "食用油#1",
        "食盐#1"
      ]
    }
  ]
  ```

  - `crafting/化学生产.json`（药剂升级/毒药等）

  ```json
  [
    {
      "title": "普通hp药剂升级（逐个）",
      "name": "抗生素",
      "value": 1,
      "price": 100,
      "kprice": 0,
      "materials": [
        "普通hp药剂#1"
      ]
    }
  ]
  ```

  - `crafting/基础防具.json` / `进阶防具.json` / `公社防具.json` / `大学装备.json` / `黑白契约.json` 等也遵循同样结构，仅产物与材料类型不同（多为防具/套装部件）。

- **Agent 使用要点**
  - 对于“**物品获取**”或“**物品展示**”类任务：
    - 可以把合成表中的 `name` 视为**可获取物品的完整列表**之一；
    - 再结合 `items` 系统中的价格（按物品名对表）作为价值参考；
  - 建议使用合成表时的策略：
    - 若任务主题是“制作/研究/料理/化学试验”等，可以优先从 `烹饪`、`化学生产` 中选取目标物品；
    - 若任务主题偏向“武器打造/强化”，则优先从 `铁枪会`、`武器合成`、`属性武器` 中选取目标；
    - 若任务主题偏向“套装/防具/时装”，则从 `基础防具`、`进阶防具`、`公社防具`、`大学装备`、`黑白契约` 等表中选取；
    - 物品如果有等级需求，则需要符合或小于玩家对应的等级区间。
  - **价值衡量**：
    - 任务对合成产物的要求（例如“合成 X 件某装备/料理/药剂”）的总价值，仍以该产物在 `items` 中的金币 `price` 为主；
    - 合成所需材料本身一般不额外重复计价，只作为“达成难度”的参考。


### 4.5 装备插件材料目录补充：`items/equipment_mods`

- **目录位置**
  - `resources/data/items/equipment_mods/`
  - 该目录下存在 `list.xml`，因此**仅 `list.xml` 中登记的 XML 视为有效插件配置文件**。

- **list.xml 结构示例**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<root>
  <!-- 低级材料 -->
  <items>低级材料_防具专用.xml</items>
  <items>低级材料_枪械专用.xml</items>
  <items>低级材料_刀专用.xml</items>
  <items>低级材料_拳专用.xml</items>
  <items>低级材料_通用.xml</items>
  <items>低级材料_下挂武器.xml</items>

  <!-- 中等材料 -->
  <items>中等材料_防具专用.xml</items>
  <items>中等材料_枪械专用.xml</items>
  <items>中等材料_刀专用.xml</items>
  <items>中等材料_拳专用.xml</items>
  <items>中等材料_通用.xml</items>
  <items>中等材料_下挂武器.xml</items>

  <!-- 高等材料 -->
  <items>高等材料_防具专用.xml</items>
  <items>高等材料_枪械专用.xml</items>
  <items>高等材料_刀专用.xml</items>
  <items>高等材料_拳专用.xml</items>
  <items>高等材料_通用.xml</items>
  <items>高等材料_下挂武器.xml</items>

  <!-- 特殊材料 -->
  <items>特殊材料_防具专用.xml</items>
  <items>特殊材料_通用.xml</items>
  <items>特殊材料_下挂武器.xml</items>
  <items>特殊材料_刀专用.xml</items>
</root>
```

- **单个插件 XML 的典型结构（示例：`高等材料_通用.xml` 中节选）**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<root>
  <mod>
    <name>动力液压杆</name>
    <use>上装装备,下装装备,手部装备,刀</use>
    <stats>
      <useSwitch>
        <use name="刀">
          <override>
            <slay>8</slay>
          </override>
          <flat>
            <power>15</power>
          </flat>
        </use>
        <use name="手部装备">
          <override>
            <slay>8</slay>
          </override>
          <flat>
            <punch>15</punch>
          </flat>
        </use>
        <use>
          <flat>
            <punch>15</punch>
            <weight>-1</weight>
          </flat>
        </use>
      </useSwitch>
    </stats>
    <description>…略…</description>
    <tag>动力接口</tag>
  </mod>
</root>
```

- **字段说明与 Agent 使用要点**
  - `<mod>`：单个插件（配件）的定义节点；
  - `<name>`：插件名称，用于界面展示与任务描述；
  - `<use>`：可安装的装备/武器类型列表，逗号分隔（如 `头部装备,上装装备,下装装备,手部装备,脚部装备,刀,手枪,长枪`）；
  - `<stats>`：属性修改集合，下含多种子节点：
    - `<flat>`：固定数值加减（如 `power`、`hp`、`weight`、`punch` 等）；
    - `<percentage>`：百分比加成（如 `damage`、`punch` 等）；
    - `<merge>`：与原属性合并的特殊效果（如 `vampirism` 吸血等）；
    - `<useSwitch>`：通过 `<use name="…">` 为不同使用场景（刀/手部装备/上装等）分别配置覆盖或附加效果；
  - `<requireTags>`：插件生效所需的装备标签（如 `电力`），用来限制可结合的装备；
  - `<description>`：多行描述文本；
  - `<tag>`：插件类别标签（如“动力接口”、“等级校正”等），可作为任务生成时的过滤条件。

- **与 Agent 任务的关系（建议用法）**
  - 当设计“改造/强化相关”的任务时，可以：
    - 通过工具查询 `equipment_mods` 下所有插件，或考虑按 `<tag>` / `<requireTags>` / `<use>` 过滤出和当前对话内容匹配的候选插件；
    - 将某个或若干插件作为任务目标（例如“收集某类插件”）；结合 `items` 系统中的价格与插件稀有度，为此类任务设置更高但受控的奖励区间。
    - 也可以作为任务奖励，但要根据玩家的进度来匹配。阶段1-6之中，低级对应1-2，中级对应2-3，高级对应4及以上。

### 5. 关卡数据（stages）与解锁条件、资源箱掉落

#### 5.1 关卡解锁配置：`__list__.xml`

- 每个 `stages/子目录` 下有一个 `__list__.xml`，用于配置关卡基础信息与主线解锁要求（`UnlockCondition`）。
- 示例：`data/stages/基地门口/__list__.xml`

```xml
<?xml version='1.0' encoding='utf-8'?>
<Stages>
  <StageInfo>
    <Type>无限过图</Type>
    <Name>商业区</Name>
    <FadeTransitionFrame>wuxianguotu_1</FadeTransitionFrame>
    <UnlockCondition>14</UnlockCondition>   <!-- 需要完成的主线任务 id -->
    <Description>城市里人流量最大的地区...</Description>
  </StageInfo>
  <StageInfo>
    <Type>无限过图</Type>
    <Name>超市废墟</Name>
    <FadeTransitionFrame>wuxianguotu_1</FadeTransitionFrame>
    <UnlockCondition>3</UnlockCondition>
    <Description>暂无资料</Description>
  </StageInfo>
  <StageInfo>
    <Type>无限过图</Type>
    <Name>新手练习场</Name>
    <FadeTransitionFrame>wuxianguotu_1</FadeTransitionFrame>
    <UnlockCondition>1</UnlockCondition>
    <Description>暂无资料</Description>
  </StageInfo>
  <StageInfo>
    <Type>无限过图</Type>
    <Name>医院</Name>
    <FadeTransitionFrame>wuxianguotu_1</FadeTransitionFrame>
    <UnlockCondition>19</UnlockCondition>
    <Description>彻底沦陷的医院，据说埋藏了些许秘密。</Description>
  </StageInfo>
  <!-- 其他关卡略 -->
</Stages>
```

- **Agent 使用规则（候选关卡筛选）**
  - 仅当满足以下条件时，某关卡才作为 agent 任务候选：
    - 对应的 `stages` 子目录是**有效大区**（可通过 `is_valid_stage_root(stage_root)` 判断）；
    - 在 `__list__.xml` 中存在且包含 `UnlockCondition`（主线任务 id）；
  - 若没有 `UnlockCondition`（或未收录在 `__list__.xml` 中），为避免各种解锁异常，**不将其作为 agent 任务候选关卡**；
  - 当需要为玩家当前进度选择合适的通关关卡时：
    - 先通过 `get_player_progress()` 获取当前阶段以及对应的主线 id 合理区间（`get_progress_stage_main_task_range`）；
    - 优先选择 `UnlockCondition` **等于或约等于** 当前阶段范围内的主线 id 的关卡（如当前主线 id 为 20，则优先 18–22 区间）；
    - 其次允许选择 `UnlockCondition` **小于** 当前主线 id 的关卡，用于布置简单任务或 NPC 仅负责低难度任务的场景；
    - 严禁选择 `UnlockCondition` 明显**大于**当前主线 id 的关卡，避免玩家接到超出当前进度的任务。
  - 对于关卡难度：
    - 地图关卡（如上示例）在游戏中有 `简单/冒险/修罗/地狱` 四种难度：
      - 通关类基础任务优先选择 `简单` / `冒险` 难度；
      - 挑战类任务可选择 `修罗` / `地狱`，并给予主要偏向经验的奖励；
    - 副本任务目录 `data/stages/副本任务` 中的关卡通常只允许 `简单` 难度；
    - 若 `mercenary_tasks.json` 中该副本配置了 `challenge` 额外难度，并且玩家等级满足其推荐等级，则可在 `difficulties` 中看到该额外难度；
    - 当 LLM 在副本中选择非 `简单` 难度时，必须在任务说明与对话台词中明确提醒玩家选择挑战模式/高难度。
      - Agent 任务引用地图关卡时允许任意难度。

#### 5.2 关卡详细结构与箱子掉落

- 每个具体关卡对应一个 `GameStage` XML 文件，例如：`data/stages/基地门口/医院.xml`。
- 结构包括 `<Rewards>`、`<SubStage>`（子场景）、`<Wave>` 敌人波次、`<Instances>` 场景实例等。
- **任务相关的关键部分是场景中的箱子掉落**（资源箱、纸箱、装备箱）。而通用随机掉落 `<Rewards>` 将来会被弱化/优化，对 agent 任务的意义较小，因此无视 `<Rewards> `的物品掉落。

- 示例：`resources\data\stages\基地门口\医院.xml` 中资源箱片段（节选）

```xml
<SubStage id="3">
  <BasicInformation>
    <Background>flashswf/backgrounds/医院电梯处.swf</Background>
  </BasicInformation>
  <Instances>
    <Instance id="0">
      <x>330</x>
      <y>290</y>
      <Identifier>资源箱</Identifier>
      <Parameters>
        <掉落物>
          <名字>抗生素</名字>
          <最小数量>20</最小数量>
          <最大数量>30</最大数量>
        </掉落物>
        <掉落物>
          <名字>肾上腺素</名字>
          <最小数量>20</最小数量>
          <最大数量>30</最大数量>
        </掉落物>
      </Parameters>
    </Instance>
  </Instances>
  <!-- 敌人波次略 -->
</SubStage>
```

- **Agent 收集类任务与关卡联动规则（建议）**
  - 仅扫描三类箱子：`纸箱`、`资源箱`、`装备箱`；
  - 对每个候选箱子的 `<掉落物>`：
    - 读取 `<名字>`、`<最小数量>`、`<最大数量>`；
    - 若物品意义较小（如剧情无关、价值过低、或显然不适合作为任务目标），后端可以提前过滤，再交给 Agent 决策；
    - 对于适合作为任务目标的物品：
      - **任务要求数量建议使用该箱子产出的最小数量**，例如最小数量为 20，则任务条件可为 `"抗生素#20"`，并增加通关要求 `医院#简单`，组成通关+收集类任务；
  - 将这些候选关卡和物品打包成结构化信息暴露给 Agent，让 Agent 决定具体用哪个物品、具体数量、以及是否组合多个物品作为任务要求。
  - 在选择关卡时，仍需遵守 5.1 中的**解锁 id 不得显著高于玩家当前主线 id**的规则，并优先使用“等于/约等于当前主线 id”的关卡，其次是低于当前进度的关卡。

- **注意**
  - 如果出现以下这种箱子，出现 `最小主线进度` 或 `最大主线进度` 任意一项属性，请将其排除，不要查询：

```xml
  <Instance id="1">
      <x>955</x>
      <y>313</y>
      <Identifier>纸箱</Identifier>
      <Parameters>
          <最小主线进度>4</最小主线进度>
          <最大主线进度>4</最大主线进度>
          <掉落物>
              <名字>抗生素</名字>
              <最小数量>2</最小数量>
              <最大数量>4</最大数量>
          </掉落物>
      </Parameters>
  </Instance>
```

### 6. Agent 功能扩展与系统架构设计

#### 6.1 Agent 任务生成的总体流程

1. **玩家与 NPC 对话（LLM 对话层）任务类型与奖励计算**
   - 对话中，Agent 可以根据上下文判断是否合适提出新任务：
     - 问候类/传话类/闲聊类（仅基础奖励）；
     - 通关类（低难度地图，基础奖励翻倍，但经验少）；
     - 挑战类（高难度，基础奖励翻倍，且经验占比较多）；
     - 资源收集类（基础材料/药剂/食材收集一定数量并提交，奖励额外增加提交品价值的 1.5~2 倍，提交品的总价值不能超过基础奖励上限的200%）；
     - 特殊物品获取类（装备/插件/食品/合成消耗品收集一个并提交，奖励额外增加提交品价值的 1.5~2 倍，提交品的总价值不能超过基础奖励上限的300%）；
     - 持有类（物品/情报收集但无需提交，奖励额外增加持有品价值的0.5倍，但额外增加的奖励上限不能超过基础奖励上限的50%）；
     - 通关 + 收集类（基础奖励翻倍，且叠加提交品价值的 1.5~2倍。必须是通关对应关卡可以从中获取到所有提交品。）；
     - 通关 + 持有类（基础奖励翻倍，且叠加持有品价值的0.5倍。必须是通关对应关卡可以从中获取到所有持有品。）；
   - 除了任务要求可以划分类别外，任务奖励也可以根据NPC设定与对话剧情来划分不同类别：
     - 金币（最常规的奖励）；
     - 经验（常规奖励，但仅挑战类可以给大量经验，其他类型给少量经验）；
     - 药剂（可选奖励）；
     - 弹夹（可选奖励）；
     - K点（阶段4及以上时可选奖励）；
     - 技能点（可选奖励）；
     - 强化石（可选奖励）；
     - 战宠灵石（可选奖励）；
     - 材料（可选奖励）；
     - 食品（可选奖励）；
     - 武器（可选奖励，仅限当前NPC的商店，且符合玩家等级区间）；
     - 防具（可选奖励，仅限当前NPC的商店，且符合玩家等级区间）；
     - 插件（可选奖励，仅限当前NPC的商店）；
   - 基础奖励区间计算公式：[阶段 * 1万金币, 阶段 * 2万金币] 的上下限区间，阶段为1-6的整数。
   - 玩家可以选择接受/拒绝/讨价还价。

2. **对话接口三阶段架构：决策 → 执行 → 生成（Decision-Execute-Generate Pipeline）**

   由于 `gemini-3.1-flash-lite-preview` 这类模型在 OpenAI 兼容模式下存在**无法同时流式输出和工具调用**的兼容性问题，也为了支撑后续越来越多的 Agent 工具，对话接口必须从当前的"单次调用"改造为**三阶段管线**。这是 LangChain / LangGraph 等主流 Agent 框架的标准做法（ReAct 循环），完全合理且必要——每次请求至少两次 LLM 调用的成本，可以通过缓存命中优化（见 6.5）来有效控制。

   **阶段 A：决策轮（Decision Round）—— 非流式**

   - 使用与最终生成**几乎完全相同的** system prompt、tool 定义、对话历史调用 LLM（非流式）；
   - LLM 决定是否需要调用工具：
     - 若返回 `tool_calls`：进入阶段 B 执行工具；
     - 若返回纯文本，但从中解析出 `tool_calls` 的相关格式：说明该模型尝试调用 `tool_calls` 失败，按现有json格式尝试解析，解析成功进入阶段 B 执行工具，解析失败直接进入阶段 C 生成对话（沿用现有的 `parse_mood_from_text` + `strip_trailing_mood_json` 逻辑并扩展至后续其他工具作为兜底）；
     - 若返回无 `tool_calls`（未解析到 `tool_calls` 相关格式的纯文本或空内容）：表示无需工具，直接进入阶段 C 生成对话；
   - 决策轮的 system prompt 末尾追加一条控制指令：*"本轮请仅判断是否需要调用工具并生成 tool_calls。若不需要任何工具，返回空内容即可，不要输出对话文本。"*
   - 本轮无需传入立绘/头像图像，但可以提供查询某NPC的立绘/头像的工具、输入关键词（如NPC名、阵营名或其他关键词）进行RAG检索的工具，供LLM选择

   **阶段 B：工具执行（Tool Execution）—— 后端处理**

   - 后端根据 `tool_calls` 列表，按需执行对应工具逻辑（详见 6.3 工具体系）；
   - 将所有工具执行结果以 `tool` role message 追加到消息列表末尾；
   - 若当前工具无需重复确认（即无需返回），调用后即产生最终结果（如 `update_npc_mood` 工具），直接进入阶段 C 生成对话；
   - 若当前工具需继续确认，**回到阶段 A** 重新调用 LLM（非流式），让 LLM 判断是否还需更多工具调用；
   - 循环终止条件：LLM 返回无 `tool_calls` 时退出循环，进入阶段 C；
   - **安全上限**：设置最大循环次数（建议 **5 轮**），超限后强制进入阶段 C，防止无限循环。

   **阶段 C：对话生成（Response Generation）—— 流式**

   - 将阶段 A/B 积累的完整消息列表（含所有 tool_calls 与 tool results），加上阶段切换指令：*"请根据以上信息，以 NPC 身份生成对话回复。"*，调用 LLM（流式）；
   - 本轮调用**不传入工具定义**，彻底规避流式 + 工具的兼容性问题；
   - 流式输出 NPC 对话内容至前端；
   - 现有的好感度与情绪工具的功能需要维持，将此前调用的工具结果返回给前端。当前NPC的立绘/头像图片也在本轮传输，沿用现在的功能。

   **关于好感度/情绪（`update_npc_mood`）的处理策略**

   `update_npc_mood` 的特殊性在于：它有可能要依赖 NPC 本轮的回复内容来匹配情绪与好感度变化，而此时回复尚未在决策阶段产生。推荐方案：

   - **解决方案**：在阶段 A 让LLM先确定是否需要调用工具来更改好感度和情绪，将此工具的调用结果拼接到阶段 C 的提示词之中，使得AI根据工具调用结果来生成匹配的对话。或者，尽管只调用 `update_npc_mood` 时，无需获取返回结果并再次让LLM调用工具，但是，同时调用多种工具中，如果包含 `update_npc_mood` ，此时仍然要把LLM调用时传的参数拼接进后续轮次，告知LLM该工具的调用结果，避免LLM反复调用该工具。

   **管线流程图**

   ```
   ask 请求入口
       │
       ▼
   ┌────────────────────────────────────────┐
   │  上下文准备                              │
   │  - RAG 检索（沿用现有 _retrieve_context） │
   │  - 构建 system prompt + 对话历史          │
   │  - 从 DB 加载待确认任务草案（如有）         │
   └─────────────────┬──────────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────────┐
   │  阶段 A: 决策轮 (非流式)                  │◄──┐
   │  LLM (tools=[全部 Agent 工具])            │   │
   │  → 返回 tool_calls?                      │   │
   │       │                                  │   │
   │      YES                NO               │   │
   │       │                 │                │   │
   │       ▼                 │                │   │
   │  ┌──────────┐           │                │   │
   │  │ 阶段 B:   │           │                │   │
   │  │ 工具执行   │           │                │   │
   │  │ (后端处理) │           │                │   │
   │  └────┬─────┘           │                │   │
   │       │ 注入 tool results                │   │
   │       │ 轮次 < 5? ──── YES ──────────────┼───┘
   │       │                                  │
   │      NO (超限 → 强制进入生成)              │
   │       │                 │                │
   │       └────────┬────────┘                │
   └────────────────┼─────────────────────────┘
                    │
                    ▼
   ┌─────────────────────────────────────────┐
   │  阶段 C: 对话生成 (流式)                  │
   │  LLM (tools=None)              │
   │  → 流式输出 NPC 对话回复                   │
   │  → 文本解析情绪/好感度变化                  │
   └─────────────────┬───────────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────────┐
   │  后处理                                  │
   │  - 保存对话历史到 DB                      │
   │  - 更新 NPC 好感度                        │
   │  - 写入任务文件（如任务已被玩家接受）        │
   │  - 清理已完成的任务草案                     │
   └─────────────────────────────────────────┘
   ```

3. **任务协商与生命周期（跨多轮对话）**

   任务从"意向"到"写入游戏文件"需要经历完整的生命周期，可能涉及多轮对话交互。

   **3.1 任务发起**

   - LLM 在对话中根据**NPC 人格、对话情节、玩家进度**（缺一不可，详见 6.6）综合判断是否适合提出新任务；
   - LLM 先调用 `prepare_task_context(task_type, reward_types)` 传入意向任务类型和奖励类型，后端根据类型一次性返回该类型所需的全部筛选后数据（玩家进度、可选关卡/物品列表、奖励预算、已有任务等）以及该类型的规则说明；
   - LLM 根据返回数据调用 `draft_agent_task` 工具，输出结构化的**任务草案（Task Draft）**（详见 6.4 结构化输出）；
   - 后端校验草案合法性（奖励上限、物品存在性、关卡解锁条件等）；
   - 校验通过后，草案**暂存到数据库** `session_task_drafts` 表（按 session_id 存储），不写入游戏文件；
   - LLM 在对话生成阶段以自然语言向玩家描述该任务（*"我这里有个事想拜托你……"*）。

   **3.2 协商与调整**

   - 在后续对话中，玩家可以：
     - **接受**（*"好的"* / *"可以"* / *"接了"*）→ 进入 3.3 确认阶段；
     - **拒绝**（*"不要"* / *"算了"*）→ 清除草案，LLM 以 NPC 身份回应，对话继续正常进行；
     - **讨价还价**（*"奖励太少了"* / *"能多给点金币吗"*）→ LLM 调用 `update_task_draft(draft_id, modify_fields)` 在允许范围内**局部调整**奖励，无需重新生成完整草案：
       - 好感度 ≥ 50 → 可在基础奖励上浮动 +10%～+50%；
       - 50 > 好感度 >= 20 → 可在基础奖励上浮动 +1%～+20%；
       - 好感度 < 20 → 几乎不让步甚至可以减少奖励；
       - 最多允许讨价还价 2 次，并提醒LLM这一限制，超过后后端拒绝调整，LLM 需同步进行取消任务或拒绝发布；
       - 每次调整后通过 `update_task_draft` 更新 `session_task_drafts` 中的草案，后端对修改后的字段重新校验（如奖励总价值是否仍在允许区间内）；
     - **修改要求**（*"换个简单点的"* / *"我不想打那个关"*）→ LLM 调用 `update_task_draft(draft_id, modify_fields)` 局部修改草案（如仅更换关卡、调整物品要求等），降低出错率。仅在需要**更改任务类型**等根本性变更时，才重新调用 `prepare_task_context` + `draft_agent_task` 生成全新草案；
   - **自动过期**：若玩家连续 **3 轮对话**未提及任务相关内容，自动清除待确认草案，避免悬挂状态。

   **3.3 确认与写入**

   - 玩家明确接受后：
     a. 后端从 `session_task_drafts` 读取最新草案；
     b. 执行最终校验（见 6.4 校验管线），不通过则向 LLM 返回错误信息，LLM 需要修正或通知玩家；
     c. 分配新任务 ID：读取 `agent_tasks.json` 当前最大 ID，+1 分配（初始为 200001，上限 300000）；
     d. 由后端根据草案 + LLM 生成的文本拼装完整的任务 JSON 和文本 JSON（详见下文第 4 点）；
     e. **原子写入** `agent_tasks.json` 与 `agent_text.json`（先写临时文件再重命名，防止写入中途崩溃导致文件损坏）；
     f. 清除 `session_task_drafts` 中的草案；
     g. LLM 在对话中确认任务已发布（类似于 *"【系统提示：请进入游戏接取任务】好，任务安排好了。"* ，但具体台词要符合NPC与情景）。

4. **任务写入格式规范**

   写入 `agent_tasks.json` 时，严格遵循现有任务 JSON 结构（参考 2.2）：

   ```json
   {
     "tasks": [
       {
         "id": 200001,
         "title": "$AGENT_TITLE_200001",
         "description": "$AGENT_DESCRIPTION_200001",
         "get_requirements": [21],
         "get_conversation": "$AGENT_GET_200001",
         "get_npc": "Andy Law",
         "finish_requirements": ["医院#简单"],
         "finish_submit_items": ["抗生素#20"],
         "finish_contain_items": [],
         "finish_conversation": "$AGENT_FINISH_200001",
         "finish_npc": "Andy Law",
         "rewards": ["金币#20000", "经验值#2000"],
         "announcement": "",
         "chain": "委托"
       }
     ]
   }
   ```

   写入 `agent_text.json` 时：

   ```json
   {
     "$AGENT_TITLE_200001": "医院的物资",
     "$AGENT_DESCRIPTION_200001": "到医院收集10个抗生素，带回给Andy Law。",
     "$AGENT_GET_200001": [
       {
         "name": "Andy Law",
         "title": "东区最强战士",
         "char": "Andy Law",
         "text": "最近基地的药品储备不太够……帮我去医院搜刮一些抗生素回来。"
       }
     ],
     "$AGENT_FINISH_200001": [
       {
         "name": "Andy Law",
         "title": "东区最强战士",
         "char": "Andy Law#微笑",
         "text": "干得不错，这些药品很有用。"
       }
     ]
   }
   ```

   - 文本 key 统一使用 `$AGENT_TITLE_{id}` / `$AGENT_DESCRIPTION_{id}` / `$AGENT_GET_{id}` / `$AGENT_FINISH_{id}` 格式；
   - `get_npc` 和 `finish_npc` 可为同一 NPC，也可未不同NPC。但`get_npc` 通常为当前对话的NPC，除非是转派的委托（如酒保或Andy Law等合适的NPC转派的其他NPC任务）；
   - `chain` 统一为 `"委托"`；
   - `get_requirements` **禁止使用 `-1`**，必须为主线任务 ID 数组或空数组；
   - `announcement` 通常为空字符串；
   - `rewards` 中每一项的格式为 `"物品名#数量"`，由后端根据 LLM 结构化输出的物品名与数量 **拼接生成**（LLM 不直接输出此格式字符串，而是输出结构化数据，避免幻觉导致格式错误）；
   - 任务文本中的 `char` 字段支持 `NPC名#情绪` 格式（如 `"Andy Law#微笑"`），情绪必须从该 NPC 的可用情绪标签中选择；也可以不带情绪后缀。

5. **游戏内接取与完成**
   - 游戏客户端与聊天前端不是同一窗口。后端写入成功后，玩家需**打开游戏项目**才能看到并接取新任务；
   - 玩家在游戏中正常与 NPC 对话接取任务，通关/收集/提交物品后完成任务并获得奖励。

#### 6.2 引入 LangGraph：状态图设计

##### 6.2.1 为什么引入 LangGraph

当前项目的 `ask` / `ask_stream` 接口已经包含隐式的多步逻辑（RAG 检索 → LLM 调用 → 工具解析 → 状态更新），但通过线性的 Python 异步代码实现，缺乏清晰的状态管理和灵活的路由控制。随着工具数量增长和任务生成流程的引入，继续在同一个大函数中堆叠逻辑将导致代码难以维护和测试。

LangGraph（`langgraph` 库，基于 LangChain 生态）提供的关键能力：

- **有向图建模**：将决策、工具执行、对话生成等环节建模为图节点，控制流为条件边，逻辑清晰、可视化、易维护；
- **条件路由**：根据 LLM 输出（有无 tool_calls）自动选择下一个节点，无需手写嵌套 if-else；
- **状态持久化**：通过 `Checkpointer`（如 `SqliteSaver`）支持跨请求的状态持久化，天然适配任务协商的多轮草案修改；
- **可观测性**：内置执行追踪，便于调试复杂的多轮工具调用链路；
- **兼容 OpenAI 格式**：可直接使用 `ChatOpenAI` 包装现有 API（支持自定义 `base_url`），无需重写 LLM 调用层。

> **注意**：引入 LangGraph 不意味着推翻现有代码。现有的 `services/llm_client.py`、`services/npc_mood_agent.py`、`services/game_rag_service.py` 中的**上下文准备逻辑（RAG 检索、prompt 构建、NPC 状态查询）** 将被重构为 LangGraph 图节点内的可复用函数，而**情绪解析、历史管理、NPC 好感度更新**等成熟逻辑将作为后处理节点保持基本不变。

##### 6.2.2 图状态定义（Graph State Schema）

```python
from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class RewardItem(TypedDict):
    """单个奖励项（结构化表示，LLM 输出此格式，后端拼接为 '物品名#数量'）"""
    item_name: str       # 物品名（必须存在于 items 系统）
    count: int           # 数量（必须在合理区间内）


class TaskDraft(TypedDict):
    """待确认的任务草案"""
    task_type: str                    # 任务类型：["问候", "传话", "通关", "清理", "挑战", "切磋", "资源收集", "装备获取", "特殊物品获取", "物品持有", "通关并收集", "通关并持有"] 
    get_requirements: list[int]       # 前置主线 ID 数组（可为空）
    finish_requirements: list[str]    # 通关要求（"关卡名#难度"），可为空
    finish_submit_items: list[RewardItem]  # 提交物品要求，可为空
    finish_contain_items: list[RewardItem] # 持有物品要求，可为空
    rewards: list[RewardItem]         # 奖励列表
    total_value: int                  # 奖励总价值（金币等价）
    get_npc: str                      # 发布 NPC
    finish_npc: str                   # 完成 NPC
    title: str                        # 任务标题（LLM 生成的自然语言）
    description: str                  # 任务描述（LLM 生成的自然语言）
    get_conversation_text: str        # 接取对话文本（LLM 生成的自然语言）
    finish_conversation_text: str     # 完成对话文本（LLM 生成的自然语言）


class AgentState(TypedDict):
    # LangGraph 消息列表（自动累加）
    messages: Annotated[list[BaseMessage], add_messages]

    # 上下文（在 prepare_context 节点中填充）
    npc_name: str
    player_progress: int              # 1-6
    npc_affinity: int                 # 0-100
    npc_relationship_level: str       # 陌生/熟悉/朋友/生死之交
    session_id: str
    retrieved_context: str            # RAG 检索结果文本

    # 任务协商状态
    pending_task_draft: Optional[TaskDraft]
    task_confirmed: bool
    task_write_result: Optional[str]  # 写入结果（成功时为任务 ID，失败时为错误信息）

    # 控制信号
    tool_call_round: int              # 当前决策-执行循环轮次（安全上限 5）
    skip_generation: bool             # 特殊场景：工具已完成所有操作，跳过生成

    # 输出（在 generate_response / post_process 节点中填充）
    final_reply: str                  # 完整的 NPC 回复文本
    emotion: str                      # 解析出的情绪标签
    favorability_change: int          # 解析出的好感度变化
```

##### 6.2.3 图节点定义

| 节点                  | 类型     | 职责                                                                 |
|-----------------------|----------|----------------------------------------------------------------------|
| `prepare_context`     | 同步/异步 | RAG 检索、构建 system prompt、加载对话历史、从 DB 加载待确认任务草案     |
| `decision`            | 异步     | 非流式调用 LLM（含全部工具定义），判断是否需要工具调用                    |
| `tool_executor`       | 异步     | 根据 tool_calls 执行后端工具逻辑，返回 tool result messages            |
| `generate_response`   | 异步     | 流式调用 LLM（无工具），生成 NPC 对话回复                               |
| `parse_mood`          | 同步     | 从完整回复中解析情绪与好感度变化等工具调用（复用并扩展现有解析逻辑）       |
| `post_process`        | 异步     | 保存对话历史、更新好感度、写入任务文件（如有已确认任务）、清理草案         |

##### 6.2.4 图拓扑与条件路由

```
                    ┌─────────────────┐
                    │ prepare_context  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
         ┌────────►│    decision      │◄───────────┐
         │         └────────┬────────┘             │
         │                  │                      │
         │          has_tool_calls?                 │
         │            │          │                  │
         │           YES        NO                 │
         │            │          │                  │
         │            ▼          │                  │
         │   ┌──────────────┐    │                  │
         │   │tool_executor  │    │                  │
         │   └───────┬──────┘    │                  │
         │           │           │                  │
         │    round < 5?         │                  │
         │     │        │        │                  │
         │    YES      NO        │                  │
         │     │        │        │                  │
         │     └────────┼────────┘                  │
         │              │                           │
         └──────────────┘ (YES: 回到 decision)
                        │ (NO 或无 tool_calls)
                        ▼
               ┌─────────────────┐
               │generate_response │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │   parse_mood     │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  post_process    │
               └────────┬────────┘
                        │
                        ▼
                      [END]
```

##### 6.2.5 LangGraph 图构建伪代码

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

graph = StateGraph(AgentState)

# 注册节点
graph.add_node("prepare_context", prepare_context_node)
graph.add_node("decision", decision_node)
graph.add_node("tool_executor", tool_executor_node)
graph.add_node("generate_response", generate_response_node)
graph.add_node("parse_mood", parse_mood_node)
graph.add_node("post_process", post_process_node)

# 入口
graph.set_entry_point("prepare_context")
graph.add_edge("prepare_context", "decision")


def route_after_decision(state: AgentState) -> str:
    """决策后路由：有工具调用 → 执行工具；无工具调用 → 生成回复"""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_executor"
    return "generate_response"


def route_after_tool(state: AgentState) -> str:
    """工具执行后路由：未超限 → 回决策；超限 → 强制生成"""
    if state.get("tool_call_round", 0) >= 5:
        return "generate_response"
    return "decision"


graph.add_conditional_edges("decision", route_after_decision)
graph.add_conditional_edges("tool_executor", route_after_tool)
graph.add_edge("generate_response", "parse_mood")
graph.add_edge("parse_mood", "post_process")
graph.add_edge("post_process", END)

# 编译（可选持久化，用于跨请求保存任务草案等状态）
checkpointer = SqliteSaver.from_conn_string("memory.db")
agent = graph.compile(checkpointer=checkpointer)
```

##### 6.2.6 与现有 ask 接口的整合

改造后的 `ask_stream` 接口伪代码：

```python
async def ask_stream(payload, npc_manager, memory):
    config = {"configurable": {"thread_id": payload.session_id}}
    initial_state = {
        "messages": [],
        "npc_name": payload.npc_name,
        "player_progress": payload.progress_stage or 1,
        "session_id": payload.session_id,
        # ... 其他初始字段
    }

    # astream_events 支持流式事件，可以在 generate_response 节点中
    # 捕获 LLM 的 streaming tokens 并实时推送给前端
    async for event in agent.astream_events(initial_state, config=config):
        if event["event"] == "on_chat_model_stream":
            # 仅转发 generate_response 节点的 streaming tokens
            if "generate_response" in event.get("tags", []):
                yield ("content", event["data"]["chunk"].content)
        elif event["event"] == "on_chain_end":
            final_state = event["data"]["output"]
            yield ("done", {
                "reply": final_state["final_reply"],
                "emotion": final_state["emotion"],
                "favorability_change": final_state["favorability_change"],
                # ...
            })
```

#### 6.3 工具体系设计

##### 6.3.1 数据访问层（Data Registry）

Agent 工具不应直接读取原始 XML/JSON 文件。后端需要在启动时（`core/startup.py` 的 `run_startup_tasks`）将所有静态游戏数据解析并加载到内存缓存中，提供高层查询 API。

**数据注册中心架构**

```python
class GameDataRegistry:
    """
    游戏数据统一注册中心。
    在 run_startup_tasks() 时初始化，全局单例。
    """
    items: ItemRegistry           # 物品数据（含价格、类型、等级）
    tasks: TaskRegistry           # 任务数据（所有任务文件聚合）
    task_texts: TaskTextRegistry   # 任务文本数据
    stages: StageRegistry         # 关卡数据（含解锁条件、箱子掉落）
    shops: ShopRegistry           # NPC 商店数据
    kshop: KShopRegistry          # K 点商城数据
    crafting: CraftingRegistry    # 合成配方数据
    agent_tasks: AgentTaskStore   # agent_tasks.json 读写管理
    agent_texts: AgentTextStore   # agent_text.json 读写管理
```

**各 Registry 关键查询方法**

| Registry          | 关键方法                                                                  |
|-------------------|--------------------------------------------------------------------------|
| `ItemRegistry`    | `get_by_name(name)` → Item; `search(keyword, type?, use?)` → list[Item]; `get_price(name)` → int; `list_by_type(type)` → list[Item]; `list_by_level_range(min, max)` → list[Item] |
| `TaskRegistry`    | `get_by_id(id)` → Task; `list_by_npc(npc_name)` → list[Task]; `get_max_agent_task_id()` → int; `list_reward_types()` → set[str] |
| `StageRegistry`   | `list_stages_for_progress(stage)` → list[StageInfo]; `get_stage_loot(area, stage_name)` → list[LootCrate]; `get_unlock_condition(area, stage_name)` → int |
| `ShopRegistry`    | `get_npc_shop(npc_name)` → list[str]; `has_shop(npc_name)` → bool       |
| `KShopRegistry`   | `list_items()` → list[KShopItem]; `get_by_name(name)` → KShopItem       |
| `CraftingRegistry`| `search(keyword)` → list[Recipe]; `get_by_product(name)` → Recipe        |
| `AgentTaskStore`  | `read_all()` → list[Task]; `append_task(task_json)` → int; `get_next_id()` → int |
| `AgentTextStore`  | `read_all()` → dict; `write_texts(texts_dict)` → None                    |

##### 6.3.2 面向 LLM 的工具定义

所有工具使用 OpenAI Function Calling 格式定义（兼容 LangChain `@tool` 装饰器）。工具按职责划分为**任务准备工具**、**任务写入工具**、**好感度工具**和**通用查询工具**四类。

现设计为**两步式调用**流程——由后端根据任务类型一次性返回所需的全部筛选后数据，减少 LLM 的决策负担：

```
Step 1: LLM 调用 prepare_task_context(task_type, reward_types)
        → 后端根据类型筛选数据，返回该类型所需的完整上下文 + 规则说明
Step 2: LLM 根据返回数据调用 draft_agent_task(TaskDraft)
        → 后端校验并暂存草案
```

---

**A. 任务准备工具**

| 工具名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `prepare_task_context` | `task_type: str, reward_types: RewardTypes` | `TaskContext`（详见下文） | 第一步：传入意向任务类型和奖励类型偏好，后端返回该类型所需的全部筛选后数据与规则说明 |

**`task_type` 枚举值：**

```json
["问候", "传话", "通关", "清理", "挑战", "切磋", "资源收集", "装备获取", "特殊物品获取", "物品持有", "通关并收集", "通关并持有"]
```
- `切磋`：仅当前NPC有切磋关卡（单独配置的challenge属性）时可选。

**`reward_types` 参数结构：**

```json
{
  "regular": ["金币", "经验"],
  "optional": ["药剂", "弹夹", "K点", "技能点", "强化石", "战宠灵石", "材料", "食品", "武器", "防具", "插件"]
}
```

- `regular`：常规奖励，可多选。金币为最常规奖励；经验为常规奖励，但仅挑战类可给大量经验，其他类型给少量经验。
- `optional`：可选奖励，可多选。其中 K 点仅阶段 4 及以上时可选；武器/防具/插件仅限当前 NPC 商店有售时可选，且武器/防具符合玩家等级区间。
- 后端根据 LLM 选择的奖励类型，返回对应的可选奖励物品列表。

**`prepare_task_context` 返回值 `TaskContext` 详细说明：**

**通用字段（所有任务类型均返回）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `level_range` | `[int, int]` | 玩家当前阶段的等级区间，如 `[1, 20]` |
| `main_task_range` | `[int, int]` | 玩家当前阶段的主线任务 ID 区间，如 `[1, 50]` |
| `reward_budget` | `object` | 奖励总额区间 `{base_min, base_max, multiplier, final_min, final_max}`，已根据 `task_type` 和好感度计算好倍率 |
| `existing_tasks` | `list` | 当前 NPC 已发布的任务列表 `[{id, title, type}]`，供 LLM 避免重复 |
| `reward_item_candidates` | `list` | 根据 `reward_types` 筛选的可选奖励物品 `[{name, type, price, level?, source}]`，`source` 标注来源（`"任务奖励常见"` / `"NPC商店"`），已根据玩家等级筛选 |
| `task_rules` | `str` | 该任务类型的思路说明、限制条件、注意事项（纯文本，供 LLM 理解规则） |

**类型专属字段：**

> 后端对所有数据均已做**进度筛选**：超出玩家当前主线 ID / 等级区间的关卡和物品一律剔除，只保留可选的部分。筛选原则：进度在玩家区间内的最优先，低于玩家区间的也保留（标注 `"below_progress": true`），因为任务符合 NPC 设定是必要条件。

**① 问候 / 传话类：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `npc_list` | `list` | 所有 NPC 列表 `[{name, faction, title, brief}]`，包含当前 NPC 自己（标注 `"is_current": true`）和其他 NPC，供 LLM 选择问候/传话对象 |

> 问候类包括闲聊，可选当前 NPC 自己为完成NPC，传话类则通常选择其他 NPC。

**② 通关 / 清理 / 挑战类：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `stage_list` | `list` | 二级结构的可选关卡列表，结构为 `[{area, area_level_range, stages: [{name, unlock_id, difficulties, is_dungeon, recommended_level?, below_progress?, challenge_modes?}]}]` |

> **后端筛选逻辑：**
> - **关卡类**（非副本）：根据解锁 ID ≤ 当前主线 ID 筛选，可选四个难度（简单/冒险/修罗/地狱）；
> - **副本类**：根据推荐等级筛选（推荐等级下限 ≤ 玩家等级上限），**未标注推荐等级的副本一律剔除**；
>   - 默认仅可选 `简单`；
>   - 若该副本在 `mercenary_tasks.json` 中配置了 `challenge` 额外难度（且额外难度的推荐等级满足玩家等级），则 `difficulties` 可能包含该额外难度，并返回 `challenge_modes` 说明；
>   - 若选择了非 `简单` 难度，LLM 必须在任务说明与对话台词中明显提醒玩家选择挑战模式/高难度；
> - 优先返回解锁 ID 在玩家当前主线 ID 区间内的关卡，低于玩家区间的关卡保留但标注 `"below_progress": true`，超出玩家进度的一律剔除。

**③ 切磋类：**

> **前置可用性检查**：后端在调用 `prepare_task_context` 时，需判断副本中是否存在当前 NPC 发布的切磋关卡（npc具有challenge属性）。**若不存在，则返回错误** `{"error": "当前NPC无可用的切磋目标，请选择其他任务类型"}`，LLM 不应选择此类型。因此，在工具声明中，应该在该npc有切磋类型时，才需要在task_type中提供，否则尽量隐藏这一任务类型。

| 字段 | 类型 | 说明 |
|------|------|------|
| `challenge_targets` | `list` | 当前 NPC 的可用切磋关卡列表 `[{dungeon_name, target_npc, difficulties, challenge_modes?}]`；`difficulties` 至少包含 `["简单"]`，若 `mercenary_tasks.json` 为该关卡配置了 `challenge` 额外难度且推荐满足玩家等级，则 `difficulties` 会额外包含该额外难度，并返回 `challenge_modes`（可选）用于描述该挑战模式 |

**④ 资源收集类：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `collectable_items` | `list` | 仅包含**食材、药剂、材料、弹夹**四类物品 `[{name, type, price, level?}]`，且必须已经存在于 `现有任务的提交物品` +  `现有任务的奖励物品`  的物品池之中；已根据玩家等级筛选，并使得总价不超出当前任务基础奖励范围的200% |

**⑤ 装备获取类：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `equipment_items` | `list` | 可提交的装备列表 `[{name, type, price, level, source}]`，包含三个来源（`source` 字段标注）：① `"非本阵营商店"` — 其他阵营 NPC 商店的装备 ② `"合成"` — 合成配方产出的装备 ③ `"K点商店"` — K 点商店的装备（根据 `kprice` 筛选一定价格以下的，避免要求过高）。已根据玩家等级筛选有效装备 |

**⑥ 特殊物品获取类：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `special_items` | `list` | 可提交的特殊物品列表 `[{name, type, price, source}]`，通常只需要1个物品，包含三个来源：① `"非本阵营商店"` — 其他阵营 NPC 商店的非装备物品（插件、药剂、菜品、贵重消耗品等） ② `"合成"` — 合成配方产出的非装备物品（插件、药剂、菜品、贵重消耗品等） ③ `"K点商店"` — K 点商店的非装备物品（根据 `kprice` 筛选一定价格以下的） |

**⑦ 物品持有类：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `holdable_items` | `list` | 可持有的物品列表 `[{name, type, price, level?, source}]`，包含两个来源：① 所有**情报**类物品 ②`"合成"` — 合成配方产出的所有类别的物品。要么是收集情报，要么是指引玩家去制作物品并检验成果。根据玩家等级筛选，并使得总价不超出当前任务基础奖励范围的200% |

**⑧ 通关并收集 / 通关并持有类：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `stage_loot_list` | `list` | **仅包含有箱子掉落的关卡**（无箱子的关卡不返回），结构为 `[{area, area_level_range, stage_name, unlock_id, is_dungeon, recommended_level?, difficulties, challenge_modes?, below_progress?, loot_items: [{item_name, min_qty, max_qty, unit_price}], total_loot_value}]` |

> **关键约束：**
> - 收集/持有要求的物品**必须是该关卡箱子的产出物品**，不可选择关卡外的物品；
> - 收集数量**建议使用箱子产出的最小数量**（`min_qty`），避免要求玩家反复刷关；
> - `total_loot_value` 为该关卡箱子产出的总价值估算（`Σ(unit_price × min_qty)`），供 LLM 参考奖励定价；
> - 后端筛选逻辑同通关类：优先返回进度区间内关卡，保留低于进度的关卡（标注 `"below_progress": true`），剔除超出进度的关卡；
> - 副本类关卡同样根据推荐等级筛选，未标注推荐等级的剔除；
>   - 默认仅可选 `简单`；
>   - 若该副本在 `mercenary_tasks.json` 配置了 `challenge` 额外难度且推荐满足玩家等级，则 `difficulties` 可能包含该额外难度，并返回 `challenge_modes` 说明；
>   - 选择非 `简单` 难度时，LLM 必须在任务说明与对话台词中明确提醒玩家选择挑战模式/高难度。

---

**B. 任务写入工具**

| 工具名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `draft_agent_task` | `TaskDraft`（完整的结构化任务草案） | `{success, draft_id, validation_errors?}` | 第二步：生成并校验任务草案，暂存到 DB |
| `update_task_draft` | `draft_id: str, modify_fields: dict` | `{success, draft_id, validation_errors?}` | 局部修改已有草案（如仅调整奖励、更换关卡），后端对修改字段重新校验，无需重新生成完整草案 |
| `confirm_agent_task` | 无（读取当前会话的 pending draft） | `{success, task_id?, error?}` | 确认并写入任务到游戏文件 |
| `cancel_agent_task` | 无 | `{success}` | 取消当前待确认的任务草案 |

**C. 好感度工具**

| 工具名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `update_npc_mood` | `favorability_change: int, emotion: str` | 无需返回 | 沿用现有定义 |

**D. 通用查询工具**

| 工具名 | 参数 | 返回值 | 说明 |
|--------|------|--------|------|
| `search_knowledge` | `keyword: str` | `str`（检索结果摘要文本） | 复用现有 RAG 检索，获取设定/情报 |

##### 6.3.3 工具返回值格式约定

- 所有工具返回 **JSON 字符串**，便于 LLM 解析；
- 列表类返回做**数量截断**（如最多 20 条），并附带 `"truncated": true` 标记，避免 token 超限；
- 错误时返回 `{"error": "错误描述"}`，LLM 可据此调整策略或通知玩家。

#### 6.4 结构化输出与校验管线

##### 6.4.1 LLM 结构化输出规范

LLM 在调用 `draft_agent_task` 工具时，必须以 **JSON 结构化参数** 的形式输出任务草案，而**不是**直接输出 `"物品名#数量"` 这样的拼接字符串。后端将 LLM 输出的结构化数据经过校验后，由 Python 代码拼接为游戏所需的 `"物品名#数量"` 格式写入文件。

这样做的核心原因：
- 避免 LLM 幻觉导致的格式错误（如缺少 `#`、物品名拼写错误等）；
- 后端可以逐项校验物品名是否存在、数量是否合理；
- 格式一致性由代码保证，不依赖 LLM 的格式遵从能力。
- 注意，"问候"、"传话"大体上是一类任务，"通关"、"清理"大体上是一类任务，"装备获取"、 "特殊物品获取"大体上是一类任务，这里细分只是为了让LLM能更好的对应当前情景，其返回的所需数据以及管控说明是基本一致的。

`draft_agent_task` 工具的参数 JSON Schema：

```json
{
  "type": "object",
  "properties": {
    "task_type": {
      "type": "string",
      "enum": ["问候", "传话", "通关", "清理", "挑战", "切磋", "资源收集", "装备获取", "特殊物品获取", "物品持有", "通关并收集", "通关并持有"]
    },
    "title": { "type": "string", "description": "任务标题，简洁明了" },
    "description": { "type": "string", "description": "任务描述，简要说明目标" },
    "get_requirements": {
      "type": "array", "items": { "type": "integer" },
      "description": "前置主线任务 ID 数组，空数组表示无前置。禁止使用 -1。"
    },
    "finish_requirements": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "stage_name": { "type": "string", "description": "关卡名，如 '商业区'" },
          "difficulty": { "type": "string", "enum": ["简单", "冒险", "修罗", "地狱"] }
        },
        "required": ["stage_name", "difficulty"]
      }
    },
    "finish_submit_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "item_name": { "type": "string" },
          "count": { "type": "integer", "minimum": 1 }
        },
        "required": ["item_name", "count"]
      }
    },
    "finish_contain_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "item_name": { "type": "string" },
          "count": { "type": "integer", "minimum": 1 }
        },
        "required": ["item_name", "count"]
      }
    },
    "rewards": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "item_name": { "type": "string" },
          "count": { "type": "integer", "minimum": 1 }
        },
        "required": ["item_name", "count"]
      }
    },
    "get_conversation_text": { "type": "string", "description": "接取时 NPC 的对话文本" },
    "finish_conversation_text": { "type": "string", "description": "完成时 NPC 的对话文本" }
  },
  "required": ["task_type", "title", "description", "get_requirements", "rewards", "get_conversation_text", "finish_conversation_text"]
}
```

`update_task_draft` 工具的参数 JSON Schema：

```json
{
  "type": "object",
  "properties": {
    "draft_id": {
      "type": "string",
      "description": "要修改的草案 ID（从 draft_agent_task 返回值获取）"
    },
    "modify_fields": {
      "type": "object",
      "description": "要修改的字段集合，仅包含需要变更的字段，未包含的字段保持原值不变",
      "properties": {
        "title": { "type": "string" },
        "description": { "type": "string" },
        "finish_requirements": { "type": "array", "description": "替换整个通关要求列表" },
        "finish_submit_items": { "type": "array", "description": "替换整个提交物品列表" },
        "finish_contain_items": { "type": "array", "description": "替换整个持有物品列表" },
        "rewards": { "type": "array", "description": "替换整个奖励列表" },
        "get_conversation_text": { "type": "string" },
        "finish_conversation_text": { "type": "string" }
      },
      "additionalProperties": false
    }
  },
  "required": ["draft_id", "modify_fields"]
}
```

> **设计说明**：`update_task_draft` 采用**整字段替换**策略（而非深层合并），即 `modify_fields` 中出现的字段会完全替换草案中对应字段的值，未出现的字段保持不变。这样 LLM 只需关注要改什么，不需要重新构造整个草案，同时语义清晰、不易出错。后端收到修改后，对变更的字段执行与 `draft_agent_task` 相同的校验管线（见 6.4.2），校验不通过则返回 `validation_errors`。注意 `task_type` 和 `get_requirements` 不允许通过此工具修改——如需更改任务类型，应重新调用 `prepare_task_context` + `draft_agent_task`。

##### 6.4.2 后端校验管线（Validation Pipeline）

当 `draft_agent_task`、`update_task_draft` 或 `confirm_agent_task` 被调用时，后端按以下顺序逐项校验，全部通过才允许暂存/写入：

| 校验步骤 | 校验内容                                                       | 失败处理                       |
|----------|----------------------------------------------------------------|-------------------------------|
| V1       | **物品存在性**：rewards / submit_items / contain_items 中的每个 `item_name` 必须在 `ItemRegistry` 中存在 | 返回不存在的物品名列表          |
| V2       | **物品数量合理性**：每种物品的数量 ≥ 1，且不超过该物品在已有任务奖励中出现过的最大数量的 2 倍（防止异常值） | 返回超限的物品及建议区间        |
| V3       | **关卡存在性与解锁**：`finish_requirements` 中的每个关卡只需要提供 `stage_name + difficulty`，关卡需存在于 `StageRegistry` 且存在解锁条件；`stage_area` 不由 LLM 提供且不参与本校验 | 返回无效关卡名                 |
| V4       | **关卡解锁条件匹配**：关卡的 `UnlockCondition` ≤ 当前阶段的 `main_task_max_id`（不得超进度） | 返回超进度关卡及其解锁 ID      |
| V5       | **副本关卡难度**：只要某个 `finish_requirements.stage_name` 在 `mercenary_tasks.json` 中存在，则始终允许 `简单`；若该 stage_name 在 `mercenary_tasks.json` 对应任务配置了 `challenge` 额外难度，则仅当玩家等级满足 `challenge.recommended_level`（或缺失则回退到根 `recommended_level`）时，才允许选择对应非 `简单` 难度 | 返回违规的副本关卡              |
| V6       | **前置任务合法性**：`get_requirements` 为空或其中的每个 ID 必须在 `TaskRegistry` 中存在且 ≠ -1 | 返回无效的前置 ID              |
| V7       | **奖励总价值**：计算 `sum(price(item) * count)` 是否在允许区间内（见下文） | 返回每种物品单价、所有物品总价值、允许区间          |
| V8       | **奖励类型合规**：奖励物品必须属于已有任务奖励中出现过的物品类型集合，或属于当前 NPC 商店的物品 | 返回不合规的物品及原因          |
| V9       | **任务不完全重复**：标题/描述/条件/NPC/奖励的组合不得与已有 agent 任务高度雷同   | 返回疑似重复的已有任务 ID       |
| V10      | **装备等级匹配**：奖励/需求中的装备类物品（武器/防具）的等级需求必须 ≤ 当前阶段的 `max_level` | 返回超等级的装备名与等级              |

**奖励总价值区间计算（V7 详细规则）**

```
base_min = stage * 10000      （阶段 × 1万金币）
base_max = stage * 20000      （阶段 × 2万金币）

任务类型倍率:
  通关类 / 挑战类:         base × 2
  通关+收集类 / 通关+持有类: base × 2（再叠加收集/持有加成）
  其他类型:                 base × 1

收集/持有加成（叠加到乘数后的基础上）:
  finish_submit_items:  额外 += 提交品总价值 × 1.5~2.0，上限 = 基础奖励 × 200% （该上限限制的是提交品总价值）
  finish_contain_items: 额外 += 持有品总价值 × 0.5，上限 = 基础奖励 × 50% （该上限限制的是叠加的奖励总价值，持有品本身的总价值上限是 基础奖励 × 200%）

好感度修正:
  affinity >= 80: +20%
  affinity >= 50: +10%
  affinity >= 20: +0%
  affinity <  20: -10%

讨价还价修正: ±0%~+50%（由 LLM 在协商阶段动态确定，且受好感度影响）

最终区间:
  final_min = (base_min × 类型倍率 + 收集/持有加成) × 好感修正
  final_max = (base_max × 类型倍率 + 收集/持有加成) × 好感修正

```

#### 6.5 Prompt 工程与缓存命中优化

##### 6.5.1 缓存命中原理

主流 LLM API 服务商（Gemini、OpenAI、DeepSeek 等）均支持某种形式的 **prompt prefix caching**：当两次请求的消息列表前缀完全相同时，服务端可以跳过该前缀的编码计算，显著降低延迟和费用（通常缓存 token 的计费为 0 或正常费率的 25%～50%）。

因此，三阶段管线中阶段 A（决策）与阶段 C（生成）的调用应遵循**前缀一致原则**。

##### 6.5.2 消息列表结构设计

```
┌─────────────────────────────────────────────────────┐
│  [system]  世界观 + NPC 设定 + 对话规则 + 工具使用指南 │  ← 完全相同
│            （固定模板）  │
├─────────────────────────────────────────────────────┤
│  [user]    RAG 检索上下文 + 对话历史 + 玩家本轮输入    │  ← 完全相同
│            + 待确认任务草案摘要（如有）                 │
├─────────────────────────────────────────────────────┤
│  ── 以上为"缓存前缀"，阶段 A 和阶段 C 完全一致 ──     │
├─────────────────────────────────────────────────────┤
│  [assistant] tool_calls（阶段 A 产出）                │  ← 阶段 C 保留
│  [tool]     工具执行结果 1                            │  ← 阶段 C 新增
│  [tool]     工具执行结果 2                            │  ← 阶段 C 新增
│  ...（可能多轮）                                      │
│  [user]     "请以 NPC 身份生成对话回复"（阶段切换指令） │  ← 仅阶段 C
└─────────────────────────────────────────────────────┘
```

- 从 `[system]` 到第一个 `[user]` 的全部 token 在两次调用中**完全一致**，实现前缀缓存命中；
- 对于无需工具调用的简单对话，阶段 A 发现无 tool_calls 后，阶段 C 的 prompt 仅在末尾追加一行阶段切换指令，前缀缓存命中率接近 100%；
- 工具结果追加在末尾，不破坏前缀一致性。

##### 6.5.3 System Prompt 分层模板

```
Layer 1 — 固定层（几乎不变，跨所有 NPC 和请求）:
  ├── 世界观背景概要（WORLD_BACKGROUND，现有逻辑）
  ├── 对话输出格式规则（现有 prompt 中的"输出方式"段落）
  └── Agent 工具使用指南（新增：何时调用哪些工具、任务发布条件等）

Layer 2 — NPC 层（按 NPC 缓存，同一 NPC 的多次请求相同）:
  ├── NPC 名、性别、阵营、称号
  ├── NPC 可用情绪标签
  └── NPC 人格约束（见 6.6：该 NPC 擅长/不擅长发布什么类型的任务，简单说明思路即可，不需要给npc单独配置）

Layer 3 — 会话层（按请求变化，但变化量小；内部按变化频率从低到高排列，最大化缓存前缀命中）:
  ├── 同阵营 NPC 列表（当前 NPC 所在阵营的其他 NPC 基本信息，阵营不变则完全一致，变化频率极低）
  ├── 玩家身份描述（变化频率低）
  ├── 玩家进度阶段（变化频率低）
  ├── 当前好感度与关系等级（每次对话可能微调，但 token 量极小）
  ├── 对话中提到的其他 NPC 信息（根据玩家输入检索，按需注入，无则为空）
  └── 待确认任务草案摘要（如有，仅在任务协商期间存在）

Layer 4 — 检索层（按请求变化）:
  ├── RAG 检索结果
  ├── 对话历史
  └── 玩家本轮输入
```

Layer 1 + Layer 2 构成了最稳定的前缀，跨请求变化极小，缓存命中率最高。Layer 3 内部按**变化频率从低到高**排列：同阵营 NPC 列表几乎不变（除非玩家切换了对话 NPC 的阵营），排在最前；玩家进度和身份描述变化缓慢，排在其次；好感度和关系等级每次对话可能微调但 token 量极小；对话中提到的其他 NPC 信息和待确认草案是最不稳定的部分，排在最后。这样即使 Layer 3 中的后半部分发生变化，前半部分仍可与 Layer 1 + Layer 2 共同构成有效的缓存前缀。Layer 4 是最大的变化量来源，但排在最后，不影响前缀。


#### 6.6 NPC 人格约束与任务情境适配

任务的发布必须**符合 NPC 的设定与角色定位**，而不仅仅是满足进度区间。这是保持游戏沉浸感的关键约束。

##### 6.6.1 NPC 任务发布能力分级

| NPC 类型                                  | 可发布的任务类型                     | 奖励倾向                            | 示例                          |
|-------------------------------------------|--------------------------------------|--------------------------------------|-------------------------------|
| 高级军事 NPC（如 Andy Law、将军等）          | 通关、挑战、通关+收集                 | 金币、经验、武器、强化石              | "去商业区清理一下丧尸"         |
| 商人 NPC（有商店的 NPC）                    | 资源收集、装备获取、持有              | 金币、商店物品、材料                  | "帮我收集一些材料"             |
| 普通成员 NPC                               | 问候、资源收集（低级）、持有（低级）   | 金币、药剂、食品、弹夹               | "帮我带点药品回来"             |
| 科技/学术 NPC（大学等）                     | 收集（特殊材料/插件）、装备获取       | K 点、技能点、插件、合成材料          | "帮我找一些研究材料"           |


##### 6.6.2 Prompt 中的人格约束注入

在 system prompt 的 Layer 2（NPC 层）中，根据 NPC 的阵营、称号和商店情况，注入如下约束：

```
【任务发布约束】
你作为「{npc_name}」（{faction}），在决定是否发布任务时：
- 只发布符合你身份和能力范围的任务；
- 即使玩家进度较高，你也不应发布超出你角色定位的高难度任务；
- 在你的身份允许的情况下，可以优先发布最符合玩家当前进度的任务，其次是低于玩家进度的任务；
- 发布任务的动机应自然融入对话（如基于当前话题、NPC 的需求或烦恼），
  不要生硬地突然提出任务；
- 只有在对话氛围合适时才考虑发布任务，不是每次对话都需要任务；
{shop_constraint}
```

其中 `{shop_constraint}` 根据 NPC 是否有商店动态生成：
- 有商店：*"你可以将自己商店的物品作为任务奖励（物品等级需匹配玩家进度）。"*
- 无商店：*"你不经营商店，奖励以金币、经验、药剂等通用物资为主。"*

##### 6.6.3 任务情境触发条件

LLM 不应在每次对话中都尝试发布任务。以下是合理的触发场景：

- **玩家主动请求**：*"有没有什么任务？"* / *"有什么能帮忙的吗？"*
- **对话自然延伸**：玩家提到某个区域 / 物品 / 困难 / 需求时，NPC 可以自然地提出相关任务；
- **好感度里程碑**：好感度跨越关系（如从"陌生"升到"熟悉"时），NPC 可以视情况考虑主动提出初次委托；

**不应发布任务的场景**：
- 玩家正在诉说烦恼 / 进行情感交流时；
- 玩家明确表示不想做任务时（近期拒绝过任务）；
- 当前会话已有一个待确认的任务草案时（一次只处理一个任务）。

#### 6.7 设计决策记录

本节记录开发文档中与原始需求描述不同的调整，以及关键设计决策的理由。

| #  | 原始需求/问题                                            | 调整/决策                                                          | 理由                                                              |
|----|--------------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|
| D1 | "是否合理、是否有必要先进行工具判断再输出对话？"             | **完全合理且必要**。采用 ReAct 循环（Decision → Execute → Generate） | 这是 LangGraph/LangChain Agent 的标准模式；解决流式兼容问题；便于扩展更多工具 |
| D2 | "每次至少两次 LLM 调用，缓存命中如何优化？"                | 前缀一致原则：system + user 消息完全相同，变化内容追加在末尾          | 多数服务商的前缀缓存可将缓存 token 费用降至 0～50%，且决策轮输出通常很短 |
| D3 | LLM 直接输出 `"物品名#数量"` 格式？                       | **不**，LLM 输出结构化 JSON（item_name + count），后端拼接字符串    | 避免 LLM 幻觉导致格式错误；后端可逐项校验物品名和数量 |
| D4 | 奖励讨价还价上限？                                       | 从"120%"调整为 **150%**；每次调整幅度受好感度约束                   | 120% 过于接近基础上限，缺乏协商空间；150% 给足弹性但仍可控 |
| D5 | 任务发布是否应匹配 NPC 人格？                             | **是**，新增 6.6 节约束；普通 NPC 即使面对高进度玩家也只发普通任务    | 核心沉浸感要求：杂货铺老板不会发"去雪山打 Boss"的任务 |
| D6 | 是否需要引入 LangGraph？                                 | **是**，但渐进引入：Phase 1-2 先建数据层和工具，Phase 3 再接 LangGraph | 避免一步到位风险；数据层和工具逻辑独立于框架，可单元测试 |
| D7 | `session_task_drafts` 用什么存储？                        | 新增 SQLite 表（复用现有 `memory.db`），而非文件或 LangGraph 内部状态 | 持久化可靠，不依赖内存状态；服务重启后草案仍在；便于查询和清理 |
| D8 | agent_tasks.json 写入是否需要原子性？                     | **是**，先写临时文件再 rename                                      | 防止写入过程中崩溃导致 JSON 损坏 |
| D9 | 挑战类任务的奖励主要是经验？                              | 是，挑战类（修罗/地狱难度）奖励中经验占比应 ≥ 50%，金币占比可降低      | 高难度战斗回报应偏向角色成长而非经济收益，平衡游戏经济 |

### 7. 分步开发计划（可执行级）

#### Phase 1：数据访问层（Data Registry）

**目标**：将所有游戏数据文件解析为内存中可查询的结构，提供统一的查询 API。

| 步骤   | 具体任务                                                                         | 产出文件                          | 依赖     |
|--------|---------------------------------------------------------------------------------|----------------------------------|----------|
| 1.1    | 创建 `services/game_data/` 包，定义数据模型（Pydantic/dataclass）：`Item`, `Task`, `StageInfo`, `LootCrate`, `Recipe`, `ShopItem` 等 | `services/game_data/models.py`   | 无       |
| 1.2    | 实现 XML/JSON 通用解析工具：`list.xml` 文件发现 → 逐文件解析                       | `services/game_data/parsers.py`  | 1.1      |
| 1.3    | 实现 `ItemRegistry`：加载 `items/` 下所有物品 XML，建立 name → Item 索引，支持按 name/type/use/level 查询 | `services/game_data/item_registry.py` | 1.1, 1.2 |
| 1.4    | 实现 `TaskRegistry`：加载 `task/` 下所有任务 JSON + `task/text/` 文本 JSON        | `services/game_data/task_registry.py` | 1.1, 1.2 |
| 1.5    | 实现 `StageRegistry`：加载 `stages/` 下 `__list__.xml` + 各关卡 XML（解锁条件 + 箱子掉落） | `services/game_data/stage_registry.py` | 1.1, 1.2 |
| 1.6    | 实现 `ShopRegistry` + `KShopRegistry` + `CraftingRegistry`                      | 各自独立文件                      | 1.1, 1.2 |
| 1.7    | 实现 `GameDataRegistry` 聚合类 + 在 `core/startup.py` 中注册启动加载              | `services/game_data/registry.py` | 1.3-1.6  |
| 1.8    | 编写单元测试：至少覆盖物品查询、关卡筛选、奖励价值计算                              | `test/test_game_data/`          | 1.3-1.7  |

**验收标准**：`GameDataRegistry` 在服务启动时自动加载全部数据，所有查询方法可在 < 1ms 内返回结果。

#### Phase 2：Agent 工具定义与校验逻辑

**目标**：定义全部 LLM 工具 schema，实现两步式任务工具的执行逻辑和校验管线。

| 步骤   | 具体任务                                                                         | 产出文件                          | 依赖     |
|--------|---------------------------------------------------------------------------------|----------------------------------|----------|
| 2.1    | 定义 `prepare_task_context` 的 OpenAI Function schema（含 `task_type` 枚举、`reward_types` 结构），以及 `search_knowledge`、`update_npc_mood` 的 schema | `services/agent_tools/schemas.py` | Phase 1  |
| 2.2    | 实现 `prepare_task_context` 执行器：根据 `task_type` 分发到不同的数据筛选逻辑，调用各 Registry 查询后组装 `TaskContext` 返回（含通用字段 + 类型专属字段，详见 6.3.2） | `services/agent_tools/context_builder.py` | 2.1      |
| 2.3    | 定义 `draft_agent_task` + `update_task_draft` 的 JSON Schema（6.4.1）            | `services/agent_tools/schemas.py` | Phase 1  |
| 2.4    | 实现校验管线（Validation Pipeline，V1-V10），支持 `draft_agent_task` 全量校验和 `update_task_draft` 增量校验（仅校验变更字段） | `services/agent_tools/validator.py` | Phase 1  |
| 2.5    | 实现 `AgentTaskStore` + `AgentTextStore`：原子读写 `agent_tasks.json` / `agent_text.json` | `services/game_data/agent_task_store.py` | Phase 1 |
| 2.6    | 实现写入工具执行器（draft / update / confirm / cancel）                           | `services/agent_tools/task_tools.py` | 2.4, 2.5 |
| 2.7    | 实现 `session_task_drafts` 数据库表（在 `MemoryManager` 中扩展或独立模块）          | `services/task_draft_store.py`   | 无       |
| 2.8    | 实现奖励预算计算模块（类型倍率 + 好感修正 + 收集/持有加成），作为 `prepare_task_context` 的内部依赖而非独立工具 | `services/agent_tools/reward_calculator.py` | Phase 1 |
| 2.9    | 编写单元测试：`prepare_task_context` 各类型返回值正确性、`update_task_draft` 增量修改、校验管线全覆盖（合法/非法用例）、奖励计算边界测试 | `test/test_agent_tools/`        | 2.2-2.8  |

**验收标准**：`prepare_task_context` 对 12 种任务类型均能返回正确的筛选后数据；`update_task_draft` 可局部修改草案并触发增量校验；校验管线能拦截所有已知的非法输入。

#### Phase 3：LangGraph 集成与 ask 接口改造

**目标**：用 LangGraph 替换当前 `ask` / `ask_stream` 中的线性调用逻辑，实现三阶段管线。

| 步骤   | 具体任务                                                                         | 产出文件                          | 依赖     |
|--------|---------------------------------------------------------------------------------|----------------------------------|----------|
| 3.1    | 安装 `langgraph` + `langchain-openai` 依赖；确认与现有 `openai` 库兼容            | `requirements.txt` 更新          | 无       |
| 3.2    | 定义 `AgentState` TypedDict（6.2.2）                                             | `services/agent_graph/state.py`  | 无       |
| 3.3    | 实现 `prepare_context` 节点：重构现有 `_prepare_ask_context` 逻辑为独立函数        | `services/agent_graph/nodes.py`  | Phase 2  |
| 3.4    | 实现 `decision` 节点：非流式 LLM 调用（使用 `ChatOpenAI` + 全部工具定义）          | `services/agent_graph/nodes.py`  | 3.2, 3.3 |
| 3.5    | 实现 `tool_executor` 节点：分发 tool_calls → 调用 Phase 2 的工具执行器 → 返回结果  | `services/agent_graph/nodes.py`  | Phase 2  |
| 3.6    | 实现 `generate_response` 节点：流式 LLM 调用（无工具或仅 mood），支持 streaming event | `services/agent_graph/nodes.py`  | 3.2      |
| 3.7    | 实现 `parse_mood` + `post_process` 节点：复用现有情绪解析 + 好感度更新逻辑          | `services/agent_graph/nodes.py`  | 3.2      |
| 3.8    | 组装 LangGraph 图（6.2.5），实现条件路由                                           | `services/agent_graph/graph.py`  | 3.3-3.7  |
| 3.9    | 改造 `GameRAGService.ask_stream` 接口：调用编译后的 graph，转发 streaming events   | `services/game_rag_service.py`   | 3.8      |
| 3.10   | 改造 `GameRAGService.ask` 接口（非流式版本）：同步调用 graph                        | `services/game_rag_service.py`   | 3.8      |
| 3.11   | **向后兼容处理**：当 `progress_stage` 未传或工具调用失败时，降级到现有的简单对话流程  | `services/agent_graph/graph.py`  | 3.8-3.10 |
| 3.12   | Prompt 模板重构：按 6.5.3 的分层结构重组 system prompt（Layer 2 仅含 NPC 本身信息；Layer 3 按变化频率排列：同阵营 NPC → 玩家进度 → 身份描述 → 好感度 → 提到的其他 NPC → 待确认草案） | `services/agent_graph/prompts.py`| 3.3      |

**验收标准**：改造后的 ask/ask_stream 接口行为与现有接口兼容（无工具调用时输出一致）；工具调用链路可正常执行且不超时。

#### Phase 4：任务生成系统端到端

**目标**：在 Phase 3 的基础上，实现完整的任务发布 → 协商 → 确认 → 写入流程。

| 步骤   | 具体任务                                                                         | 产出文件                          | 依赖     |
|--------|---------------------------------------------------------------------------------|----------------------------------|----------|
| 4.1    | 完善两步式任务工具的 LLM 提示词：在 system prompt 中注入 `prepare_task_context` → `draft_agent_task` 的调用流程说明和任务发布规则（6.6） | `services/agent_graph/prompts.py`| Phase 3  |
| 4.2    | 实现任务草案的 DB 持久化（`session_task_drafts` 表的 CRUD + 局部更新操作）          | `services/task_draft_store.py`   | 2.7      |
| 4.3    | 实现协商状态管理：在 `prepare_context` 节点中加载待确认草案并注入 prompt；支持 `update_task_draft` 的讨价还价和局部修改流程 | `services/agent_graph/nodes.py`  | 4.2      |
| 4.4    | 实现 `confirm_agent_task` 的完整流程：校验 → 分配 ID → 原子写入 → 清理草案          | `services/agent_tools/task_tools.py` | 2.5, 2.6 |
| 4.5    | 实现草案自动过期清理（连续 3 轮未提及任务）                                        | `services/agent_graph/nodes.py`  | 4.2      |
| 4.6    | 端到端集成测试：模拟完整的 "对话 → `prepare_task_context` → `draft_agent_task` → 讨价还价（`update_task_draft`）→ 接受 → 写入" 流程，以及 "修改要求 → 局部修改" 和 "更改类型 → 重新走全流程" 两种协商路径 | `test/test_task_generation/`    | 4.1-4.5  |
| 4.7    | 前端接口适配（如需）：确认 SSE 事件格式不变，新增任务相关事件字段（可选）             | `api/game_api.py`                | 4.6      |

**验收标准**：从对话中自然发起任务 → 协商 → 接受 → `agent_tasks.json` 和 `agent_text.json` 中正确写入新任务 → 在游戏项目中可正常加载和接取。

#### Phase 5：优化、测试与迭代

**目标**：性能优化、边界测试、提示词调优、游戏平衡性验证。

| 步骤   | 具体任务                                                                         |
|--------|---------------------------------------------------------------------------------|
| 5.1    | Prompt 调优：针对不同模型（Gemini Flash / DeepSeek 等）测试任务生成质量，调整提示词  |
| 5.2    | 奖励平衡测试：生成大量测试任务，统计奖励价值分布，调整区间参数                        |
| 5.3    | 性能测试：测量三阶段管线的端到端延迟（含缓存命中率），确认对首字延迟的影响可接受       |
| 5.4    | 异常场景测试：LLM 输出格式异常 / 工具执行失败 / 文件写入失败的降级处理                |
| 5.5    | 建立物品/关卡黑名单（不适合作为任务目标的物品，如剧情道具、特殊 K 点商品等）           |
| 5.6    | 考虑后续扩展：技能数据（skills）的引入、NPC 间的任务联动、任务链（多步骤任务）等       |

**各 Phase 预估工作量**

| Phase   | 预估周期   | 核心风险                                     |
|---------|-----------|---------------------------------------------|
| Phase 1 | 3-5 天    | XML/JSON 格式边界情况（缺失字段、编码问题等）   |
| Phase 2 | 3-5 天    | 校验规则的完备性；奖励计算公式的平衡性           |
| Phase 3 | 5-7 天    | LangGraph 与现有 OpenAI 兼容 API 的集成调试    |
| Phase 4 | 3-5 天    | 任务协商的多轮状态管理；文件原子写入可靠性       |
| Phase 5 | 持续迭代   | 需要实际游戏测试数据来调优                      |

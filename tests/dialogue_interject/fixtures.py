"""共享契约 fixture（照抄游戏仓《AI对话内嵌插话-交接与契约fixture-2026-10-18》§3）。

只用于离线假流测试：不调真实模型、不读游戏仓、不写生产 data root。
编号与文档一一对应；不得为了迁就实现改写 fixture。
"""

from __future__ import annotations

# 固定样例身份（fixture §1）
DAILY_DIALOGUE_ID = "11111111-1111-4111-8111-111111111111"
DAILY_TURN_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
DAILY_TURN_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
TASK_DIALOGUE_ID = "22222222-2222-4222-8222-222222222222"
TASK_TURN = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
NPC = "Andy Law"


# ---------------------------------------------------------------------------
# 请求
# ---------------------------------------------------------------------------

DAILY_REMAINING = [
    {"index": 1, "name": NPC, "text": "有事说事。", "emotion": "普通"},
    {"index": 2, "name": NPC, "text": "没事就别挡道。", "emotion": "普通"},
]

DAILY_PLAYED_FIRST = [
    {"seq": 0, "role": "npc", "name": NPC, "text": "又是你。", "emotion": "普通"},
]


def daily_request(**overrides) -> dict:
    """F-01 请求（日常普通）；用 overrides 改字段派生 F-02/F-03/F-10/F-11。"""
    payload = {
        "mode": "daily",
        "dialogue_instance_id": DAILY_DIALOGUE_ID,
        "turn_id": DAILY_TURN_A,
        "npc": NPC,
        "player_input": "最近怎么样？",
        "session_title": "日常插话 · Andy Law · 2026-10-18 14:03",
        "dialogue": {
            "kind": "daily",
            "group_id": "daily_andy_01",
            "played": [dict(row) for row in DAILY_PLAYED_FIRST],
            "remaining": [dict(row) for row in DAILY_REMAINING],
            "reference_continuation": [],
        },
        "task_context": None,
        "limits": {"line_hard_max": 165, "line_prompt_max": 135},
    }
    payload.update(overrides)
    return payload


def daily_turn_b_request() -> dict:
    """F-04 Turn B：同一组第二轮；played 含上一轮已播，未播的进参考续写。"""
    payload = daily_request(turn_id=DAILY_TURN_B, player_input="先别说这个，我想问问任务。")
    payload["dialogue"]["played"] = [
        {"seq": 0, "role": "npc", "name": NPC, "text": "又是你。", "emotion": "普通"},
        {"seq": 1, "role": "player", "name": "$PC", "text": "最近怎么样？"},
        {"seq": 2, "role": "ai", "name": NPC, "text": "还能怎么样，老样子。", "emotion": "普通"},
    ]
    payload["dialogue"]["reference_continuation"] = [
        {"kind": "add", "name": NPC, "text": "你倒是难得过来一趟。", "emotion": "普通"},
        {"kind": "closing", "name": NPC, "text": "说吧，想聊什么？", "emotion": "普通"},
    ]
    return payload


TASK_REMAINING = [
    {"index": 1, "name": NPC, "text": "你先别急着答应。", "emotion": "普通"},
    {"index": 2, "name": NPC, "text": "北边仓库昨晚丢了东西。", "emotion": "普通"},
    {"index": 3, "name": NPC, "text": "去把清单上的货找回来。", "emotion": "普通"},
    {"index": 4, "name": NPC, "text": "回来之前别让我等太久。", "emotion": "普通"},
]


def task_request(turn_id: str = TASK_TURN, **overrides) -> dict:
    """F-05 请求（任务前）；F-05b/F-06/F-07/F-08/F-09 复用。"""
    payload = {
        "mode": "task",
        "dialogue_instance_id": TASK_DIALOGUE_ID,
        "turn_id": turn_id,
        "npc": NPC,
        "player_input": "这活儿我接了，先跟我说说细节。",
        "session_title": "任务插话 · 北边仓库 · 任务前 · 2026-10-18 14:10",
        "dialogue": {
            "kind": "task_pre",
            "group_id": "task_200124_pre",
            "played": [
                {"seq": 0, "role": "npc", "name": NPC, "text": "你来了。", "emotion": "普通"},
                {"seq": 1, "role": "npc", "name": NPC, "text": "先别急着走。", "emotion": "普通"},
            ],
            "remaining": [dict(row) for row in TASK_REMAINING],
            "reference_continuation": [],
        },
        "task_context": {
            "completed": {
                "task_id": 200123,
                "title": "南边哨站补给",
                "npc": NPC,
                "objectives": ["送回 3 箱弹药"],
                "rewards": ["金币 1200"],
                "outcome": "玩家已送回 3 箱弹药",
            },
            "upcoming": {
                "task_id": 200124,
                "title": "北边仓库失窃",
                "npc": NPC,
                "objectives": ["找回失窃货物", "返回交付"],
                "rewards": ["金币 2400", "声望 10"],
            },
        },
        "limits": {"line_hard_max": 165, "line_prompt_max": 135},
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# 假 LLM 原文（JSONL）
# ---------------------------------------------------------------------------

F01_JSONL = [
    '{"add":"还能怎么样，老样子。","emo":"普通"}\n',
    '{"add":"你倒是难得过来一趟。","emo":"普通"}\n',
    '{"closing":"说吧，想聊什么？","emo":"普通"}\n',
]

F02_LONG_ADD = (
    "北边那条路前几天塌了，现在只能绕南边过去；南边的桥也不稳，晚上没人敢走。"
    "你要是真打算去，先去货栈找老周，他知道一条旧路，不过那条路要穿过废楼，白天都容易迷路。"
    "别一个人去，带上家伙，也别信路上遇到的任何人，这年头谁都不容易，谁都想从你身上捞点好处。"
    "再说了，老周那人嘴碎，但消息准，问他之前先递包烟，别空着手去。"
    "要是路上真出了事，就往东边的旧水塔跑，那里地势高，能看见整个厂区，也方便我找人接你。"
)
F02_JSONL = [
    '{"add":"' + F02_LONG_ADD + '","emo":"普通"}\n',
    '{"closing":"记住了就早点回来。","emo":"普通"}\n',
]

F05_JSONL = [
    '{"add":"你倒是爽快。","emo":"普通"}\n',
    '{"n":2,"line":"北边仓库昨晚丢了一批货。","emo":"普通"}\n',
    '{"end":3}\n',
]

F05B_LONG_EDIT = (
    "北边仓库昨晚丢的那批货，我已经让人把清单理出来了，一共三箱弹药、两箱药品，还有一箱是给哨站的口粮。"
    "门锁是被撬开的，地上有拖拽的痕迹，对方不止一个人，而且对仓库的布局很熟。"
    "你先去后巷找看门的老赵，他昨晚值夜，应该记得有什么人进出；问他的时候别提货的事，就说是我让你去的。"
    "看完别急着追，先回来把情况告诉我，我们再决定从哪条线查。"
    "记住，别一个人进仓库，也别碰里面的东西，那些货上有记号，少了哪一箱我都看得出来。"
)
F05B_JSONL = [
    '{"add":"你倒是爽快。","emo":"普通"}\n',
    '{"n":3,"line":"' + F05B_LONG_EDIT + '","emo":"普通"}\n',
    '{"end":4}\n',
]

F06_JSONL = [
    '{"add":"这事不好办。","emo":"普通"}\n',
    '{"n":2,"remove":true}\n',
    '{"end":3}\n',
]

F07_JSONL = [
    '{"add":"……","emo":"普通"}\n',
    '{"n":3,"line":"清单就压在柜台底下。","emo":"普通"}\n',
    '{"end":4}\n',
]

F08_JSONL = [
    '{"add":"……","emo":"普通"}\n',
    '{"n":1,"line":"这事我只跟你说一遍。","emo":"普通"}\n',
    '{"n":2,"remove":true}\n',
]

F09_JSONL = [
    '{"mute":true}\n',
    '{"add":"行了，这事不用你操心。","emo":"普通"}\n',
    '{"end":1}\n',
]

F10_TEXT = [
    "好的，我明白了。玩家想和我聊天，我应该先安抚他，然后再……\n",
    "（这里不是 JSONL）\n",
]

F11_JSONL = [
    '{"add":"你先听我说。","emo":"普通"}\n',
]


# ---------------------------------------------------------------------------
# finalize（fixture §4.4）
# ---------------------------------------------------------------------------

F01_FINALIZE = {
    "dialogue_instance_id": DAILY_DIALOGUE_ID,
    "turn_id": DAILY_TURN_A,
    "transcript": [
        {"seq": 0, "role": "npc", "name": NPC, "text": "又是你。", "emotion": "普通"},
        {"seq": 1, "role": "player", "name": "$PC", "text": "最近怎么样？"},
        {"seq": 2, "role": "ai", "name": NPC, "text": "还能怎么样，老样子。", "emotion": "普通"},
        {"seq": 3, "role": "ai", "name": NPC, "text": "你倒是难得过来一趟。", "emotion": "普通"},
        {"seq": 4, "role": "ai", "name": NPC, "text": "说吧，想聊什么？", "emotion": "普通"},
    ],
}

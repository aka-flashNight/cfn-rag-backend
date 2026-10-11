"""任务发布临时关闭名单（用户裁定 2026-10-11）。

名单＝开发计划 §8.2「npc_state_db 有记录、map 无驻地」的 17 个 NPC
（含 `$PC_CHAR`）。这批 NPC 无驻地点，派任务时场景信息缺失；
正式修法（数据标注＋按标注拦截）另立项（§8.5-3 连带登记），
在此之前按临时措施处理：不注册任务路由工具、提示词不再引导玩家打开终端、
提示词改为「该 NPC 不可发布任务」约束；编排层对硬发路由调用一律忽略。

匹配口径＝小写＋去空白归一（与其它跨库名字匹配一致）。
"""

from __future__ import annotations

TASK_PUBLISH_DISABLED_NPCS: frozenset[str] = frozenset({
    "lochan",
    "暮光闪闪",
    "歌留多",
    "black-l",
    "枫",
    "王道",
    "lsy大人",
    "pd421",
    "和平",
    "乔恩",
    "劳埃德",
    "清水结衣",
    "千夏沐晓",
    "云吞魔芋",
    "prophet2",
    "短波无线电",
    "$pc_char",
})


def _norm(name: str) -> str:
    return "".join(str(name or "").lower().split())


def is_task_publish_disabled(npc_name: str) -> bool:
    return _norm(npc_name) in TASK_PUBLISH_DISABLED_NPCS

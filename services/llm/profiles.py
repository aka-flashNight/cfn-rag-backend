"""模型 Profile 注册表：按模型名匹配思考参数 / 视觉能力 / 特殊行为。

对应 docs/v3-developer/02-LLM接入层.md §3.1。表中参数全部视为「可能错」，
正确性由 client.py 的降级链兜底；新增/修正模型只改这一处。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    """单个模型的调用配置。thinking_body 中的键全部经 extra_body 进入请求体。"""

    match: str                        # 小写子串匹配（先匹配先中）
    vision: bool | None               # None = 探测型（发图失败自动降级）
    thinking_body: dict[str, Any] = field(default_factory=dict)
    retransmit_reasoning: bool = False  # Qwen 系开思考时需要把 reasoning_content 回传
    note: str = ""


# 先匹配先中：更具体的子串放前面（deepseek-v4-flash-vision-exp 必须在 deepseek-v4-flash 前）
PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        match="glm-5.3-flash",
        vision=True,
        thinking_body={"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
        note="思考不能关（disabled 会报错）；强度参数名以官方文档为准，报错走降级链",
    ),
    ModelProfile(
        match="glm-5.3",
        vision=True,
        thinking_body={"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
        note="同 glm-5.3-flash",
    ),
    ModelProfile(
        match="deepseek-v4-flash-vision-exp",
        vision=True,
        thinking_body={"thinking": {"type": "disabled"}},
    ),
    ModelProfile(
        match="deepseek-v4-flash",
        vision=False,
        thinking_body={"thinking": {"type": "disabled"}},
        note="纯文本版",
    ),
    ModelProfile(
        match="qwen3.8-flash",
        vision=True,
        thinking_body={"enable_thinking": False},
        retransmit_reasoning=True,
        note="百炼默认即关；另有 /no_think 软开关可写进 prompt 兜底",
    ),
    ModelProfile(
        match="gpt-5.6-luna",
        vision=True,
        thinking_body={"reasoning_effort": "none"},
        note="none=真·非思考（官方）",
    ),
    ModelProfile(
        match="doubao-seed-2-0-lite",
        vision=True,
        thinking_body={"thinking": {"type": "disabled"}},
        note="方舟统一参数",
    ),
    ModelProfile(
        match="kimi-k3",
        vision=True,
        thinking_body={},
        note="思考恒开无法关；不传 temperature",
    ),
    ModelProfile(
        match="minimax-m3",
        vision=True,
        thinking_body={"thinking": {"type": "disabled"}},
    ),
    ModelProfile(
        match="gemini-3",
        vision=True,
        thinking_body={"thinking_level": "low"},
        note="未见完全关闭档；low 为最低",
    ),
)

# 匹配不到 = DEFAULT_PROFILE：不传任何思考参数、vision 探测型，新模型零配置可用
DEFAULT_PROFILE = ModelProfile(match="", vision=None, thinking_body={})


def get_profile(model_name: str) -> ModelProfile:
    """按模型名小写子串匹配 Profile；匹配不到返回 DEFAULT_PROFILE。"""
    key = (model_name or "").strip().lower()
    if not key:
        return DEFAULT_PROFILE
    for profile in PROFILES:
        if profile.match and profile.match in key:
            return profile
    return DEFAULT_PROFILE

"""模型 Profile 注册表测试（02 §6.2：表中每个模型命中正确 Profile；未知模型落 DEFAULT）。"""

from __future__ import annotations

from services.llm.profiles import DEFAULT_PROFILE, ModelProfile, get_profile


def test_glm_53_flash():
    p = get_profile("glm-5.3-flash")
    assert p.match == "glm-5.3-flash"  # 更具体的子串先命中，不落到 glm-5.3
    assert p.vision is True
    assert p.thinking_body == {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}


def test_glm_53_other_variant():
    p = get_profile("GLM-5.3-Air")
    assert p.match == "glm-5.3"
    assert p.vision is True


def test_deepseek_flash_profile():
    """官方只有 deepseek-flash 一个 flash 模型：纯文本 + 关思考。"""
    p = get_profile("deepseek-flash")
    assert p.match == "deepseek-flash"
    assert p.vision is False
    assert p.thinking_body == {"thinking": {"type": "disabled"}}
    assert p.retransmit_reasoning is True  # 思考开时工具多轮必须回传


def test_qwen_profile():
    p = get_profile("qwen3.8-flash")
    assert p.vision is True
    assert p.thinking_body == {"enable_thinking": False}
    assert p.retransmit_reasoning is True


def test_gpt_luna_reasoning_none():
    p = get_profile("gpt-5.6-luna")
    assert p.vision is True
    assert p.thinking_body == {"reasoning_effort": "none"}


def test_doubao_minimax_gemini_kimi():
    assert get_profile("doubao-seed-2-0-lite").thinking_body == {"thinking": {"type": "disabled"}}
    minimax = get_profile("minimax-m3")
    assert minimax.thinking_body == {"thinking": {"type": "disabled"}}
    assert minimax.retransmit_reasoning is True
    gemini = get_profile("gemini-3-flash")
    assert gemini.match == "gemini-3"
    assert gemini.thinking_body == {"thinking_level": "low"}
    assert gemini.retransmit_reasoning is False  # 需要的是 thought_signature，不是 reasoning_content
    kimi = get_profile("kimi-k3")
    assert kimi.thinking_body == {}  # 恒开无法关
    assert kimi.vision is True
    assert kimi.retransmit_reasoning is True
    assert get_profile("kimi-k3.1-pro").retransmit_reasoning is True


def test_unknown_model_falls_back_to_default():
    p = get_profile("totally-new-model-9000")
    assert p is DEFAULT_PROFILE
    assert p.vision is None  # 探测型
    assert p.thinking_body == {}  # 不传任何思考参数


def test_empty_model_name_uses_default():
    assert get_profile("") is DEFAULT_PROFILE


def test_case_insensitive_matching():
    assert get_profile("DeepSeek-Flash").match == "deepseek-flash"


def test_default_profile_is_detection_type():
    assert DEFAULT_PROFILE.vision is None
    assert isinstance(DEFAULT_PROFILE, ModelProfile)

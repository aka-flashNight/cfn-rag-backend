"""模型 Profile 注册表测试（02 §6.2：表中每个模型命中正确 Profile；未知模型落 DEFAULT）。"""

from __future__ import annotations

from services.llm.profiles import DEFAULT_PROFILE, ModelProfile, get_profile


def test_glm_53_flash():
    p = get_profile("glm-5.3-flash")
    assert p.vision is True
    assert p.thinking_body == {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}


def test_glm_53_other_variant():
    p = get_profile("GLM-5.3-Air")
    assert p.match == "glm-5.3"
    assert p.vision is True


def test_deepseek_vision_precedence():
    """更具体的子串必须先命中（vision-exp 不落到纯文本档）。"""
    p = get_profile("deepseek-v4-flash-vision-exp")
    assert p.match == "deepseek-v4-flash-vision-exp"
    assert p.vision is True
    p2 = get_profile("deepseek-v4-flash")
    assert p2.match == "deepseek-v4-flash"
    assert p2.vision is False
    assert p2.thinking_body == {"thinking": {"type": "disabled"}}


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
    assert get_profile("minimax-m3").thinking_body == {"thinking": {"type": "disabled"}}
    gemini = get_profile("gemini-3-flash")
    assert gemini.match == "gemini-3"
    assert gemini.thinking_body == {"thinking_level": "low"}
    kimi = get_profile("kimi-k3")
    assert kimi.thinking_body == {}  # 恒开无法关
    assert kimi.vision is True


def test_unknown_model_falls_back_to_default():
    p = get_profile("totally-new-model-9000")
    assert p is DEFAULT_PROFILE
    assert p.vision is None  # 探测型
    assert p.thinking_body == {}  # 不传任何思考参数


def test_empty_model_name_uses_default():
    assert get_profile("") is DEFAULT_PROFILE


def test_case_insensitive_matching():
    assert get_profile("DeepSeek-V4-Flash-Vision-Exp").match == "deepseek-v4-flash-vision-exp"


def test_default_profile_is_detection_type():
    assert DEFAULT_PROFILE.vision is None
    assert isinstance(DEFAULT_PROFILE, ModelProfile)

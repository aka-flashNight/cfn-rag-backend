"""services/llm：LLM 接入层（client / profiles / meta / errors）。

对应 docs/v3-developer/02-LLM接入层.md。对外主要入口：
- LLMClient.for_config(cfg)：缓存复用的统一调用客户端
- ChatRequest / ChatResult / StreamEvent
- get_profile(model)：模型 Profile
- split_meta / split_meta_events / meta_prompt_block：meta 行协议
"""

from services.llm.client import (
    ChatRequest,
    ChatResult,
    LLMClient,
    LLMConfig,
    LLMPurpose,
    StreamEvent,
    is_image_unsupported_model,
    mark_image_unsupported,
    normalize_usage,
)
from services.llm.errors import (
    LLMAuthError,
    LLMError,
    LLMNetworkError,
    LLMParamError,
    LLMQuotaError,
    LLMUnsupportedImageError,
    LLMUnsupportedToolsError,
    is_image_unsupported_error,
    is_param_error,
    is_tools_unsupported_error,
)
from services.llm.meta import (
    Meta,
    MetaAct,
    meta_prompt_block,
    parse_meta_obj,
    split_meta,
    split_meta_events,
)
from services.llm.profiles import DEFAULT_PROFILE, ModelProfile, get_profile

__all__ = [
    "ChatRequest",
    "ChatResult",
    "LLMClient",
    "LLMConfig",
    "LLMPurpose",
    "StreamEvent",
    "is_image_unsupported_model",
    "mark_image_unsupported",
    "normalize_usage",
    "LLMAuthError",
    "LLMError",
    "LLMNetworkError",
    "LLMParamError",
    "LLMQuotaError",
    "LLMUnsupportedImageError",
    "LLMUnsupportedToolsError",
    "is_image_unsupported_error",
    "is_param_error",
    "is_tools_unsupported_error",
    "Meta",
    "MetaAct",
    "meta_prompt_block",
    "parse_meta_obj",
    "split_meta",
    "split_meta_events",
    "DEFAULT_PROFILE",
    "ModelProfile",
    "get_profile",
]

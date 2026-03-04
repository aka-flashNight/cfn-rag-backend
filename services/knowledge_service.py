from typing import Any, Dict, List

import httpx

from core.config import Settings
from schemas.knowledge_schema import AskRequest, AskResponse


class KnowledgeService:
    """
    基于 Gemini 的知识问答 Service。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def ask_question(self, payload: AskRequest) -> AskResponse:
        """
        调用 Gemini 2.5 Flash 生成回答。
        """

        if not self._settings.gemini_api_key:
            return AskResponse(
                answer="Gemini API Key 未配置，请在环境变量或 .env 中设置 GEMINI_API_KEY。",
                sources=[],
            )

        url: str = (
            f"{self._settings.gemini_api_base}/models/"
            f"{self._settings.gemini_model}:generateContent"
        )

        request_body: Dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": payload.query,
                        }
                    ]
                }
            ]
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json=request_body,
                params={"key": self._settings.gemini_api_key},
            )
            response.raise_for_status()
            data: Dict[str, Any] = response.json()

        answer_text: str = self._extract_answer_text(data)
        sources: List[str] = []

        if not answer_text:
            answer_text = "未从 Gemini 获取到有效回答。"

        return AskResponse(answer=answer_text, sources=sources)

    @staticmethod
    def _extract_answer_text(data: Dict[str, Any]) -> str:
        """
        从 Gemini 响应中提取文本答案。
        """

        candidates: List[Dict[str, Any]] = data.get("candidates") or []
        if not candidates:
            return ""

        content: Dict[str, Any] = candidates[0].get("content") or {}
        parts: List[Dict[str, Any]] = content.get("parts") or []

        texts: List[str] = []
        for part in parts:
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)

        return "".join(texts).strip()


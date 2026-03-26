from __future__ import annotations

from typing import Iterable, Mapping, Any
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class LLMClient:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._llm = None

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self._llm = ChatOpenAI(
                model=self.model,
                api_key=api_key,
                temperature=0,
                streaming=True,
            )

    async def stream(
        self,
        prompt: str,
        tools: list[Mapping[str, object]] | None = None,
    ):
        del tools

        if not self._llm:
            for token in _fallback_stream(prompt):
                yield token
            return

        async for chunk in self._llm.astream(prompt):
            content = _normalize_chunk_content(chunk.content)
            if content:
                yield content


def get_model() -> LLMClient:
    return LLMClient()


def _normalize_chunk_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    return ""


def _fallback_stream(prompt: str) -> Iterable[str]:
    text = (
        "LLM provider not configured. "
        "Set OPENAI_API_KEY to stream real model output. "
        f"You asked: {prompt}"
    )
    for word in text.split(" "):
        yield word + " "

from __future__ import annotations

from typing import Any, Iterable, Sequence
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage
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

    def has_model(self) -> bool:
        return self._llm is not None

    @property
    def chat_model(self) -> ChatOpenAI | None:
        return self._llm

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage | tuple[str, str]],
        tools: list[Any] | None = None,
    ) -> AIMessage:
        if not self._llm:
            text = "LLM provider not configured. Set OPENAI_API_KEY to use model output."
            return AIMessage(content=text)

        runnable = self._llm.bind_tools(tools) if tools else self._llm
        return await runnable.ainvoke(messages)

    async def astream(
        self,
        messages: Sequence[BaseMessage | tuple[str, str]],
        tools: list[Any] | None = None,
    ):
        if not self._llm:
            fallback = "LLM provider not configured. Set OPENAI_API_KEY to stream real model output."
            for token in _fallback_stream(fallback):
                yield token
            return

        runnable = self._llm.bind_tools(tools) if tools else self._llm
        async for chunk in runnable.astream(messages):
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


def _fallback_stream(text: str) -> Iterable[str]:
    for word in text.split(" "):
        yield word + " "

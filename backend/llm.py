from __future__ import annotations

from typing import Iterable, Mapping
import os
from dotenv import load_dotenv

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - optional dependency fallback
    AsyncOpenAI = None

load_dotenv()


class LLMClient:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        self._client = AsyncOpenAI(api_key=api_key) if AsyncOpenAI and api_key else None

    async def stream(
        self,
        prompt: str,
        tools: list[Mapping[str, object]] | None = None,
    ):
        if not self._client:
            for token in _fallback_stream(prompt):
                yield token
            return

        messages = [{"role": "user", "content": prompt}]
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            tools=tools,
        )

        async for event in stream:
            choice = event.choices[0]
            delta = choice.delta.content
            if delta:
                yield delta


def get_model() -> LLMClient:
    return LLMClient()


def _fallback_stream(prompt: str) -> Iterable[str]:
    text = (
        "LLM provider not configured. "
        "Set OPENAI_API_KEY to stream real model output. "
        f"You asked: {prompt}"
    )
    for word in text.split(" "):
        yield word + " "

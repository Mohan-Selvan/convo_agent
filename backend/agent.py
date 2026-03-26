from __future__ import annotations

import asyncio
from typing import Iterable

from backend.llm import get_model


async def run_agent(
    query: str,
    history: list[dict[str, str]] | None = None,
    use_rag: bool = False,
    use_tools: bool = False,
):
    llm = get_model()

    context_parts = []
    if use_rag:
        context_parts.append("RAG is enabled (placeholder context).")
    if use_tools:
        context_parts.append("Tools are enabled (placeholder outputs).")

    context = "\n".join(context_parts) if context_parts else "No external context."
    prompt = f"""You are a helpful assistant.

Context:
{context}

User query:
{query}
"""

    async for token in llm.stream(prompt):
        yield token


def run_agent_sync(
    query: str,
    history: list[dict[str, str]] | None = None,
    use_rag: bool = False,
    use_tools: bool = False,
) -> Iterable[str]:
    loop = asyncio.new_event_loop()
    agen = run_agent(
        query=query,
        history=history,
        use_rag=use_rag,
        use_tools=use_tools,
    )
    try:
        while True:
            yield loop.run_until_complete(agen.__anext__())
    except StopAsyncIteration:
        return
    finally:
        loop.close()

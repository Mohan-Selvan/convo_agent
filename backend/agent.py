from __future__ import annotations

import asyncio
from typing import Any, Iterable

from backend.llm import get_model
from backend.rag import Retriever


def build_context(
    query: str,
    use_rag: bool = False,
    use_tools: bool = False,
) -> dict[str, Any]:
    retrieved_docs: list[str] = []
    context_parts: list[str] = []

    if use_rag:
        retrieved_docs = Retriever().search(query)
        if retrieved_docs:
            rag_block = "\n".join(f"- {doc}" for doc in retrieved_docs)
            context_parts.append(f"Retrieved knowledge:\n{rag_block}")
        else:
            context_parts.append("RAG is enabled, but no documents were found.")

    if use_tools:
        context_parts.append("Tools are enabled (placeholder outputs).")

    context = "\n\n".join(context_parts) if context_parts else "No external context."
    return {
        "context": context,
        "retrieved_docs": retrieved_docs,
        "tool_outputs": [],
    }


async def run_agent(
    query: str,
    history: list[dict[str, str]] | None = None,
    use_rag: bool = False,
    use_tools: bool = False,
    context_payload: dict[str, Any] | None = None,
):
    llm = get_model()
    payload = context_payload or build_context(
        query=query,
        use_rag=use_rag,
        use_tools=use_tools,
    )

    prompt = f"""You are a helpful assistant.

Context:
{payload['context']}

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
    context_payload: dict[str, Any] | None = None,
) -> Iterable[str]:
    loop = asyncio.new_event_loop()
    agen = run_agent(
        query=query,
        history=history,
        use_rag=use_rag,
        use_tools=use_tools,
        context_payload=context_payload,
    )
    try:
        while True:
            yield loop.run_until_complete(agen.__anext__())
    except StopAsyncIteration:
        return
    finally:
        loop.close()

from __future__ import annotations

import asyncio
from typing import Any, Iterable

from backend.llm import get_model
from backend.memory import InMemoryDatabase
from backend.rag import Retriever

_db = InMemoryDatabase()


def get_conversation_history(session_id: str) -> list[dict[str, str]]:
    return _db.get_messages(session_id)


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


def _format_history(messages: list[dict[str, str]], limit: int = 12) -> str:
    tail = messages[-limit:]
    if not tail:
        return "(none)"

    lines: list[str] = []
    for msg in tail:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def run_agent(
    session_id: str,
    query: str,
    use_rag: bool = False,
    use_tools: bool = False,
    context_payload: dict[str, Any] | None = None,
):
    _db.add_message(session_id=session_id, role="user", content=query)

    llm = get_model()
    payload = context_payload or build_context(
        query=query,
        use_rag=use_rag,
        use_tools=use_tools,
    )
    history = _db.get_messages(session_id)
    history_text = _format_history(history)

    prompt = f"""You are a helpful assistant.

Conversation history:
{history_text}

Context:
{payload['context']}

User query:
{query}
"""

    chunks: list[str] = []
    async for token in llm.stream(prompt):
        chunks.append(token)
        yield token

    _db.add_message(session_id=session_id, role="assistant", content="".join(chunks))


def run_agent_sync(
    session_id: str,
    query: str,
    use_rag: bool = False,
    use_tools: bool = False,
    context_payload: dict[str, Any] | None = None,
) -> Iterable[str]:
    loop = asyncio.new_event_loop()
    agen = run_agent(
        session_id=session_id,
        query=query,
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

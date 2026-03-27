from __future__ import annotations

import asyncio
from typing import Any, Iterable

from langchain.agents import create_agent

from backend.llm import get_model
from backend.memory import InMemoryDatabase
from backend.rag import Retriever
from backend.tools import get_tools

_db = InMemoryDatabase()


def get_conversation_history(session_id: str) -> list[dict[str, str]]:
    return _db.get_messages(session_id)


def get_debug_state(session_id: str) -> dict[str, Any]:
    return _db.get_debug_state(session_id)


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
):
    _db.add_message(session_id=session_id, role="user", content=query)

    llm = get_model()
    history = _db.get_messages(session_id)
    history_text = _format_history(history)

    retrieved_docs = Retriever().search(query)
    rag_block = "\n".join(f"- {doc}" for doc in retrieved_docs) if retrieved_docs else "- No retrieved documents found."

    system_prompt = (
        "You are a helpful assistant. Use tools when needed for live finance data. "
        "Ground answers in retrieved knowledge and tool outputs when available."
    )
    user_prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Retrieved knowledge:\n{rag_block}\n\n"
        f"User query:\n{query}"
    )

    route = "rag"
    tool_outputs: list[dict[str, Any]] = []
    chunks: list[str] = []
    pending_tool_args: dict[str, Any] = {}

    if llm.has_model():
        agent = create_agent(
            model=llm.chat_model,
            tools=get_tools(),
            system_prompt=system_prompt,
        )

        async for event in agent.astream_events(
            {"messages": [("user", user_prompt)]},
            version="v2",
        ):
            event_type = event.get("event", "")
            run_id = str(event.get("run_id", ""))

            if event_type == "on_tool_start":
                pending_tool_args[run_id] = event.get("data", {}).get("input")

            if event_type == "on_tool_end":
                route = "hybrid"
                tool_outputs.append(
                    {
                        "tool": event.get("name", "unknown_tool"),
                        "args": pending_tool_args.pop(run_id, None),
                        "ok": True,
                        "data": event.get("data", {}).get("output"),
                    }
                )

            if event_type == "on_tool_error":
                route = "hybrid"
                tool_outputs.append(
                    {
                        "tool": event.get("name", "unknown_tool"),
                        "args": pending_tool_args.pop(run_id, None),
                        "ok": False,
                        "error": str(event.get("data", {}).get("error", "tool execution failed")),
                    }
                )

            if event_type == "on_chat_model_stream":
                text = _extract_stream_text(event.get("data", {}).get("chunk"))
                if text:
                    chunks.append(text)
                    yield text
    else:
        messages: list[Any] = [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
        async for token in llm.astream(messages):
            chunks.append(token)
            yield token

    _db.set_debug_state(
        session_id,
        {
            "route": route,
            "retrieved_docs": retrieved_docs,
            "tool_outputs": tool_outputs,
            "latency_ms": None,
        },
    )

    _db.add_message(session_id=session_id, role="assistant", content="".join(chunks))


def run_agent_sync(
    session_id: str,
    query: str,
) -> Iterable[str]:
    loop = asyncio.new_event_loop()
    agen = run_agent(
        session_id=session_id,
        query=query,
    )
    try:
        while True:
            yield loop.run_until_complete(agen.__anext__())
    except StopAsyncIteration:
        return
    finally:
        loop.close()


def _extract_stream_text(chunk: Any) -> str:
    if chunk is None:
        return ""

    content = getattr(chunk, "content", None)
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

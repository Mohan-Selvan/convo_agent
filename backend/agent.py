from __future__ import annotations

import os
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from backend.llm import get_chat_model
from backend.memory import get_memory_db
from backend.rag import Retriever
from backend.tools import get_tools


_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Use retrieved context as the primary source of truth. "
    "Use tools when the user needs live financial data."
)


def answer_query(query: str, session_id: str = "default") -> str:
    if not query or not query.strip():
        return "Please provide a query."

    memory = get_memory_db()
    history = memory.get_messages(session_id)

    top_k = int(os.getenv("RAG_TOP_K", "4"))
    docs = Retriever().search(query, limit=top_k)

    if docs:
        context_block = "\n".join(f"- {doc}" for doc in docs)
    else:
        context_block = "- No relevant documents found in the knowledge base."

    current_turn = (
        f"Retrieved knowledge base context:\n{context_block}\n\n"
        f"User query:\n{query}"
    )

    agent = create_agent(
        model=get_chat_model(),
        tools=get_tools(),
        system_prompt=_SYSTEM_PROMPT,
    )

    result = agent.invoke({"messages": [*history, ("user", current_turn)]})
    answer = _extract_answer_text(result)

    memory.append_message(session_id, "user", query)
    memory.append_message(session_id, "assistant", answer)
    return answer


def _extract_answer_text(result: Any) -> str:
    messages = result.get("messages") if isinstance(result, dict) else None
    if not isinstance(messages, list):
        return "No response generated."

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str):
                return content.strip() or "No response generated."
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
                joined = "".join(text_parts).strip()
                if joined:
                    return joined

    return "No response generated."

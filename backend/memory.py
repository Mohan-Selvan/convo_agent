from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Message:
    role: str
    content: str


class InMemoryDatabase:
    def __init__(self):
        self._conversations: dict[str, list[Message]] = {}
        self._debug_state: dict[str, dict[str, Any]] = {}

    # TODO: Replace this in-memory store with a persistent database before production.
    def add_message(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._conversations:
            self._conversations[session_id] = []
        self._conversations[session_id].append(Message(role=role, content=content))

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        messages = self._conversations.get(session_id, [])
        return [{"role": m.role, "content": m.content} for m in messages]

    def set_debug_state(self, session_id: str, state: dict[str, Any]) -> None:
        self._debug_state[session_id] = state

    def get_debug_state(self, session_id: str) -> dict[str, Any]:
        return self._debug_state.get(
            session_id,
            {
                "route": "rag",
                "retrieved_docs": [],
                "tool_outputs": [],
                "latency_ms": None,
            },
        )

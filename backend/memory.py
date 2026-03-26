from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str


class InMemoryDatabase:
    def __init__(self):
        self._conversations: dict[str, list[Message]] = {}

    # TODO: Replace this in-memory store with a persistent database (something like.. PostgreSQL/Redis).
    def add_message(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._conversations:
            self._conversations[session_id] = []
        self._conversations[session_id].append(Message(role=role, content=content))

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        messages = self._conversations.get(session_id, [])
        return [{"role": m.role, "content": m.content} for m in messages]

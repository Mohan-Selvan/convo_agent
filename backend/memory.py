from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict


class InMemoryDatabase:
    """Simple process-local chat memory store."""

    # TODO: Replace with persistent storage (PostgreSQL/Redis) for production.
    def __init__(self):
        self._messages: DefaultDict[str, list[tuple[str, str]]] = defaultdict(list)

    def get_messages(self, session_id: str) -> list[tuple[str, str]]:
        return list(self._messages.get(session_id, []))

    def append_message(self, session_id: str, role: str, content: str) -> None:
        self._messages[session_id].append((role, content))

    def clear_session(self, session_id: str) -> None:
        self._messages.pop(session_id, None)


_db = InMemoryDatabase()


def get_memory_db() -> InMemoryDatabase:
    return _db

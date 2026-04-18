"""Short-term memory — current session context, held in RAM."""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ShortEntry:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class ShortTermMemory:
    """In-memory deque — wiped when the process ends."""

    def __init__(self, max_size: int = 100):
        self._entries: deque[ShortEntry] = deque(maxlen=max_size)
        self.session_start = datetime.now()

    def add(self, role: str, content: str, metadata: Optional[dict] = None):
        self._entries.append(ShortEntry(role=role, content=content, metadata=metadata or {}))

    def get_messages(self, limit: Optional[int] = None) -> list[dict]:
        entries = list(self._entries)
        if limit:
            entries = entries[-limit:]
        return [{"role": e.role, "content": e.content} for e in entries if e.role in ("user", "assistant")]

    def get_recent_text(self, n: int = 5) -> str:
        recent = list(self._entries)[-n:]
        lines = []
        for e in recent:
            ts = e.timestamp.strftime("%H:%M")
            lines.append(f"[{ts}] {e.role.upper()}: {e.content[:200]}")
        return "\n".join(lines)

    def clear(self):
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

"""Sacred memory — immutable core facts about the user. Never deleted, only appended."""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from jailson.config.settings import MEMORY_DIR


class SacredMemory:
    """Protected JSON store. Once written, entries are never deleted.

    This holds the most fundamental truths known about the user — name,
    core values, immutable preferences, and critical personal facts.
    """

    def __init__(self):
        self.path = MEMORY_DIR / "sacred.json"
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {"entries": [], "created_at": datetime.now().isoformat()}

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def add(self, content: str, category: str = "fact", source: str = "user"):
        """Add a sacred entry. Cannot be deleted."""
        entry = {
            "id": len(self._data["entries"]) + 1,
            "timestamp": datetime.now().isoformat(),
            "content": content,
            "category": category,
            "source": source,
        }
        self._data["entries"].append(entry)
        self._save()

    def get_all(self, category: Optional[str] = None) -> list[dict]:
        entries = self._data.get("entries", [])
        if category:
            entries = [e for e in entries if e.get("category") == category]
        return entries

    def get_context_for_prompt(self) -> str:
        entries = self.get_all()
        if not entries:
            return ""
        lines = ["## Memória Sagrada (núcleo imutável)"]
        for e in entries:
            lines.append(f"- [{e['category']}] {e['content']}")
        return "\n".join(lines)

    def search(self, query: str) -> list[dict]:
        query_lower = query.lower()
        return [e for e in self._data.get("entries", []) if query_lower in e["content"].lower()]

    def __len__(self) -> int:
        return len(self._data.get("entries", []))

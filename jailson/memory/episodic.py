"""Episodic memory — specific events with temporal context (SQLite)."""
import sqlite3
import json
from datetime import datetime
from typing import Optional

from jailson.config.settings import MEMORY_DIR


class EpisodicMemory:
    """Stores memorable events with importance ratings and tags."""

    def __init__(self):
        self.db_path = MEMORY_DIR / "episodic.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    importance INTEGER DEFAULT 5,
                    tags TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_importance ON events(importance);
            """)

    def add_event(self, title: str, description: str,
                  importance: int = 5, tags: list = None, metadata: dict = None):
        """Record a memorable event. Importance: 1 (trivial) to 10 (life-changing)."""
        importance = max(1, min(10, importance))
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO events (timestamp, title, description, importance, tags, metadata) VALUES (?,?,?,?,?,?)",
                (
                    datetime.now().isoformat(),
                    title,
                    description,
                    importance,
                    json.dumps(tags or []),
                    json.dumps(metadata or {}),
                ),
            )

    def get_recent_events(self, limit: int = 20, min_importance: int = 1) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT title, description, importance, tags, timestamp FROM events "
                "WHERE importance >= ? ORDER BY timestamp DESC LIMIT ?",
                (min_importance, limit),
            ).fetchall()
        return [
            {
                "title": r[0],
                "description": r[1],
                "importance": r[2],
                "tags": json.loads(r[3]),
                "timestamp": r[4],
            }
            for r in rows
        ]

    def get_significant_events(self, min_importance: int = 7, limit: int = 10) -> list[dict]:
        return self.get_recent_events(limit=limit, min_importance=min_importance)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT title, description, importance, timestamp FROM events "
                "WHERE title LIKE ? OR description LIKE ? ORDER BY importance DESC, timestamp DESC LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [{"title": r[0], "description": r[1], "importance": r[2], "timestamp": r[3]} for r in rows]

    def get_context_for_prompt(self) -> str:
        significant = self.get_significant_events(min_importance=6, limit=5)
        if not significant:
            return ""
        lines = ["## Eventos Significativos (memória episódica)"]
        for e in significant:
            ts = e["timestamp"][:10]
            lines.append(f"- [{ts}] ({e['importance']}/10) **{e['title']}**: {e['description'][:150]}")
        return "\n".join(lines)

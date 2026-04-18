"""Medium-term memory — interactions and insights from the last 30 days (SQLite)."""
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from jailson.config.settings import MEMORY_DIR, MEDIUM_TERM_DAYS


class MediumTermMemory:
    """SQLite store for recent interactions and short-horizon insights."""

    def __init__(self):
        self.db_path = MEMORY_DIR / "medium_term.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    session_id TEXT,
                    metadata TEXT DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT,
                    metadata TEXT DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_inter_ts ON interactions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_insights_ts ON insights(timestamp);
            """)

    def add_interaction(self, role: str, content: str, session_id: str = "", metadata: dict = None):
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO interactions (timestamp, role, content, session_id, metadata) VALUES (?,?,?,?,?)",
                (datetime.now().isoformat(), role, content, session_id, json.dumps(metadata or {})),
            )

    def add_insight(self, content: str, source: str = "session", metadata: dict = None):
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO insights (timestamp, content, source, metadata) VALUES (?,?,?,?)",
                (datetime.now().isoformat(), content, source, json.dumps(metadata or {})),
            )

    def get_recent_interactions(self, days: int = 7, limit: int = 50) -> list[dict]:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT role, content, timestamp FROM interactions WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]

    def get_recent_insights(self, days: int = 14, limit: int = 20) -> list[str]:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT content FROM insights WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        return [r[0] for r in rows]

    def purge_old(self):
        """Delete entries older than MEDIUM_TERM_DAYS."""
        cutoff = (datetime.now() - timedelta(days=MEDIUM_TERM_DAYS)).isoformat()
        with sqlite3.connect(self.db_path) as con:
            con.execute("DELETE FROM interactions WHERE timestamp < ?", (cutoff,))
            con.execute("DELETE FROM insights WHERE timestamp < ?", (cutoff,))

    def get_summary_context(self) -> str:
        insights = self.get_recent_insights()
        if not insights:
            return ""
        lines = ["## Insights Recentes (últimas 2 semanas)"]
        for ins in insights[:10]:
            lines.append(f"- {ins}")
        return "\n".join(lines)

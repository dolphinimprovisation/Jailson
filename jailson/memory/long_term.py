"""Long-term memory — permanent facts, patterns, and learned preferences (SQLite)."""
import sqlite3
import json
from datetime import datetime
from typing import Optional

from jailson.config.settings import MEMORY_DIR


class LongTermMemory:
    """Permanent knowledge store. Facts and patterns survive indefinitely."""

    def __init__(self):
        self.db_path = MEMORY_DIR / "long_term.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    confidence REAL DEFAULT 1.0,
                    source TEXT,
                    tags TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    content TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_seen TEXT,
                    tags TEXT DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS idx_facts_cat ON facts(category);
                CREATE INDEX IF NOT EXISTS idx_facts_ts ON facts(timestamp);
            """)

    def add_fact(self, content: str, category: str = "general",
                 confidence: float = 1.0, source: str = "", tags: list = None):
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO facts (timestamp, content, category, confidence, source, tags) VALUES (?,?,?,?,?,?)",
                (datetime.now().isoformat(), content, category, confidence, source, json.dumps(tags or [])),
            )

    def add_pattern(self, content: str, tags: list = None):
        """Upsert a pattern — increment frequency if similar content exists."""
        with sqlite3.connect(self.db_path) as con:
            existing = con.execute(
                "SELECT id, frequency FROM patterns WHERE content = ?", (content,)
            ).fetchone()
            if existing:
                con.execute(
                    "UPDATE patterns SET frequency = ?, last_seen = ? WHERE id = ?",
                    (existing[1] + 1, datetime.now().isoformat(), existing[0]),
                )
            else:
                now = datetime.now().isoformat()
                con.execute(
                    "INSERT INTO patterns (timestamp, content, frequency, last_seen, tags) VALUES (?,?,1,?,?)",
                    (now, content, now, json.dumps(tags or [])),
                )

    def get_all_facts(self, category: Optional[str] = None, limit: int = 100) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
            if category:
                rows = con.execute(
                    "SELECT content, category, confidence, tags FROM facts WHERE category = ? ORDER BY timestamp DESC LIMIT ?",
                    (category, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT content, category, confidence, tags FROM facts ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [{"content": r[0], "category": r[1], "confidence": r[2], "tags": json.loads(r[3])} for r in rows]

    def get_top_patterns(self, limit: int = 20) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT content, frequency, last_seen FROM patterns ORDER BY frequency DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"content": r[0], "frequency": r[1], "last_seen": r[2]} for r in rows]

    def search_facts(self, query: str, limit: int = 10) -> list[str]:
        """Simple text search across facts."""
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT content FROM facts WHERE content LIKE ? ORDER BY confidence DESC, timestamp DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        return [r[0] for r in rows]

    def get_context_for_prompt(self) -> str:
        facts = self.get_all_facts(limit=30)
        patterns = self.get_top_patterns(limit=10)
        if not facts and not patterns:
            return ""
        lines = ["## O que sei sobre o utilizador (memória longa)"]
        if facts:
            lines.append("\n### Factos")
            for f in facts:
                conf = f" (confiança: {f['confidence']:.0%})" if f["confidence"] < 1.0 else ""
                lines.append(f"- [{f['category']}] {f['content']}{conf}")
        if patterns:
            lines.append("\n### Padrões de Comportamento")
            for p in patterns[:5]:
                lines.append(f"- {p['content']} (observado {p['frequency']}x)")
        return "\n".join(lines)

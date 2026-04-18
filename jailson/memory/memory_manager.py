"""Unified memory manager — single interface to all memory tiers."""
import uuid
from datetime import datetime
from typing import Optional

from jailson.memory.short_term import ShortTermMemory
from jailson.memory.medium_term import MediumTermMemory
from jailson.memory.long_term import LongTermMemory
from jailson.memory.sacred import SacredMemory
from jailson.memory.episodic import EpisodicMemory
from jailson.memory.semantic import SemanticMemory
from jailson.config.settings import ensure_dirs


class MemoryManager:
    """Orchestrates all memory tiers for Jailson."""

    def __init__(self):
        ensure_dirs()
        self.short = ShortTermMemory()
        self.medium = MediumTermMemory()
        self.long = LongTermMemory()
        self.sacred = SacredMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self._session_id = str(uuid.uuid4())[:8]

    # ── Write helpers ──────────────────────────────────────────────────────────

    def record_turn(self, role: str, content: str):
        """Record a conversation turn in short + medium memory + semantic index."""
        self.short.add(role, content)
        self.medium.add_interaction(role, content, session_id=self._session_id)
        doc_id = f"turn_{self._session_id}_{datetime.now().timestamp():.0f}"
        self.semantic.add(
            content=f"{role}: {content}",
            doc_id=doc_id,
            metadata={"type": "interaction", "role": role},
        )

    def store(self, content: str, memory_type: str, tags: list = None,
              category: str = "general", importance: int = 5,
              title: str = "", source: str = "jailson"):
        """Store to a specific memory tier."""
        tags = tags or []
        if memory_type == "long_term":
            self.long.add_fact(content, category=category, source=source, tags=tags)
            self.semantic.add(content, doc_id=f"lt_{uuid.uuid4().hex[:8]}",
                              metadata={"type": "long_term", "category": category})
        elif memory_type == "sacred":
            self.sacred.add(content, category=category, source=source)
            self.semantic.add(content, doc_id=f"sacred_{uuid.uuid4().hex[:8]}",
                              metadata={"type": "sacred"})
        elif memory_type == "episodic":
            self.episodic.add_event(
                title=title or content[:50],
                description=content,
                importance=importance,
                tags=tags,
            )
            self.semantic.add(content, doc_id=f"ep_{uuid.uuid4().hex[:8]}",
                              metadata={"type": "episodic", "importance": str(importance)})
        elif memory_type == "pattern":
            self.long.add_pattern(content, tags=tags)
        elif memory_type == "insight":
            self.medium.add_insight(content, source=source)

    # ── Read helpers ───────────────────────────────────────────────────────────

    def search(self, query: str, n: int = 5) -> list[dict]:
        """Semantic search + SQLite fallback across all tiers."""
        results = self.semantic.search(query, n_results=n)
        if not results:
            # Fallback: simple text search in long-term facts
            facts = self.long.search_facts(query, limit=n)
            results = [{"content": f, "metadata": {"type": "long_term"}, "distance": 0.5} for f in facts]
        return results

    def search_summary(self, query: str, n: int = 5) -> str:
        results = self.search(query, n)
        if not results:
            return "Nenhuma memória encontrada para esta pesquisa."
        lines = [f"## Resultados da pesquisa: '{query}'"]
        for r in results:
            mem_type = r.get("metadata", {}).get("type", "?")
            lines.append(f"- [{mem_type}] {r['content'][:200]}")
        return "\n".join(lines)

    # ── Context builders ───────────────────────────────────────────────────────

    def get_stable_context(self) -> str:
        """Long-lived context: sacred + long-term facts. Suitable for prompt caching."""
        parts = []
        sacred_ctx = self.sacred.get_context_for_prompt()
        if sacred_ctx:
            parts.append(sacred_ctx)
        lt_ctx = self.long.get_context_for_prompt()
        if lt_ctx:
            parts.append(lt_ctx)
        ep_ctx = self.episodic.get_context_for_prompt()
        if ep_ctx:
            parts.append(ep_ctx)
        return "\n\n".join(parts)

    def get_recent_context(self) -> str:
        """Volatile context: recent insights and session summary."""
        return self.medium.get_summary_context()

    def get_short_context_messages(self, limit: int = 20) -> list[dict]:
        """Recent conversation as message history."""
        return self.short.get_messages(limit=limit)

    # ── Maintenance ────────────────────────────────────────────────────────────

    def purge_old(self):
        """Clean up expired medium-term entries."""
        self.medium.purge_old()

    def stats(self) -> dict:
        return {
            "short_term": len(self.short),
            "sacred": len(self.sacred),
            "semantic": len(self.semantic),
            "session_id": self._session_id,
        }

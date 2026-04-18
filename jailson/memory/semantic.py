"""Semantic memory — vector embeddings for cross-memory semantic search (ChromaDB)."""
from datetime import datetime
from typing import Optional

from jailson.config.settings import DATA_DIR, SEMANTIC_COLLECTION


class SemanticMemory:
    """ChromaDB-backed vector store for semantic similarity search.

    Falls back gracefully if ChromaDB is unavailable.
    """

    def __init__(self):
        self._client = None
        self._collection = None
        self._init()

    def _init(self):
        try:
            import chromadb

            chroma_path = DATA_DIR / "chroma"
            chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(chroma_path))
            self._collection = self._client.get_or_create_collection(
                name=SEMANTIC_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            pass
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._collection is not None

    def add(self, content: str, doc_id: str, metadata: dict = None):
        """Add a document to the vector store."""
        if not self.available:
            return
        try:
            meta = {"timestamp": datetime.now().isoformat(), **(metadata or {})}
            self._collection.upsert(
                ids=[doc_id],
                documents=[content],
                metadatas=[meta],
            )
        except Exception:
            pass

    def search(self, query: str, n_results: int = 5,
               where: Optional[dict] = None) -> list[dict]:
        """Semantic similarity search. Returns list of {content, metadata, distance}."""
        if not self.available:
            return []
        try:
            kwargs = {"query_texts": [query], "n_results": min(n_results, self._count())}
            if where:
                kwargs["where"] = where
            results = self._collection.query(**kwargs)
            if not results["documents"]:
                return []
            output = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                output.append({"content": doc, "metadata": meta, "distance": dist})
            return output
        except Exception:
            return []

    def delete(self, doc_id: str):
        if self.available:
            try:
                self._collection.delete(ids=[doc_id])
            except Exception:
                pass

    def _count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0

    def __len__(self) -> int:
        return self._count()

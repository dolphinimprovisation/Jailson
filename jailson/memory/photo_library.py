"""Photo Library — persistent SQLite store for Horus analyzed photos."""
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from jailson.config.settings import MEMORY_DIR


class PhotoLibrary:
    """Local DB of every photo Horus has analyzed. Zero tokens after first analysis."""

    def __init__(self):
        self.db_path = MEMORY_DIR / "photo_library.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS photos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    file_hash TEXT,
                    file_size INTEGER,
                    phash TEXT,
                    analyzed_at TEXT,
                    description TEXT,
                    tags TEXT DEFAULT '[]',
                    faces_count INTEGER DEFAULT 0,
                    faces_info TEXT DEFAULT '[]',
                    exif_data TEXT DEFAULT '{}',
                    thumbnail_b64 TEXT,
                    width INTEGER,
                    height INTEGER,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_photos_path ON photos(path);
                CREATE INDEX IF NOT EXISTS idx_photos_hash ON photos(file_hash);
                CREATE INDEX IF NOT EXISTS idx_photos_phash ON photos(phash);
                CREATE TABLE IF NOT EXISTS trash (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_path TEXT NOT NULL,
                    trash_path TEXT NOT NULL,
                    moved_at TEXT NOT NULL,
                    reason TEXT
                );
            """)

    # ── Write ──────────────────────────────────────────────────────────────────

    def upsert(self, path: str, data: dict):
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                INSERT INTO photos
                    (path, file_hash, file_size, phash, analyzed_at, description,
                     tags, faces_count, faces_info, exif_data, thumbnail_b64,
                     width, height, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(path) DO UPDATE SET
                    file_hash=excluded.file_hash,
                    file_size=excluded.file_size,
                    phash=excluded.phash,
                    analyzed_at=excluded.analyzed_at,
                    description=excluded.description,
                    tags=excluded.tags,
                    faces_count=excluded.faces_count,
                    faces_info=excluded.faces_info,
                    exif_data=excluded.exif_data,
                    thumbnail_b64=excluded.thumbnail_b64,
                    width=excluded.width,
                    height=excluded.height
            """, (
                path,
                data.get("file_hash"),
                data.get("file_size"),
                data.get("phash"),
                data.get("analyzed_at", now),
                data.get("description", ""),
                json.dumps(data.get("tags", [])),
                data.get("faces_count", 0),
                json.dumps(data.get("faces_info", [])),
                json.dumps(data.get("exif_data", {})),
                data.get("thumbnail_b64"),
                data.get("width"),
                data.get("height"),
                now,
            ))

    def update_tags(self, path: str, tags: list[str]):
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "UPDATE photos SET tags=? WHERE path=?",
                (json.dumps(tags), path),
            )

    def record_trash(self, original_path: str, trash_path: str, reason: str = ""):
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO trash (original_path, trash_path, moved_at, reason) VALUES (?,?,?,?)",
                (original_path, trash_path, datetime.now().isoformat(), reason),
            )
            con.execute("DELETE FROM photos WHERE path=?", (original_path,))

    # ── Read ───────────────────────────────────────────────────────────────────

    def get(self, path: str) -> dict | None:
        with sqlite3.connect(self.db_path) as con:
            row = con.execute(
                "SELECT * FROM photos WHERE path=?", (path,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def is_analyzed(self, path: str) -> bool:
        with sqlite3.connect(self.db_path) as con:
            row = con.execute(
                "SELECT analyzed_at FROM photos WHERE path=? AND analyzed_at IS NOT NULL",
                (path,),
            ).fetchone()
        return row is not None

    def find_duplicates(self, folder: str | None = None) -> list[list[dict]]:
        """Group photos with identical hash OR (same name + same size)."""
        with sqlite3.connect(self.db_path) as con:
            if folder:
                rows = con.execute(
                    "SELECT * FROM photos WHERE path LIKE ?", (f"{folder}%",)
                ).fetchall()
            else:
                rows = con.execute("SELECT * FROM photos").fetchall()

        photos = [self._row_to_dict(r) for r in rows]
        groups: dict[str, list[dict]] = {}

        for p in photos:
            # Group by hash
            key = p.get("file_hash") or ""
            if key:
                groups.setdefault(key, []).append(p)

        # Also group by name+size for photos without hash
        name_size_groups: dict[str, list[dict]] = {}
        for p in photos:
            name = Path(p["path"]).name
            size = p.get("file_size") or 0
            if name and size:
                name_size_groups.setdefault(f"{name}_{size}", []).append(p)

        result = []
        seen_paths: set[str] = set()
        for group in list(groups.values()) + list(name_size_groups.values()):
            if len(group) > 1:
                paths = tuple(sorted(p["path"] for p in group))
                if paths not in seen_paths:
                    seen_paths.add(paths)
                    result.append(group)

        return result

    def find_similar(self, folder: str | None = None, threshold: int = 10) -> list[list[dict]]:
        """Group visually similar photos using perceptual hash distance."""
        try:
            import imagehash
        except ImportError:
            return []

        with sqlite3.connect(self.db_path) as con:
            if folder:
                rows = con.execute(
                    "SELECT * FROM photos WHERE path LIKE ? AND phash IS NOT NULL",
                    (f"{folder}%",),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM photos WHERE phash IS NOT NULL"
                ).fetchall()

        photos = [self._row_to_dict(r) for r in rows]
        grouped: list[list[dict]] = []
        used: set[str] = set()

        for i, p in enumerate(photos):
            if p["path"] in used:
                continue
            h1 = imagehash.hex_to_hash(p["phash"])
            group = [p]
            for p2 in photos[i + 1:]:
                if p2["path"] in used:
                    continue
                h2 = imagehash.hex_to_hash(p2["phash"])
                if h1 - h2 <= threshold:
                    group.append(p2)
            if len(group) > 1:
                for p in group:
                    used.add(p["path"])
                grouped.append(group)

        return grouped

    def search_by_tags(self, query: str, limit: int = 50) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT * FROM photos WHERE tags LIKE ? OR description LIKE ? LIMIT ?",
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def list_library_folders(self) -> list[str]:
        """Return unique parent directories of all analyzed photos in the DB."""
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute("SELECT DISTINCT path FROM photos").fetchall()
        folders: set[str] = set()
        for (path,) in rows:
            folders.add(str(Path(path).parent))
        return sorted(folders)

    def list_folder(self, folder: str, limit: int = 200, offset: int = 0) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT * FROM photos WHERE path LIKE ? ORDER BY path LIMIT ? OFFSET ?",
                (f"{folder}%", limit, offset),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as con:
            total = con.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
            analyzed = con.execute(
                "SELECT COUNT(*) FROM photos WHERE analyzed_at IS NOT NULL"
            ).fetchone()[0]
            in_trash = con.execute("SELECT COUNT(*) FROM trash").fetchone()[0]
        return {"total": total, "analyzed": analyzed, "in_trash": in_trash}

    def get_trash(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT original_path, trash_path, moved_at, reason FROM trash ORDER BY moved_at DESC"
            ).fetchall()
        return [{"original_path": r[0], "trash_path": r[1], "moved_at": r[2], "reason": r[3]} for r in rows]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _row_to_dict(self, row) -> dict:
        if row is None:
            return {}
        cols = ["id", "path", "file_hash", "file_size", "phash", "analyzed_at",
                "description", "tags", "faces_count", "faces_info", "exif_data",
                "thumbnail_b64", "width", "height", "created_at"]
        d = dict(zip(cols, row))
        for key in ("tags", "faces_info", "exif_data"):
            try:
                d[key] = json.loads(d[key] or "[]" if key != "exif_data" else d[key] or "{}")
            except Exception:
                d[key] = [] if key != "exif_data" else {}
        return d


def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

"""Configuration management for Jailson."""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent.parent
CHARTER_PATH = BASE_DIR / "charter.json"

DATA_DIR = Path(os.getenv("JAILSON_DATA_DIR", "~/.jailson")).expanduser()
MEMORY_DIR = DATA_DIR / "memory"
LOGS_DIR = DATA_DIR / "logs"

JAILSON_MODEL = os.getenv("JAILSON_MODEL", "claude-sonnet-4-6")
SUBAGENT_MODEL = os.getenv("SUBAGENT_MODEL", "claude-haiku-4-5")

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
SHOW_THINKING = os.getenv("SHOW_THINKING", "false").lower() == "true"

IMAGE_FOLDERS = [
    Path(p.strip()).expanduser()
    for p in os.getenv("IMAGE_FOLDERS", "~/Pictures").split(",")
    if p.strip()
]

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
    ".tiff", ".tif", ".heic", ".heif", ".avif",
}

SUPPORTED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".html",
    ".css", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".sh", ".bash", ".zsh", ".fish", ".csv", ".xml",
    ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb",
    ".php", ".swift", ".kt", ".scala", ".lua", ".r",
}

SHORT_TERM_MAX = 100
MEDIUM_TERM_DAYS = 30
SEMANTIC_COLLECTION = "jailson_memories"


def ensure_dirs():
    """Create required directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_charter() -> dict:
    """Load the charter from JSON."""
    with open(CHARTER_PATH, encoding="utf-8") as f:
        return json.load(f)


def charter_to_text(charter: dict) -> str:
    """Convert charter dict to formatted text for the system prompt."""
    lines = [
        f"# {charter['nome']} — Carta de Princípios",
        f"\n## Missão\n{charter['missao']}",
        "\n## Princípios",
    ]
    for i, p in enumerate(charter["principios"], 1):
        lines.append(f"{i}. {p}")
    lines.append("\n## Objetivos")
    for key, val in charter["objetivos"].items():
        lines.append(f"- **{key.capitalize()}**: {val}")
    lines.append("\n## Sistema de Memória")
    for key, val in charter["tipos_de_memoria"].items():
        lines.append(f"- **{key.capitalize()}**: {val}")
    return "\n".join(lines)

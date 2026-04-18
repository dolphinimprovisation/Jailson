"""Photo Manager — orchestrates Horus analysis with library caching and cost estimation."""
import base64
import json
import platform
import subprocess
from pathlib import Path

import anthropic

from jailson.config.settings import SUBAGENT_MODEL, DATA_DIR, SUPPORTED_IMAGE_EXTENSIONS
from jailson.memory.photo_library import PhotoLibrary, file_hash
from jailson.tools.image_analyzer import get_image_metadata, scan_image_folder, encode_image_for_claude

TRASH_DIR = DATA_DIR / "trash"

# Vision API cost estimate (claude-haiku-4-5 input ~$0.80/1M tokens; image ≈ 1500 tokens)
_COST_PER_IMAGE_USD = 0.0012


def _ensure_local(path: str) -> bool:
    """Force OneDrive to download a cloud-only file. Returns True if file is now local."""
    if platform.system() != "Windows":
        return True
    try:
        # Reading the file forces OneDrive to download it
        subprocess.run(
            ["powershell", "-Command", f'$null = [System.IO.File]::ReadAllBytes("{path}")'],
            timeout=30, capture_output=True,
        )
        return Path(path).stat().st_size > 0
    except Exception:
        return False


def get_library() -> PhotoLibrary:
    _lib = getattr(get_library, "_instance", None)
    if _lib is None:
        get_library._instance = PhotoLibrary()
    return get_library._instance


# ── Cost estimation ────────────────────────────────────────────────────────────

def estimate_folder_cost(folder_path: str) -> dict:
    """Return count of new vs cached photos and estimated API cost."""
    lib = get_library()
    result = scan_image_folder(folder_path, recursive=True, max_images=5000)
    if not result.get("success"):
        return {"success": False, "error": result.get("error", "Erro ao varrer pasta")}

    images = result.get("images") or result.get("images_sample") or []
    total = result.get("total_images", len(images))

    already_analyzed = sum(1 for img in images if lib.is_analyzed(img["path"]))
    new_count = total - already_analyzed
    cost_usd = round(new_count * _COST_PER_IMAGE_USD, 4)

    return {
        "success": True,
        "folder": folder_path,
        "total_images": total,
        "already_in_library": already_analyzed,
        "new_to_analyze": new_count,
        "estimated_cost_usd": cost_usd,
        "estimated_cost_eur": round(cost_usd * 0.93, 4),
        "message": (
            f"Pasta tem {total} fotos. "
            f"{already_analyzed} já na biblioteca (grátis). "
            f"{new_count} novas → estimativa: ~${cost_usd:.4f} em tokens."
        ),
    }


# ── Single photo analysis ──────────────────────────────────────────────────────

def analyze_photo(path: str, client: anthropic.Anthropic = None, force: bool = False) -> dict:
    """Analyze a photo. Returns cached result if already in library."""
    lib = get_library()

    if not force and lib.is_analyzed(path):
        cached = lib.get(path)
        cached["from_cache"] = True
        return cached

    if client is None:
        client = anthropic.Anthropic()

    p = Path(path)
    if not p.exists():
        return {"success": False, "error": f"Ficheiro não existe: {path}"}

    # Force OneDrive to download cloud-only files before reading
    _ensure_local(path)

    meta = get_image_metadata(path)
    exif = meta if meta.get("success") else {}

    phash_str = _compute_phash(path)
    fhash = _safe_hash(path)
    thumb = _make_thumbnail_b64(path)

    description, tags, faces_count, faces_info = _vision_analyze(path, client)

    data = {
        "file_hash": fhash,
        "file_size": p.stat().st_size,
        "phash": phash_str,
        "analyzed_at": None,
        "description": description,
        "tags": tags,
        "faces_count": faces_count,
        "faces_info": faces_info,
        "exif_data": exif,
        "thumbnail_b64": thumb,
        "width": exif.get("width"),
        "height": exif.get("height"),
    }

    from datetime import datetime
    data["analyzed_at"] = datetime.now().isoformat()
    lib.upsert(path, data)

    return {"success": True, "path": path, "from_cache": False, **data}


# ── Batch analysis ─────────────────────────────────────────────────────────────

def analyze_folder_batch(folder_path: str, max_new: int = 20,
                          client: anthropic.Anthropic = None) -> dict:
    """Analyze up to max_new new photos in a folder. Skips already cached."""
    lib = get_library()
    result = scan_image_folder(folder_path, recursive=True, max_images=5000)
    if not result.get("success"):
        return {"success": False, "error": result.get("error")}

    images = result.get("images") or result.get("images_sample") or []
    if client is None:
        client = anthropic.Anthropic()

    analyzed = []
    skipped = 0
    errors = []

    for img in images:
        p = img["path"]
        if lib.is_analyzed(p):
            skipped += 1
            continue
        if len(analyzed) >= max_new:
            break
        r = analyze_photo(p, client=client)
        if r.get("success") or r.get("from_cache"):
            analyzed.append({"path": p, "tags": r.get("tags", []), "faces": r.get("faces_count", 0)})
        else:
            errors.append({"path": p, "error": r.get("error")})

    return {
        "success": True,
        "folder": folder_path,
        "newly_analyzed": len(analyzed),
        "skipped_cached": skipped,
        "errors": len(errors),
        "results": analyzed,
    }


# ── Trash ──────────────────────────────────────────────────────────────────────

def move_to_trash(path: str, reason: str = "") -> dict:
    """Move a photo to the trash folder."""
    try:
        TRASH_DIR.mkdir(parents=True, exist_ok=True)
        p = Path(path)
        if not p.exists():
            return {"success": False, "error": f"Não existe: {path}"}
        dest = TRASH_DIR / p.name
        # Avoid collision
        if dest.exists():
            dest = TRASH_DIR / f"{p.stem}_{p.stat().st_ino}{p.suffix}"
        p.rename(dest)
        get_library().record_trash(path, str(dest), reason)
        return {"success": True, "message": f"Movida para lixo: {dest}", "trash_path": str(dest)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def empty_trash() -> dict:
    """Permanently delete everything in the trash folder."""
    import shutil
    try:
        if not TRASH_DIR.exists():
            return {"success": True, "message": "Lixo já estava vazio."}
        count = sum(1 for _ in TRASH_DIR.iterdir())
        shutil.rmtree(str(TRASH_DIR))
        TRASH_DIR.mkdir(parents=True, exist_ok=True)
        return {"success": True, "message": f"{count} ficheiros eliminados permanentemente."}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Duplicates & similar ───────────────────────────────────────────────────────

def find_duplicates(folder_path: str = "") -> dict:
    groups = get_library().find_duplicates(folder_path or None)
    return {
        "success": True,
        "groups": len(groups),
        "duplicates": [
            [{"path": p["path"], "size": p.get("file_size"), "tags": p.get("tags")} for p in g]
            for g in groups
        ],
    }


def find_similar(folder_path: str = "", threshold: int = 10) -> dict:
    groups = get_library().find_similar(folder_path or None, threshold)
    return {
        "success": True,
        "groups": len(groups),
        "similar": [
            [{"path": p["path"], "phash": p.get("phash"), "tags": p.get("tags")} for p in g]
            for g in groups
        ],
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _compute_phash(path: str) -> str | None:
    try:
        import imagehash
        from PIL import Image
        return str(imagehash.phash(Image.open(path)))
    except Exception:
        return None


def _safe_hash(path: str) -> str | None:
    try:
        return file_hash(path)
    except Exception:
        return None


def _make_thumbnail_b64(path: str, size: int = 200) -> str | None:
    try:
        from PIL import Image
        import io
        if Path(path).suffix.lower() in (".heic", ".heif"):
            import pillow_heif
            pillow_heif.register_heif_opener()
        img = Image.open(path)
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _vision_analyze(path: str, client: anthropic.Anthropic) -> tuple:
    """Call Vision API. Returns (description, tags, faces_count, faces_info)."""
    encoded = encode_image_for_claude(path)
    if not encoded:
        return ("", [], 0, [])

    prompt = (
        "Analisa esta imagem e responde em JSON com exactamente estas chaves:\n"
        '{"description": "descrição concisa em português", '
        '"tags": ["tag1", "tag2", ...], '
        '"faces_count": 0, '
        '"faces_info": ["descrição rosto 1", ...]}\n'
        "Tags: objectos, pessoas, local, actividade, ambiente, cores dominantes. Máx 15 tags."
    )

    try:
        response = client.messages.create(
            model=SUBAGENT_MODEL,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": encoded["media_type"],
                        "data": encoded["data"],
                    }},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = response.content[0].text if response.content else "{}"
        # Extract JSON even if model adds extra text
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end]) if start >= 0 else {}
        return (
            data.get("description", ""),
            data.get("tags", []),
            data.get("faces_count", 0),
            data.get("faces_info", []),
        )
    except Exception:
        return ("", [], 0, [])

"""Image analysis tools — metadata, face detection via Claude vision, folder scanning."""
import base64
import json
from pathlib import Path
from typing import Optional

from jailson.config.settings import SUPPORTED_IMAGE_EXTENSIONS


def get_image_metadata(path: str) -> dict:
    """Return image metadata: size, format, EXIF data."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {"success": False, "error": f"Ficheiro não encontrado: {path}"}

    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(p)
        meta = {
            "success": True,
            "path": str(p),
            "filename": p.name,
            "format": img.format,
            "mode": img.mode,
            "width": img.width,
            "height": img.height,
            "size_bytes": p.stat().st_size,
        }

        # Extract EXIF
        exif_data = {}
        try:
            exif = img._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, str(tag_id))
                    if tag in ("DateTime", "DateTimeOriginal", "Make", "Model",
                               "GPSInfo", "ImageDescription", "Artist", "Copyright",
                               "Software", "ExposureTime", "FNumber", "ISOSpeedRatings"):
                        exif_data[tag] = str(value)[:200]
        except Exception:
            pass

        meta["exif"] = exif_data
        return meta
    except ImportError:
        return {"success": False, "error": "Pillow não instalado: pip install Pillow"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def scan_image_folder(folder_path: str, recursive: bool = True, max_images: int = 500) -> dict:
    """Scan a folder and return info about all images found."""
    p = Path(folder_path).expanduser().resolve()
    if not p.exists():
        return {"success": False, "error": f"Pasta não encontrada: {folder_path}"}
    if not p.is_dir():
        return {"success": False, "error": f"Não é uma pasta: {folder_path}"}

    images = []
    glob = p.rglob("*") if recursive else p.glob("*")

    for entry in glob:
        if len(images) >= max_images:
            break
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            if entry.name.startswith("."):
                continue
            try:
                stat = entry.stat()
                images.append({
                    "path": str(entry),
                    "name": entry.name,
                    "extension": entry.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                    "relative_path": str(entry.relative_to(p)),
                })
            except (PermissionError, OSError):
                continue

    by_ext: dict = {}
    for img in images:
        ext = img["extension"]
        by_ext[ext] = by_ext.get(ext, 0) + 1

    total_size = sum(i["size_bytes"] for i in images)

    return {
        "success": True,
        "folder": str(p),
        "images": images,
        "total": len(images),
        "by_extension": by_ext,
        "total_size_mb": round(total_size / 1024 / 1024, 1),
    }


def encode_image_for_claude(path: str, max_size_kb: int = 4096) -> Optional[dict]:
    """Encode an image as base64 for Claude vision API.

    Returns: {data, media_type} or None if encoding fails.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return None

    size_kb = p.stat().st_size / 1024
    if size_kb > max_size_kb:
        # Resize if too large
        try:
            from PIL import Image
            import io
            img = Image.open(p)
            img.thumbnail((1920, 1920))
            buf = io.BytesIO()
            fmt = img.format or "JPEG"
            img.save(buf, format=fmt)
            data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
            media_type = _get_media_type(p.suffix.lower())
            return {"data": data, "media_type": media_type}
        except ImportError:
            return None

    try:
        with open(p, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        media_type = _get_media_type(p.suffix.lower())
        return {"data": data, "media_type": media_type}
    except Exception:
        return None


def detect_faces_basic(path: str) -> dict:
    """Basic face detection using face_recognition library (optional dependency).

    Falls back to a placeholder if not installed.
    """
    try:
        import face_recognition
        import numpy as np

        img = face_recognition.load_image_file(path)
        locations = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, locations)

        return {
            "success": True,
            "face_count": len(locations),
            "faces": [
                {
                    "index": i,
                    "location": {"top": loc[0], "right": loc[1], "bottom": loc[2], "left": loc[3]},
                    "has_encoding": True,
                }
                for i, loc in enumerate(locations)
            ],
        }
    except ImportError:
        return {
            "success": False,
            "error": "face_recognition não instalado. Para instalar: pip install face-recognition (requer cmake e dlib).",
            "fallback": "Usa o Claude Vision para deteção de rostos.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def suggest_image_tags(metadata: dict, analysis: str) -> list[str]:
    """Extract tag suggestions from image metadata and analysis text."""
    tags = []

    if metadata.get("exif"):
        exif = metadata["exif"]
        if exif.get("DateTimeOriginal"):
            date = exif["DateTimeOriginal"][:10]
            tags.append(f"data:{date}")
        if exif.get("Make"):
            tags.append(f"camera:{exif['Make'].lower().strip()}")

    keywords = [
        "landscape", "portrait", "selfie", "food", "travel", "nature",
        "sunset", "sunrise", "architecture", "street", "family", "friends",
        "dog", "cat", "animal", "car", "beach", "mountain", "city", "night",
        "black and white", "macro", "event", "party", "birthday", "wedding",
        "paisagem", "retrato", "comida", "viagem", "natureza", "pôr do sol",
        "família", "amigos", "animal", "praia", "montanha", "cidade", "noite",
    ]
    analysis_lower = analysis.lower()
    for kw in keywords:
        if kw in analysis_lower:
            tags.append(kw.replace(" ", "_"))

    return list(dict.fromkeys(tags))[:20]


def _get_media_type(ext: str) -> str:
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    return mapping.get(ext, "image/jpeg")

"""File writer tools — move, delete, copy, rename, create directory."""
import platform
import shutil
import subprocess
from pathlib import Path


def move_item(src: str, dst: str) -> dict:
    try:
        src_path = Path(src)
        if not src_path.exists():
            return {"success": False, "error": f"Origem não existe: {src}"}
        shutil.move(str(src_path), dst)
        return {"success": True, "message": f"Movido: {src} → {dst}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_item(path: str, safe_delete: bool = True) -> dict:
    try:
        p = Path(path)
        if not p.exists():
            return {"success": False, "error": f"Não existe: {path}"}
        if safe_delete and p.is_dir():
            items = list(p.rglob("*"))
            if len(items) > 100:
                return {"success": False, "error": f"Pasta tem {len(items)} itens. Usa safe_delete=false para confirmar."}
        if platform.system() == "Windows":
            _onedrive_pause()

        if p.is_dir():
            try:
                shutil.rmtree(str(p))
            except Exception:
                if platform.system() == "Windows":
                    result = subprocess.run(
                        ["cmd", "/c", "rd", "/s", "/q", str(p)],
                        capture_output=True, text=True, timeout=30,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(result.stderr or result.stdout)
                else:
                    raise
        else:
            try:
                p.unlink()
            except Exception:
                if platform.system() == "Windows":
                    result = subprocess.run(
                        ["cmd", "/c", "del", "/f", "/q", str(p)],
                        capture_output=True, text=True, timeout=10,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(result.stderr or result.stdout)
                else:
                    raise
        return {"success": True, "message": f"Eliminado: {path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def create_directory(path: str) -> dict:
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return {"success": True, "message": f"Pasta criada: {path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def copy_item(src: str, dst: str) -> dict:
    try:
        src_path = Path(src)
        if not src_path.exists():
            return {"success": False, "error": f"Origem não existe: {src}"}
        if src_path.is_dir():
            shutil.copytree(str(src_path), dst)
        else:
            shutil.copy2(str(src_path), dst)
        return {"success": True, "message": f"Copiado: {src} → {dst}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _onedrive_pause():
    """Briefly pause OneDrive sync so file locks are released."""
    try:
        subprocess.run(
            ["taskkill", "/f", "/im", "OneDrive.exe"],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass


def rename_item(path: str, new_name: str) -> dict:
    try:
        p = Path(path)
        if not p.exists():
            return {"success": False, "error": f"Não existe: {path}"}
        new_path = p.parent / new_name
        p.rename(new_path)
        return {"success": True, "message": f"Renomeado para: {str(new_path)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

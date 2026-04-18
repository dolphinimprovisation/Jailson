"""System information tools — hardware, software, disk, processes."""
import os
import platform
import subprocess
import json
from pathlib import Path
from typing import Optional


def get_hardware_info() -> dict:
    """Return CPU, RAM, disk, and GPU information."""
    try:
        import psutil

        cpu = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "frequency_mhz": round(psutil.cpu_freq().current) if psutil.cpu_freq() else None,
            "usage_percent": psutil.cpu_percent(interval=1),
            "model": _get_cpu_model(),
        }

        ram = psutil.virtual_memory()
        memory = {
            "total_gb": round(ram.total / 1024**3, 1),
            "available_gb": round(ram.available / 1024**3, 1),
            "used_percent": ram.percent,
        }

        disks = []
        for part in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(part.mountpoint)
                disks.append({
                    "device": part.device,
                    "mountpoint": part.mountpoint,
                    "fstype": part.fstype,
                    "total_gb": round(usage.total / 1024**3, 1),
                    "free_gb": round(usage.free / 1024**3, 1),
                    "used_percent": usage.percent,
                })
            except (PermissionError, OSError):
                continue

        return {
            "success": True,
            "os": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "hostname": platform.node(),
            },
            "cpu": cpu,
            "memory": memory,
            "disks": disks,
        }
    except ImportError:
        return {"success": False, "error": "psutil não instalado. Instala com: pip install psutil"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_running_processes(top_n: int = 20, sort_by: str = "memory") -> dict:
    """Return list of running processes sorted by CPU or memory usage."""
    try:
        import psutil

        processes = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
            try:
                info = proc.info
                processes.append({
                    "pid": info["pid"],
                    "name": info["name"],
                    "cpu_percent": round(info["cpu_percent"] or 0, 1),
                    "memory_percent": round(info["memory_percent"] or 0, 2),
                    "status": info["status"],
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        key = "memory_percent" if sort_by == "memory" else "cpu_percent"
        processes.sort(key=lambda p: p[key], reverse=True)

        return {"success": True, "processes": processes[:top_n], "total": len(processes)}
    except ImportError:
        return {"success": False, "error": "psutil não instalado"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_disk_usage(path: str = "/") -> dict:
    """Get detailed disk usage for a path."""
    try:
        import psutil
        p = Path(path).expanduser().resolve()
        usage = psutil.disk_usage(str(p))
        return {
            "success": True,
            "path": str(p),
            "total_gb": round(usage.total / 1024**3, 2),
            "used_gb": round(usage.used / 1024**3, 2),
            "free_gb": round(usage.free / 1024**3, 2),
            "used_percent": usage.percent,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_installed_software() -> dict:
    """Return list of installed applications (cross-platform)."""
    system = platform.system()
    apps = []

    try:
        if system == "Linux":
            apps = _get_linux_packages()
        elif system == "Darwin":
            apps = _get_macos_apps()
        elif system == "Windows":
            apps = _get_windows_apps()
        return {"success": True, "apps": apps, "total": len(apps), "os": system}
    except Exception as e:
        return {"success": False, "error": str(e), "os": system}


def get_memory_modules() -> dict:
    """Return detailed info about physical RAM modules (manufacturer, part number, speed, capacity)."""
    system = platform.system()
    modules = []

    try:
        if system == "Windows":
            result = subprocess.run(
                ["wmic", "memorychip", "get",
                 "Manufacturer,PartNumber,Speed,Capacity,FormFactor,MemoryType,DeviceLocator",
                 "/format:csv"],
                capture_output=True, text=True, timeout=10,
            )
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            if len(lines) >= 2:
                headers = [h.strip() for h in lines[0].split(",")]
                for line in lines[1:]:
                    vals = [v.strip() for v in line.split(",")]
                    if len(vals) == len(headers):
                        m = dict(zip(headers, vals))
                        capacity_gb = round(int(m.get("Capacity", 0)) / 1024**3, 1) if m.get("Capacity", "").isdigit() else m.get("Capacity", "?")
                        modules.append({
                            "slot": m.get("DeviceLocator", ""),
                            "manufacturer": m.get("Manufacturer", "").strip(),
                            "part_number": m.get("PartNumber", "").strip(),
                            "capacity_gb": capacity_gb,
                            "speed_mhz": m.get("Speed", ""),
                            "form_factor": _ram_form_factor(m.get("FormFactor", "")),
                            "memory_type": _ram_type(m.get("MemoryType", "")),
                        })

        elif system == "Linux":
            result = subprocess.run(
                ["dmidecode", "--type", "memory"],
                capture_output=True, text=True, timeout=10,
            )
            current = {}
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("Memory Device"):
                    if current:
                        modules.append(current)
                    current = {}
                elif ":" in line:
                    key, _, val = line.partition(":")
                    current[key.strip()] = val.strip()
            if current:
                modules.append(current)

        elif system == "Darwin":
            result = subprocess.run(
                ["system_profiler", "SPMemoryDataType"],
                capture_output=True, text=True, timeout=10,
            )
            modules.append({"raw": result.stdout[:2000]})

        if not modules:
            return {"success": False, "error": "Não foi possível obter detalhes dos módulos de RAM."}

        return {"success": True, "modules": modules, "count": len(modules)}

    except Exception as e:
        return {"success": False, "error": str(e)}


def _ram_form_factor(code: str) -> str:
    mapping = {"8": "DIMM", "12": "SO-DIMM", "13": "SO-DIMM", "0": "Desconhecido"}
    return mapping.get(code, f"FormFactor({code})")


def _ram_type(code: str) -> str:
    mapping = {
        "20": "DDR", "21": "DDR2", "24": "DDR3", "26": "DDR4", "34": "DDR5",
        "0": "Desconhecido", "2": "DRAM",
    }
    return mapping.get(code, f"Type({code})")


def get_environment_info() -> dict:
    """Return Python environment, PATH, and key env variables."""
    import sys
    env_vars = {}
    safe_keys = ["PATH", "HOME", "USER", "SHELL", "LANG", "TERM", "EDITOR"]
    for k in safe_keys:
        v = os.environ.get(k, "")
        if v:
            env_vars[k] = v

    return {
        "success": True,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "path": sys.path[:5],
        },
        "environment": env_vars,
        "cwd": os.getcwd(),
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _get_cpu_model() -> str:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        elif system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip()
        elif system == "Windows":
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=5,
            )
            lines = result.stdout.strip().split("\n")
            return lines[1].strip() if len(lines) > 1 else platform.processor()
    except Exception:
        pass
    return platform.processor()


def _get_linux_packages() -> list:
    apps = []
    managers = [
        (["dpkg", "--list"], lambda l: l.split()[1] if l.startswith("ii") else None),
        (["rpm", "-qa", "--queryformat", "%{NAME}\n"], lambda l: l.strip() or None),
        (["pacman", "-Qq"], lambda l: l.strip() or None),
        (["snap", "list"], lambda l: l.split()[0] if l and not l.startswith("Name") else None),
    ]
    for cmd, parser in managers:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    name = parser(line)
                    if name:
                        apps.append({"name": name, "source": cmd[0]})
            if apps:
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return apps[:500]


def _get_macos_apps() -> list:
    apps = []
    app_dirs = [Path("/Applications"), Path.home() / "Applications"]
    for app_dir in app_dirs:
        if app_dir.exists():
            for app in app_dir.glob("*.app"):
                apps.append({"name": app.stem, "path": str(app), "source": "Applications"})
    try:
        result = subprocess.run(["brew", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for pkg in result.stdout.splitlines():
                if pkg.strip():
                    apps.append({"name": pkg.strip(), "source": "homebrew"})
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return apps


def _get_windows_apps() -> list:
    apps = []
    try:
        import winreg
        for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for path in (
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
            ):
                try:
                    key = winreg.OpenKey(root, path)
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            subkey = winreg.OpenKey(key, winreg.EnumKey(key, i))
                            name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                            if name:
                                apps.append({"name": name, "source": "registry"})
                        except OSError:
                            continue
                except OSError:
                    continue
    except ImportError:
        pass
    return apps

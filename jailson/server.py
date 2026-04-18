"""FastAPI web server for Jailson."""
import asyncio
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from jailson.agent import JailsonAgent
from jailson.config.settings import ensure_dirs, IMAGE_FOLDERS, SUPPORTED_IMAGE_EXTENSIONS
from jailson.tools.photo_manager import (
    estimate_folder_cost, analyze_photo, analyze_folder_batch,
    move_to_trash, empty_trash, find_duplicates, find_similar, get_library,
)

ensure_dirs()

app = FastAPI(title="Jailson")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_agent: JailsonAgent | None = None
_executor = ThreadPoolExecutor(max_workers=1)

_WEB_DIR = Path(__file__).parent.parent / "web"


def get_agent() -> JailsonAgent:
    global _agent
    if _agent is None:
        _agent = JailsonAgent()
    return _agent


class ChatRequest(BaseModel):
    message: str


class MemoryRequest(BaseModel):
    content: str
    memory_type: str = "long_term"


@app.get("/")
def index():
    return HTMLResponse((_WEB_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/chat")
async def chat(req: ChatRequest):
    agent = get_agent()
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run_in_thread():
        try:
            for chunk in agent.chat_stream(req.message):
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put(f"\n[Erro: {e}]"), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    _executor.submit(run_in_thread)

    async def event_stream():
        while True:
            chunk = await queue.get()
            if chunk is None:
                yield "data: [DONE]\n\n"
                break
            data = json.dumps({"chunk": chunk}, ensure_ascii=False)
            yield f"data: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/stats")
def stats():
    return get_agent().get_memory_stats()


@app.post("/memory")
def store_memory(req: MemoryRequest):
    get_agent().store_to_memory(req.content, req.memory_type)
    return {"success": True}


@app.get("/search")
def search(q: str):
    return {"result": get_agent().search_memory(q)}


@app.post("/reset")
def reset():
    get_agent().reset_conversation()
    return {"success": True}


# ── Gallery endpoints ──────────────────────────────────────────────────────────

@app.get("/gallery")
def gallery_page():
    return HTMLResponse((_WEB_DIR / "gallery.html").read_text(encoding="utf-8"))


@app.get("/gallery/folders")
def gallery_folders():
    folders = [{"path": str(f), "name": f.name, "exists": f.exists()} for f in IMAGE_FOLDERS]
    return {"folders": folders}


@app.get("/gallery/list")
def gallery_list(folder: str = "", limit: int = 100, offset: int = 0):
    lib = get_library()
    if folder:
        photos = lib.list_folder(folder, limit=limit, offset=offset)
    else:
        photos = lib.list_folder(str(IMAGE_FOLDERS[0]) if IMAGE_FOLDERS else "", limit=limit, offset=offset)
    return {"photos": [{k: v for k, v in p.items() if k != "thumbnail_b64"} for p in photos], "total": len(photos)}


@app.get("/gallery/thumb")
def gallery_thumb(path: str, size: int = 200):
    _assert_allowed_path(path)
    try:
        from PIL import Image
        img = Image.open(path)
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=75)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/gallery/photo")
def gallery_photo(path: str):
    _assert_allowed_path(path)
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404)
    suffix = p.suffix.lower()
    media = "image/jpeg" if suffix in (".jpg", ".jpeg") else f"image/{suffix.lstrip('.')}"
    return Response(content=p.read_bytes(), media_type=media)


@app.get("/gallery/meta")
def gallery_meta(path: str):
    _assert_allowed_path(path)
    cached = get_library().get(path)
    if cached:
        return cached
    from jailson.tools.image_analyzer import get_image_metadata
    return get_image_metadata(path)


class AnalyzeRequest(BaseModel):
    path: str
    force: bool = False

class TagsRequest(BaseModel):
    path: str
    tags: list[str]

class TrashRequest(BaseModel):
    path: str
    reason: str = ""

class BatchRequest(BaseModel):
    folder_path: str
    max_new: int = 20


@app.post("/gallery/analyze")
def gallery_analyze(req: AnalyzeRequest):
    _assert_allowed_path(req.path)
    import anthropic
    return analyze_photo(req.path, client=anthropic.Anthropic(), force=req.force)


@app.post("/gallery/analyze-batch")
def gallery_analyze_batch(req: BatchRequest):
    import anthropic
    return analyze_folder_batch(req.folder_path, max_new=req.max_new, client=anthropic.Anthropic())


@app.get("/gallery/estimate")
def gallery_estimate(folder: str):
    return estimate_folder_cost(folder)


@app.post("/gallery/tags")
def gallery_save_tags(req: TagsRequest):
    get_library().update_tags(req.path, req.tags)
    return {"success": True}


@app.post("/gallery/trash")
def gallery_trash(req: TrashRequest):
    _assert_allowed_path(req.path)
    return move_to_trash(req.path, req.reason)


@app.post("/gallery/trash/empty")
def gallery_empty_trash():
    return empty_trash()


@app.get("/gallery/trash/list")
def gallery_trash_list():
    return {"items": get_library().get_trash()}


@app.get("/gallery/duplicates")
def gallery_duplicates(folder: str = ""):
    return find_duplicates(folder)


@app.get("/gallery/similar")
def gallery_similar(folder: str = "", threshold: int = 10):
    return find_similar(folder, threshold)


@app.get("/gallery/stats")
def gallery_stats():
    return get_library().stats()


def _assert_allowed_path(path: str):
    p = Path(path).resolve()
    allowed = [f.resolve() for f in IMAGE_FOLDERS]
    trash = (Path("~/.jailson/trash").expanduser()).resolve()
    if not any(str(p).startswith(str(a)) for a in allowed) and not str(p).startswith(str(trash)):
        raise HTTPException(status_code=403, detail="Caminho não permitido")

"""FastAPI web server for Jailson."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from jailson.agent import JailsonAgent
from jailson.config.settings import ensure_dirs

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

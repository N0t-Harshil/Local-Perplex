"""FastAPI web UI and JSON API for Local Perplex."""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import uvicorn
from cachetools import TTLCache
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from local_perplex import __version__
from local_perplex.config import save_settings
from local_perplex.documents import SUPPORTED_EXTS
from local_perplex.engine import LocalPerplex, is_greeting
from local_perplex.mindmap import MindMapGenerator
from local_perplex.providers import PROVIDERS, build_provider, provider_catalog
from local_perplex.search import SEARCH_BACKENDS, search_catalog

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent.parent.parent
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MODES = {"industry_standard", "all_references", "pro"}
FOCUS_MODES = ["All", "Academic", "Technical", "Reddit", "YouTube", "Computational", "Writing"]

GREETING_REPLIES = {
    "hi": "Hi! What would you like to find out?",
    "hello": "Hello! Ask me anything and I'll research it.",
    "hey": "Hey! What's on your mind?",
    "thanks": "You're welcome. Anything else to look into?",
    "thank you": "You're welcome. Anything else to look into?",
}


def _sessions_dir() -> Path:
    path = Path(os.getenv("LOCAL_PERPLEX_SESSIONS_DIR") or PROJECT_ROOT / "sessions")
    path.mkdir(parents=True, exist_ok=True)
    legacy = APP_DIR.parent.parent / "sessions"  # older builds stored chats inside the package
    if legacy.is_dir() and legacy.resolve() != path.resolve():
        for f in legacy.glob("*.json"):
            if not (path / f.name).exists():
                shutil.move(str(f), path / f.name)
    return path


sessions_dir = _sessions_dir()

app = FastAPI(title="Local Perplex", version=__version__,
              description="Private, local-first AI research assistant.")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
templates.env.filters["split"] = lambda value, sep: value.split(sep)
templates.env.filters["domain"] = lambda url: (urlparse(url or "").netloc or "").removeprefix("www.")


# ── security: block cross-site writes to this local server ──────────────
@app.middleware("http")
async def same_origin_writes(request: Request, call_next):
    """A web page in another tab must not be able to POST here (e.g. to swap
    the provider URL and exfiltrate an API key). Browsers always send Origin
    on cross-site POSTs, so rejecting foreign origins closes that hole."""
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and origin != "null":
            host = request.headers.get("host", "")
            if urlparse(origin).netloc != host:
                return JSONResponse({"detail": "Cross-origin request blocked"}, status_code=403)
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


# ── engine ──────────────────────────────────────────────────────────────
def _pop_flag(*flags: str) -> str | None:
    for flag in flags:
        if flag in sys.argv:
            idx = sys.argv.index(flag)
            if idx + 1 < len(sys.argv):
                value = sys.argv[idx + 1]
                del sys.argv[idx:idx + 2]
                return value
    return None


_model_arg = _pop_flag("--model", "-m")
_provider_arg = _pop_flag("--provider")
engine = LocalPerplex(model=_model_arg, provider=_provider_arg)
mm_generator = MindMapGenerator()
pending = TTLCache(maxsize=500, ttl=3600)       # research jobs waiting for their stream
_session_lock = threading.Lock()


# ── chat sessions (multi-turn conversations) ────────────────────────────
def _load_sessions() -> dict:
    sessions = {}
    for f in sessions_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("history"):
            data.setdefault("updated", f.stat().st_mtime)
            sessions[f.stem] = data
    return sessions


def _save_session(sid: str, data: dict) -> None:
    if not _valid_sid(sid):
        return
    try:
        (sessions_dir / f"{sid}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not save session: %s", exc)


def _valid_sid(sid: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f\-]{36}", sid or ""))


chat_sessions = _load_sessions()


def _status() -> dict:
    return engine.refresh_status(force=False)


def _ctx(request: Request, page: str, **extra) -> dict:
    status = _status()
    return {"request": request, "page": page, "status": status, "version": __version__,
            "current_model": engine.model, "ollama_online": engine.llm_online,
            "available_models": engine.available_models, **extra}


def _source_card(s, i: int) -> dict:
    get = s.get if isinstance(s, dict) else lambda k, d=None: getattr(s, k, d)
    url = get("url", "") or ""
    content = get("snippet") or get("content") or ""
    return {"n": i, "title": get("title", "") or url, "url": url,
            "domain": (urlparse(url).netloc or "").removeprefix("www."),
            "relevance": round(float(get("relevance", 0) or 0), 3),
            "category": get("category", "General") or "General",
            "snippet": " ".join(content.split())[:240]}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ── pages ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    recent = engine.history.get_all_sessions()[:6]
    return templates.TemplateResponse(request, "index.html", _ctx(
        request, "home", recent=recent, focus_modes=FOCUS_MODES,
        graph_stats=engine.graph_stats(), doc_count=len(engine.docs.files())))


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    return templates.TemplateResponse(request, "history.html", _ctx(
        request, "history", history=engine.history.get_all_sessions()))


@app.get("/graph", response_class=HTMLResponse)
async def graph_page(request: Request):
    return templates.TemplateResponse(request, "graph.html", _ctx(
        request, "graph", stats=engine.graph_stats(), nodes=engine.graph.recent_nodes(limit=40)))


@app.get("/vault", response_class=HTMLResponse)
async def vault_page(request: Request):
    docs = [{"name": f.name, "size": f.stat().st_size, "modified": f.stat().st_mtime,
             "ext": f.suffix.lower().lstrip(".")} for f in engine.docs.files()]
    return templates.TemplateResponse(request, "vault.html", _ctx(
        request, "vault", docs=docs, supported=sorted(e.lstrip(".") for e in SUPPORTED_EXTS)))


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse(request, "settings.html", _ctx(
        request, "settings", providers=provider_catalog(engine.settings),
        search_backends=search_catalog(engine.settings), settings=engine.settings,
        model=engine.model, graph_stats=engine.graph_stats()))


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    return templates.TemplateResponse(request, "privacy.html", _ctx(
        request, "privacy", search_label=SEARCH_BACKENDS[engine.search_engine.backend].label))


@app.post("/settings")
async def settings_post(model: str = Form(...)):
    engine.model = model.strip()
    engine.settings.model = engine.model
    save_settings(engine.settings)
    return RedirectResponse(url="/settings", status_code=303)


# ── asking ──────────────────────────────────────────────────────────────
@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, question: str = Form(...), mode: str = Form("industry_standard"),
              focus_mode: str = Form("All"), conversation_history: str = Form(""),
              image: UploadFile = File(None), tag: str = Form("General"),
              privacy_mode: str = Form("false"), session_id: str = Form("")):
    """Register a research job and render the answer page immediately.

    The slow work (search, reading, synthesis) happens in ``/stream-answer``,
    so the page appears instantly and shows live progress.
    """
    question = question.strip()[:4000]
    incognito = str(privacy_mode).lower() in ("true", "1", "yes", "on")
    mode = mode if mode in MODES else "industry_standard"
    focus_mode = focus_mode if focus_mode in FOCUS_MODES else "All"
    tag = "[INCOGNITO]" if incognito else ((tag or "General").strip()[:40] or "General")
    status = engine.refresh_status()

    with _session_lock:
        if session_id in chat_sessions:
            history = list(chat_sessions[session_id]["history"])
        else:
            session_id = str(uuid.uuid4())
            title = question if len(question) <= 60 else question[:57].rstrip() + "…"
            chat_sessions[session_id] = {"history": [], "title": title, "auto_title": True,
                                         "incognito": incognito}
            history = []

    image_b64 = None
    if image is not None and image.filename:
        data = await image.read()
        if data and len(data) <= MAX_IMAGE_BYTES:
            image_b64 = base64.b64encode(data).decode("ascii")

    error = None
    if not status["llm_online"]:
        error = (f"{status['provider_label']} isn't reachable. "
                 + ("Start Ollama with `ollama serve`, or choose another provider in Settings."
                    if status["provider"] == "ollama" else "Check its API key and URL in Settings."))
    job_id = uuid.uuid4().hex
    if not error:
        pending[job_id] = {"question": question, "mode": mode, "focus_mode": focus_mode, "tag": tag,
                           "image_b64": image_b64, "session_id": session_id}
    return templates.TemplateResponse(request, "result.html", _ctx(
        request, "result", question=question, mode=mode, focus_mode=focus_mode, tag=tag,
        privacy_mode=incognito, error=error, job_id="" if error else job_id,
        session_id=session_id, history_list=history, sources=[], focus_modes=FOCUS_MODES,
        title=chat_sessions[session_id]["title"]))


@app.get("/stream-answer")
async def stream_answer(job: str = "", session_id: str = ""):
    params = pending.pop(job, None) if job else None
    if params is None:
        return StreamingResponse(iter([_sse({"error": "This research request expired. Ask again."})]),
                                 media_type="text/event-stream")
    return StreamingResponse(_run_job(params), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _run_job(p: dict):
    question, sid = p["question"], p["session_id"]
    session = chat_sessions.get(sid, {"history": []})
    history = session.get("history", [])
    try:
        if is_greeting(question) and not history:
            key = question.strip().rstrip(".!?").lower()
            reply = GREETING_REPLIES.get(key, GREETING_REPLIES["hi"])
            yield _sse({"chunk": reply})
            _append_turn(sid, question, reply, [], {})
            yield _sse({"done": True, "related": [], "session_id": sid, "metrics": {}})
            return

        selected, context = [], ""
        for kind, payload in engine.research_events(question, p["mode"], p["image_b64"], p["focus_mode"]):
            if kind == "status":
                yield _sse({"status": payload})
            elif kind == "sources":
                yield _sse({"update_sources": [_source_card(s, i) for i, s in enumerate(payload, 1)]})
            elif kind == "result":
                _, selected, context = payload
        metrics = dict(engine.last_metrics)

        yield _sse({"status": f"Writing the answer with {engine.model}…"})
        answer = ""
        for chunk in engine.ask_stream(question, context, history, p["mode"], sources=selected):
            answer += chunk
            yield _sse({"chunk": chunk})
        metrics["generation"] = engine.llm.last_stats.as_dict()
        metrics["model"] = engine.model
        metrics["provider"] = engine.provider_label

        cards = [_source_card(s, i) for i, s in enumerate(selected, 1)]
        _append_turn(sid, question, answer, cards, metrics)
        related = engine.finalize_research(question, answer, selected, p["tag"])
        yield _sse({"done": True, "related": related, "session_id": sid, "metrics": metrics})
    except Exception as exc:  # noqa: BLE001 - surface any failure to the page
        logger.exception("Research failed")
        yield _sse({"error": f"Research failed: {exc}"})


def _append_turn(sid: str, question: str, answer: str, sources: list, metrics: dict) -> None:
    with _session_lock:
        session = chat_sessions.get(sid)
        if session is None:
            return
        session["history"].append({"question": question, "answer": answer,
                                   "sources": sources, "metrics": metrics})
        session["updated"] = time.time()
        if not session.get("incognito"):
            _save_session(sid, session)
    if session.get("auto_title") and len(session["history"]) == 1 and not session.get("incognito"):
        threading.Thread(target=_retitle, args=(sid, question), daemon=True).start()


def _retitle(sid: str, question: str) -> None:
    title = engine.generate_chat_title(question)
    with _session_lock:
        if sid in chat_sessions and chat_sessions[sid].get("auto_title"):
            chat_sessions[sid]["title"] = title
            chat_sessions[sid]["auto_title"] = False
            _save_session(sid, chat_sessions[sid])


@app.get("/chat/{session_id}", response_class=HTMLResponse)
async def view_chat(request: Request, session_id: str):
    session = chat_sessions.get(session_id)
    if not session or not session.get("history"):
        return RedirectResponse(url="/", status_code=303)
    history = list(session["history"])
    last = history[-1]
    return templates.TemplateResponse(request, "result.html", _ctx(
        request, "result", question=last["question"], answer=last["answer"],
        sources=last.get("sources", []), metrics=last.get("metrics", {}), mode="industry_standard",
        focus_mode="All", tag="General", privacy_mode=False, error=None, job_id="",
        session_id=session_id, history_list=history[:-1], focus_modes=FOCUS_MODES,
        title=session.get("title", "")))


@app.get("/history/{index}", response_class=HTMLResponse)
async def view_history(request: Request, index: int):
    history = engine.history.get_all_sessions()
    if not 0 <= index < len(history):
        return RedirectResponse(url="/history", status_code=303)
    s = history[index]
    return templates.TemplateResponse(request, "result.html", _ctx(
        request, "result", question=s["question"], answer=s["answer"],
        sources=[_source_card(x, i) for i, x in enumerate(s.get("sources", []), 1)],
        metrics=s.get("metrics", {}), mode="industry_standard", focus_mode="All",
        tag=s.get("tag", "General"), privacy_mode=False, error=None, job_id="", session_id="",
        history_list=[], focus_modes=FOCUS_MODES, title=s["question"], export_index=index))


@app.get("/export/{index}", response_class=PlainTextResponse)
async def export_research(index: int):
    history = engine.history.get_all_sessions()
    if not 0 <= index < len(history):
        raise HTTPException(status_code=404, detail="Session not found")
    s = history[index]
    lines = [f"# {s['question']}", "", f"*Researched {s['timestamp']} with Local Perplex*", "",
             s["answer"], "", "## Sources", ""]
    for i, src in enumerate(s.get("sources", []), 1):
        lines.append(f"{i}. [{src.get('title', src.get('url'))}]({src.get('url')})")
    return PlainTextResponse("\n".join(lines), headers={
        "Content-Disposition": f'attachment; filename="research-{index}.md"'})


# ── documents ───────────────────────────────────────────────────────────
@app.post("/upload-docs")
async def upload_docs(files: list[UploadFile] = File(...)):
    for file in files:
        name = os.path.basename(file.filename or "")
        if not name or Path(name).suffix.lower() not in SUPPORTED_EXTS:
            continue
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if not data or len(data) > MAX_UPLOAD_BYTES:
            continue
        engine.docs.add_bytes(name, data)
        try:
            engine.index_document(name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Indexing %s failed: %s", name, exc)
    return RedirectResponse(url="/vault", status_code=303)


@app.post("/vault/delete")
async def vault_delete(filename: str = Form(...)):
    path = engine.docs.doc_dir / os.path.basename(filename)
    if path.is_file():
        path.unlink()
    return RedirectResponse(url="/vault", status_code=303)


# ── JSON API ────────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz():
    return {"ok": True, "version": __version__}


@app.get("/api/status")
async def api_status():
    return {**engine.refresh_status(), "graph": engine.graph_stats()}


class AskBody(BaseModel):
    question: str
    mode: str = "industry_standard"
    focus_mode: str = "All"
    incognito: bool = True


@app.post("/api/ask")
def api_ask(body: AskBody):
    """Non-streaming research endpoint for scripts and integrations."""
    if not engine.refresh_status()["llm_online"]:
        raise HTTPException(status_code=503, detail=f"{engine.provider_label} is not reachable")
    mode = body.mode if body.mode in MODES else "industry_standard"
    q, selected, context = engine.research_step(body.question, mode=mode, focus_mode=body.focus_mode)
    metrics = dict(engine.last_metrics)
    answer = engine.ask(body.question, context, mode=mode, sources=selected)
    metrics["generation"] = engine.llm.last_stats.as_dict()
    if not body.incognito:
        engine.finalize_research(body.question, answer, selected)
    return {"question": body.question, "answer": answer, "model": engine.model,
            "provider": engine.provider_id,
            "sources": [_source_card(s, i) for i, s in enumerate(selected, 1)], "metrics": metrics}


@app.get("/api/providers")
async def api_providers():
    return {"current": engine.provider_id, "model": engine.model,
            "providers": provider_catalog(engine.settings),
            "search": {"current": engine.search_engine.backend,
                       "backends": search_catalog(engine.settings)}}


class ProviderTest(BaseModel):
    provider: str
    api_key: str = ""
    base_url: str = ""


@app.post("/api/providers/models")
def api_provider_models(body: ProviderTest):
    """List models for a provider, optionally with an unsaved key/URL (the Test button)."""
    if body.provider not in PROVIDERS:
        raise HTTPException(status_code=404, detail="Unknown provider")
    spec = PROVIDERS[body.provider]
    llm = build_provider(body.provider, engine.settings, api_key=body.api_key or None,
                         base_url=(body.base_url if spec.editable_url else "") or None)
    try:
        models = llm.list_models(timeout=6)
        if spec.kind == "ollama":
            models = [m for m in models if "embed" not in m]
        return {"ok": True, "models": models[:300], "default": spec.default_model}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "models": [], "default": spec.default_model, "error": str(exc)[:300]}


class SettingsBody(BaseModel):
    provider: str
    model: str = ""
    api_key: str | None = None        # None or "" = keep the saved key
    base_url: str = ""
    search_provider: str = "duckduckgo"
    search_api_key: str | None = None
    search_base_url: str = ""
    temperature: float = 0.3


@app.post("/api/settings")
def api_settings(body: SettingsBody):
    if body.provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail="Unknown provider")
    if body.search_provider not in SEARCH_BACKENDS:
        raise HTTPException(status_code=400, detail="Unknown search backend")
    spec = PROVIDERS[body.provider]
    s = engine.settings
    if body.api_key:
        s.set_provider_value(spec.id, "api_key", body.api_key.strip())
    if spec.editable_url and body.base_url:
        url = body.base_url.strip().rstrip("/")
        if not re.match(r"^https?://", url):
            raise HTTPException(status_code=400, detail="Base URL must start with http:// or https://")
        s.set_provider_value(spec.id, "base_url", url)
    s.search_provider = body.search_provider
    if body.search_api_key:
        s.set_search_value(body.search_provider, "api_key", body.search_api_key.strip())
    if body.search_base_url:
        s.set_search_value(body.search_provider, "base_url", body.search_base_url.strip().rstrip("/"))
    s.temperature = max(0.0, min(float(body.temperature), 1.5))
    status = engine.switch_provider(body.provider, body.model.strip() or None)
    return {"ok": True, "status": status}


@app.get("/api/graph")
async def api_graph():
    return {"stats": engine.graph_stats(), "recent_nodes": engine.graph.recent_nodes(limit=50)}


@app.get("/api/graph/data")
async def api_graph_data(limit: int = 160):
    limit = max(10, min(limit, 600))
    nodes = [{"id": n["id"], "label": n["label"], "kind": n["kind"], "weight": n["weight"]}
             for n in engine.graph.top_nodes(limit=limit)]
    ids = {n["id"] for n in nodes}
    edges = [e for e in engine.graph.get_all_edges(limit=limit * 6)
             if e["source_id"] in ids and e["target_id"] in ids]
    return {"nodes": nodes, "edges": edges, "stats": engine.graph_stats()}


@app.get("/api/mindmap")
async def api_mindmap():
    data = mm_generator.load_from_graph(engine.graph)
    return {"mindmap": mm_generator.to_mermaid_mindmap(data), "graph": mm_generator.to_mermaid_graph(data),
            "timeline": mm_generator.to_mermaid_timeline(data), "stats": data["stats"],
            "concepts": data["concepts"], "research": data["research"], "sources": data["sources"]}


@app.post("/api/clear-graph")
async def api_clear_graph():
    engine.graph.clear()
    return {"status": "ok", "message": "Knowledge graph cleared"}


@app.get("/api/sessions")
async def api_list_sessions():
    items = sorted(chat_sessions.items(), key=lambda kv: kv[1].get("updated", 0), reverse=True)
    return {"sessions": [{"session_id": sid, "title": d.get("title") or "Untitled",
                          "message_count": len(d.get("history", [])),
                          "last_question": d["history"][-1]["question"] if d.get("history") else ""}
                         for sid, d in items if d.get("history") and not d.get("incognito")]}


@app.post("/api/sessions/{session_id}/rename")
async def api_rename_session(session_id: str, title: str = Form(...)):
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    chat_sessions[session_id]["title"] = title.strip()[:120] or "Untitled"
    chat_sessions[session_id]["auto_title"] = False
    _save_session(session_id, chat_sessions[session_id])
    return {"status": "ok"}


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del chat_sessions[session_id]
    if _valid_sid(session_id):
        (sessions_dir / f"{session_id}.json").unlink(missing_ok=True)
    return {"status": "ok"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local Perplex web UI")
    parser.add_argument("--model", "-m", help="Model to use (default: auto-select)")
    parser.add_argument("--provider", help=f"LLM provider: {', '.join(PROVIDERS)}")
    parser.add_argument("--port", "-p", type=int, default=int(os.getenv("PORT", 8000)))
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"),
                        help="Interface to bind (default 127.0.0.1; use 0.0.0.0 to expose on your network)")
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    if args.provider and args.provider != engine.provider_id:
        engine.switch_provider(args.provider, args.model, persist=False)
    elif args.model and args.model != engine.model:
        engine.model = args.model
    print(f"\n  Local Perplex {__version__}  ->  http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}")
    print(f"  {engine.provider_label} / {engine.model}  ({'online' if engine.llm_online else 'offline'})\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

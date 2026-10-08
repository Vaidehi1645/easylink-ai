"""EasyLink AI API server (Phase 2).

Wraps the existing RAGEngine — zero new AI logic. Provides:
  POST /chat     -> SSE-streamed, grounded answers (the widget's only endpoint)
  GET  /widget.js-> the embeddable web component
  GET  /test     -> hostile-CSS isolation test page (dev only)
  GET  /health   -> liveness + config summary
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from api.security import Guard
from easylink.config import load_settings
from easylink.gaps import record_gap
from easylink.providers import ProviderError, get_provider
from easylink.rag import RAGEngine
from easylink.store import Database, get_vector_store

settings = load_settings()
db = Database()
vectors = get_vector_store(db)
provider = get_provider(settings)
guard = Guard(settings)
_engines: dict[str, RAGEngine] = {}

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="EasyLink AI API", version="0.2.0")


def engine_for(site) -> RAGEngine:
    eng = _engines.get(site["id"])
    if eng is None:
        eng = RAGEngine(provider, vectors, settings, site["name"], site["url"])
        _engines[site["id"]] = eng
    return eng


def _clean_history(raw) -> list[dict]:
    out: list[dict] = []
    if isinstance(raw, list):
        for turn in raw[:8]:
            if isinstance(turn, dict):
                q = str(turn.get("q", ""))[:500]
                a = str(turn.get("a", ""))[:1500]
                if q and a:
                    out.append({"q": q, "a": a})
    return out


# ---------------------------------------------------------------- CORS ----
# CORS = browser cooperation headers. The SECURITY check (origin_allowed)
# lives in Guard.check — a foreign origin gets 403 even if we echo headers.
@app.middleware("http")
async def cors(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method == "OPTIONS":
        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": origin or "null",
                "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
                "Access-Control-Allow-Headers": "content-type",
                "Access-Control-Max-Age": "600",
                "Vary": "Origin",
            },
        )
    response = await call_next(request)
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    return response


# ----------------------------------------------------------------- chat ----
@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    token = str(body.get("token", ""))[:128]
    question = str(body.get("question", "")).strip()[:2000]
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)

    site = db.resolve_token(token)
    if site is None:
        return JSONResponse({"error": "invalid site token"}, status_code=401)

    ip = request.client.host if request.client else "unknown"
    verdict = guard.check(
        ip, token, request.headers.get("origin"), site, db.usage_today(site["id"])
    )
    if verdict:
        code, msg = verdict
        return JSONResponse({"error": msg}, status_code=code)

    db.bump_usage(site["id"])
    history = _clean_history(body.get("history"))
    engine = engine_for(site)

    def generate():
        try:
            for event in engine.stream_answer(question, site["id"], history):
                if "delta" in event:
                    yield "data: " + json.dumps({"delta": event["delta"]}, ensure_ascii=False) + "\n\n"
                else:
                    ans = event["answer"]
                    if ans.abstained:
                        record_gap(db, site["id"], question, "no_match")
                    payload = {
                        "answer": {
                            "text": ans.text,
                            "sources": ans.sources,
                            "abstained": ans.abstained,
                        }
                    }
                    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
        except ProviderError as exc:
            yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"
        except Exception:
            # never leak internals to the embedder
            yield "data: " + json.dumps({"error": "assistant error — please retry"}) + "\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------- widget ------
@app.get("/widget.js")
def widget_js():
    return FileResponse(STATIC / "widget.js", media_type="application/javascript")


@app.get("/test")
def test_page(request: Request):
    token = request.query_params.get("token")
    site = db.resolve_token(token) if token else None
    if site is None:
        site = db.get_site(None)  # latest crawled site
    if site is None:
        return JSONResponse(
            {"error": "no site crawled yet — run: python -m easylink crawl <url>"},
            status_code=404,
        )
    real_token = db.get_or_create_token(site["id"])
    html = (STATIC / "test.html").read_text(encoding="utf-8")
    html = html.replace("__TOKEN__", real_token).replace("__SITE__", site["name"])
    return HTMLResponse(html)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": settings.provider,
        "sites": len(db.list_sites()),
        "daily_quota": guard.daily_quota,
    }

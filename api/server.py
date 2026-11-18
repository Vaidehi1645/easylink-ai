"""EasyLink AI API server (Phase 2 + Phase 6 self-serve).

Wraps the existing RAGEngine — zero new AI logic. Provides:
  POST /chat        -> SSE-streamed, grounded answers (the widget's only endpoint)
  POST /ingest      -> SSE crawl feed: paste a URL, watch it crawl, get a demo link
  GET  /widget.js   -> the embeddable web component
  GET  /test        -> agent intake page (no token) / demo page (?token=...)
  GET  /demo/<slug> -> shareable demo page for a crawled site
  GET  /health      -> liveness + config summary
"""

from __future__ import annotations

import dataclasses
import json
import queue
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from api.security import Guard
from easylink.config import load_settings
from easylink.crawler import crawl, fetch_verdict
from easylink.gaps import record_gap
from easylink.indexer import index_pages
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

# Self-serve crawls are deliberately small: enough for a useful bot,
# small enough that one IP can't soak the server or the OpenAI bill.
# Wall-clock + attempt budgets keep a pasted link answered in well under a
# minute — a huge frontier must not drain for 9 minutes (palmonas.com did).
INGEST_MAX_PAGES = 20
INGEST_MAX_SECONDS = 22.0
INGEST_MAX_FETCHES = 60
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

# One crawl per host: a visitor who pastes the same link WHILE it is
# crawling JOINS that run (each visitor gets the same feed and demo link;
# only the starter spends quota). Guards the natural double-paste reflex.
_jobs: dict[str, list[queue.Queue]] = {}
_jobs_lock = threading.Lock()

app = FastAPI(title="EasyLink AI API", version="0.3.0")


def engine_for(site) -> RAGEngine:
    eng = _engines.get(site["id"])
    if eng is None:
        eng = RAGEngine(provider, vectors, settings, site["name"], site["url"], db=db)
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


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _origin(request: Request) -> str:
    """This server's own origin as the CLIENT sees it (proxy-aware), used to
    build shareable demo links + embed snippets."""
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or request.url.netloc
    )
    return f"{proto}://{host}"


def _demo_html(site) -> str:
    """The demo page (test.html) wired to a specific site's widget token."""
    real_token = db.get_or_create_token(site["id"])
    html = (STATIC / "test.html").read_text(encoding="utf-8")
    return html.replace("__TOKEN__", real_token).replace("__SITE__", site["name"])


# ---------------------------------------------------------------- CORS ----
# CORS = browser cooperation headers. The SECURITY check (origin_allowed)
# lives in Guard.check — a foreign origin gets 403 even if we echo headers.
@app.middleware("http")
async def cors(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": origin or "null",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "content-type",
            "Access-Control-Max-Age": "600",
            "Vary": "Origin",
        }
        # A page on a PUBLIC site calling a localhost API gets a Private
        # Network Access preflight from Chrome; without this answer the
        # widget on a live site is refused before it even starts.
        if request.headers.get("access-control-request-private-network"):
            headers["Access-Control-Allow-Private-Network"] = "true"
        return Response(status_code=204, headers=headers)
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
        ip, token, request.headers.get("origin"), site, db.usage_today(site["id"]),
        own_host=request.headers.get("host", ""),
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
                        record_gap(db, site["id"], question, "no_match", provider=provider)
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
        headers=SSE_HEADERS,
    )


# ------------------------------------------------------------- widget ------
@app.get("/widget.js")
def widget_js():
    return FileResponse(STATIC / "widget.js", media_type="application/javascript")


@app.get("/test")
def test_page(request: Request):
    token = request.query_params.get("token")
    if token:
        site = db.resolve_token(token)
        if site is None:
            return JSONResponse({"error": "invalid site token"}, status_code=401)
        return HTMLResponse(_demo_html(site))
    # No token -> the agent intake page (Phase 6): paste a link, watch the
    # crawl, get a shareable /demo/<slug> link back.
    return HTMLResponse((STATIC / "intake.html").read_text(encoding="utf-8"))


@app.get("/demo/{slug}")
def demo_page(slug: str):
    site = db.get_site_by_slug(slug)
    if site is None:
        return JSONResponse({"error": "no such demo — check the link"}, status_code=404)
    return HTMLResponse(_demo_html(site))


# ------------------------------------------------------------- ingest ------
@app.post("/ingest")
async def ingest(request: Request):
    """Self-serve onboarding (Phase 6): paste a URL -> crawl -> demo link.

    Streams SSE: {"type":"start"} -> live {"fetch"|"log"|"index"} feed ->
    {"type":"done"} carrying the demo link + embed snippet, or
    {"type":"error"} with a friendly message. Guards: URL shape, SSRF
    (public addresses only), per-IP crawl quota, host-level reuse
    (an already-crawled site is free and instant)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    url = str(body.get("url", "")).strip()[:2048]
    if not url:
        return JSONResponse({"error": "paste a website link first"}, status_code=400)
    try:
        parts = urlparse(url)
    except Exception:
        parts = None
    if parts is None or parts.scheme not in ("http", "https") or not parts.netloc:
        return JSONResponse(
            {"error": "that doesn't look like a valid web address — try https://example.com"},
            status_code=400,
        )
    try:
        host = parts.hostname or ""
    except ValueError:
        host = ""
    if not host:
        return JSONResponse(
            {"error": "that doesn't look like a valid web address — try https://example.com"},
            status_code=400,
        )
    ok, why = fetch_verdict(url)
    if not ok:
        return JSONResponse({"error": why}, status_code=400)

    origin = _origin(request)
    site_name = host[4:] if host.startswith("www.") else host
    ip = request.client.host if request.client else "unknown"

    def done_payload(site_row, slug: str, reused: bool, extra: dict | None = None) -> dict:
        token = db.get_or_create_token(site_row["id"])
        payload = {
            "type": "done",
            "link": f"{origin}/demo/{slug}",
            "snippet": f'<script src="{origin}/widget.js" data-site="{token}"></script>',
            "site": site_row["name"],
            "pages": int(site_row["page_count"] or 0),
            "chunks": int(site_row["chunk_count"] or 0),
            "reused": reused,
        }
        if extra:
            payload.update(extra)
        return payload

    # Decide under one lock:
    #   JOIN  — this host is being crawled right now (double-paste): share
    #           the running crawl, no second fetch, no second quota spend;
    #   REUSE — this host is already crawled: free + instant;
    #   START — new crawl: the only path that spends quota, and only the
    #           first visitor for this host spends it.
    joining = False
    existing = None
    events: queue.Queue | None = None
    with _jobs_lock:
        if host in _jobs:
            events = queue.Queue()
            _jobs[host].append(events)
            joining = True
        else:
            existing = db.find_site_by_host(host)
            if existing is None:
                verdict = guard.check_crawl(ip)
                if verdict:
                    code, msg = verdict
                    return JSONResponse({"error": msg}, status_code=code)
                events = queue.Queue()
                _jobs[host] = [events]

    if existing is not None:
        def reuse_stream():
            yield _sse({"type": "start", "url": url})
            yield _sse({"type": "reuse", "site": existing["name"]})
            yield _sse(
                done_payload(
                    existing,
                    db.ensure_slug(existing["id"]),
                    True,
                    extra={"learned": db.page_titles(existing["id"]), "seconds": 0.0},
                )
            )

        return StreamingResponse(reuse_stream(), media_type="text/event-stream", headers=SSE_HEADERS)

    crawl_settings = dataclasses.replace(
        settings,
        max_pages=INGEST_MAX_PAGES,
        max_depth=3,
        page_timeout=8.0,
        politeness_delay=0.1,
        sitemap_limit=100,
    )

    def publish(ev: dict) -> None:
        with _jobs_lock:
            for sub in _jobs.get(host, []):
                sub.put(ev)

    def close_job() -> None:
        with _jobs_lock:
            for sub in _jobs.get(host, []):
                sub.put(None)  # poison pill: every stream ends
            _jobs.pop(host, None)

    def job() -> None:
        t0 = time.monotonic()
        try:
            def log(msg):
                publish({"type": "log", "text": str(msg)})

            def on_fetch(fetched: str, status):
                publish({"type": "fetch", "url": fetched[:160], "status": status})

            result = crawl(
                url,
                crawl_settings,
                log=log,
                safe=True,
                on_fetch=on_fetch,
                max_seconds=INGEST_MAX_SECONDS,
                max_fetches=INGEST_MAX_FETCHES,
            )
            if not result.pages:
                publish({"type": "error", "message": "that site blocked us or had no readable content — try another link"})
                return
            # Honest heads-up: when most fetches came back with no readable
            # text, the site renders via JavaScript and answers will be thin.
            if result.skipped_thin >= 5 and result.skipped_thin > len(result.pages):
                publish({"type": "warn", "message": "this site renders most of its content with JavaScript — the bot learned only what the site's HTML provides, so answers may be limited"})
            publish({"type": "index", "text": f"indexing {len(result.pages)} pages — teaching your bot…"})

            site_id = db.create_site(url, site_name)
            pages_n, chunks_n = index_pages(
                site_id, result.pages, provider, db, vectors, crawl_settings, log=log
            )
            db.update_site_counts(site_id, pages_n, chunks_n)
            slug = db.ensure_slug(site_id)
            row = db.get_site(site_id)
            publish(
                done_payload(
                    row,
                    slug,
                    False,
                    extra={
                        "learned": [p.title for p in result.pages if p.title][:20],
                        "seconds": round(time.monotonic() - t0, 1),
                    },
                )
            )
        except ValueError as exc:
            publish({"type": "error", "message": str(exc)})
        except ProviderError:
            publish({"type": "error", "message": "indexing failed — the server's AI key may be missing"})
        except Exception:
            # never leak internals to the visitor
            publish({"type": "error", "message": "something went wrong while crawling — try another link"})
        finally:
            close_job()

    def stream():
        if not joining:
            threading.Thread(target=job, daemon=True).start()
        yield _sse({"type": "start", "url": url})
        if joining:
            yield _sse({"type": "note", "text": "already crawling this site — joining that run…"})
        while True:
            ev = events.get()
            if ev is None:
                break
            yield _sse(ev)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": settings.provider,
        "sites": len(db.list_sites()),
        "daily_quota": guard.daily_quota,
    }

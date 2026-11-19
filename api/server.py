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
import hmac
import json
import queue
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import quote, urlparse

from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from api.security import Guard
from easylink import auth as oauth
from easylink.config import load_settings
from easylink.crawler import crawl, crawl_warning, fetch_verdict
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
INGEST_MAX_PAGES = 30
INGEST_MAX_SECONDS = 35.0
INGEST_MAX_FETCHES = 90
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
    # Say how much the bot actually knows — a4-page bot built from an
    # Amazon-class wall must not present itself as a full-fledged one.
    pages = int(site["page_count"] or 0)
    note = f"Built from {pages} page{'s' if pages != 1 else ''} of {site['name']}"
    if 0 < pages <= 4:
        note += " — this site shares very little of its content, so answers will be limited"
    # "What this bot learned": the crawled page titles — the same list the
    # intake page shows, so a visitor sees the bot's actual knowledge (and
    # its limits) before asking. Injected as JSON inside <script>: escape
    # line separators and "</" so a hostile page title can't break out.
    titles = db.page_titles(site["id"])
    learned = (
        json.dumps(titles, ensure_ascii=False)
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
        .replace("</", "<\\/")
    )
    return (
        html.replace("__TOKEN__", real_token)
        .replace("__SITE__", site["name"])
        .replace("__NOTE__", note)
        .replace("__LEARNED__", learned)
    )


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
                            "verified": ans.verified,
                        }
                    }
                    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
        except ProviderError as exc:
            yield "data: " + json.dumps({"error": str(exc)}) + "\n\n"
        except Exception:
            # never leak internals to the embedder — but the SERVER must
            # see the real traceback, else every stream failure is silent.
            print(f"[chat] FAILED for site={site['id']!r} q={question[:80]!r}",
                  flush=True)
            traceback.print_exc()
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


# --------------------------------------------------------------- brand ------
@app.get("/logo.png")
def logo_png():
    """The EasyLink robot artwork: header logo on every page + the favicon."""
    if not (STATIC / "logo.png").exists():
        return HTMLResponse("logo not installed", status_code=404)
    return FileResponse(STATIC / "logo.png", media_type="image/png")


@app.get("/favicon.ico")
def favicon():
    # Browsers that ignore <link rel="icon"> still ask for this; same art.
    return logo_png()


@app.get("/test")
def test_page(request: Request):
    token = request.query_params.get("token")
    if token:
        blocked = _gate(request)
        if blocked:
            return blocked
        site = db.resolve_token(token)
        if site is None:
            return JSONResponse({"error": "invalid site token"}, status_code=401)
        return HTMLResponse(_demo_html(site))
    # No token -> the agent intake page (Phase 6): paste a link, watch the
    # crawl, get a shareable /demo/<slug> link back. Intake stays public —
    # the login modal there is a dismissible nudge, never a wall.
    return HTMLResponse((STATIC / "intake.html").read_text(encoding="utf-8"))


@app.get("/demo/{slug}")
def demo_page(request: Request, slug: str):
    blocked = _gate(request)
    if blocked:
        return blocked
    site = db.get_site_by_slug(slug)
    if site is None:
        return JSONResponse({"error": "no such demo — check the link"}, status_code=404)
    return HTMLResponse(_demo_html(site))


# ------------------------------------------------------------- auth ---------
# Owner sign-in (Google + GitHub). Everything here is feature-flagged: with
# no client IDs in .env, `oauth.configured()` is empty and the app behaves
# exactly as it did before the feature existed (no gates, no modal).
def _session_user(request: Request):
    """The signed-in owner from the session cookie (None = visitor)."""
    tok = request.cookies.get(oauth.SESSION_COOKIE)
    if not tok:
        return None
    return db.session_user(oauth.hash_token(tok))


def _auth_on() -> bool:
    return bool(oauth.configured(settings))


def _redirect_uri(request: Request, provider: str) -> str:
    """Where the provider sends the browser back.

    EASYLINK_PUBLIC_URL (https://your-domain) wins — required once the app
    sits behind a proxy that hides the real host. Google accepts plain
    http://localhost for development, but demands HTTPS anywhere else.
    """
    base = settings.public_url or str(request.base_url).rstrip("/")
    return f"{base}/auth/callback/{provider}"


def _gate(request: Request) -> Response | None:
    """Hard gate for demo pages: signed-in owners only.

    Off entirely when login is unconfigured or EASYLINK_DEMO_GATE=false.
    """
    if not _auth_on() or not settings.demo_gate or _session_user(request):
        return None
    path = request.url.path
    if request.url.query:
        path += "?" + request.url.query
    return RedirectResponse(
        f"/auth/login?next={quote(path, safe='/')}", status_code=303
    )


@app.get("/auth/me")
def auth_me(request: Request):
    """What the pages need to render the modal/chip — never any secrets."""
    user = _session_user(request)
    return {
        "providers": oauth.configured(settings),
        "user": ({"name": user["name"], "picture": user["picture"]} if user else None),
    }


@app.get("/auth/login")
def auth_login(request: Request, provider: str | None = None, next: str | None = None):
    """The sign-in card (no provider) or the bounce to the provider."""
    if not _auth_on():
        return RedirectResponse("/test", status_code=303)
    if provider is None:
        return HTMLResponse(oauth.login_page(settings, oauth.safe_next(next)))
    if provider not in oauth.configured(settings):
        return JSONResponse(
            {"error": "that sign-in method is not configured"}, status_code=400
        )
    state, nonce = oauth.new_state()
    resp = RedirectResponse(
        oauth.authorize_url(
            provider, settings, _redirect_uri(request, provider), state, nonce
        ),
        status_code=302,
    )
    resp.set_cookie(
        oauth.STATE_COOKIE,
        oauth.state_cookie_value(state, nonce, oauth.safe_next(next)),
        max_age=oauth.STATE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return resp


@app.get("/auth/callback/{provider}")
def auth_callback(
    request: Request,
    provider: str,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
):
    """Verify state -> exchange code -> verify token -> create the session."""

    def fail(msg: str, status_code: int = 400) -> Response:
        resp = JSONResponse({"error": msg}, status_code=status_code)
        resp.delete_cookie(oauth.STATE_COOKIE)
        return resp

    if error:
        return fail(f"sign-in was cancelled or refused ({error})")
    raw = request.cookies.get(oauth.STATE_COOKIE)
    if not raw or not state:
        return fail("sign-in attempt expired — please try again")
    try:
        saved = json.loads(raw)
    except (ValueError, TypeError):
        return fail("sign-in attempt corrupted — please try again")
    if not hmac.compare_digest(str(saved.get("state", "")), str(state)):
        return fail("state mismatch — sign-in rejected (possible CSRF)", status_code=403)
    if not code:
        return fail("no authorization code was returned")
    if provider not in oauth.configured(settings):
        return fail("sign-in method not configured", status_code=400)
    try:
        tokens = oauth.exchange_code(
            provider, settings, code, _redirect_uri(request, provider)
        )
        ident = oauth.identity(provider, settings, tokens, saved.get("nonce"))
    except oauth.AuthError as exc:
        return fail(str(exc), status_code=502)
    if not ident.get("provider_id"):
        return fail("the provider returned no account id", status_code=502)
    user_id = db.upsert_user(
        provider,
        ident["provider_id"],
        ident.get("email", ""),
        ident.get("name", ""),
        ident.get("picture", ""),
    )
    token = oauth.new_session_token()
    db.create_session(oauth.hash_token(token), user_id, settings.session_days)
    resp = RedirectResponse(oauth.safe_next(saved.get("next")), status_code=303)
    resp.set_cookie(
        oauth.SESSION_COOKIE,
        token,
        max_age=settings.session_days * 86400,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    resp.delete_cookie(oauth.STATE_COOKIE)
    return resp


@app.post("/auth/logout")
def auth_logout(request: Request):
    """POST on purpose: a forged <img src=/auth/logout> must not sign you out."""
    tok = request.cookies.get(oauth.SESSION_COOKIE)
    if tok:
        db.delete_session(oauth.hash_token(tok))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(oauth.SESSION_COOKIE)
    return resp


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
        # A4-page crawl (Amazon-class wall) must not present itself as a
        # healthy bot on re-open — carry the same honesty as a fresh warn.
        rpages = int(existing["page_count"] or 0)
        reuse_warn = (
            f"this site only ever gave us {rpages} page"
            f"{'s' if rpages != 1 else ''} — answers will be limited"
            if 0 < rpages <= 4
            else None
        )

        def reuse_stream():
            yield _sse({"type": "start", "url": url})
            yield _sse({"type": "reuse", "site": existing["name"]})
            if reuse_warn:
                yield _sse({"type": "warn", "message": reuse_warn})
            yield _sse(
                done_payload(
                    existing,
                    db.ensure_slug(existing["id"]),
                    True,
                    extra={
                        "learned": db.page_titles(existing["id"]),
                        "seconds": 0.0,
                        "warn": reuse_warn,
                    },
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
            # Honest heads-up when the SITE fought the crawl (blocked-heavy)
            # or serves its body via JavaScript (thin-heavy): say it out loud
            # so "I don't know" in chat reads as the site's refusal, not a bug.
            warn = crawl_warning(
                len(result.pages), result.skipped_blocked, result.skipped_thin
            )
            if warn:
                publish({"type": "warn", "message": warn})
            publish({"type": "index", "text": f"indexing {len(result.pages)} pages — teaching your bot…"})

            site_id = db.create_site(url, site_name)
            pages_n, chunks_n = index_pages(
                site_id, result.pages, provider, db, vectors, crawl_settings, log=log
            )
            db.update_site_counts(site_id, pages_n, chunks_n)
            # Remember what FIGHTED the crawl (bot walls, JS-only bodies) so
            # chat abstentions can say the truth instead of "I don't know".
            db.set_crawl_stats(
                site_id, blocked=result.skipped_blocked, thin=result.skipped_thin
            )
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
                        "warn": warn,
                        "blocked": result.skipped_blocked,
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

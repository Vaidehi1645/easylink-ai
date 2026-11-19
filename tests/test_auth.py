"""Auth suite: Google/GitHub sign-in, demo gates, sessions, CSRF state.

Run:
    venv\\Scripts\\python.exe tests\\test_auth.py

Runs the FastAPI app in-process (fastapi.testclient) with fake OAuth
credentials in env, so the whole flow is provable with zero real
credentials and zero network calls — the provider exchange/identity are
mocked at the module boundary (easylink.auth).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Fake credentials BEFORE api.server imports easylink.config — these make
# oauth.configured() return both providers, i.e. login fully "on".
os.environ["GOOGLE_CLIENT_ID"] = "utest-google-id.apps.googleusercontent.com"
os.environ["GOOGLE_CLIENT_SECRET"] = "utest-google-secret"
os.environ["GITHUB_CLIENT_ID"] = "utest-github-id"
os.environ["GITHUB_CLIENT_SECRET"] = "utest-github-secret"
os.environ.pop("EASYLINK_DEMO_GATE", None)  # default = hard demo gate

from fastapi.testclient import TestClient  # noqa: E402

import api.server as server  # noqa: E402
import easylink.auth as oauth  # noqa: E402

# Force OUR fake credentials onto the loaded settings — .env may contain real
# ones (load_dotenv overrides ambient env), and the bounce test asserts on
# the exact client id. Behaviour must not depend on the developer's .env.
server.settings.google_client_id = "utest-google-id.apps.googleusercontent.com"
server.settings.google_client_secret = "utest-google-secret"
server.settings.github_client_id = "utest-github-id"
server.settings.github_client_secret = "utest-github-secret"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  {mark}  {name}" + (f"  [{detail}]" if detail else ""))


def fake_exchange(provider, settings, code, redirect_uri):
    return {"access_token": "fake-token", "id_token": "fake.jwt.token"}


def fake_identity_google(provider, settings, tokens, nonce):
    return {
        "provider_id": "utest-g-1",
        "email": "utest@example.com",
        "name": "Unit Tester",
        "picture": "https://example.com/p.png",
    }


def fake_identity_github(provider, settings, tokens, nonce):
    return {
        "provider_id": "utest-gh-1",
        "email": "utest@example.com",
        "name": "Unit Tester",
        "picture": "",
    }


def sign_in(client: TestClient, provider: str = "google", next_url: str = "/test"):
    """Run the whole dance with mocked provider calls -> the callback response."""
    orig = (oauth.exchange_code, oauth.identity)
    oauth.exchange_code = fake_exchange
    oauth.identity = (
        fake_identity_google if provider == "google" else fake_identity_github
    )
    try:
        r = client.get(
            f"/auth/login?provider={provider}&next={next_url}",
            follow_redirects=False,
        )
        state = None
        if r.status_code in (302, 303):
            q = parse_qs(urlparse(r.headers.get("location", "")).query)
            state = (q.get("state") or [None])[0]
        return client.get(
            f"/auth/callback/{provider}?code=fake-code&state={state}",
            follow_redirects=False,
        )
    finally:
        oauth.exchange_code, oauth.identity = orig


def set_cookie(resp, name: str) -> str:
    """The Set-Cookie header for `name` ("" when absent)."""
    for h in resp.headers.get_list("set-cookie"):
        if h.startswith(name + "="):
            return h
    return ""


def main() -> int:
    c = TestClient(server.app)

    print("\n== auth: config + login card ==")
    r = c.get("/auth/me")
    d = r.json()
    check(
        "/auth/me reports both providers and no user",
        r.status_code == 200 and d["providers"] == ["google", "github"] and d["user"] is None,
        f"{r.status_code} {d}",
    )

    r = c.get("/auth/login")
    html = r.text
    check(
        "login card lists Google + GitHub buttons",
        r.status_code == 200
        and "Continue with Google" in html
        and "Continue with GitHub" in html
        and "provider=google" in html
        and "provider=github" in html,
        str(r.status_code),
    )

    r = c.get("/auth/login?next=https://evil.example.com")
    check(
        "login card sanitises an evil next (open-redirect blocked)",
        r.status_code == 200 and "evil" not in r.text and "next=/test" in r.text,
        "",
    )

    print("\n== auth: provider bounce + state ==")
    r = c.get("/auth/login?provider=google&next=/demo/x", follow_redirects=False)
    loc = r.headers.get("location", "")
    sc = set_cookie(r, oauth.STATE_COOKIE)
    check(
        "provider bounce goes to Google with state + our client id",
        r.status_code == 302
        and loc.startswith("https://accounts.google.com/")
        and "utest-google-id" in loc
        and "state=" in loc
        and "HttpOnly" in sc
        and "SameSite=lax" in sc,
        f"{r.status_code} {loc[:60]}",
    )

    r = c.get("/auth/login?provider=bogus", follow_redirects=False)
    check("unknown provider -> 400", r.status_code == 400, str(r.status_code))

    fresh = TestClient(server.app)
    r = fresh.get(
        "/auth/callback/google?code=x&state=whatever", follow_redirects=False
    )
    check(
        "callback without state cookie -> rejected",
        r.status_code == 400 and "expired" in r.json().get("error", ""),
        f"{r.status_code} {r.json()}",
    )

    r = c.get(
        "/auth/callback/google?code=x&state=WRONG-STATE", follow_redirects=False
    )
    check(
        "callback with wrong state -> 403 (CSRF)",
        r.status_code == 403 and "CSRF" in r.json().get("error", ""),
        str(r.status_code),
    )

    print("\n== auth: happy path (mocked provider) ==")
    r = sign_in(c, "google", "/demo/does-not-matter")
    sc = set_cookie(r, oauth.SESSION_COOKIE)
    check(
        "full Google sign-in: 303 back, session cookie HttpOnly + SameSite",
        r.status_code == 303
        and r.headers.get("location", "").startswith("/demo/")
        and "HttpOnly" in sc
        and "SameSite=lax" in sc,
        f"{r.status_code} -> {r.headers.get('location', '')}",
    )

    d = c.get("/auth/me").json()
    check(
        "/auth/me shows the signed-in user",
        d["user"] is not None and d["user"]["name"] == "Unit Tester",
        str(d),
    )

    tok = c.cookies.get(oauth.SESSION_COOKIE) or ""
    raw_n = server.db.conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE token_hash=?", (tok,)
    ).fetchone()[0]
    hashed_n = server.db.conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE token_hash=?", (oauth.hash_token(tok),)
    ).fetchone()[0]
    check(
        "session cookie is stored only as a SHA-256 hash",
        bool(tok) and raw_n == 0 and hashed_n == 1,
        f"raw={raw_n} hashed={hashed_n}",
    )

    print("\n== auth: gates ==")
    slug_row = server.db.conn.execute(
        "SELECT slug FROM sites WHERE slug IS NOT NULL AND slug != '' LIMIT 1"
    ).fetchone()
    created_sid = None
    if slug_row:
        slug = slug_row["slug"]
    else:
        created_sid = server.db.create_site(
            "https://utest-auth.example.com", "UTest Auth"
        )
        slug = server.db.ensure_slug(created_sid)

    visitor = TestClient(server.app)
    r = visitor.get(f"/demo/{slug}", follow_redirects=False)
    check(
        "demo page redirects visitors to sign-in (hard gate)",
        r.status_code == 303
        and r.headers.get("location", "").startswith("/auth/login?next=")
        and "demo" in r.headers.get("location", ""),
        f"{r.status_code} {r.headers.get('location', '')}",
    )

    r = c.get(f"/demo/{slug}", follow_redirects=False)
    check(
        "demo page opens for a signed-in owner",
        r.status_code == 200 and "Live Demo" in r.text,
        str(r.status_code),
    )

    site = server.db.get_site_by_slug(slug)
    token = server.db.get_or_create_token(site["id"])
    r = visitor.get(f"/test?token={token}", follow_redirects=False)
    check(
        "/test?token= demo view is gated too",
        r.status_code == 303
        and r.headers.get("location", "").startswith("/auth/login?next="),
        f"{r.status_code} {r.headers.get('location', '')}",
    )
    r = c.get(f"/test?token={token}", follow_redirects=False)
    check("signed-in owner sees the token demo view", r.status_code == 200, str(r.status_code))

    r = visitor.get("/test", follow_redirects=False)
    check(
        "intake page stays public (no login wall)",
        r.status_code == 200 and "loginmodal" in r.text,
        str(r.status_code),
    )

    r = visitor.post("/ingest", json={}, follow_redirects=False)
    check(
        "/ingest stays public without a session",
        not (300 <= r.status_code < 400),
        str(r.status_code),
    )

    r = visitor.post("/chat", json={}, follow_redirects=False)
    check(
        "/chat stays public without a session",
        not (300 <= r.status_code < 400),
        str(r.status_code),
    )

    print("\n== auth: logout + expiry ==")
    r = c.post("/auth/logout", follow_redirects=False)
    d = c.get("/auth/me").json()
    check(
        "logout kills the session",
        r.status_code == 200 and d["user"] is None,
        f"{r.status_code} {d}",
    )
    r = visitor.get(f"/demo/{slug}", follow_redirects=False)
    check(
        "after logout the gate is back",
        r.status_code == 303,
        str(r.status_code),
    )

    server.db.conn.execute(
        "INSERT INTO users (id, provider, provider_id, email, name, picture, created_at) "
        "VALUES ('utest-exp','google','utest-exp','','Expired','','2020-01-01T00:00:00')"
    )
    server.db.conn.execute(
        "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) "
        "VALUES ('utest-expired-hash','utest-exp','2020-01-01T00:00:00','2020-01-02T00:00:00')"
    )
    server.db.conn.commit()
    check(
        "expired session is rejected",
        server.db.session_user("utest-expired-hash") is None,
        "",
    )

    print("\n== auth: GitHub flow + unit checks ==")
    g = TestClient(server.app)
    r = sign_in(g, "github", "/test")
    d = g.get("/auth/me").json()
    check(
        "GitHub sign-in works with the same flow",
        r.status_code == 303 and d["user"] is not None and d["user"]["name"] == "Unit Tester",
        f"{r.status_code} {d}",
    )

    check(
        "safe_next unit: only local paths pass",
        oauth.safe_next("https://evil.com") == "/test"
        and oauth.safe_next("//evil.com") == "/test"
        and oauth.safe_next("/\\evil") == "/test"
        and oauth.safe_next(None) == "/test"
        and oauth.safe_next("/demo/x") == "/demo/x",
        "",
    )

    # ---- cleanup: test users/sessions (+ throwaway site) from the real DB ----
    server.db.conn.execute(
        "DELETE FROM sessions WHERE user_id IN (SELECT id FROM users WHERE provider_id LIKE 'utest%')"
    )
    server.db.conn.execute("DELETE FROM users WHERE provider_id LIKE 'utest%'")
    if created_sid:
        server.db.delete_site(created_sid)
    server.db.conn.commit()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

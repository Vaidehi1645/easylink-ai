"""Owner sign-in: Google + GitHub OAuth (authorization-code flow).

The load-bearing security decisions — each one is a line in an interview:

- `state`  — random value minted per attempt, parked in an HttpOnly cookie,
             compared on the callback. A forged/CSRF'd callback dies here.
- `nonce`  — second random value that must come back *inside* Google's ID
             token; an intercepted authorization response cannot be replayed.
- Google ID tokens are verified by `google-auth.verify_oauth2_token`, which
             checks the signature against Google's published certificates, the
             audience (our client id) and the expiry. We never parse a JWT by
             hand — hand-rolled JWT verification is how takeovers start.
- The client secret only ever lives in .env and travels server-to-server.
- Sessions: 256-bit random token; only its SHA-256 hash is stored in SQLite,
             delivered in an HttpOnly + SameSite=Lax cookie. A database leak
             does not hand over live sessions.
"""

from __future__ import annotations

import hashlib
import html
import json
import secrets
from urllib.parse import quote, urlencode

import requests

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"

STATE_COOKIE = "el_oauth_state"
SESSION_COOKIE = "el_session"
STATE_MAX_AGE = 600  # 10 minutes to finish the sign-in dance


class AuthError(Exception):
    """A sign-in failure with a message that is safe to show a human."""


# ----------------------------------------------------------------- helpers ---
def configured(settings) -> list[str]:
    """Providers with credentials in .env — [] means login is off entirely."""
    out: list[str] = []
    if settings.google_client_id and settings.google_client_secret:
        out.append("google")
    if settings.github_client_id and settings.github_client_secret:
        out.append("github")
    return out


def _creds(settings, provider: str) -> tuple[str, str]:
    if provider == "google":
        return settings.google_client_id, settings.google_client_secret
    return settings.github_client_id, settings.github_client_secret


def safe_next(next_url: str | None, default: str = "/test") -> str:
    """Only local paths may be redirected back to (open-redirect defence)."""
    if not next_url:
        return default
    if not next_url.startswith("/") or next_url.startswith("//"):
        return default
    if "\\" in next_url or "\n" in next_url or "\r" in next_url:
        return default
    return next_url


def new_state() -> tuple[str, str]:
    """(state, nonce) for one sign-in attempt."""
    return secrets.token_urlsafe(24), secrets.token_urlsafe(24)


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def state_cookie_value(state: str, nonce: str, next_url: str) -> str:
    return json.dumps({"state": state, "nonce": nonce, "next": next_url})


# ------------------------------------------------------------- the flow ------
def authorize_url(
    provider: str, settings, redirect_uri: str, state: str, nonce: str
) -> str:
    """Where the browser is sent to collect the user's consent."""
    client_id, _ = _creds(settings, provider)
    if provider == "google":
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "nonce": nonce,
            "prompt": "select_account",
        }
        return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": "read:user user:email",
        "state": state,
    }
    return f"{GITHUB_AUTH_URL}?{urlencode(params)}"


def exchange_code(provider: str, settings, code: str, redirect_uri: str) -> dict:
    """Trade the one-time code for tokens (server-to-server; secret stays here)."""
    client_id, client_secret = _creds(settings, provider)
    url = GOOGLE_TOKEN_URL if provider == "google" else GITHUB_TOKEN_URL
    try:
        resp = requests.post(
            url,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
            timeout=20,
        )
    except requests.RequestException as exc:
        raise AuthError("could not reach the sign-in provider — try again") from exc
    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    if resp.status_code >= 400 or "error" in payload:
        desc = str(
            payload.get("error_description") or payload.get("error") or "rejected"
        )
        raise AuthError(f"the provider refused the code ({desc})")
    return payload


def identity(provider: str, settings, tokens: dict, nonce: str | None) -> dict:
    """-> {provider_id, email, name, picture}, with the token *verified*."""
    if provider == "google":
        return _google_identity(settings, tokens, nonce)
    return _github_identity(tokens)


def _google_identity(settings, tokens: dict, nonce: str | None) -> dict:
    id_token = tokens.get("id_token")
    if not id_token:
        raise AuthError("Google returned no ID token")
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.id_token import verify_oauth2_token

        # Checks signature (Google's JWKS), audience and expiry for us.
        claims = verify_oauth2_token(
            id_token, Request(), audience=settings.google_client_id
        )
    except Exception as exc:  # google-auth raises BadAssertionError & friends
        raise AuthError("Google's ID token could not be verified") from exc
    if nonce and claims.get("nonce") != nonce:
        raise AuthError("sign-in nonce mismatch — rejected (possible replay)")
    return {
        "provider_id": str(claims.get("sub", "")),
        "email": str(claims.get("email") or ""),
        "name": str(claims.get("name") or claims.get("email") or "Google user"),
        "picture": str(claims.get("picture") or ""),
    }


def _github_identity(tokens: dict) -> dict:
    access = tokens.get("access_token")
    if not access:
        raise AuthError("GitHub returned no access token")
    headers = {
        "Authorization": f"Bearer {access}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "EasyLinkAI",
    }
    try:
        resp = requests.get(GITHUB_USER_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        user = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise AuthError("could not fetch your GitHub profile") from exc
    email = str(user.get("email") or "")
    if not email:
        # The primary e-mail may be private — ask for the verified list.
        try:
            er = requests.get(GITHUB_EMAILS_URL, headers=headers, timeout=20)
            emails = er.json()
            if isinstance(emails, list):
                pick = next(
                    (e for e in emails if e.get("primary") and e.get("verified")), None
                )
                pick = pick or next((e for e in emails if e.get("verified")), None)
                email = str((pick or {}).get("email") or "")
        except (requests.RequestException, ValueError):
            pass
    return {
        "provider_id": str(user.get("id", "")),
        "email": email,
        "name": str(user.get("name") or user.get("login") or "GitHub user"),
        "picture": str(user.get("avatar_url") or ""),
    }


# ------------------------------------------------------------- the card ------
_G_SVG = (
    '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">'
    '<path fill="#4285F4" d="M23.5 12.3c0-.8-.1-1.6-.2-2.3H12v4.5h6.5a5.6 5.6 0 0 1-2.4 3.6v3h3.9c2.3-2.1 3.5-5.2 3.5-8.8z"/>'
    '<path fill="#34A853" d="M12 24c3.2 0 6-1.1 8-2.9l-3.9-3c-1.1.7-2.5 1.2-4.1 1.2-3.1 0-5.8-2.1-6.8-5H1.2v3.1C3.2 21.3 7.3 24 12 24z"/>'
    '<path fill="#FBBC05" d="M5.2 14.3a7.2 7.2 0 0 1 0-4.6V6.6H1.2a12 12 0 0 0 0 10.8l4-3.1z"/>'
    '<path fill="#EA4335" d="M12 4.8c1.8 0 3.4.6 4.6 1.8l3.4-3.4C18 1.2 15.2 0 12 0 7.3 0 3.2 2.7 1.2 6.6l4 3.1c1-2.9 3.7-4.9 6.8-4.9z"/></svg>'
)
_GH_SVG = (
    '<svg viewBox="0 0 16 16" width="18" height="18" fill="currentColor" aria-hidden="true">'
    '<path d="M8 0C3.6 0 0 3.6 0 8c0 3.5 2.3 6.5 5.5 7.6.4.1.5-.2.5-.4v-1.4c-2.2.5-2.7-1-2.7-1-.4-.9-.9-1.2-.9-1.2-.7-.5.1-.5.1-.5.8.1 1.2.8 1.2.8.7 1.2 1.9.9 2.3.7.1-.5.3-.9.5-1.1-1.8-.2-3.6-.9-3.6-4 0-.9.3-1.6.8-2.2-.1-.2-.4-1 .1-2.1 0 0 .7-.2 2.2.8a7.5 7.5 0 0 1 4 0c1.5-1 2.2-.8 2.2-.8.5 1.1.2 1.9.1 2.1.5.6.8 1.3.8 2.2 0 3.1-1.9 3.8-3.6 4 .3.3.5.8.5 1.5v2.2c0 .2.1.5.5.4A8 8 0 0 0 16 8c0-4.4-3.6-8-8-8z"/></svg>'
)

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in — EasyLink AI</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: linear-gradient(160deg, #eef0ff 0%, #f6f7fb 45%, #eef0ff 100%);
    color: #1f2430; padding: 24px;
  }
  .card {
    width: 100%; max-width: 380px; background: #fff; border: 1px solid #e6e8f0;
    border-radius: 18px; padding: 30px 26px 26px; text-align: center;
    box-shadow: 0 18px 45px rgba(31,36,48,.12);
  }
  .logo { font-size: 13px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: #4f46e5; text-decoration: none; }
  h1 { font-size: 21px; margin: 14px 0 4px; }
  .sub { color: #8a90a3; font-size: 13.5px; margin: 0 0 20px; }
  .btn {
    display: flex; align-items: center; justify-content: center; gap: 10px;
    width: 100%; padding: 12px; margin: 9px 0; border-radius: 11px;
    font-size: 14.5px; font-weight: 600; text-decoration: none; cursor: pointer;
    border: 1.5px solid #d8dbe7; background: #fff; color: #1f2430;
    font-family: inherit;
  }
  .btn:hover { border-color: #4f46e5; }
  .btn.github { background: #14162b; border-color: #14162b; color: #fff; }
  .btn.github:hover { filter: brightness(1.15); }
  .fine { font-size: 12px; color: #8a90a3; margin-top: 16px; line-height: 1.7; }
  .fine a { color: #4f46e5; text-decoration: none; font-weight: 600; }
</style>
</head>
<body>
  <div class="card">
    <a class="logo" href="/test">&#128279; EasyLink AI</a>
    <h1>Sign in to continue</h1>
    <p class="sub">One click — we only ever ask for your name and e-mail.</p>
__BUTTONS__
    <p class="fine">Signing in keeps your bots and demo links in one place.<br>
      <a href="__BACK__">&#8592; Back to where you were</a></p>
  </div>
</body>
</html>
"""


def login_page(settings, next_url: str) -> str:
    """The centered sign-in card served when a gated page bounces you here."""
    next_q = quote(next_url, safe="/")
    provs = configured(settings)
    buttons = ""
    if "google" in provs:
        buttons += (
            f'    <a class="btn google" href="/auth/login?provider=google&amp;next={next_q}">'
            f"{_G_SVG}Continue with Google</a>\n"
        )
    if "github" in provs:
        buttons += (
            f'    <a class="btn github" href="/auth/login?provider=github&amp;next={next_q}">'
            f"{_GH_SVG}Continue with GitHub</a>\n"
        )
    return (
        _PAGE.replace("__BUTTONS__", buttons)
        .replace("__BACK__", html.escape(next_url))
    )

"""One shared owner session for the suites, for runs with owner login ON.

The demo pages (/demo/<slug>, /test?token=...) redirect to the sign-in card
WHEN Google/GitHub credentials exist in .env. The suites must pass either
way, so the ones that open gated pages seed the same suite owner here and
send its session as the el_session cookie. With no credentials the gates
are off and the cookie is simply ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from easylink import auth as oauth  # noqa: E402
from easylink.store import Database  # noqa: E402

COOKIE_NAME = "el_session"


def owner_session() -> str:
    """Create (or refresh) the suite owner; returns the raw session token."""
    db = Database()
    user_id = db.upsert_user(
        "test", "suite-owner", "suite-owner@example.test", "Suite Owner", ""
    )
    # keep exactly one live suite session (refresh instead of piling up)
    db.conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    db.conn.commit()
    token = oauth.new_session_token()
    db.create_session(oauth.hash_token(token), user_id, 30)
    return token

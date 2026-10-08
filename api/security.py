"""Widget API security — layered exactly as the risk register demands:

1. per-site token        -> random internet users can't call your API
2. origin check          -> stolen tokens don't work on foreign sites
3. rate limit (IP+token) -> bots can't hammer you
4. daily quota           -> cost explosions are impossible (your pricing caps)

CORS headers are handled separately by the server (CORS is a browser
cooperation mechanism, NOT a security boundary — the checks above are).
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from urllib.parse import urlparse

from easylink.config import Settings


def origin_allowed(origin: str | None, site_url: str) -> bool:
    """Allow: the site's own host, localhost (dev test pages), file:// (null),
    and missing origins (curl / server-to-server)."""
    if not origin or origin == "null":
        return True
    o = urlparse(origin)
    host = o.netloc or o.path
    site_host = urlparse(site_url).netloc
    if host == site_host:
        return True
    if host.startswith("localhost") or host.startswith("127.0.0.1"):
        return True
    return False


class Guard:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ip_per_minute = int(os.environ.get("EASYLINK_RATE_IP", "30"))
        self.token_per_minute = int(os.environ.get("EASYLINK_RATE_TOKEN", "120"))
        self.daily_quota = int(os.environ.get("EASYLINK_DAILY_QUOTA", "500"))
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def _allow(self, key: str, limit: int) -> bool:
        now = time.time()
        with self._lock:
            q = self._hits[key]
            while q and q[0] < now - 60.0:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def check(
        self,
        ip: str,
        token: str,
        origin: str | None,
        site_row,
        usage_today: int,
    ) -> tuple[int, str] | None:
        """None = allowed. (status, message) = blocked."""
        if not origin_allowed(origin, site_row["url"]):
            return 403, "this widget is not allowed on that website"
        if not self._allow(f"ip:{ip}", self.ip_per_minute):
            return 429, "too many requests — please slow down"
        if not self._allow(f"tok:{token}", self.token_per_minute):
            return 429, "this widget is receiving too many messages"
        if usage_today >= self.daily_quota:
            return 429, "daily message limit reached — try again tomorrow"
        return None

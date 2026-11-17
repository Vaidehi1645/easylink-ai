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


def origin_allowed(origin: str | None, site_url: str, own_host: str | None = None) -> bool:
    """Allow: the site's own host, this server's own host (the demo page is
    served by us, so it is trusted), localhost (dev test pages), file:// (null),
    and missing origins (curl / server-to-server)."""
    if not origin or origin == "null":
        return True
    o = urlparse(origin)
    host = o.netloc or o.path
    site_host = urlparse(site_url).netloc
    if host == site_host:
        return True
    if own_host and host == own_host:
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
        # Self-serve crawl quota (Phase 6): crawls cost fetches + embeddings.
        self.crawl_daily = int(os.environ.get("EASYLINK_CRAWL_DAILY", "10"))
        self._hits: dict[str, deque] = defaultdict(deque)
        self._crawls: dict[str, tuple[str, int]] = {}
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
        own_host: str | None = None,
    ) -> tuple[int, str] | None:
        """None = allowed. (status, message) = blocked."""
        if not origin_allowed(origin, site_row["url"], own_host):
            return 403, "this widget is not allowed on that website"
        if not self._allow(f"ip:{ip}", self.ip_per_minute):
            return 429, "too many requests — please slow down"
        if not self._allow(f"tok:{token}", self.token_per_minute):
            return 429, "this widget is receiving too many messages"
        if usage_today >= self.daily_quota:
            return 429, "daily message limit reached — try again tomorrow"
        return None

    def check_crawl(self, ip: str) -> tuple[int, str] | None:
        """Crawl quota (Phase 6): N crawls per IP per day.

        Called ONLY when a crawl will actually run — reusing an existing
        crawl is free (no fetches, no embeddings), so it never spends quota.
        None = allowed to crawl; (429, msg) = blocked."""
        day = time.strftime("%Y-%m-%d")
        with self._lock:
            seen_day, n = self._crawls.get(ip, (day, 0))
            if seen_day != day:
                n = 0  # new day -> counter resets
            if n >= self.crawl_daily:
                return 429, "daily crawl limit reached — try again tomorrow"
            self._crawls[ip] = (day, n + 1)
        return None

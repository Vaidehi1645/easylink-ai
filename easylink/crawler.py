"""Responsible crawler (Phase 1).

Built-in safety from the risk register:
- sitemap-first discovery
- hard page cap + depth limit
- robots.txt respected (honest User-Agent)
- URL normalization + content-hash dedupe (loop killers)
- trap detection (pagination / calendars / session ids)
- per-page timeout, one job never crashes the system
"""

from __future__ import annotations

import hashlib
import re
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from easylink.config import Settings

# Query params that are noise, not content
_STRIP_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "msclkid", "ref", "source", "mc_cid", "mc_eid",
}
# Extensions we never fetch (binary / non-HTML)
_BAD_EXT = re.compile(
    r"\.(jpg|jpeg|png|gif|webp|svg|ico|pdf|zip|gz|mp3|mp4|avi|mov|docx?|xlsx?"
    r"|pptx?|css|js|json|xml|rss|atom|woff2?|ttf|eot)(?:$|\?)",
    re.I,
)
# URL shapes that eat crawlers: pagination, calendars, facets, sort loops
_TRAP_PARAMS = {"page", "paged", "p", "seite", "seite_nr", "sort", "order", "filter"}
_TRAP_PATTERNS = re.compile(
    r"(?:/page/\d+|/calendar|/date/\d{4}|/feed/?$|/wp-json/|/cart|/checkout"
    r"|/login|/signin|/search\b|/tag/|/author/)",
    re.I,
)


def normalize_url(url: str) -> str | None:
    """Canonical form so 'the same page' is only visited once."""
    try:
        p = urlparse(url)
    except ValueError:
        return None
    if p.scheme not in ("http", "https") or not p.netloc:
        return None

    # drop noise params, sort the rest for stable identity
    pairs = [
        (k, v)
        for k, v in parse_qsl(p.query, keep_blank_values=False)
        if k.lower() not in _STRIP_PARAMS
    ]
    pairs.sort()
    query = urlencode(pairs)

    path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", query, ""))


def is_trap(url: str) -> bool:
    p = urlparse(url)
    if _TRAP_PATTERNS.search(p.path):
        return True
    for key, value in parse_qsl(p.query):
        if key.lower() in _TRAP_PARAMS:
            return True
        if len(p.query) > 120:  # runaway query strings
            return True
        if not value and key.lower() in {"output", "format"}:
            return True
    return False


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_content(html: str, url: str) -> tuple[str, str]:
    """Returns (title, main_text). Strips nav/boilerplate, prefers <main>/<article>."""
    soup = BeautifulSoup(html, "lxml")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title = og["content"].strip()
    if not title and soup.h1:
        title = soup.h1.get_text(" ", strip=True)

    for tag in soup(["script", "style", "noscript", "nav", "header", "footer",
                     "aside", "form", "svg", "iframe", "template", "button"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article")
    root = main if main else soup.body or soup
    lines: list[str] = []
    seen: set[str] = set()
    for raw in root.get_text("\n").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if len(line) < 2:
            continue
        if line in seen:  # repeated nav/menu items
            continue
        seen.add(line)
        lines.append(line)

    text = "\n".join(lines)
    return title or url, text


def collect_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        absolute = urljoin(base_url, href)
        if _BAD_EXT.search(urlparse(absolute).path):
            continue
        out.append(absolute)
    return out


# --------------------------------------------------------------------------- #
# Sitemap + robots
# --------------------------------------------------------------------------- #
def load_robots(base_url: str, session: requests.Session, timeout: float) -> urllib.robotparser.RobotFileParser:
    """Fetch robots.txt with the SAME identity the crawler uses.

    urllib's default 'Python-urllib' UA gets blocked by some WAFs, and
    robotparser silently turns that into disallow_all=True — which would ban
    us from the whole site. Using our own session keeps identity consistent.
    RFC 9309: 200=obey rules, 404/410=no rules (allow), 401/403=no crawling.
    """
    rp = urllib.robotparser.RobotFileParser()
    robots_url = urljoin(base_url, "/robots.txt")
    rp.set_url(robots_url)
    try:
        resp = session.get(robots_url, timeout=timeout)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        elif resp.status_code in (401, 403):
            rp.disallow_all = True
        else:  # 404, 410, 5xx, redirects...
            rp.allow_all = True
    except Exception:
        rp.allow_all = True
    return rp


def sitemap_urls(base_url: str, session: requests.Session, timeout: float, limit: int = 500) -> list[str]:
    """sitemap.xml first — the civilized way to discover pages."""
    found: list[str] = []
    queue = [urljoin(base_url, "/sitemap.xml")]
    visited: set[str] = set()

    while queue and len(visited) < 3 and len(found) < limit:
        sm_url = queue.pop(0)
        if sm_url in visited:
            continue
        visited.add(sm_url)
        try:
            resp = session.get(sm_url, timeout=timeout)
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
        except Exception:
            continue

        tag = root.tag.split("}")[-1]
        ns = {"sm": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
        if tag == "sitemapindex":
            for loc in root.findall(".//sm:sitemap/sm:loc", ns) or root.findall(".//loc"):
                queue.append((loc.text or "").strip())
        else:
            for loc in root.findall(".//sm:url/sm:loc", ns) or root.findall(".//loc"):
                u = (loc.text or "").strip()
                if u:
                    found.append(u)
                if len(found) >= limit:
                    break
    return found


# --------------------------------------------------------------------------- #
# Crawl
# --------------------------------------------------------------------------- #
@dataclass
class Page:
    url: str
    title: str
    text: str
    content_hash: str


@dataclass
class CrawlResult:
    pages: list[Page] = field(default_factory=list)
    skipped_blocked: int = 0
    skipped_thin: int = 0
    skipped_trap: int = 0
    skipped_dup: int = 0
    errors: int = 0


def crawl(start_url: str, settings: Settings, log=print) -> CrawlResult:
    start = normalize_url(start_url)
    if not start:
        raise ValueError(f"Not a valid http(s) URL: {start_url}")

    host = urlparse(start).netloc
    session = requests.Session()
    session.headers.update({
        "User-Agent": settings.user_agent,
        "Accept": "text/html,application/xhtml+xml",
    })
    robots = load_robots(start, session, settings.page_timeout)

    result = CrawlResult()
    seen: set[str] = set()
    visited_hashes: set[str] = set()
    queue: deque[tuple[str, int]] = deque()

    # --- phase A: sitemap-first ---
    try:
        for u in sitemap_urls(start, session, settings.page_timeout):
            n = normalize_url(u)
            if not n or n in seen or urlparse(n).netloc != host or is_trap(n):
                continue
            if not robots.can_fetch(settings.user_agent, n):
                continue
            seen.add(n)
            queue.append((n, 1))
    except Exception:
        pass

    if start not in seen:
        seen.add(start)
        queue.appendleft((start, 0))

    log(f"  queue: {len(queue)} urls discovered (sitemap + seed)")

    # --- phase B: bounded BFS ---
    while queue and len(result.pages) < settings.max_pages:
        url, depth = queue.popleft()
        if depth > settings.max_depth:
            continue
        if is_trap(url):
            result.skipped_trap += 1
            continue
        if not robots.can_fetch(settings.user_agent, url):
            result.skipped_blocked += 1
            continue

        try:
            resp = session.get(url, timeout=settings.page_timeout, allow_redirects=True)
        except Exception:
            result.errors += 1
            continue

        time.sleep(settings.politeness_delay)  # politeness, same host only

        if resp.status_code != 200:
            result.skipped_blocked += 1
            continue
        ctype = resp.headers.get("Content-Type", "")
        if "html" not in ctype and ctype:
            continue
        # Fix mojibake ("Â£51.77"): if the server didn't declare a charset,
        # detect the real one instead of assuming latin-1.
        if "charset" not in ctype.lower():
            detected = resp.apparent_encoding
            if detected:
                resp.encoding = detected

        final_url = normalize_url(resp.url) or url
        title, text = extract_content(resp.text, final_url)

        if len(text) < 100:
            result.skipped_thin += 1
        else:
            h = content_hash(text)
            if h in visited_hashes:
                result.skipped_dup += 1
            else:
                visited_hashes.add(h)
                result.pages.append(Page(final_url, title, text, h))
                if len(result.pages) % 10 == 0 or len(result.pages) == 1:
                    log(f"  [{len(result.pages)}/{settings.max_pages}] {final_url[:90]}")

        # discover more (only when depth budget remains)
        if depth < settings.max_depth:
            try:
                for link in collect_links(resp.text, resp.url):
                    n = normalize_url(link)
                    if (
                        n
                        and n not in seen
                        and urlparse(n).netloc == host
                        and not is_trap(n)
                        and robots.can_fetch(settings.user_agent, n)
                    ):
                        seen.add(n)
                        queue.append((n, depth + 1))
            except Exception:
                result.errors += 1

    log(
        f"  done: {len(result.pages)} pages | skipped: "
        f"{result.skipped_blocked} blocked, {result.skipped_thin} thin, "
        f"{result.skipped_dup} dup, {result.skipped_trap} traps, {result.errors} errors"
    )
    return result

"""Responsible crawler (Phase 1).

Built-in safety from the risk register:
- sitemap-first discovery
- hard page cap + depth limit
- robots.txt respected (honest User-Agent)
- URL normalization + content-hash dedupe (loop killers)
- trap detection (pagination / calendars / session ids)
- per-page timeout, one job never crashes the system
- SSRF guard (fetch_verdict / GuardedSession): when the server crawls a
  user-pasted URL, only publicly routable addresses are ever fetched —
  including every hop of a redirect chain
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
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


# ---------------------------------------------------------------- safety ----
def fetch_verdict(url: str) -> tuple[bool, str]:
    """SSRF guard: may OUR SERVER safely fetch this URL?

    Returns (True, "") only for http(s) URLs whose host resolves EXCLUSIVELY
    to publicly routable addresses. Blocks localhost, private ranges
    (10/8, 172.16/12, 192.168/16), link-local (cloud metadata 169.254.x),
    CGNAT, and anything else `ipaddress` considers non-global — the classic
    SSRF targets. Used both as a pre-flight check (/ingest) and on every
    request a GuardedSession sends (each redirect hop re-checked)."""
    try:
        parts = urlparse(url)
    except Exception:
        return False, "that doesn't look like a valid web address"
    if parts.scheme not in ("http", "https"):
        return False, "only http:// and https:// links are supported"
    host = parts.hostname
    if not host:
        return False, "that doesn't look like a valid web address"
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return False, "that address has an invalid port"
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, OSError):
        return False, "couldn't find that website — check the spelling"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False, "that address doesn't look valid"
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped  # ::ffff:127.0.0.1 -> check the IPv4 truth
        if not ip.is_global:
            return False, "that link points to a private/internal address — not allowed"
    return True, ""


class GuardedSession(requests.Session):
    """Session that re-runs fetch_verdict on EVERY request it sends.

    requests calls Session.send() for the initial request AND for each
    redirect hop, so one override covers the whole chain: a public site
    that 302s to 169.254.169.254 (cloud metadata) or localhost never gets
    fetched. Redirects are also capped (default 5)."""

    def __init__(self, max_redirects: int = 5):
        super().__init__()
        self.max_redirects = max_redirects

    def send(self, request, **kwargs):
        ok, why = fetch_verdict(request.url)
        if not ok:
            raise requests.RequestException(f"blocked by SSRF guard: {why}")
        return super().send(request, **kwargs)


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


def _json_strings(node, depth: int = 0) -> list[str]:
    """All string values in a JSON-LD block (bounded depth)."""
    if depth > 4:
        return []
    out: list[str] = []
    if isinstance(node, dict):
        for v in node.values():
            out.extend(_json_strings(v, depth + 1))
    elif isinstance(node, list):
        for v in node:
            out.extend(_json_strings(v, depth + 1))
    elif isinstance(node, str):
        out.append(node)
    return out


def extract_stub(html: str) -> str:
    """Server-provided summary for pages whose body text never rendered.

    JS-heavy stores (Shopify SPAs etc.) ship a shell of HTML with the real
    content behind JavaScript — extract_content sees zero text and the page
    would be dropped as "thin", which is exactly how a freshly crawled
    palmonas.com ended up knowing only its 8 policy pages. The <head>
    metadata (meta description, og:description, JSON-LD blocks) IS
    server-rendered, and so is the site's own link text (product tiles,
    collection cards) — we index all of it instead of dropping the page.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return ""
    parts: list[str] = []

    def add(value) -> None:
        if not value:
            return
        v = " ".join(str(value).split())
        if 20 <= len(v) <= 400 and v not in parts:
            parts.append(v)

    md = soup.find("meta", attrs={"name": "description"})
    add(md.get("content") if md else None)
    og = soup.find("meta", attrs={"property": "og:description"})
    add(og.get("content") if og else None)
    for blk in soup.find_all("script", attrs={"type": "application/ld+json"}):
        if len(parts) >= 6:
            break
        try:
            data = json.loads(blk.string or "")
        except Exception:
            continue
        for s in _json_strings(data):
            if len(parts) >= 6:
                break
            if s.startswith(("http://", "https://")):
                continue
            add(s)
    head = " | ".join(parts)[:600]

    # Anchor text: on a JS shell the site's own links (product tiles,
    # collection cards) are often the only server-rendered catalog — the
    # homepage of a store lists every product name here even though its
    # body text renders at zero characters. Image tiles carry no text, so
    # fall back to the URL slug, which IS the label (/products/mini-
    # solitaire-ring-13689 -> "mini solitaire ring"). Without this, a
    # 20-page crawl learns three products and honestly answers "I don't
    # know" to "do you have rings?".
    anchors: list[str] = []
    for a in soup.find_all("a", href=True):
        t = " ".join(a.get_text(" ", strip=True).split())
        if len(t) < 3:
            seg = urlparse(a["href"]).path.rstrip("/").split("/")[-1]
            t = re.sub(r"\s+", " ", re.sub(r"[-_]+", " ", seg)).strip()
            t = re.sub(r"\s+\d{4,}$", "", t)  # trailing Shopify product id
        if 3 <= len(t) <= 80 and re.search(r"[a-z]", t, re.I) and t not in anchors:
            anchors.append(t)
        if len(anchors) >= 60:
            break
    links = f"Links: {' · '.join(anchors)}"[:800] if anchors else ""

    return " | ".join(x for x in (head, links) if x)


def collect_links(html: str, base_url: str) -> list[tuple[str, bool]]:
    """(absolute_url, in_chrome) for every crawlable link, in DOM order.

    in_chrome marks site furniture — header/nav/footer/aside links that
    appear on every page of the site (global marketing nav, legal footer).
    The crawler ranks content links ahead of chrome; see link_tier.
    """
    soup = BeautifulSoup(html, "lxml")
    out: list[tuple[str, bool]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        absolute = urljoin(base_url, href)
        if _BAD_EXT.search(urlparse(absolute).path):
            continue
        if a.find_parent(["nav", "footer", "aside"]):
            in_chrome = True
        elif a.find_parent("header") and not a.find_parent(["main", "article"]):
            in_chrome = True
        else:
            in_chrome = False
        out.append((absolute, in_chrome))
    return out


# Legal/policy/boilerplate paths (matched on the URL path only).
_LOW_VALUE = re.compile(
    r"/(privacy|policy|policies|terms|shipping|delivery|returns?|refunds?|"
    r"warranty|gift-card|grievance|cookie|disclaimer|legal|notice)([-/]|$)",
    re.I,
)


def is_low_value(url: str) -> bool:
    """Legal/policy pages answer almost no buyer question, but they are
    numerous on store sites — demote them to the LAST tier so a time-limited
    crawl spends its budget on products first. Demoted, not excluded: a small
    site still gets them crawled once everything else is done.
    """
    return bool(_LOW_VALUE.search(urlparse(url).path))


def link_tier(url: str, seed_root: str, in_chrome: bool) -> int:
    """Frontier rank — lower tiers crawl first; the queue only falls to a
    higher tier when every cheaper tier is empty.

      0 = on-subject: shares the seed's first path segment (profile tabs,
          section menus, pages under the seeded area) — even when in the nav
      1 = content link elsewhere on the same site
      2 = site chrome (global header/footer links)

    Why this exists: a plain FIFO queue followed links in DOM order, and
    every site's <header> marketing nav comes before its <main>. A crawl of
    github.com/someone spent 18 of its 20 pages on GitHub's marketing pages
    and never reached the person's repositories — so answers were built from
    a polluted page pool. (tested: tests/test_crawler.py)
    """
    path = urlparse(url).path
    if seed_root and seed_root != "/" and (
        path == seed_root or path.startswith(seed_root + "/")
    ):
        return 0
    if in_chrome or is_low_value(url):
        return 2
    return 1


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


def crawl(
    start_url: str,
    settings: Settings,
    log=print,
    safe: bool = False,
    on_fetch=None,
    max_seconds: float | None = None,
    max_fetches: int | None = None,
) -> CrawlResult:
    """Crawl start_url. `safe=True` (user-pasted URLs, Phase 6) swaps in a
    GuardedSession — SSRF-checked on every request/redirect hop. `on_fetch`
    (if given) is called after each attempt as on_fetch(url, status|None)
    to drive the live crawl feed.

    `max_seconds` (wall clock) and `max_fetches` (attempt budget) bound the
    crawl however the caller needs — Phase 6 ingest sets both so a pasted
    link finishes in well under a minute: whatever was collected by the
    deadline gets indexed instead of the frontier draining for minutes.
    """
    start = normalize_url(start_url)
    if not start:
        raise ValueError(f"Not a valid http(s) URL: {start_url}")

    t0 = time.monotonic()
    fetches = 0
    host = urlparse(start).netloc
    session: requests.Session = GuardedSession() if safe else requests.Session()
    session.headers.update({
        "User-Agent": settings.user_agent,
        "Accept": "text/html,application/xhtml+xml",
    })
    robots = load_robots(start, session, settings.page_timeout)

    result = CrawlResult()
    seen: set[str] = set()
    visited_hashes: set[str] = set()

    # Three-tier frontier (see link_tier): on-subject links first, then
    # other content, then site chrome. Plain FIFO put the global header nav
    # ahead of the page's own content, so small page budgets were spent on
    # marketing pages instead of the site's actual subject.
    tiers: list[deque[tuple[str, int]]] = [deque(), deque(), deque()]
    seed_path = urlparse(start).path.strip("/")
    seed_root = "/" + seed_path.split("/")[0] if seed_path else "/"

    # --- phase A: sitemap-first ---
    try:
        for u in sitemap_urls(start, session, settings.page_timeout, settings.sitemap_limit):
            n = normalize_url(u)
            if not n or n in seen or urlparse(n).netloc != host or is_trap(n):
                continue
            if not robots.can_fetch(settings.user_agent, n):
                continue
            seen.add(n)
            # sitemap entries bypass link_tier, so apply the same demotion:
            # policy pages crawl last, content (products) first.
            tiers[2 if is_low_value(n) else 1].append((n, 1))
    except Exception:
        pass

    if start not in seen:
        seen.add(start)
        tiers[0].append((start, 0))

    log(f"  queue: {sum(len(t) for t in tiers)} urls discovered (sitemap + seed)")

    # --- phase B: bounded BFS, cheapest tier first ---
    while any(tiers) and len(result.pages) < settings.max_pages:
        if max_fetches is not None and fetches >= max_fetches:
            log(f"  fetch budget reached ({max_fetches} attempts) — indexing what we have")
            break
        if max_seconds is not None and time.monotonic() - t0 >= max_seconds:
            log(f"  time budget reached ({max_seconds:g}s) — indexing what we have")
            break
        url, depth = next(t.popleft() for t in tiers if t)
        if depth > settings.max_depth:
            continue
        if is_trap(url):
            result.skipped_trap += 1
            continue
        if not robots.can_fetch(settings.user_agent, url):
            result.skipped_blocked += 1
            continue

        fetches += 1
        try:
            resp = session.get(url, timeout=settings.page_timeout, allow_redirects=True)
        except Exception:
            result.errors += 1
            if on_fetch:
                on_fetch(url, None)
            continue

        if on_fetch:
            on_fetch(url, resp.status_code)

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

        def _accept(page_text: str) -> None:
            h = content_hash(page_text)
            if h in visited_hashes:
                result.skipped_dup += 1
                return
            visited_hashes.add(h)
            result.pages.append(Page(final_url, title, page_text, h))
            if len(result.pages) % 10 == 0 or len(result.pages) == 1:
                log(f"  [{len(result.pages)}/{settings.max_pages}] {final_url[:90]}")

        if len(text) >= 100:
            _accept(text)
        else:
            # JS-heavy page: body text never rendered server-side. Index the
            # server-provided summary (meta/og/JSON-LD) instead of dropping
            # the page — this is how a pasted Shopify store stays answerable.
            stub = extract_stub(resp.text)
            if stub:
                _accept(f"{title}: {stub}" if title else stub)
            elif url == start and title:
                # The seed always earns a line: the site's own name/descrip-
                # tion is the minimum a visitor must learn from any crawl.
                _accept(title)
            else:
                result.skipped_thin += 1

        # discover more (only when depth budget remains)
        if depth < settings.max_depth:
            try:
                for link, in_chrome in collect_links(resp.text, resp.url):
                    n = normalize_url(link)
                    if (
                        n
                        and n not in seen
                        and urlparse(n).netloc == host
                        and not is_trap(n)
                        and robots.can_fetch(settings.user_agent, n)
                    ):
                        seen.add(n)
                        tiers[link_tier(n, seed_root, in_chrome)].append(
                            (n, depth + 1)
                        )
            except Exception:
                result.errors += 1

    log(
        f"  done: {len(result.pages)} pages | skipped: "
        f"{result.skipped_blocked} blocked, {result.skipped_thin} thin, "
        f"{result.skipped_dup} dup, {result.skipped_trap} traps, {result.errors} errors"
    )
    return result

"""Crawler prioritization suite — offline (fake session, zero network, zero cost).

Covers the fix for a real bug reported on a live crawl of
github.com/Vaidehi1645: the frontier was plain FIFO, links are collected in
DOM order, and every site's <header> marketing nav comes before its <main>.
18 of 20 pages fetched were GitHub marketing pages — the owner's repos were
never crawled, answers were built from a polluted pool, and citations
pointed at irrelevant pages.

Run:  venv\\Scripts\\python.exe tests\\test_crawler.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from easylink import crawler
from easylink.config import Settings
from easylink.crawler import collect_links, crawl, link_tier, normalize_url

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# --------------------------------------------------------------------------- #
# Fixtures — a miniature GitHub-style profile site
# --------------------------------------------------------------------------- #
BASE = "https://ex.test"

# Header marketing nav (DOM order: comes FIRST), then main content:
# profile tab nav (a <nav> — chrome by tag, rescued by path affinity),
# pinned repo links, then the legal footer.
PROFILE_HTML = f"""
<html><body>
<header><nav>
  <a href="/features">Features</a>
  <a href="/pricing">Pricing</a>
  <a href="/solutions">Solutions</a>
</nav></header>
<main>
  <nav class="tabs">
    <a href="/vaidehi?tab=repositories">Repositories</a>
    <a href="/vaidehi?tab=stars">Stars</a>
  </nav>
  <a href="/vaidehi/easylink-ai">easylink-ai</a>
  <a href="/vaidehi/sias">sias</a>
  <p>Profile of Vaidehi — AI developer. Pinned projects and activity live
     on this page, repositories are one tab away.</p>
</main>
<footer>
  <a href="/site/terms">Terms</a>
  <a href="/site/privacy">Privacy</a>
</footer>
</body></html>
"""

HOME_HTML = f"""
<html><body>
<header><nav><a href="/features">Features</a></nav></header>
<main>
  <h1>Welcome</h1>
  <p>{"Home page content for the extractor. " * 8}</p>
  <a href="/about">About us</a>
</main>
</body></html>
"""


def page(title: str) -> str:
    body = f"<h1>{title}</h1><p>{(title + ' page content. ') * 12}</p>"
    return f"<html><head><title>{title}</title></head><body><main>{body}</main></body></html>"


class FakeResp:
    def __init__(self, url: str, text: str = "", status: int = 200):
        self.url = url
        self.text = text
        self.content = text.encode("utf-8")
        self.status_code = status
        self.headers = {"Content-Type": "text/html; charset=utf-8"}
        self.apparent_encoding = "utf-8"
        self.encoding = "utf-8"


class FakeSession:
    """Serves canned pages for the fake site and records fetch order."""

    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.headers: dict[str, str] = {}
        self.fetched: list[str] = []

    def get(self, url, **kwargs):
        self.fetched.append(url)
        if url.endswith(("robots.txt", "sitemap.xml")):
            return FakeResp(url, "", 404)
        if url in self.pages:
            return FakeResp(url, self.pages[url])
        return FakeResp(url, "", 404)


def run_crawl(pages: dict[str, str], start: str, max_pages: int, **kw) -> tuple[list[str], object]:
    """Crawl the fake site offline; return (content fetches in order, result)."""
    made: list[FakeSession] = []

    def factory():
        s = FakeSession(pages)
        made.append(s)
        return s

    original = crawler.requests.Session
    crawler.requests.Session = factory
    try:
        settings = Settings(
            max_pages=max_pages, max_depth=3, politeness_delay=0.0, page_timeout=5.0
        )
        result = crawl(start, settings, log=lambda *a, **k: None, **kw)
    finally:
        crawler.requests.Session = original
    order = [u for u in made[0].fetched if not u.endswith(("robots.txt", "sitemap.xml"))]
    return order, result


# --------------------------------------------------------------------------- #
print("\n===== URL + link classification =====")

check(
    "tab query survives normalize (?tab=repositories stays distinct)",
    normalize_url("https://ex.test/vaidehi?tab=repositories")
    == "https://ex.test/vaidehi?tab=repositories",
    "tab is content, not noise",
)

links = dict(collect_links(PROFILE_HTML, f"{BASE}/vaidehi"))
check(
    "header marketing links marked chrome",
    links.get(f"{BASE}/features") is True,
)
check(
    "profile tab nav marked chrome by tag (affinity rescues it in link_tier)",
    links.get(f"{BASE}/vaidehi?tab=repositories") is True,
)
check(
    "pinned repo link in main marked content",
    links.get(f"{BASE}/vaidehi/easylink-ai") is False,
)
check("footer links marked chrome", links.get(f"{BASE}/site/terms") is True)

# --------------------------------------------------------------------------- #
print("\n===== link_tier ranking =====")

check(
    "tier 0: path affinity beats chrome (nav tab under seed root)",
    link_tier(f"{BASE}/vaidehi?tab=repositories", "/vaidehi", True) == 0,
)
check(
    "tier 0: child of seed root is on-subject",
    link_tier(f"{BASE}/vaidehi/easylink-ai", "/vaidehi", False) == 0,
)
check(
    "tier 1: off-subject content link",
    link_tier(f"{BASE}/blog/post", "/vaidehi", False) == 1,
)
check(
    "tier 2: off-subject chrome link",
    link_tier(f"{BASE}/features", "/vaidehi", True) == 2,
)
check(
    "homepage seed (root='/'): content stays tier 1, chrome tier 2",
    link_tier(f"{BASE}/about", "/", False) == 1
    and link_tier(f"{BASE}/features", "/", True) == 2,
)

# --------------------------------------------------------------------------- #
print("\n===== crawl order: profile budget goes to repos, not marketing =====")

profile_pages = {
    f"{BASE}/": page("Home"),
    f"{BASE}/vaidehi": PROFILE_HTML,
    f"{BASE}/vaidehi?tab=repositories": page("Repositories"),
    f"{BASE}/vaidehi?tab=stars": page("Stars"),
    f"{BASE}/vaidehi/easylink-ai": page("easylink-ai"),
    f"{BASE}/vaidehi/sias": page("sias"),
    f"{BASE}/features": page("Features"),
    f"{BASE}/pricing": page("Pricing"),
    f"{BASE}/solutions": page("Solutions"),
    f"{BASE}/site/terms": page("Terms"),
    f"{BASE}/site/privacy": page("Privacy"),
}
# NOTE: the seed is the profile itself, matching the reported live crawl.
order, result = run_crawl(profile_pages, f"{BASE}/vaidehi", max_pages=4)

expected = [
    f"{BASE}/vaidehi",
    f"{BASE}/vaidehi?tab=repositories",
    f"{BASE}/vaidehi?tab=stars",
    f"{BASE}/vaidehi/easylink-ai",
]
check(
    "crawl starts at the seed",
    order and order[0] == f"{BASE}/vaidehi",
    f"first={order[0] if order else None}",
)
check(
    "tabs + pinned repos fill the 4-page budget, in DOM order",
    order == expected,
    f"got={[u.replace(BASE, '') for u in order]}",
)
check(
    "NO marketing/chrome page fetched while tier 0 remains "
    "(the reported bug: /features was page 10 and 20)",
    not any(u in (f"{BASE}/features", f"{BASE}/pricing") for u in order),
    "header nav untouched",
)
check(
    "pages actually captured match the budget",
    len(result.pages) == 4,
    f"pages={len(result.pages)}",
)

# --------------------------------------------------------------------------- #
print("\n===== crawl order: homepage falls back to content, then chrome =====")

home_pages = {
    f"{BASE}/": HOME_HTML,
    f"{BASE}/about": page("About"),
    f"{BASE}/features": page("Features"),
}
home_order, _ = run_crawl(home_pages, f"{BASE}/", max_pages=3)

check(
    "homepage: content link (/about) before chrome (/features)",
    home_order == [f"{BASE}/", f"{BASE}/about", f"{BASE}/features"],
    f"got={[u.replace(BASE, '') for u in home_order]}",
)
check(
    "chrome still reachable when cheaper tiers drain (no starvation)",
    f"{BASE}/features" in home_order,
)

# --------------------------------------------------------------------------- #
print("\n===== policy pages demoted to last tier =====")

check(
    "policy paths demoted to tier 2 (fetched only when budget remains)",
    link_tier(f"{BASE}/pages/privacy-policy", "/", False) == 2
    and link_tier(f"{BASE}/pages/terms-of-service", "/", False) == 2
    and link_tier(f"{BASE}/policies/refund-policy", "/", False) == 2,
)
check(
    "normal content NOT demoted (products/about stay tier 1)",
    link_tier(f"{BASE}/collections/rings", "/", False) == 1
    and link_tier(f"{BASE}/about", "/", False) == 1,
)
check(
    "is_low_value: shipping/warranty demoted, product pages not",
    crawler.is_low_value(f"{BASE}/pages/shipping-policy")
    and crawler.is_low_value(f"{BASE}/warranty")
    and not crawler.is_low_value(f"{BASE}/collections/necklaces"),
)

# --------------------------------------------------------------------------- #
print("\n===== crawl budgets: wall clock + fetch attempts =====")

order, result = run_crawl(profile_pages, f"{BASE}/vaidehi", max_pages=10, max_fetches=3)
check(
    "fetch budget caps content attempts at exactly 3",
    len(order) == 3,
    f"fetches={len(order)}",
)
check(
    "crawl still returns what it collected under budget",
    1 <= len(result.pages) <= 3,
    f"pages={len(result.pages)}",
)

order, result = run_crawl(profile_pages, f"{BASE}/vaidehi", max_pages=10, max_seconds=0.0)
check(
    "wall-clock budget of 0s stops before any content fetch",
    order == [] and result.pages == [],
    f"fetches={len(order)}",
)

# --------------------------------------------------------------------------- #
print("\n===== JS-heavy pages: stub rescue + seed guarantee =====")

THIN_PRODUCT = """
<html><head>
<title>Glow Ring — Palmonas</title>
<meta name="description" content="Handcrafted glow ring in sterling silver with a soft inner glow. Ships across India in 3-5 days.">
</head><body><div id="app"></div></body></html>
"""

THIN_EMPTY = """
<html><head><title>Boring Page</title></head><body><div id="app"></div></body></html>
"""

THIN_LINKS = """
<html><head><title>Shop</title></head><body>
<div id="app">
  <a href="/p/solitaire-ring">18k Solitaire Ring</a>
  <a href="/p/heart-ring">Heart Promise Ring</a>
</div></body></html>
"""

THIN_TILES = """
<html><head><title>Store</title></head><body>
<div id="app">
  <a href="/products/mini-solitaire-ring-13689"><img src="x.jpg"></a>
  <a href="/products/love-hug-ring-18k-gold-plated-81409"><img src="y.jpg"></a>
</div></body></html>
"""


def store_home(*links: str) -> str:
    items = "".join(f"<a href='{l}'>{l}</a>" for l in links)
    body = "Store home content for the extractor. " * 10
    return f"<html><head><title>Store</title></head><body><main><p>{body}</p>{items}</main></body></html>"


js_pages = {
    f"{BASE}/": store_home("/products/glow-ring", "/products/boring"),
    f"{BASE}/products/glow-ring": THIN_PRODUCT,
    f"{BASE}/products/boring": THIN_EMPTY,
}
_, result = run_crawl(js_pages, f"{BASE}/", max_pages=5)
by_url = {p.url: p for p in result.pages}
check(
    "thin product page rescued via its meta description",
    f"{BASE}/products/glow-ring" in by_url
    and "sterling silver" in by_url[f"{BASE}/products/glow-ring"].text,
    f"pages={len(result.pages)}",
)
check(
    "rescued page carries its title (identity for retrieval)",
    f"{BASE}/products/glow-ring" in by_url
    and "Glow Ring" in by_url[f"{BASE}/products/glow-ring"].text,
)
check(
    "thin page with no metadata still rejected (no junk indexed)",
    f"{BASE}/products/boring" not in by_url and result.skipped_thin == 1,
    f"skipped_thin={result.skipped_thin}",
)

_, result = run_crawl({f"{BASE}/shop": THIN_LINKS}, f"{BASE}/shop", max_pages=3)
check(
    "thin page with no metadata rescued via its link text (product names)",
    len(result.pages) == 1 and "Solitaire Ring" in result.pages[0].text
    and "Heart Promise Ring" in result.pages[0].text,
    f"pages={len(result.pages)}",
)

_, result = run_crawl({f"{BASE}/tiles": THIN_TILES}, f"{BASE}/tiles", max_pages=3)
check(
    "empty image tiles rescued via URL slug labels (id stripped)",
    len(result.pages) == 1
    and "mini solitaire ring" in result.pages[0].text
    and "love hug ring 18k gold plated" in result.pages[0].text
    and "13689" not in result.pages[0].text,
    result.pages[0].text[:110] if result.pages else "no pages",
)

_, result = run_crawl({f"{BASE}/": THIN_EMPTY}, f"{BASE}/", max_pages=3)
check(
    "seed with zero extractable text still earns a line (title)",
    len(result.pages) == 1 and "Boring" in result.pages[0].text,
    f"pages={len(result.pages)}",
)

stub = crawler.extract_stub(
    "<html><head>"
    "<meta property='og:description' content='Luxury rings that ship across India in 3 days.'>"
    '<script type="application/ld+json">{"@context": "https://schema.org", '
    '"name": "Palmonas", "description": "A jewelry store selling handcrafted silver pieces."}'
    "</script></head><body></body></html>"
)
check(
    "extract_stub pulls og:description + JSON-LD, skips bare URLs/names",
    "Luxury rings" in stub and "jewelry store" in stub and "schema.org" not in stub,
    stub[:90],
)

# --------------------------------------------------------------------------- #
passed = sum(1 for _, ok, _ in RESULTS if ok)
failed = len(RESULTS) - passed
print(f"\n{'=' * 60}\nCRAWLER SUITE: {passed}/{len(RESULTS)} passed"
      + (f" — {failed} FAILED" if failed else ""))
sys.exit(1 if failed else 0)

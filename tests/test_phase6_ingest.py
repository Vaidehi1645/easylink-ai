"""Phase 6 self-serve ingest suite (server must already be running).

Run:
    venv\\Scripts\\python.exe tests\\test_phase6_ingest.py

Covers the agent intake page, URL validation, SSRF blocks, crawl-quota
wiring, host-level crawl reuse, /demo/<slug> serving and slug stability.

This suite NEVER crawls: blocked requests die before the crawler, quota
tests die before the crawler, and the happy path reuses an already-crawled
host — so no embeddings are spent and no site is fetched.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Override to test a second server instance:  $env:EL_TEST_BASE="http://127.0.0.1:8002"
BASE = os.environ.get("EL_TEST_BASE", "http://127.0.0.1:8000")
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  {mark}  {name}" + (f"  [{detail}]" if detail else ""))


def req(path, data=None, method=None, base=BASE, timeout=60):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    r = urllib.request.Request(
        base + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method or ("POST" if data is not None else "GET"),
    )
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "ignore"), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "ignore"), dict(e.headers)


def parse_sse(raw: str) -> list[dict]:
    events = []
    for block in raw.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                try:
                    events.append(json.loads(line[5:].strip()))
                except ValueError:
                    pass
    return events


def wait_for_server(base=BASE) -> bool:
    for _ in range(30):
        try:
            code, _, _ = req("/health", base=base, timeout=3)
            if code == 200:
                return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> int:
    if not wait_for_server():
        print("Server not reachable at " + BASE + " — start it:  python -m api")
        return 1

    # ------------------------------------------------------- intake page
    code, raw, _ = req("/test")
    check("GET /test (no token) -> agent intake page",
          code == 200 and "Build my bot" in raw and "HOSTILE" not in raw
          and "data-site" not in raw,
          f"{code} {len(raw)} bytes")

    # ---------------------------------------------------- URL validation
    code, raw, _ = req("/ingest", {"url": ""})
    check("POST /ingest empty url -> 400", code == 400, raw[:60])

    code, raw, _ = req("/ingest", {"url": "ftp://files.example.com/x"})
    check("POST /ingest ftp:// -> 400 http(s) only",
          code == 400 and "http" in raw, raw[:70])

    code, raw, _ = req("/ingest", {"url": "example.com"})
    check("POST /ingest missing scheme -> 400", code == 400, raw[:70])

    # ------------------------------------------------------- SSRF blocks
    ssrf_targets = {
        "loopback 127.0.0.1": "http://127.0.0.1/",
        "localhost": "http://localhost:8080/",
        "cloud metadata 169.254.169.254": "http://169.254.169.254/latest/meta-data/",
        "private 10.x": "http://10.0.0.1/",
        "private 192.168.x": "http://192.168.1.10/admin",
        "IPv6 loopback [::1]": "http://[::1]/",
    }
    for label, target in ssrf_targets.items():
        code, raw, _ = req("/ingest", {"url": target})
        check(f"SSRF: {label} -> 400",
              code == 400 and ("private" in raw or "internal" in raw),
              f"{code} {raw[:60]}")

    code, raw, _ = req("/ingest", {"url": "https://no-such-site-ez9k2.invalid/"})
    check("unresolvable host -> 400 friendly",
          code == 400 and "couldn't find" in raw, f"{code} {raw[:70]}")

    # -------------------------------------------- fetch_verdict (unit)
    from easylink.crawler import fetch_verdict

    blocked = all(
        fetch_verdict(u)[0] is False
        for u in (
            "http://localhost/",
            "http://127.0.0.1/",
            "http://169.254.169.254/",
            "http://10.1.2.3/",
            "ftp://x.example.com/",
        )
    )
    check("fetch_verdict unit: private targets + bad scheme blocked", blocked)
    check("fetch_verdict unit: public example.com allowed",
          fetch_verdict("https://example.com/")[0] is True)

    # --------------------------------------------- crawl quota (unit)
    from api.security import Guard
    from easylink.config import load_settings

    old_env = os.environ.get("EASYLINK_CRAWL_DAILY")
    os.environ["EASYLINK_CRAWL_DAILY"] = "3"
    try:
        g = Guard(load_settings())
    finally:
        if old_env is None:
            os.environ.pop("EASYLINK_CRAWL_DAILY", None)
        else:
            os.environ["EASYLINK_CRAWL_DAILY"] = old_env
    verdicts = [g.check_crawl("203.0.113.7") for _ in range(4)]
    check("crawl quota unit: 3 allowed then 429",
          verdicts[:3] == [None, None, None]
          and verdicts[3] is not None and verdicts[3][0] == 429,
          str(verdicts[3]))

    # ------------------------------------ crawl quota (server wiring)
    # Fresh process with EASYLINK_CRAWL_DAILY=0: a valid, never-crawled
    # public URL must be refused BEFORE any fetching happens.
    port = 8002
    env = {**os.environ, "EASYLINK_PORT": str(port), "EASYLINK_CRAWL_DAILY": "0"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "api"],
        env=env,
        cwd=str(Path(__file__).resolve().parent.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        ok = wait_for_server(f"http://127.0.0.1:{port}")
        check("quota server starts (crawl daily = 0)", ok)
        if ok:
            code, raw, _ = req(
                "/ingest", {"url": "https://www.wikipedia.org/"},
                base=f"http://127.0.0.1:{port}", timeout=30,
            )
            check("quota server: valid fresh URL -> 429 crawl limit",
                  code == 429 and "crawl limit" in raw, f"{code} {raw[:70]}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    # --------------------------------- reuse: example.com is already crawled
    code, raw, _ = req("/ingest", {"url": "https://example.com/"}, timeout=30)
    events = parse_sse(raw) if code == 200 else []
    types = [e.get("type") for e in events]
    check("reuse: existing host -> reuse + done, no crawling",
          code == 200 and "reuse" in types and "done" in types
          and "fetch" not in types and "error" not in types,
          f"{code} {types}")

    done = next((e for e in events if e.get("type") == "done"), {})
    link = str(done.get("link", ""))
    snippet = str(done.get("snippet", ""))
    check("done: demo link + embed snippet present",
          "/demo/" in link and "/widget.js" in snippet
          and re.search(r'data-site="[0-9a-f]{16,}"', snippet) is not None,
          link[:80])
    check("reuse flagged (reused=true)", done.get("reused") is True,
          str(done.get("reused")))
    check("done carries what the bot learned + timing",
          isinstance(done.get("learned"), list) and len(done.get("learned", [])) >= 1
          and isinstance(done.get("seconds"), (int, float)),
          f"learned={done.get('learned')} seconds={done.get('seconds')}")

    # ------------------------------------------- the demo link itself
    token_m = re.search(r'data-site="([0-9a-f]+)"', snippet)
    token = token_m.group(1) if token_m else ""
    path = urllib.parse.urlparse(link).path or "/demo/"
    code, raw, _ = req(path)
    check("GET demo link -> 200 + widget token + how-it-works",
          code == 200 and token in raw and "How it works" in raw,
          f"{code} {path}")

    # ------------------------------------- slug stability on re-ingest
    code2, raw2, _ = req("/ingest", {"url": "https://example.com/"}, timeout=30)
    link2 = str(next((e for e in parse_sse(raw2)
                      if e.get("type") == "done"), {}).get("link", ""))
    check("same URL again -> identical demo link (stable slug)",
          code2 == 200 and link2 == link, f"{link2[:80]} vs {link[:80]}")

    # ------------------------------------------------- misc route guards
    code, _, _ = req("/demo/no-such-site-xyz")
    check("GET /demo/unknown -> 404", code == 404, str(code))

    code, _, _ = req("/test?token=deadbeefdeadbeefdeadbeef")
    check("GET /test?token=invalid -> 401", code == 401, str(code))

    print(f"\n{'='*60}")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name}  {detail}")
    print(f"{passed}/{len(RESULTS)} passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

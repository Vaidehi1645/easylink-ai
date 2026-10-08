"""Phase 2 API test suite.

Run (with the server already running in another terminal):
    venv\Scripts\python.exe tests\test_phase2_api.py            # core + security
    venv\Scripts\python.exe tests\test_phase2_api.py --flood    # rate-limit test (run LAST)
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from easylink.store import Database  # noqa: E402

BASE = "http://127.0.0.1:8000"
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  {mark}  {name}" + (f"  [{detail}]" if detail else ""))


def req(path, data=None, headers=None, method=None, base=BASE, timeout=120):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    r = urllib.request.Request(
        base + path,
        data=body,
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method or ("POST" if data is not None else "GET"),
    )
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "ignore"), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "ignore"), dict(e.headers)


def get_h(hdrs: dict, key: str) -> str | None:
    """Case-insensitive header lookup (HTTP headers are case-insensitive)."""
    for k, v in hdrs.items():
        if k.lower() == key.lower():
            return v
    return None


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


def wait_for_server() -> bool:
    for _ in range(30):
        try:
            code, _, _ = req("/health", timeout=3)
            if code == 200:
                return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "core"

    db = Database()
    site = db.get_site("Bookstore Test")
    if site is None:
        print("No 'Bookstore Test' site — crawl it first.")
        return 1
    token = db.get_or_create_token(site["id"])

    if not wait_for_server():
        print("Server not reachable at " + BASE + " — start it:  python -m api")
        return 1

    # ---------------------------------------------------------- rate limit
    if mode == "--flood":
        blocked = 0
        for i in range(40):
            code, raw, _ = req(
                "/chat",
                {"token": token, "question": "state the shipping policy"},
                headers={"Origin": "http://127.0.0.1:8000"},
                timeout=60,
            )
            if code == 429:
                blocked += 1
        check("rate limit: flood of 40 -> some 429s", blocked >= 10, f"{blocked} blocked")
        print(f"\n{'='*60}\n{sum(1 for _, ok, _ in RESULTS if ok)}/{len(RESULTS)} passed")
        return 0 if all(ok for _, ok, _ in RESULTS) else 1

    # ------------------------------------------------------- daily quota
    if mode == "--quota":
        import os
        import subprocess

        # usage already consumed by earlier tests -> quota = usage + 1
        usage = db.usage_today(site["id"])
        env = {**os.environ, "EASYLINK_PORT": "8001", "EASYLINK_DAILY_QUOTA": str(usage + 1)}
        proc = subprocess.Popen(
            [sys.executable, "-m", "api"],
            env=env,
            cwd=str(Path(__file__).resolve().parent.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            ok = False
            for _ in range(30):
                try:
                    c, _, _ = req("/health", base="http://127.0.0.1:8001", timeout=3)
                    ok = c == 200
                    break
                except Exception:
                    time.sleep(0.5)
            check("quota server starts", ok)
            if ok:
                code1, _, _ = req("/chat",
                                  {"token": token, "question": "Do you have fiction?"},
                                  base="http://127.0.0.1:8001", timeout=60)
                code2, raw2, _ = req("/chat",
                                     {"token": token, "question": "Do you have mystery?"},
                                     base="http://127.0.0.1:8001", timeout=60)
                check("quota: message within limit -> allowed", code1 == 200, str(code1))
                check("quota: next message -> 429 quota message",
                      code2 == 429 and "daily message" in raw2, f"{code2} {raw2[:60]}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        print(f"\n{'='*60}\n{sum(1 for _, ok, _ in RESULTS if ok)}/{len(RESULTS)} passed")
        return 0 if all(ok for _, ok, _ in RESULTS) else 1

    # ------------------------------------------------------------- health
    code, raw, _ = req("/health")
    data = json.loads(raw) if raw else {}
    check("GET /health -> 200 + status ok", code == 200 and data.get("status") == "ok",
          f"provider={data.get('provider')}")

    # ---------------------------------------------------------- widget.js
    code, raw, _ = req("/widget.js")
    check("GET /widget.js -> 200 + shadow DOM code",
          code == 200 and "attachShadow" in raw and "data-site" in raw, f"{len(raw)} bytes")

    # ---------------------------------------------------------- test page
    code, raw, _ = req(f"/test?token={token}")
    check("GET /test -> 200 + hostile CSS + token",
          code == 200 and "HOSTILE" in raw and token in raw)

    # ------------------------------------------------------- auth: bad token
    code, raw, _ = req("/chat", {"token": "deadbeef" * 4, "question": "hi"})
    check("POST /chat bad token -> 401", code == 401, raw[:60])

    # ------------------------------------------------------- auth: no question
    code, raw, _ = req("/chat", {"token": token, "question": "   "})
    check("POST /chat empty question -> 400", code == 400, raw[:60])

    # ------------------------------------------------- security: wrong origin
    code, raw, _ = req("/chat", {"token": token, "question": "hi"},
                       headers={"Origin": "https://evil.com"})
    check("foreign Origin -> 403", code == 403, raw[:70])

    # ------------------------------------------------- CORS preflight (allowed)
    code, raw, hdrs = req("/chat", method="OPTIONS",
                          headers={"Origin": "http://localhost:9999",
                                   "Access-Control-Request-Method": "POST"})
    check("preflight OPTIONS -> 204 + ACAO echo",
          code == 204 and get_h(hdrs, "Access-Control-Allow-Origin") == "http://localhost:9999",
          str(get_h(hdrs, "Access-Control-Allow-Origin")))

    # ------------------------------------------------- allowed cross-origin chat
    code, raw, hdrs = req("/chat",
                          {"token": token, "question": "Do you have poetry books?"},
                          headers={"Origin": "http://localhost:9999"})
    events = parse_sse(raw)
    deltas = [e["delta"] for e in events if "delta" in e]
    answers = [e["answer"] for e in events if "answer" in e]
    check("localhost cross-origin chat -> 200 + ACAO header",
          code == 200 and get_h(hdrs, "Access-Control-Allow-Origin") == "http://localhost:9999",
          str(get_h(hdrs, "Access-Control-Allow-Origin")))
    check("streaming: >= 2 delta events before answer",
          len(deltas) >= 2 and len(answers) == 1, f"{len(deltas)} deltas")
    ans = answers[0] if answers else {}
    check("grounded answer has >= 1 source + not abstained",
          bool(ans.get("sources")) and not ans.get("abstained"),
          f"{len(ans.get('sources', []))} sources")

    # ------------------------------------------------- abstention + gap logging
    before = len(db.list_gaps(site["id"], 999))
    code, raw, _ = req("/chat",
                       {"token": token, "question": "What is the airspeed velocity of an unladen swallow?"},
                       headers={"Origin": "http://127.0.0.1:8000"})
    events = parse_sse(raw)
    answers = [e["answer"] for e in events if "answer" in e]
    ans = answers[0] if answers else {}
    time.sleep(0.5)
    after = len(db.list_gaps(site["id"], 999))
    check("off-topic question -> honest abstention", ans.get("abstained") is True,
          str(ans.get("text", ""))[:50])
    check("abstention auto-logged as a gap", after > before, f"{before} -> {after}")

    # ------------------------------------------------- per-site isolation
    other = db.get_site("Example Test")
    if other:
        other_token = db.get_or_create_token(other["id"])
        check("tokens are per-site (different tokens)",
              other_token != token)

    print(f"\n{'='*60}")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name}  {detail}")
    print(f"{passed}/{total} passed")
    if passed == total:
        print("Run the flood test next:  tests\\test_phase2_api.py --flood")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

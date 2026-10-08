"""Phase 2 widget browser test (headless Chromium via Playwright).

Run with the API server already running:
    venv\\Scripts\\python.exe tests\\test_phase2_widget.py

Covers what curl can't: Shadow DOM isolation against hostile CSS,
real streaming render, source links, abstention, and a cross-origin
file:// embed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from easylink.store import Database  # noqa: E402

BASE = "http://127.0.0.1:8000"
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def run() -> int:
    from playwright.sync_api import sync_playwright

    db = Database()
    site = db.get_site("Bookstore Test")
    token = db.get_or_create_token(site["id"])

    shot_dir = Path(__file__).parent / "screenshots"
    shot_dir.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 800})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        # --------------------------------------------- same-origin test page
        page.goto(f"{BASE}/test", wait_until="networkidle")
        bubble = page.locator(".bubble")
        check("widget bubble mounts", bubble.count() == 1)

        styles = bubble.evaluate(
            """el => { const s = getComputedStyle(el);
                  return {radius: s.borderRadius, bg: s.backgroundColor}; }"""
        )
        check("bubble survives hostile CSS (indigo, round)",
              styles["bg"] == "rgb(79, 70, 229)" and styles["radius"] == "50%",
              f"{styles['bg']} {styles['radius']}")

        page.screenshot(path=str(shot_dir / "01-widget-closed.png"))

        # --------------------------------------------- open panel + stream
        bubble.click()
        panel = page.locator(".panel")
        check("panel opens", panel.is_visible())

        pstyles = panel.evaluate(
            """el => { const s = getComputedStyle(el);
                  return {radius: s.borderRadius, bg: s.backgroundColor,
                          font: s.fontFamily}; }"""
        )
        check("panel isolated (white bg, rounded, own font)",
              pstyles["bg"] == "rgb(255, 255, 255)"
              and pstyles["radius"] != "0px"
              and "system-ui" in pstyles["font"],
              f"{pstyles['radius']} / {pstyles['font'][:30]}")

        inp = page.locator("input")
        inp.fill("What books are available on this site?")
        page.locator("form button.send").click()

        # sample the streaming bubble growth
        lengths: set[int] = set()
        sources_visible = False
        deadline = time.time() + 45
        while time.time() < deadline:
            src = page.locator(".msgs .src")
            if src.count() and src.first.is_visible():
                sources_visible = True
                lengths.add(-1)
                break
            answer = page.locator(".msgs .m.b").last
            if answer.count():
                try:
                    t = answer.inner_text(timeout=500)
                    if t.strip():
                        lengths.add(len(t))
                except Exception:
                    pass
            time.sleep(0.15)

        check("answer streams in (grows over time)", len(lengths) >= 3,
              f"{len(lengths)} growth snapshots")
        check("source links render", sources_visible)

        final = page.locator(".msgs .m.b").last.inner_text()
        check("answer mentions books/£ (grounded)", "£" in final or "book" in final.lower(),
              final[:60].replace("\n", " "))
        page.screenshot(path=str(shot_dir / "02-widget-answered.png"))

        # --------------------------------------------- honest abstention
        inp.fill("What is the airspeed velocity of an unladen swallow?")
        page.locator("form button.send").click()
        try:
            page.locator(".msgs .m.b", has_text="information on this website").last.wait_for(
                timeout=45000
            )
            check("widget abstains honestly", True)
        except Exception:
            last = page.locator(".msgs .m.b").last.inner_text()
            check("widget abstains honestly", False, last[:70])

        page.screenshot(path=str(shot_dir / "03-widget-abstain.png"))

        # --------------------------------------------- case preserved (no uppercase)
        typed = page.locator(".msgs .m.u").first.inner_text()
        check("hostile text-transform did NOT leak in", typed[:1] == typed[:1].upper() and "books" in typed.lower() and "BOOKS" not in typed,
              typed[:40])

        check("no JS console errors", not errors, "; ".join(errors)[:120])

        # --------------------------------------------- cross-origin file:// embed
        page2 = browser.new_page()
        ferrors: list[str] = []
        page2.on("pageerror", lambda e: ferrors.append(str(e)))
        tmp = Path(__file__).parent / "_cross_origin_test.html"
        tmp.write_text(
            f"""<!doctype html><html><body>
            <h1>Some other website entirely</h1>
            <p>Origin here is file:// (opaque "null" origin) — real cross-origin embed.</p>
            <script src="{BASE}/widget.js" data-site="{token}"></script>
            </body></html>""",
            encoding="utf-8",
        )
        page2.goto(tmp.as_uri())
        page2.locator(".bubble").wait_for(timeout=10000)
        check("cross-origin file:// embed mounts", page2.locator(".bubble").count() == 1)
        page2.locator(".bubble").click()
        page2.locator("input").fill("Do you have poetry?")
        page2.locator("form button.send").click()
        try:
            page2.locator(".msgs .src").first.wait_for(timeout=45000)
            check("cross-origin chat streams a sourced answer", True)
        except Exception:
            err = page2.locator(".msgs").inner_text()
            check("cross-origin chat streams a sourced answer", False, err[:80])
        check("no JS errors on cross-origin page", not ferrors, "; ".join(ferrors)[:120])
        tmp.unlink(missing_ok=True)

        browser.close()

    print(f"\n{'='*60}")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name}  {detail}")
    print(f"{passed}/{len(RESULTS)} passed")
    print(f"screenshots: {shot_dir}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(run())

"""Phase 2 widget browser test (headless Chromium via Playwright).

Run with the API server already running:
    venv\\Scripts\\python.exe tests\\test_phase2_widget.py

Covers what curl can't: Shadow DOM isolation (widget styles always win),
real streaming render, source links, abstention, and a cross-origin
file:// embed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from easylink.store import Database  # noqa: E402
from suite_session import COOKIE_NAME, owner_session  # noqa: E402

# Override to test a second server instance:  $env:EL_TEST_BASE="http://127.0.0.1:8002"
BASE = os.environ.get("EL_TEST_BASE", "http://127.0.0.1:8000")
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def run() -> int:
    from playwright.sync_api import sync_playwright

    db = Database()
    site = db.get_site("Bookstore Test")
    token = db.get_or_create_token(site["id"])
    # Owner login may be configured (.env) -> /test?token= is gated; sign
    # the suite in so the widget demo opens with login ON or OFF.
    owner_token = owner_session()

    shot_dir = Path(__file__).parent / "screenshots"
    shot_dir.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 800})
        page.context.add_cookies(
            [{"name": COOKIE_NAME, "value": owner_token, "url": BASE}]
        )
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        # --------------------------------------------- same-origin test page
        # Pin the token: /test without one shows the agent intake page
        # (paste-a-link flow) — the widget demo lives behind a token.
        page.goto(f"{BASE}/test?token={token}", wait_until="networkidle")
        bubble = page.locator(".bubble")
        check("widget bubble mounts", bubble.count() == 1)

        styles = bubble.evaluate(
            """el => { const s = getComputedStyle(el);
                  return {radius: s.borderRadius, bg: s.backgroundColor}; }"""
        )
        check("bubble renders in widget's own style (indigo, round)",
              styles["bg"] == "rgb(79, 70, 229)" and styles["radius"] == "50%",
              f"{styles['bg']} {styles['radius']}")

        page.screenshot(path=str(shot_dir / "01-widget-closed.png"))

        # --------------------------------------------- open panel + stream
        bubble.click()
        panel = page.locator(".panel")
        check("panel opens", panel.is_visible())

        wdot = page.locator(".hdr .dot")
        wcls = (wdot.get_attribute("class") or "") if wdot.count() else "missing"
        check("widget status dot present + green",
              wdot.count() == 1 and "off" not in wcls, wcls)

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
        inp.fill("Tell me about the book A Light in the Attic")
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
        gtext = final.lower()
        check("answer is grounded (title/£)",
              "£" in final or any(k in gtext for k in ("book", "light", "attic", "poetry", "shakespeare")),
              final[:60].replace("\n", " "))
        page.screenshot(path=str(shot_dir / "02-widget-answered.png"))

        # --------------------------------------------- honest abstention
        inp.fill("What is the airspeed velocity of an unladen swallow?")
        page.locator("form button.send").click()
        # Two honest forms: the instant layer-1 ABSTAIN_TEXT, or a layer-2
        # model refusal — judged by the product's OWN detector (is_refusal),
        # so the test and the runtime can never disagree on what a refusal is.
        from easylink.rag import is_refusal

        deadline = time.time() + 45
        honest = False
        last_text = ""
        while time.time() < deadline:
            last_text = page.locator(".msgs .m.b").last.inner_text()
            if "information on this website" in last_text or is_refusal(last_text):
                honest = True
                break
            time.sleep(0.3)
        check("widget abstains honestly", honest, last_text[:70])

        page.screenshot(path=str(shot_dir / "03-widget-abstain.png"))

        # --------------------------------------------- case preserved (no uppercase)
        typed = page.locator(".msgs .m.u").first.inner_text()
        check("typed question keeps original case (no CSS transform)",
              typed[:1] == typed[:1].upper() and "book" in typed.lower() and "BOOK" not in typed,
              typed[:40])

        # --------------------------------------------- owner-verified fact badge
        # Bookstore Test carries the owner fact "shipping ... 1200 rupees";
        # facts have no URL, so the answer must arrive with the badge.
        page.locator("form button.send:not([disabled])").wait_for(timeout=45000)
        inp.fill("What are the shipping options and costs?")
        page.locator("form button.send").click()
        try:
            page.locator(".msgs .ver").first.wait_for(timeout=45000)
            check("owner-fact answer shows the verified badge", True)
            page.screenshot(path=str(shot_dir / "07-owner-verified-badge.png"))
        except Exception:
            check("owner-fact answer shows the verified badge", False,
                  page.locator(".msgs").inner_text()[-140:])

        # --------------------------------------------- "what this bot learned"
        # the demo page lists the crawled titles: 8 + "+N more" toggle,
        # and the header logo links to the maker's GitHub.
        learned_ok = toggle_ok = logo_ok = False
        ldetail = ""
        try:
            total = page.evaluate("window.LEARNED_TITLES.length")
            page.locator("#learnedcard:not([hidden])").wait_for(timeout=5000)
            first = page.locator("#learned li:not(.more)").count()
            has_more = page.locator("#learned li.more").count() == 1
            learned_ok = first == min(8, total) and has_more == (total > 8)
            ldetail = f"total={total} first={first} more={has_more}"
            page.locator("#learned li.more").click()
            page.wait_for_timeout(150)
            expanded = page.locator("#learned li:not(.more)").count()
            less_txt = page.locator("#learned li.more").inner_text()
            toggle_ok = expanded == total and "show less" in less_txt
            ldetail += f" -> expanded={expanded} {less_txt!r}"
            page.screenshot(path=str(shot_dir / "08-demo-learned.png"))
            logo_href = page.locator("header .logo").get_attribute("href") or ""
            logo_ok = "github.com/Vaidehi1645" in logo_href
            ldetail += f" logo={logo_href!r}"
        except Exception as e:
            ldetail = str(e)[:140]
        check("demo learned card: 8 titles + '+N more' toggle", learned_ok, ldetail)
        check("toggle expands to every crawled page + 'show less'", toggle_ok, ldetail)
        check("demo logo links to GitHub", logo_ok, ldetail)

        # --------------------------------------------- live status: offline / online
        page.context.set_offline(True)
        page.wait_for_timeout(500)
        wcls = page.locator(".hdr .dot").get_attribute("class") or ""
        wpill = page.locator(".msgs .sys.off")
        check("offline -> red dot", "off" in wcls, wcls)
        check("offline -> 'network gone' pill in chat",
              wpill.count() >= 1 and "network gone" in wpill.last.inner_text(),
              wpill.last.inner_text() if wpill.count() else "missing")

        page.context.set_offline(False)
        green = False
        deadline = time.time() + 6
        while time.time() < deadline:
            wcls = page.locator(".hdr .dot").get_attribute("class") or ""
            if "off" not in wcls:
                green = True
                break
            page.wait_for_timeout(200)
        check("back online -> green dot", green, wcls)
        page.screenshot(path=str(shot_dir / "06-widget-offline.png"))

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
        page2.locator("input").fill("Tell me about the book A Light in the Attic")
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

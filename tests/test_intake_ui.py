"""Phase 6 intake UI test (headless Chromium via Playwright).

Covers what the API suites can't — the two client-side behaviors on the
build page (/test):
  1. "What your bot learned": 8 titles shown, "+N more pages" expands,
     "show less" collapses (toggle).
  2. Live agent status: green pulsing dot while reachable, red
     "network gone" + disabled input while offline, green again after.

Run with the API server already running:
    venv\\Scripts\\python.exe tests\\test_intake_ui.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = os.environ.get("EL_TEST_BASE", "http://127.0.0.1:8000")
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def run() -> int:
    from playwright.sync_api import sync_playwright

    shot_dir = Path(__file__).parent / "screenshots"
    shot_dir.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1200, "height": 800})
        # Owner login may be configured (.env): the first-visit sign-in modal
        # must not cover the crawl controls this suite clicks through.
        context.add_init_script(
            "try { localStorage.setItem('el_login_dismissed', '1'); } catch (e) {}"
        )
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        page.goto(f"{BASE}/test", wait_until="networkidle")

        # --------------------------------------------- status starts green
        dot = page.locator("#dot")
        check("status dot present", dot.count() == 1)
        cls = dot.get_attribute("class") or ""
        txt = page.locator("#statustext").inner_text()
        check("starts green + online", "off" not in cls and "online" in txt,
              f"class={cls!r} text={txt!r}")

        # --------------------------------------------- logo links to GitHub
        href = page.locator(".logo").get_attribute("href") or ""
        check("intake logo links to GitHub", "github.com/Vaidehi1645" in href,
              href or "no href")
        check("intake header shows the robot logo artwork",
              page.locator(".logo img").count() == 1)

        # --------------------------------------------- build (palmonas reuse: instant, free)
        page.fill("#url", "https://palmonas.com/")
        page.click("#go")
        page.locator("#result").wait_for(timeout=30000)
        page.wait_for_selector("#learned li", timeout=10000)

        first_count = page.locator("#learned li:not(.more)").count()
        more = page.locator("#learned li.more")
        check("learned list shows first 8", first_count == 8, f"{first_count} items")
        check("'+N more pages' line present", more.count() == 1,
              more.inner_text() if more.count() else "missing")

        # --------------------------------------------- expand / collapse toggle
        more.click()
        page.wait_for_timeout(150)
        expanded = page.locator("#learned li:not(.more)").count()
        less_txt = page.locator("#learned li.more").inner_text()
        check("click expands to full list", expanded > 8,
              f"{expanded} items, toggle={less_txt!r}")
        check("toggle flips to 'show less'", "show less" in less_txt, less_txt)
        page.screenshot(path=str(shot_dir / "04-intake-learned-expanded.png"))

        page.locator("#learned li.more").click()
        page.wait_for_timeout(150)
        collapsed = page.locator("#learned li:not(.more)").count()
        back_txt = page.locator("#learned li.more").inner_text()
        check("click again collapses back to 8", collapsed == 8,
              f"{collapsed} items, toggle={back_txt!r}")
        check("toggle flips back to '+N more'", "more pages" in back_txt, back_txt)

        # --------------------------------------------- offline -> red + message
        context.set_offline(True)
        page.wait_for_timeout(500)
        cls = dot.get_attribute("class") or ""
        txt = page.locator("#statustext").inner_text()
        check("offline: red dot + 'network gone'",
              "off" in cls and "network gone" in txt,
              f"class={cls!r} text={txt!r}")
        check("offline: input disabled", page.locator("#url").is_disabled())
        page.screenshot(path=str(shot_dir / "05-intake-offline.png"))

        # --------------------------------------------- back online -> green
        context.set_offline(False)
        deadline = time.time() + 6
        green = False
        while time.time() < deadline:
            cls = dot.get_attribute("class") or ""
            if "off" not in cls and page.locator("#url").is_enabled():
                green = True
                break
            page.wait_for_timeout(200)
        txt = page.locator("#statustext").inner_text()
        check("back online: green dot + input re-enabled", green,
              f"class={cls!r} text={txt!r}")

        check("no JS console errors", not errors, "; ".join(errors)[:160])
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

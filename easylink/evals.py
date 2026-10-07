"""Eval harness — trust but verify (layer 7).

Two modes:
- auto (no LLM cost): takes sentences from indexed chunks as pseudo-questions
  and checks retrieval brings back their source page. Measures index health.
- golden file: JSONL of {"question", "expect_url"?} — runs the full answer
  pipeline and reports retrieval hit rate + abstain rate.
"""

from __future__ import annotations

import json
from pathlib import Path

from easylink.providers import _content_words
from easylink.rag import RAGEngine
from easylink.store import Database


def _fact_found(question: str, expect_url: str, hits: list[dict]) -> bool:
    """A retrieval counts as correct if it surfaced the expected page OR
    another page containing the same fact (listings repeat product info —
    retrieving either is a correct answer for a real visitor).

    expect_url is matched as a SUBSTRING of the URL (as documented in the
    golden file): 'example.com' matches 'https://example.com/'."""
    if not expect_url:
        return False
    if any(expect_url in h["url"] for h in hits):
        return True
    q_words = _content_words(question)
    if not q_words:
        return False
    for h in hits:
        overlap = len(q_words & _content_words(h.get("text", ""))) / len(q_words)
        if overlap >= 0.6:
            return True
    return False


def _auto_questions(db: Database, site_id: str, limit: int = 15) -> list[dict]:
    rows = db.conn.execute(
        "SELECT page_url, text FROM chunks WHERE site_id=? ORDER BY RANDOM() LIMIT 40",
        (site_id,),
    ).fetchall()
    out: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        lines = [ln.strip() for ln in r["text"].split("\n") if ln.strip()]
        # Prefer a line that reads like a standalone fact (mid-length),
        # checking the end of the chunk first — that's where descriptions live.
        cand = next((ln for ln in reversed(lines) if 30 <= len(ln) <= 200), None)
        if cand is None:
            cand = next((ln for ln in lines if 30 <= len(ln) <= 200), None)
        if not cand or cand in seen:
            continue
        seen.add(cand)
        out.append({"question": cand, "expect_url": r["page_url"]})
        if len(out) >= limit:
            break
    return out


def run_eval(
    engine: RAGEngine,
    db: Database,
    site_id: str,
    golden_file: str | None = None,
    use_llm: bool = False,
    log=print,
) -> dict:
    if golden_file:
        path = Path(golden_file)
        if not path.exists():
            raise FileNotFoundError(f"Golden file not found: {golden_file}")
        cases = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        mode = f"golden ({path.name})"
    else:
        cases = _auto_questions(db, site_id)
        mode = "auto (retrieval self-test)"

    if not cases:
        return {"error": "No eval cases available."}

    log(f"Running eval: {mode} — {len(cases)} questions\n")

    retrieved_ok = answered = abstained = with_expect = 0
    rows_out = []
    for case in cases:
        q = case["question"]
        expect = case.get("expect_url", "")

        try:
            if use_llm:
                ans = engine.answer(q, site_id)
                hit = _fact_found(q, expect, ans.hits)
                abstained += int(ans.abstained)
                if not ans.abstained:
                    answered += 1
                shown = ans.text[:80]
            else:
                hits, ok = engine.retrieve(q, site_id)
                hit = _fact_found(q, expect, hits)
                answered += int(ok)
                abstained += int(not ok)
                shown = (hits[0]["text"][:60] + "...") if hits else "(no match)"
        except Exception as exc:
            # Rate limit / network failure: report what we have, don't crash
            log(f"\n  ! stopped early: {exc}")
            log("  (likely rate-limited — wait a minute and re-run)")
            break

        retrieved_ok += int(hit)
        with_expect += int(bool(expect))
        mark = "✓" if hit else ("✗" if expect else "·")  # · = no expectation set
        rows_out.append(f"  {mark} {q[:70]}")
        log(f"  {mark} {q[:70]}")
        log(f"      -> {shown}")

    total = max(len(rows_out), 1)  # processed count (may stop early on errors)
    denom = with_expect or total   # hit rate only over cases that had an expectation
    log("")
    log("── Summary ─────────────────────────────")
    log(f"  retrieval hit rate : {retrieved_ok}/{denom} ({100 * retrieved_ok // denom}%)")
    log(f"  above threshold    : {answered}/{total} ({100 * answered // total}%)")
    log(f"  abstained          : {abstained}/{total}")

    if not golden_file:
        db.add_eval_run(site_id, total, retrieved_ok, answered, abstained)

    return {
        "mode": mode,
        "total": total,
        "retrieved_ok": retrieved_ok,
        "answered": answered,
        "abstained": abstained,
    }

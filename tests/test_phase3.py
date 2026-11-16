"""Phase 3 test suite — the Gap Engine.

Part 1: offline (temp DB + deterministic fake provider, zero API cost)
Part 2: live end-to-end with the configured provider (openai by default)

Run:  venv\\Scripts\\python.exe tests\\test_phase3.py
"""

from __future__ import annotations

import json
import math
import re
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# --------------------------------------------------------------------------- #
# Deterministic fake provider: bag-of-words embeddings + canned JSON answers.
# --------------------------------------------------------------------------- #
class FakeProvider:
    name = "fake"
    default_threshold = 0.15
    default_chat_model = "fake"
    default_embed_model = "fake"

    def __init__(self):
        self.complete_calls = 0

    def _vec(self, text: str) -> list[float]:
        # Faithful to real embeddings: punctuation-insensitive. Strip non-word
        # chars first, else "costs?" and "costs" would hash differently.
        # crc32, not hash(): PYTHONHASHSEED would reshuffle buckets every run
        # and move cosines — every threshold assertion must be reproducible.
        text = re.sub(r"[^\w\s]", " ", text.lower())
        v = [0.0] * 64
        for w in text.split():
            v[zlib.crc32(w.encode("utf-8")) % 64] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def complete(self, system: str, user: str, history=None) -> str:
        self.complete_calls += 1
        # consolidation: regroup visitor questions by meaning.
        # Fake "semantics": any question about shipping/delivery is ONE topic.
        if "satisfy BOTH" in system:
            # pending-question dedupe (stage 2): judge each pair. Fake
            # "semantics" = share a shipping word => same underlying ask.
            ship_words = ("ship", "deliver", "postage", "courier", "cost",
                          "price", "charge", "fee", "option", "policy")
            same = []
            num = 0
            lines = user.splitlines()
            i = 0
            while i < len(lines):
                m = re.match(r"(\d+)\. A: (.+)", lines[i])
                b = re.match(r"\s*B: (.+)", lines[i + 1]) if i + 1 < len(lines) else None
                if m and b:
                    num = int(m.group(1))
                    a_txt, b_txt = m.group(2).lower(), b.group(1).lower()
                    a_hit = any(w in a_txt for w in ship_words)
                    b_hit = any(w in b_txt for w in ship_words)
                    if a_hit and b_hit:
                        same.append(num)
                i += 1
            return json.dumps({"same": same})
        if "UNDERLYING MEANING" in system:
            ship_words = ("ship", "deliver", "postage", "courier", "arrival", "arrive")
            clusters: list[dict] = []
            shipping_ids: list[int] = []
            for line in user.splitlines():
                m = re.match(r"id=(\d+) \[asked x\d+\] (.+)", line.strip())
                if not m:
                    continue
                gid, text = int(m.group(1)), m.group(2).lower()
                if any(w in text for w in ship_words):
                    shipping_ids.append(gid)
                else:
                    clusters.append({"topic": f"topic {gid}", "gap_ids": [gid],
                                     "action": "draft"})
            if shipping_ids:
                clusters.insert(0, {"topic": "shipping policy",
                                    "gap_ids": shipping_ids, "action": "draft"})
            return json.dumps({"clusters": clusters})
        # refusal-mode trigger for the citation test (question text only)
        if "REFUSE" in user:
            return "I don't have that information in the context provided."
        # partial refusal: refuses one part but still names a real product
        if "PARTIAL" in user:
            return (
                "I don't know if pearl earrings are available. However, "
                "there is a gold bracelet called the Bejewelled Geometric "
                "Gold Bracelet in stock."
            )
        # answer echoes exactly one word of the hit -> filter says no,
        # the guarded top-hit fallback says yes
        if "TOPHIT" in user:
            return "This widget matters."
        # answer shares ATTRIBUTES with every page ("anti tarnish") but
        # NAMES only one product by title
        if "ATTR" in user:
            return "The stainless steel bracelet is anti tarnish and gold plated."
        if "VISITOR QUESTION" in user:
            return json.dumps(
                {"topic": "delivery policy",
                 "question": "Do you deliver, to which areas, what fee and how long?",
                 "why": "Visitors ask but the site is silent."}
            )
        if "PAGE:" in user:
            return json.dumps([{"topic": "pricing clarity",
                                "question": "What exactly does 'starting at' include?",
                                "why": "A visitor can't tell what they'd pay."}])
        return "The answer is 30 days, as stated by the owner."

    def complete_stream(self, system, user, history=None):
        yield self.complete(system, user, history)


def part1_offline() -> None:
    print("\n===== PART 1: offline (temp DB + fake provider) =====")
    from easylink.config import load_settings
    from easylink.facts import add_fact, load_facts
    from easylink.gaps import rank_gaps, record_gap, score_gap
    from easylink.questions import _parse_json, generate_questions, scan_site, show_questions
    from easylink.rag import (
        ABSTAIN_TEXT,
        RAGEngine,
        _today_text,
        build_system_prompt,
        build_user_prompt,
        clock_answer,
    )
    from easylink.store import Database, NumpyVectorStore

    tmp = Path(tempfile.gettempdir()) / "easylink_phase3_offline.db"
    if tmp.exists():
        tmp.unlink()
    db = Database(tmp)
    fake = FakeProvider()
    settings = load_settings()
    site_id = db.create_site("https://offline.example", "Offline P3")

    # --- dedupe: exact repeat
    record_gap(db, site_id, "Do you deliver?", "no_match", provider=fake)
    record_gap(db, site_id, "Do you deliver?", "no_match", provider=fake)
    record_gap(db, site_id, "do you deliver?", "no_match", provider=fake)
    rows = db.list_gaps(site_id, 50, status="open")
    check("exact repeats collapse to 1 gap", len(rows) == 1, f"{len(rows)} rows")
    check("repeat count tracked", rows and int(rows[0]["count"]) == 3,
          f"count={rows[0]['count'] if rows else '?'}")

    # --- dedupe: embedding merge — reworded twin (norms differ, cosine ~0.95)
    long_q = "order deliver price refund support shipping hours warranty stock available"
    record_gap(db, site_id, long_q, "no_match", provider=fake)
    record_gap(db, site_id, long_q + " please", "no_match", provider=fake)
    rows = db.list_gaps(site_id, 50, status="open")
    check("embedding merge: reworded twin joins existing gap",
          len(rows) == 2 and any(int(r["count"]) == 2 for r in rows),
          f"rows={len(rows)} counts={[r['count'] for r in rows]}")

    # --- semantic different stays separate
    record_gap(db, site_id, "What color is the sky in winter?", "no_match", provider=fake)
    rows = db.list_gaps(site_id, 50, status="open")
    check("unrelated question stays a separate gap", len(rows) == 3, f"{len(rows)} rows")

    # --- resolve + reopen
    gid = rows[0]["id"]
    db.resolve_gap(gid)
    check("resolve flips status", db.list_gaps(site_id, 50, status="resolved") != [])
    before = int(db.list_gaps(site_id, 50, status="resolved")[0]["count"])
    record_gap(db, site_id, "What color is the sky in winter?", "no_match", provider=fake)
    reopened = [g for g in db.list_gaps(site_id, 50, status="open") if g["id"] == gid]
    check("resolved gap REOPENS when re-asked", bool(reopened),
          "asked again -> back in the queue")
    if reopened:
        check("reopened gap count grew",
              int(reopened[0]["count"]) >= before + 1,
              f"{before} -> {reopened[0]['count']}")

    # --- ranking: frequency + intent
    for i in range(5):
        record_gap(db, site_id, f"random chat question number {i}", "no_match", provider=fake)
    ranked = rank_gaps(db.list_gaps(site_id, 100, status=None))
    top = ranked[0]
    check("ranking: repeated commercial question floats to top",
          "deliver" in top["question"].lower(),
          f"top='{top['question'][:40]}' score={score_gap(top)}")
    check("score reflects count",
          score_gap(top) >= 2.0, f"score={score_gap(top)}")

    # --- facts
    add_fact(db, site_id, "What is the return window?", "30 days, free pickup.", provider=fake)
    facts = load_facts(db, site_id)
    check("fact stored + loadable", len(facts) == 1 and "30 days" in facts[0]["answer"])

    # --- engine: no context + no fact -> instant abstain (no LLM call).
    #     Uses a FRESH site: site1 already has a fact, so layer 8 would
    #     (correctly!) consult the model there.
    site2 = db.create_site("https://no-facts.example", "NoFacts")
    engine2 = RAGEngine(fake, NumpyVectorStore(db), settings, "NoFacts",
                        "https://no-facts.example", db=db)
    ans = engine2.answer("quantum blockchain opportunities?", site2)
    check("no context + no fact -> instant abstain", ans.abstained and fake.complete_calls == 0,
          f"llm_calls={fake.complete_calls}")

    # --- engine: no context BUT fact exists -> answers from the fact
    engine = RAGEngine(fake, NumpyVectorStore(db), settings, "Offline",
                       "https://offline.example", db=db)
    ans = engine.answer("What is the return window?", site_id)
    check("facts override abstention (layer 8)",
          (not ans.abstained) and "30 days" in ans.text,
          f"'{ans.text[:50]}'")

    # --- engine streaming path uses facts too
    events = list(engine.stream_answer("What is the return window?", site_id))
    final = [e for e in events if "answer" in e][0]["answer"]
    check("stream_answer also uses facts", not final.abstained and "30 days" in final.text)

    # --- date awareness: the model must know TODAY. Live bug this fixes:
    #     asked "did I commit anything today? and what date is today?" it
    #     answered "October 8 ... which is today" (Oct 10) because nothing
    #     told it the date and it anchored on dates in the crawled pages.
    before_txt, calls_before = _today_text(), fake.complete_calls

    ans = engine2.answer("What is today's date?", site2)
    check("date question answered from the clock, zero LLM calls",
          (not ans.abstained)
          and ans.text in (f"Today is {before_txt}.", f"Today is {_today_text()}.")
          and fake.complete_calls == calls_before,
          f"'{ans.text}' llm_calls={fake.complete_calls - calls_before}")

    events = list(engine2.stream_answer("what date is today?", site2))
    final = [e for e in events if "answer" in e][0]["answer"]
    check("stream path answers date questions too (with delta)",
          events and "delta" in events[0]
          and not final.abstained
          and final.text in (f"Today is {before_txt}.", f"Today is {_today_text()}."),
          f"'{final.text}'")

    ans = engine2.answer("did i commit anything today?", site2)
    check("questions merely MENTIONING today take the normal path",
          ans.abstained and ans.text == ABSTAIN_TEXT,
          f"'{ans.text[:45]}' (not hijacked by the clock)")

    check("clock regex: date asks match, statements/mentions don't",
          clock_answer("What's today's date?") is not None
          and clock_answer("what day is it today?") is not None
          and clock_answer("Tell me the date") is not None
          and clock_answer("did i commit anything today? and what date is today?") is None
          and clock_answer("no today is 10th oct 2026. just remember it") is None,
          "anchored on asking, not mentioning")

    prompt = build_user_prompt("Q?", [], [])
    check("every model prompt carries today's date",
          f"TODAY'S DATE: {before_txt}" in prompt, before_txt)

    # --- citations relate to the ANSWER, not just the question: a bare
    #     "I don't know" must not dump unrelated links on the visitor, a
    #     partial answer still cites the product it named, and a normal
    #     answer never renders linkless (top-hit fallback).
    engine3 = RAGEngine(
        fake, NumpyVectorStore(db), settings, "RefuseCo", "https://refuse.example", db=db
    )
    fake_hit = {
        "url": "https://refuse.example/blue-widget",
        "score": 0.91,
        "text": "The Blue Widget costs 5 dollars.",
        "title": "Blue Widget",
    }
    engine3.retrieve = lambda q, s, embedding=None: ([fake_hit], True)

    ans = engine3.answer("REFUSE the visitor here", site_id)
    check("refusal matching no page cites NO links",
          ans.abstained and ans.sources == [],
          f"abstained={ans.abstained} sources={ans.sources}")

    events = list(engine3.stream_answer("REFUSE the visitor here", site_id))
    final = [e for e in events if "answer" in e][0]["answer"]
    check("stream refusal matching no page cites no links too",
          final.abstained and final.sources == [],
          f"sources={final.sources}")

    ans = engine3.answer("anything normal here", site_id)
    check("answered reply echoing no page cites nothing",
          (not ans.abstained) and ans.sources == [],
          f"abstained={ans.abstained} sources={ans.sources}")

    ans = engine3.answer("TOPHIT question here", site_id)
    check("one specific word -> guarded top-hit fallback keeps a link",
          (not ans.abstained) and ans.sources == [fake_hit["url"]],
          f"abstained={ans.abstained} sources={ans.sources}")

    events = list(engine3.stream_answer("TOPHIT question here", site_id))
    final = [e for e in events if "answer" in e][0]["answer"]
    check("stream answered reply falls back to its top hit too",
          (not final.abstained) and final.sources == [fake_hit["url"]],
          f"sources={final.sources}")

    bracelet_hit = {
        "url": "https://refuse.example/bejewelled-bracelet",
        "score": 0.88,
        "text": "The Bejewelled Geometric Gold Bracelet is a stunning piece in stock.",
        "title": "Bejewelled Geometric Gold Bracelet",
    }
    jar_hit = {
        "url": "https://refuse.example/mystery-jar",
        "score": 0.55,
        "text": "Mystery Jar worth 7999 rupees, size large.",
        "title": "Mystery Jar",
    }
    engine3.retrieve = lambda q, s, embedding=None: ([bracelet_hit, jar_hit], True)

    ans = engine3.answer("PARTIAL answer with a real product", site_id)
    check("partial answer cites the product it named, drops the rest",
          ans.abstained and ans.sources == [bracelet_hit["url"]],
          f"abstained={ans.abstained} sources={ans.sources}")

    # cross-sell carousels repeat shared attributes ("anti tarnish") on
    # many pages — 5 hits so the attribute pair stays "specific" by the
    # frequency mask; the UNIQUE-word anchor must still refuse it, while
    # the page the answer names by title is cited.
    attr_a = {
        "url": "https://refuse.example/steel-bracelet",
        "score": 0.9,
        "text": "Stainless steel bracelet, anti tarnish, gold plated.",
        "title": "Stainless Steel Bracelet",
    }
    attr_b = {
        "url": "https://refuse.example/mystery-jar",
        "score": 0.7,
        "text": "Gift jar containing anti tarnish necklace and more.",
        "title": "Jewelsmars Mystery Jar",
    }
    fillers = [
        {"url": f"https://refuse.example/f{i}", "score": 0.5,
         "text": f"Filler page number {i} about other things.",
         "title": f"Filler {i}"}
        for i in range(1, 4)
    ]
    engine3.retrieve = lambda q, s, embedding=None: ([attr_a, attr_b, *fillers], True)

    ans = engine3.answer("ATTR answer here", site_id)
    check("shared attributes never cite; the named page's title does",
          ans.sources == [attr_a["url"]],
          f"sources={ans.sources}")

    sys_prompt = build_system_prompt("RefuseCo", "https://refuse.example", [])
    check("system prompt: partial context -> answer, not refusal",
          "PARTIALLY" in sys_prompt and "instead of refusing" in sys_prompt,
          "rule 1 amended")

    # --- question drafting (gap-driven)
    drafted = generate_questions(db, fake, NumpyVectorStore(db), settings,
                                 db.get_site(site_id), top_n=3)
    check("gap-driven drafting creates questions", len(drafted) >= 1,
          f"{len(drafted)} drafted")
    check("drafted question is a question for the OWNER",
          drafted and drafted[0]["question"].endswith("?"),
          drafted[0]["question"][:60] if drafted else "")

    # --- question drafting: re-run skips gaps that already have pending drafts
    pending_before = {q["gap_id"] for q in db.list_questions(site_id, "pending")
                      if q["gap_id"]}
    ids_before = {q["id"] for q in db.list_questions(site_id, "pending")}
    generate_questions(db, fake, NumpyVectorStore(db), settings,
                       db.get_site(site_id), top_n=3)
    fresh = [q for q in db.list_questions(site_id, "pending")
             if q["id"] not in ids_before]
    collided = [q for q in fresh if q["gap_id"] in pending_before]
    check("re-run skips gaps that already have pending questions",
          not collided,
          f"{len(fresh)} new drafts, {len(collided)} collided")

    # --- proactive scan
    # needs a chunk in the index for the page-walk; store one directly
    db.add_chunks([(f"p3t-{site_id}", site_id, "https://offline.example/pricing",
                    "Pricing", 0, "Plans start at 99. All the details are here maybe.")])
    NumpyVectorStore(db).add(site_id, [f"p3t-{site_id}"], [fake._vec("pricing")],
                             ["Plans start at 99..."],
                             [{"url": "https://offline.example/pricing", "title": "Pricing"}])
    scanned = scan_site(db, fake, settings, db.get_site(site_id), max_pages=5)
    check("proactive scan drafts from site content", len(scanned) >= 1,
          f"{len(scanned)} drafted")

    # --- display
    from easylink.gaps import show_gaps

    g = show_gaps(db, site_id)
    q = show_questions(db, db.get_site(site_id))
    check("gaps display shows ranked scores", "score" in g and "open" in g)
    check("questions display shows pending + answer hint", "Pending" in q and "--id" in q)

    # --- JSON parser hardening
    ok1 = _parse_json('```json\n{"topic":"t","question":"Q?","why":"w"}\n```', False)
    ok2 = _parse_json("no json here at all", False)
    check("JSON parser: handles code fences", isinstance(ok1, dict))
    check("JSON parser: rejects garbage", ok2 is None)

    # ------------------------------------------------------------------ #
    # Consolidation (the fix for "same meaning, different words")
    # ------------------------------------------------------------------ #
    from easylink.questions import consolidate_gaps, dedupe_pending

    site3 = db.create_site("https://ship.example", "ShipCo")
    # Four DIFFERENT wordings of one shipping question. Bag-of-words keeps
    # them as separate rows (cosine < 0.90) — exactly the bug the owner saw.
    shipping_qs = [
        "Do you ship to Pune?",
        "What is the delivery charge?",
        "How long does shipping take?",
        "Which courier do you use for postage?",
    ]
    for sq in shipping_qs:
        record_gap(db, site3, sq, "no_match", provider=fake)
    record_gap(db, site3, "What color is the sky?", "no_match", provider=fake)

    open_before = db.list_gaps(site3, 50, status="open")
    check("consolidation setup: 5 separate wordings", len(open_before) == 5,
          f"{len(open_before)} rows")

    clusters = consolidate_gaps(db, fake, db.get_site(site3))
    open_after = db.list_gaps(site3, 50, status="open")
    ship_rows = [g for g in open_after
                 if any(w in g["question"].lower()
                        for w in ("ship", "deliver", "postage", "courier"))]
    check("consolidation: 4 shipping wordings merged into 1 gap",
          len(ship_rows) == 1,
          f"{len(ship_rows)} shipping rows remain of 4")
    check("consolidation: merged count sums the asks",
          ship_rows and int(ship_rows[0]["count"]) == 4,
          f"count={ship_rows[0]['count'] if ship_rows else '?'} (expect 4)")
    check("consolidation: unrelated question untouched",
          any("color is the sky" in g["question"].lower() for g in open_after))
    check("consolidation: total rows dropped 5 -> 2",
          len(open_after) == 2, f"{len(open_after)} rows")
    check("consolidation: LLM named a shipping topic",
          any(c["topic"].lower() == "shipping policy" for c in clusters),
          f"topics={[c['topic'] for c in clusters]}")

    # drafting now produces ONE question for the shipping topic, not four.
    # Fresh site: let generate_questions do the merge itself, so the single
    # draft must carry merged_from=4 (the four wordings it collapsed).
    site4 = db.create_site("https://ship2.example", "ShipCo2")
    for sq in shipping_qs:
        record_gap(db, site4, sq, "no_match", provider=fake)
    drafted3 = generate_questions(db, fake, NumpyVectorStore(db), settings,
                                  db.get_site(site4), top_n=5)
    ship_drafts = [d for d in drafted3 if d.get("merged_from") == 4]
    check("drafting: exactly one question for the merged shipping topic",
          len(ship_drafts) == 1 and len(drafted3) == 1,
          f"{len(ship_drafts)} drafts with merged_from=4 (of {len(drafted3)} total)")

    # re-run: no NEW shipping question (its gap already has a pending one)
    drafted4 = generate_questions(db, fake, NumpyVectorStore(db), settings,
                                  db.get_site(site4), top_n=5)
    check("re-run: no duplicate shipping question drafted",
          len([d for d in drafted4 if d.get("merged_from") == 4]) == 0,
          f"{len(drafted4)} new drafts on re-run")

    # --- dedupe_pending: near-duplicate PENDING questions collapse --------
    # Newest-wins: the survivor is the more recent wording, the older twin
    # is hidden as 'duplicate' (owner never loses a question, just de-noised).
    p1 = db.add_question(site3, "What are the shipping options and costs?",
                         "shipping", "older draft")
    p2 = db.add_question(site3, "What are the shipping options and costs for orders?",
                         "shipping", "newer twin")
    marked = dedupe_pending(db, fake, site3, db.get_site(site3))
    still_pending = {q["id"] for q in db.list_questions(site3, "pending")}
    dup_ids = {q["id"] for q in db.list_questions(site3, "duplicate")}
    check("dedupe_pending: near-duplicate pending marked 'duplicate'",
          marked == 1 and p1 in dup_ids and p2 in still_pending,
          f"marked={marked} dup={sorted(dup_ids)} newer_survives={p2 in still_pending}")
    check("questions display hides duplicates",
          "hidden" in show_questions(db, db.get_site(site3)),
          "show_questions mentions hidden duplicates")

    # --- dedupe stage 2: recall band (0.60-0.85) -> ONE LLM call ---------
    # The case a fixed threshold cannot solve: duplicates that are similar
    # but not near-identical. Embeddings only PROPOSE the pair; the model
    # decides whether one owner answer would satisfy both.
    site5 = db.create_site("https://band.example", "BandCo")
    record_gap(db, site5, "What are the shipping options for orders?", "no_match",
               provider=fake)
    g5 = db.list_gaps(site5, 5, status="open")[0]["id"]
    q_a = db.add_question(site5, "What are the shipping options for orders?",
                          "shipping", "older wording", gap_id=g5, source="gap")
    q_b = db.add_question(site5,
                          "What are the shipping options cost for deliveries now?",
                          "shipping", "newer near-twin")
    q_c = db.add_question(site5, "How do refunds work for damaged books?",
                          "refunds", "genuinely different")
    va, vb, vc = fake.embed([
        "What are the shipping options for orders?",
        "What are the shipping options cost for deliveries now?",
        "How do refunds work for damaged books?",
    ])
    cos_ab = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
    cos_ac = float(np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc)))
    check("harness: twin pair sits in the recall band [0.60, 0.85)",
          0.60 <= cos_ab < 0.85, f"cos_ab={cos_ab:.3f}")
    check("harness: unrelated pair sits below the recall floor",
          cos_ac < 0.60, f"cos_ac={cos_ac:.3f}")

    calls_before = fake.complete_calls
    marked5 = dedupe_pending(db, fake, site5, db.get_site(site5))
    pend5 = {q["id"] for q in db.list_questions(site5, "pending")}
    dup5 = {q["id"] for q in db.list_questions(site5, "duplicate")}
    check("dedupe stage 2: recall-band twin hidden, unrelated kept",
          marked5 == 1 and q_a in dup5 and q_b in pend5 and q_c in pend5,
          f"marked={marked5} dup={sorted(dup5)} pending={sorted(pend5)}")
    check("dedupe stage 2: exactly one model call for all ambiguous pairs",
          fake.complete_calls - calls_before == 1,
          f"{fake.complete_calls - calls_before} call(s)")
    q_b_row = [q for q in db.list_questions(site5, "pending") if q["id"] == q_b][0]
    check("dedupe: survivor inherits the hidden twin's visitor-gap link",
          q_b_row["gap_id"] == g5,
          f"gap_id={q_b_row['gap_id']} (expect {g5})")

    # --- dedupe link-collision: survivor already linked to ANOTHER gap ----
    # Two duplicate questions = two duplicate gaps. Merge instead of dropping
    # the hidden twin's link, or the orphaned gap gets re-drafted every run.
    site6 = db.create_site("https://merge.example", "MergeCo")
    record_gap(db, site6, "Do you ship worldwide?", "no_match", provider=fake)
    record_gap(db, site6, "What are delivery charges?", "no_match", provider=fake)
    ga, gb = sorted(g["id"] for g in db.list_gaps(site6, 5, status="open"))
    q1 = db.add_question(site6, "What are the shipping options for orders?",
                         "shipping", "older", gap_id=ga, source="gap")
    q2 = db.add_question(site6, "What are the shipping options cost for deliveries now?",
                         "shipping", "newer", gap_id=gb, source="gap")
    marked6 = dedupe_pending(db, fake, site6, db.get_site(site6))
    gaps6 = db.list_gaps(site6, 5, status="open")
    q2_row = [q for q in db.list_questions(site6, "pending") if q["id"] == q2][0]
    check("dedupe: duplicate questions merge their duplicate gaps",
          marked6 == 1 and len(gaps6) == 1 and gaps6[0]["id"] == gb
          and int(gaps6[0]["count"]) == 2 and q2_row["gap_id"] == gb,
          f"marked={marked6} gaps={[(g['id'], g['count']) for g in gaps6]} "
          f"survivor_gap={q2_row['gap_id']} (expect {gb})")

    # --- site-confusion fix: an empty site must never dead-end -----------
    # The owner ran `answer` with no --site, so it checked the LATEST site
    # (Cap Test 2) and said "No drafted questions yet" — while Bookstore had
    # 7 waiting. The empty state must name the site checked AND point at the
    # site that actually has work.
    site7 = db.create_site("https://empty.example", "EmptyCo")
    others = db.pending_by_site(exclude_site_id=site7)
    empty_msg = show_questions(db, db.get_site(site7))
    check("empty site: pending_by_site finds where the questions ARE",
          len(others) >= 2 and all(r["n"] >= 1 for r in others),
          f"{[(r['name'], r['n']) for r in others]}")
    check("empty site: message names a site + the exact --site command",
          (others and f"'{others[0]['name']}'" in empty_msg)
          and '--site "' in empty_msg,
          "suggests a --site target")
    check("empty site: message says which site was checked",
          "EmptyCo" in empty_msg, "names the checked site")

    # --- duplicate site names resolve to the NEWEST row --------------------
    # Re-crawling with the same --name creates a second row; SQLite would
    # return the OLD one first, silently chatting against stale vectors.
    site_old = db.create_site("https://dup.example", "DupCo")
    site_new = db.create_site("https://dup.example", "DupCo")
    resolved = db.get_site("DupCo")
    check("duplicate site name resolves to the newest row",
          resolved is not None and resolved["id"] == site_new,
          f"got={resolved['id'] if resolved else None} expect={site_new}")

    db.conn.close()


def part2_live() -> None:
    print("\n===== PART 2: live end-to-end (real provider) =====")
    from easylink.config import load_settings
    from easylink.facts import add_fact
    from easylink.gaps import rank_gaps, record_gap, show_gaps
    from easylink.providers import get_provider
    from easylink.questions import generate_questions, scan_site, show_questions
    from easylink.rag import RAGEngine
    from easylink.store import Database, get_vector_store

    s = load_settings()
    db = Database()
    provider = get_provider(s)
    vs = get_vector_store(db)
    # pick a site indexed in THIS provider's embedding space (else dimension clash)
    site = None
    for cand in ("Example Test", "Bookstore Test", "Cap Test"):
        c = db.get_site(cand)
        if c is None or vs.count(c["id"]) == 0:
            continue
        try:
            [probe] = provider.embed(["dimension probe"])
            vs.query(c["id"], probe, 1)
            site = c
            break
        except Exception:
            continue
    if site is None:
        check("live: compatible site found", False,
              "no site indexed with the current provider")
        return
    site_id = site["id"]
    engine = RAGEngine(provider, vs, s, site["name"], site["url"], db=db)
    print(f"  site: {site['name']} ({site_id}) · provider: {provider.name}")

    # 1. off-topic -> honest abstention (no LLM call: threshold gate)
    q_offtopic = "Do you offer scuba diving lessons?"
    ans = engine.answer(q_offtopic, site_id)
    check("live: off-topic abstains honestly", ans.abstained, ans.text[:50])

    # 2. gap recorded + dedupe on repeat
    record_gap(db, site_id, q_offtopic, "no_match", provider=provider)
    record_gap(db, site_id, q_offtopic, "no_match", provider=provider)
    matching = [g for g in db.list_gaps(site_id, 500, status="open")
                if g["question"].lower().startswith("do you offer scuba")]
    check("live: repeat question = 1 gap with count >= 2",
          len(matching) == 1 and int(matching[0]["count"]) >= 2,
          f"rows={len(matching)} count={matching[0]['count'] if matching else '?'}")

    # 3. AI drafts questions for the top gaps (real JSON from the model).
    #    regen=True so the suite is re-runnable (supersedes prior drafts).
    drafted = generate_questions(db, provider, vs, s, site, top_n=3, regen=True)
    check("live: AI drafted questions for owner", len(drafted) >= 1,
          f"{len(drafted)} new")
    if drafted:
        print(f"      sample: {drafted[0]['question']}")

    # 4. owner answers -> fact -> AI now knows (money shot)
    test_q = "What is our test coverage policy for this website?"
    test_a = "We test everything twice before shipping, even the test. [TEST]"
    add_fact(db, site_id, test_q, test_a, provider=provider)
    ans2 = engine.answer(test_q, site_id)
    check("live: fact answers a question the SITE never covers",
          (not ans2.abstained) and "test" in ans2.text.lower(),
          ans2.text[:60])

    # 5. facts are durable but let's clean the test one up
    db._exec("DELETE FROM facts WHERE site_id=? AND answer LIKE '%[TEST]%'",
             (site_id,))
    check("live: test fact cleaned up", db.count_facts(site_id) == 0
          or all("[TEST]" not in (f["answer"] or "") for f in db.list_facts(site_id)))

    # 6. resolve + reopen on a real gap
    if matching:
        gid = matching[0]["id"]
        db.resolve_gap(gid)
        check("live: gap resolves", any(
            g["id"] == gid for g in db.list_gaps(site_id, 500, status="resolved")))
        record_gap(db, site_id, q_offtopic, "no_match", provider=provider)
        back = [g for g in db.list_gaps(site_id, 500, status="open") if g["id"] == gid]
        check("live: re-ask reopens it", bool(back))

    # 7. proactive scan (real model reading the crawled page)
    scanned = scan_site(db, provider, s, site, max_pages=3)
    check("live: proactive site scan runs", isinstance(scanned, list),
          f"{len(scanned)} drafted this run")
    if scanned:
        print(f"      sample: {scanned[0]['question']}")

    # 7b. scan a REAL business site — this is what the owner experience is
    store_site = db.get_site("Bookstore Test")
    if store_site and store_site["id"] != site_id:
        scanned2 = scan_site(db, provider, s, store_site, max_pages=3)
        check("live: business-site scan runs", isinstance(scanned2, list),
              f"{len(scanned2)} drafted from bookstore pages")
        for item in scanned2[:2]:
            print(f"      sample: {item['question']}  ({item['page'].split('/')[-1]})")

    # 8. ranking sanity: commercial questions outrank chit-chat
    ranked = rank_gaps(db.list_gaps(site_id, 500, status="open"))
    check("live: ranked list produced", len(ranked) >= 1, f"{len(ranked)} open gaps")

    # 9. displays
    g = show_gaps(db, site_id)
    check("live: gaps display", "score" in g and "stats" in g)
    q = show_questions(db, site)
    check("live: questions display", "Pending questions" in q or "No pending" in q)

    print(f"\n  pending questions left for YOUR hands-on: "
          f"{len(db.list_questions(site_id, 'pending'))}")


def main() -> int:
    part1_offline()
    part2_live()
    print(f"\n{'='*60}")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name}  {detail}")
    print(f"{passed}/{len(RESULTS)} passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

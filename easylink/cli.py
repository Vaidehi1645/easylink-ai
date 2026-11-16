"""EasyLink AI — Phase 1 CLI.

Commands:
  crawl <url>   crawl a site, chunk it, embed it, store it
  chat          terminal RAG chat with citations + abstention
  sites         list indexed sites
  gaps          show questions the AI couldn't answer
  eval          quality report (retrieval + optionally full answers)
  doctor        environment / provider / store health check
"""

from __future__ import annotations

import argparse
import sys

from easylink import __version__
from easylink.config import load_settings


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _require_site(db, args):
    site = db.get_site(getattr(args, "site", None))
    if not site:
        print("No site indexed yet. Run:  python -m easylink crawl <url>")
        raise SystemExit(1)
    return site


# --------------------------------------------------------------------------- #
def cmd_doctor(args) -> int:
    from easylink.store import Database, get_vector_store

    s = load_settings()
    print(f"EasyLink AI v{__version__} — doctor")
    print(f"  provider      : {s.provider}")
    if s.provider in ("openai", "gemini"):
        if s.provider == "gemini":
            key = s.gemini_key or s.api_key
            default_models = ("gemini-3.8-flash", "gemini-embedding-001")
        else:
            key = s.api_key
            default_models = ("gpt-4o-mini", "text-embedding-3-small")
        masked = f"{key[:6]}...{key[-4:]}" if len(key) > 12 else ("set" if key else "MISSING")
        print(f"  api key       : {masked}")
        print(f"  chat model    : {s.chat_model or default_models[0]}")
        print(f"  embed model   : {s.embed_model or default_models[1]}")
        try:
            from easylink.providers import get_provider

            get_provider(s)
            print("  provider check: OK")
        except Exception as exc:
            print(f"  provider check: FAILED — {exc}")
            return 1
    else:
        print("  (mock provider — offline, free, no key needed)")

    db = Database()
    vs = get_vector_store(db)
    print(f"  vector store  : {vs.backend}")
    print(f"  sites indexed : {len(db.list_sites())}")
    print(f"  caps          : {s.max_pages} pages, depth {s.max_depth}, "
          f"top_k {s.top_k}, threshold {s.min_similarity if s.min_similarity is not None else 'provider default'}")
    return 0


def cmd_crawl(args) -> int:
    from easylink.crawler import crawl
    from easylink.indexer import index_pages
    from easylink.providers import get_provider
    from easylink.store import Database, get_vector_store

    s = load_settings()
    if args.max_pages:
        s.max_pages = args.max_pages
    if args.max_depth is not None:
        s.max_depth = args.max_depth

    print(f"[1/3] Crawling {args.url}  (max {s.max_pages} pages, depth {s.max_depth})")
    try:
        result = crawl(args.url, s)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    except Exception as exc:
        print(f"crawl failed: {exc}")
        return 1

    if not result.pages:
        print("No content crawled. The site may block bots — check robots.txt.")
        return 1

    db = Database()
    vs = get_vector_store(db)
    provider = get_provider(s)

    site_id = db.create_site(args.url, args.name or args.url)
    print(f"[2/3] Indexed site id: {site_id}")

    print(f"[3/3] Chunking + embedding with {provider.name} "
          f"(store: {vs.backend})")
    pages, chunks = index_pages(site_id, result.pages, provider, db, vs, s)
    db.update_site_counts(site_id, pages, chunks)

    print(f"\nDone: {pages} pages, {chunks} chunks  (site id: {site_id})")
    print("Chat with it:   python -m easylink chat")
    return 0


def cmd_chat(args) -> int:
    from easylink.gaps import record_gap, show_gaps
    from easylink.providers import get_provider
    from easylink.rag import RAGEngine
    from easylink.store import Database, get_vector_store

    s = load_settings()
    db = Database()
    site = _require_site(db, args)
    site_id = site["id"]

    provider = get_provider(s)
    vs = get_vector_store(db)
    if vs.count(site_id) == 0:
        print("This site has no vectors. Re-run crawl.")
        return 1

    engine = RAGEngine(provider, vs, s, site["name"], site["url"], db=db)
    print(f"\nEasyLink chat — {site['name']}")
    if getattr(args, "site", None) in (None, "latest"):
        print(f"  (latest site — switch any time: chat --site \"<name>\")")
    print(f"  {site['chunk_count']} chunks | provider {provider.name} | "
          f"threshold {engine.threshold:.2f}")
    print("  Commands: /quit  /gaps  /help\n")

    last_question = ""
    history: list[dict] = []  # last few turns for follow-up questions
    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit", ":q", "quit", "exit"):
            break
        if user == "/help":
            print("  /quit exit  |  /gaps show unanswered questions  |  "
                  "type 👎 after a bad answer to log it as a gap")
            continue
        if user == "/gaps":
            print(show_gaps(db, site_id))
            continue
        if user in ("\U0001F44E", "down", "bad") and last_question:
            record_gap(db, site_id, last_question, "owner_downvote", provider=provider)
            print("  noted — logged as a knowledge gap for the owner.\n")
            continue

        from easylink.providers import ProviderError as _ProviderError

        try:
            answer = engine.answer(user, site_id, history=history)
        except (_ProviderError, RuntimeError) as exc:
            # Transient API failures and provider mismatches must never kill the session
            print(f"\nbot > [temporarily unavailable: {exc}]")
            print("      please try again in a moment.\n")
            continue
        last_question = user

        print(f"\nbot > {answer.text}")
        if answer.abstained:
            record_gap(db, site_id, user, "no_match", provider=provider)
            print("      (no match — logged to gap queue)")
        else:
            history.append({"q": user, "a": answer.text})
            history = history[-4:]  # keep context small & cheap
        if answer.sources:
            print(f"      sources: {', '.join(answer.sources[:4])}")
        print()
    return 0


def cmd_sites(args) -> int:
    from easylink.store import Database

    db = Database()
    sites = db.list_sites()
    if not sites:
        print("No sites indexed yet.")
        return 0
    print(f"Sites ({len(sites)}):")
    for st in sites:
        print(f"  {st['id']}  {st['name'][:45]:45}  "
              f"pages={st['page_count']} chunks={st['chunk_count']}  {st['created_at']}")
    return 0


def cmd_gaps(args) -> int:
    from easylink.gaps import show_gaps
    from easylink.store import Database

    db = Database()
    # With --site: filter to that site. Without it: show ALL gaps —
    # silently defaulting to "latest site" hid every other site's questions.
    site_id = None
    if getattr(args, "site", None):
        site = db.get_site(args.site)
        site_id = site["id"] if site else None
    print(show_gaps(db, site_id, resolved=getattr(args, "resolved", False)))
    return 0


def cmd_questions(args) -> int:
    """Phase 3: AI drafts precise questions for the OWNER."""
    from easylink.providers import get_provider
    from easylink.questions import generate_questions, scan_site, show_questions
    from easylink.store import Database, get_vector_store

    s = load_settings()
    db = Database()
    site = _require_site(db, args)
    provider = get_provider(s)

    if args.scan:
        print(f"Scanning {site['name']} — what does the site fail to explain?")
        drafted = scan_site(db, provider, s, site, max_pages=args.pages)
        if not drafted:
            print("  Nothing new found — either the site explains itself well")
            print("  or these questions were already drafted:")
        else:
            print(f"  {len(drafted)} new question(s) drafted from the site itself:\n")
            for item in drafted:
                print(f"  · {item['topic']}")
                print(f"    Q: {item['question']}")
                print(f"    Why: {item['why']}")
                print(f"    (page: {item['page']})\n")
    else:
        vs = get_vector_store(db)
        print(f"Drafting questions for {site['name']} (top {args.top} open gaps)...")
        drafted = generate_questions(
            db, provider, vs, s, site, top_n=args.top, regen=args.regen
        )
        if not drafted:
            print("  Nothing new — top gaps already have pending questions.")
        else:
            print(f"  {len(drafted)} question(s) drafted:\n")
            for item in drafted:
                print(f"  · {item['topic']}  (visitors asked x{item.get('gap_count', 1)})")
                print(f"    Q: {item['question']}")
                print(f"    Why: {item['why']}\n")

    print(show_questions(db, site))
    return 0


def cmd_answer(args) -> int:
    """Phase 3: owner answers a drafted question -> fact + gap resolved."""
    from easylink.facts import add_fact
    from easylink.providers import get_provider
    from easylink.store import Database

    s = load_settings()
    db = Database()
    site = _require_site(db, args)
    site_id = site["id"]

    def store(question: str, answer: str, qid: int | None) -> None:
        provider = get_provider(s)
        add_fact(db, site_id, question, answer, provider=provider)
        if qid is not None:
            row = db.get_question(qid)
            if row:
                db.set_question_status(qid, "answered")
                if row["gap_id"]:
                    db.resolve_gap(row["gap_id"])
        print(f"\n  \u2713 Learned! The AI will now answer:")
        print(f'    Q: {question}')
        print(f'    A: {answer}')
        if qid is not None:
            print("  \u2713 Related gap marked resolved.")
        print("\n  Test it: easylink chat, then ask the same question.\n")

    # Non-interactive path (scripts/tests): --id + --answer
    if args.id is not None:
        row = db.get_question(args.id)
        if row is None or row["site_id"] != site_id:
            print(f"No question #{args.id} on this site.")
            return 1
        if not args.answer:
            print("Missing --answer \"your answer\"")
            return 1
        store(row["question"], args.answer, args.id)
        return 0

    # Interactive: list pending questions, pick one, type the answer
    from easylink.questions import show_questions

    pending = db.list_questions(site_id, "pending")
    if not pending:
        # Never a dead end: say WHICH site was checked, why, and where the
        # questions actually are (default = latest site, which silently
        # switching contexts is exactly how owners get confused).
        default_used = not getattr(args, "site", None)
        print(f"No pending questions for '{site['name']}'"
              + (" (default: latest site — you passed no --site)" if default_used else "")
              + ".")
        others = db.pending_by_site(exclude_site_id=site_id)
        for r in others:
            print(f"  -> {r['n']} waiting on '{r['name']}'  ->  "
                  f'easylink answer --site "{r["name"]}"')
        if not others:
            print("First run:  easylink questions        (from visitor gaps)")
            print("       or:  easylink questions --scan  (proactive site review)")
        return 0
    print(show_questions(db, site))
    try:
        choice = input("\nQuestion number to answer (Enter to quit): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    if not choice:
        return 0
    try:
        qid = int(choice)
    except ValueError:
        print("Not a number.")
        return 1
    row = db.get_question(qid)
    if row is None or row["site_id"] != site_id:
        print(f"No question #{qid} on this site.")
        return 1
    print(f"\n  Q: {row['question']}")
    try:
        answer = input("  A: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0
    if not answer:
        print("  (no answer given — nothing saved)")
        return 0
    store(row["question"], answer, qid)
    return 0


def cmd_eval(args) -> int:
    from easylink.evals import run_eval
    from easylink.providers import get_provider
    from easylink.rag import RAGEngine
    from easylink.store import Database, get_vector_store

    s = load_settings()
    db = Database()
    site = _require_site(db, args)
    provider = get_provider(s)
    vs = get_vector_store(db)
    engine = RAGEngine(provider, vs, s, site["name"], site["url"], db=db)
    run_eval(engine, db, site["id"], golden_file=args.file, use_llm=args.llm)
    return 0


def cmd_embed(args) -> int:
    """Print this site's widget token + copy-paste snippet (Phase 2)."""
    import os

    from easylink.store import Database

    db = Database()
    site = db.get_site(args.site)
    if not site:
        print("No site found. Crawl one first: easylink crawl <url>")
        return 1
    token = db.get_or_create_token(site["id"])
    port = os.environ.get("EASYLINK_PORT", "8000")
    base = f"http://127.0.0.1:{port}"
    print(f"Site:  {site['name']}  (id {site['id']})")
    print(f"Token: {token}")
    print()
    print("Embed snippet (paste before </body> on your website):")
    print(f'    <script src="{base}/widget.js" data-site="{token}"></script>')
    print()
    print(f"Local test page : {base}/test?token={token}")
    print("Start the server: venv\\Scripts\\python.exe -m api")
    print("(localhost only in Phase 2 — public HTTPS hosting arrives in Phase 5)")
    return 0


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    _utf8()
    p = argparse.ArgumentParser(
        prog="easylink",
        description="EasyLink AI Phase 1 — crawl, index, and chat with any website.",
    )
    sub = p.add_subparsers(dest="command")

    d = sub.add_parser("doctor", help="check environment, provider, and store")
    d.set_defaults(func=cmd_doctor)

    c = sub.add_parser("crawl", help="crawl + index a website")
    c.add_argument("url", help="site to crawl (https://...)")
    c.add_argument("--max-pages", type=int, default=None, help="hard page cap")
    c.add_argument("--max-depth", type=int, default=None, help="max link depth")
    c.add_argument("--name", default="", help="friendly name for this site")
    c.add_argument("--site", help=argparse.SUPPRESS)
    c.set_defaults(func=cmd_crawl)

    ch = sub.add_parser("chat", help="terminal RAG chat")
    ch.add_argument("--site", default="latest", help="site id or 'latest'")
    ch.set_defaults(func=cmd_chat)

    st = sub.add_parser("sites", help="list indexed sites")
    st.set_defaults(func=cmd_sites)

    g = sub.add_parser("gaps", help="show unanswered questions")
    g.add_argument("--site", default=None)
    g.add_argument("--resolved", action="store_true", help="show closed gaps")
    g.set_defaults(func=cmd_gaps)

    q = sub.add_parser("questions", help="AI drafts questions for YOU (Phase 3)")
    q.add_argument("--site", default=None, help="site id or name (default: latest)")
    q.add_argument("--top", type=int, default=5, help="how many top gaps to draft for")
    q.add_argument("--scan", action="store_true", help="proactive: review the site itself")
    q.add_argument("--pages", type=int, default=10, help="max pages to scan")
    q.add_argument("--regen", action="store_true", help="redraft pending gap questions")
    q.set_defaults(func=cmd_questions)

    a = sub.add_parser("answer", help="answer a drafted question -> AI learns (Phase 3)")
    a.add_argument("--site", default=None, help="site id or name (default: latest)")
    a.add_argument("--id", type=int, help="question id to answer (non-interactive)")
    a.add_argument("--answer", help="your answer (required with --id)")
    a.set_defaults(func=cmd_answer)

    e = sub.add_parser("eval", help="quality report")
    e.add_argument("--site", default="latest")
    e.add_argument("--file", default=None, help="golden questions JSONL")
    e.add_argument("--llm", action="store_true", help="generate real answers (uses API credits)")
    e.set_defaults(func=cmd_eval)

    em = sub.add_parser("embed", help="widget token + embed snippet (Phase 2)")
    em.add_argument("--site", default="latest", help="site id or 'latest'")
    em.set_defaults(func=cmd_embed)

    args = p.parse_args(argv)
    if not getattr(args, "command", None):
        p.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

# EasyLink AI — Phase 1 (Core RAG CLI) + Phase 2 (API & Embeddable Widget)

Give any website a brain you can talk to.

**Phase 1** proves the engine: can this AI answer questions about a website
correctly, honestly, and safely? **Phase 2** wraps that engine in a local API
and ships a one-line embeddable chat widget any site can drop in.

## Setup

```powershell
pip install -r requirements.txt
copy .env.example .env      # then paste your API key
python -m easylink doctor   # verify provider + store
```

**Your key, auto-detected:** a Gemini key (`AIza...`) and an OpenAI key (`sk-`)
are each picked up automatically — no config needed. Both work; the adapter
layer means neither is ever locked in.

No API key at all? Test the whole pipeline offline for free:

```powershell
# in .env:  EASYLINK_PROVIDER=mock
```

## Usage

```powershell
# 1. Crawl + index a site (caps enforced, robots.txt respected)
python -m easylink crawl https://books.toscrape.com --max-pages 15 --name "Test Bookstore"

# 2. Chat with it — answers cite their sources, and it admits when it doesn't know
python -m easylink chat

# 3. See what it failed to answer (your future gap queue)
python -m easylink gaps

# 4. Quality report (free retrieval self-test; add --llm to test real answers)
python -m easylink eval

# Extras
python -m easylink sites
python -m easylink doctor
python -m easylink embed       # Phase 2: widget token + embed snippet
python -m easylink questions   # Phase 3: AI drafts questions for YOU
python -m easylink answer      # Phase 3: you answer -> the AI learns
```

## Phase 3 — the Gap Engine (self-improvement loop)

```powershell
# 1. See what visitors asked that the AI couldn't — ranked by priority
python -m easylink gaps

# 2. The AI drafts precise questions for YOU (top 5 gaps)
python -m easylink questions

# 3. Proactive mode: the AI reviews your crawled site and asks about
#    whatever your pages fail to explain — before anyone complains
python -m easylink questions --scan

# 4. Answer one (30 seconds) -> stored as an owner-verified fact
python -m easylink answer              # interactive picker
python -m easylink answer --id 3 --answer "We deliver in 2 days, flat 49 fee."

# 5. Ask the same question in chat — the AI now knows it
python -m easylink chat
```

- **Dedupe:** repeat questions merge (exact + embedding cosine ≥ 0.90) — one gap, `count ×N`.
- **Consolidation:** before drafting, embeddings pre-group open gaps (cosine ≥ 0.75)
  and **one LLM call regroups them by underlying meaning** — "do you ship?",
  "delivery charges?", "shipping policy?" become ONE topic with summed counts,
  named and drafted once. Embeddings propose, the model disposes.
- **No redundant drafts:** drafted questions are embedding-checked against what's
  already pending (≥ 0.85 hides instantly; the 0.60–0.85 "maybe" band — measured
  on real data where the similarity bands *overlap* — is settled by one batched
  LLM call that asks: *would one owner answer satisfy both?*). Duplicate questions
  also merge their underlying gaps, so nothing gets re-drafted on later runs.
  Duplicates are hidden, never deleted — `questions` tells you how many it de-noised.
- **Ranking:** `frequency × recency × buying intent` — money questions first.
- **Owner-verified facts** are the highest authority in the prompt: they can
  answer even when page retrieval fails (layer 8 of the anti-hallucination stack).
  Facts have no page URL to link, so those answers carry a green
  **"✓ owner-verified"** badge instead — a claim never reaches the visitor
  without a stated source.
- **Reopening:** if a "resolved" gap gets re-asked, it reopens — your first
  answer wasn't enough, and the queue says so.

## Phase 2 — API server & embeddable widget

```powershell
# 1. Get your site's token + one-line snippet
python -m easylink embed

# 2. Start the local API server (http://127.0.0.1:8000)
python -m api

# 3. Open the agent intake page — paste any legal website link, watch
#    it crawl live, and get back a shareable /demo/<slug> link
#    http://127.0.0.1:8000/test
```

Embed on any page:

```html
<script src="http://127.0.0.1:8000/widget.js" data-site="YOUR_TOKEN"></script>
```

- **Streaming:** answers render word-by-word (SSE under the hood).
- **Shadow DOM:** the widget cannot be broken by the host page's CSS (and can't break it back).
- **Security layers:** per-site token → origin check → per-IP + per-token rate
  limits → per-site daily quota. Foreign origins get 403; floods get 429.
- **Gaps:** web abstentions are logged to the same gap queue as CLI chats.

Tests (server must be running):

```powershell
python tests\test_phase2_api.py            # API + security (13 checks)
python tests\test_phase2_api.py --flood    # rate limiting (run last)
python tests\test_phase2_api.py --quota    # daily quota (spawns its own server)
python tests\test_phase2_widget.py         # headless-browser widget test (13 checks)
python tests\test_phase6_ingest.py         # self-serve intake + SSRF + quota (24 checks)
```

### Inside chat

| Question type | Behavior |
|---|---|
| answerable question | grounded answer + cited sources |
| off-corpus question | "I don't know" + **logged to gap queue** |
| `👎` | flag the previous answer as a knowledge gap |
| `/gaps` | list unanswered questions |
| `/quit` | exit |

## Phase 6 — self-serve agent intake

`GET /test` (no token) is now an **agent**: paste any valid & legal
website link and watch it crawl live, then take back two outputs:

```text
demo link    http://your-host/demo/<slug>   -> clean page + widget for that site
snippet      <script src=".../widget.js" data-site="TOKEN"></script>
```

- **`POST /ingest`** streams the crawl as SSE: `start` → `fetch` (every
  page + status) → `log`/`warn` → `index` → `done`/`error`.
- **Guards:** http(s) only · **SSRF blocked** (only publicly routable
  addresses are ever fetched — every redirect hop re-checked via
  `GuardedSession`) · **10 crawls/day/IP** (`EASYLINK_CRAWL_DAILY`) ·
  **30 pages max** per crawl · robots.txt + politeness delays (built in).
- **Speed budgets:** every ingest crawl is bounded — **35 s wall clock**
  (`INGEST_MAX_SECONDS`), **90 fetch attempts** (`INGEST_MAX_FETCHES`),
  8 s per page, 0.1 s politeness, sitemap capped at 100 URLs. Whatever is
  collected by the deadline gets indexed: a pasted link answers in
  seconds, never minutes (a live palmonas.com crawl runs ~9 s).
- **Double-paste safe:** pasting a link *while it is still crawling* joins
  the run in progress — one crawl, every visitor gets the same feed and
  demo link, only the first visitor spends quota.
- **Reuse:** the same host is crawled once — re-pasting it returns the
  same link instantly and spends nothing (no fetches, no embeddings).
- **JS-heavy sites stay answerable:** stores that render their body with
  JavaScript are indexed from what the server *does* send — meta
  description / og:description / JSON-LD summaries plus the site's own
  link text (product tiles fall back to their URL-slug labels), with an
  honest warning when most fetches came back empty.
- **`done` carries `learned`** — the page titles the bot actually knows —
  plus crawl `seconds`; the intake UI lists them under "What your bot
  learned", showing 8 with a clickable **"+N more pages"** that expands
  to the full list and **"↑ show less"** that collapses it back. The
  **demo page shows the same list** ("📚 What this bot learned" card,
  server-rendered and HTML-escaped), so anyone opening a copied demo
  link sees exactly what the bot knows — and what it doesn't — before
  asking. The 🔗 EasyLink AI logo on both the intake and demo headers
  now links to the maker's GitHub.
- **Live status, for real:** the intake agent card and the embeddable
  widget both show a pulsing green dot while the server answers its
  health ping, and flip to a red **"network gone — check your
  connection"** state the moment the network or server drops (input
  disabled on intake, an in-chat pill on the widget) — restored
  automatically, with a "back online" note.
- **Bot-hostile sites say so out loud:** when most requests come back
  blocked (Amazon-class walls), the feed warns, `done` carries `warn` +
  `blocked`, the intake shows a persistent "answers will be limited"
  badge, and the demo page states how many pages it was built from — so
  "I don't have information" reads as the site's refusal, not a storage
  bug. Reopening a ≤4-page site repeats the warning.
- `GET /demo/<slug>` serves the demo page; slugs are domain-derived and
  stable (`jewelsmars.com` → `/demo/jewelsmars`, `…-2` on collision).

## Owner sign-in (Google + GitHub OAuth)

The intake page is the **owner console** — it gets a real login; visitors
never need accounts.

- **Feature-flagged:** with no client IDs in `.env`, login is off and the
  app behaves exactly as before (no modal, no gates, tests unchanged).
- **Intake (`/test`)** shows a **dismissible sign-in modal** on first visit
  (Google + GitHub; click outside / × closes it) — crawling never requires
  an account, and a **Sign in** link stays in the header for later.
- **Demo pages** (`/demo/<slug>`, `/test?token=…`) are **gated**: visitors
  bounce to a sign-in card and land back where they were after authenticating
  (`EASYLINK_DEMO_GATE=false` opens them up). The **embeddable widget stays
  public** — embeds are authenticated by the per-site token.
- **Sessions:** 256-bit random token, only its SHA-256 hash stored in
  SQLite, `HttpOnly` + `SameSite=Lax` cookie, 30-day expiry
  (`EASYLINK_SESSION_DAYS`), logout revokes server-side (POST only — a
  forged `<img src>` can't log you out).
- **Security:** `state` cookie (CSRF) + `nonce` (replay) verified on the
  callback; Google ID tokens checked against Google's published
  certificates via `google-auth` (signature + audience + expiry — never
  hand-parsed); client secrets only in `.env`; `next` restricted to local
  paths (no open redirects); signed-in owners see avatar + **Sign out** in
  the header.
- **Setup:** create a Google OAuth client (redirect URI
  `http://127.0.0.1:8000/auth/callback/google`) and a GitHub OAuth app
  (callback `http://127.0.0.1:8000/auth/callback/github`), paste the four
  values into `.env` (see `.env.example`), restart the server. Production
  needs `EASYLINK_PUBLIC_URL=https://your-domain` — Google only accepts
  HTTPS callbacks (localhost is the one exception).

## The 7 anti-hallucination layers (1–7 live in Phase 1)

1. **Retrieval threshold** — below the similarity floor, it never even asks the model; auto "I don't know"
2. **Grounded system prompt** — answer ONLY from provided context
3. **Low temperature** — facts, not creativity
4. **Source citations** — links are filtered to the pages the reply actually drew on; a bare "I don't know" cites nothing, no grounded answer can render sourceless (owner-verified badge when an owner fact answered, first-page fallback otherwise), and error pages (`/404`) are never cited
5. **Never-say list** — `never_say.txt`, editable per your rules
6. **Gap logging** — every failure becomes a question for the owner (Phase 3 = the queue UI)
7. **Eval suite** — `eval` command with a golden-question file

**Date awareness** — every question carries today's date (the model never
infers it from crawled page content — no more "October 8 … which is
today"), and pure date questions are answered from the clock with zero
LLM calls.

## Crawler safety (built in)

sitemap-first · hard page cap · depth limit · robots.txt respected · honest
User-Agent · URL normalization · content-hash dedupe · trap detection
(pagination/calendars/sort loops) · content-first link prioritization
(on-subject pages → other content → header/footer chrome) · per-page timeout ·
politeness delay · charset detection (no `Â£` mojibake) · duplicate-chunk
suppression.

## Reliability (built in)

- **Retries with backoff** on 429/500/503 — a Gemini traffic spike never kills a session
- **Chat survives API failures** — friendly message, session continues
- **Eval survives rate limits** — partial report instead of a crash

## Known limits (Phase 1)

- Gemini **free tier has small chat quotas** — heavy testing can exhaust it for the day.
  Fixes: wait for reset, enable billing on your key, add an OpenAI `sk-` key, or use `EASYLINK_PROVIDER=mock`.

## Files

```
easylink/
├── cli.py        crawl / chat / gaps / eval / sites / doctor / embed / questions / answer
├── config.py     all caps, thresholds, .env secrets
├── providers.py  OpenAI + Gemini + mock (with streaming for all three)
├── crawler.py    responsible crawler
├── indexer.py    chunker + batched embeddings
├── store.py      SQLite records + Chroma/NumPy vectors (namespaced by site)
├── rag.py        retrieval + grounded answering (8 layers, + stream_answer)
├── gaps.py       Phase 3: normalize, embedding dedupe, priority ranking
├── questions.py  Phase 3: AI question drafting + proactive site scan
├── facts.py      Phase 3: owner-verified facts (highest prompt authority)
└── evals.py      quality harness
api/
├── server.py     FastAPI: /chat (SSE) · /ingest (SSE crawl) · /widget.js · /test · /demo/<slug> · /health
├── security.py   token + origin + rate limit + daily quota + crawl quota
├── static/       widget.js + demo test page + agent intake page
└── __main__.py   python -m api
tests/            API + browser + Phase 3 gap-engine + Phase 6 ingest suites
never_say.txt     owner-editable forbidden answers
evals/golden.example.jsonl
```

## Phase 3 does NOT include (by design)

Web dashboard · email digests · auth · payments · auto-editing the website ·
publishing answers as public pages — those are Phase 4+.

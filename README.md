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
- **Reopening:** if a "resolved" gap gets re-asked, it reopens — your first
  answer wasn't enough, and the queue says so.

## Phase 2 — API server & embeddable widget

```powershell
# 1. Get your site's token + one-line snippet
python -m easylink embed

# 2. Start the local API server (http://127.0.0.1:8000)
python -m api

# 3. Open the demo page and try the widget (flip the hostile-CSS switch
#    to watch Shadow DOM isolation survive a real CSS attack)
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
```

### Inside chat

| Question type | Behavior |
|---|---|
| answerable question | grounded answer + cited sources |
| off-corpus question | "I don't know" + **logged to gap queue** |
| `👎` | flag the previous answer as a knowledge gap |
| `/gaps` | list unanswered questions |
| `/quit` | exit |

## The 7 anti-hallucination layers (1–7 live in Phase 1)

1. **Retrieval threshold** — below the similarity floor, it never even asks the model; auto "I don't know"
2. **Grounded system prompt** — answer ONLY from provided context
3. **Low temperature** — facts, not creativity
4. **Source citations** — links are filtered to the pages the reply actually drew on; a bare "I don't know" cites nothing
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
├── server.py     FastAPI: /chat (SSE) · /widget.js · /test · /health
├── security.py   token + origin + rate limit + daily quota
├── static/       widget.js (Shadow DOM web component) + demo test page
└── __main__.py   python -m api
tests/            API suite + browser suite + Phase 3 gap-engine suite
never_say.txt     owner-editable forbidden answers
evals/golden.example.jsonl
```

## Phase 3 does NOT include (by design)

Web dashboard · email digests · auth · payments · auto-editing the website ·
publishing answers as public pages — those are Phase 4+.

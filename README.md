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
```

## Phase 2 — API server & embeddable widget

```powershell
# 1. Get your site's token + one-line snippet
python -m easylink embed

# 2. Start the local API server (http://127.0.0.1:8000)
python -m api

# 3. Open the hostile-CSS test page and try the widget
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
4. **Source citations** — every answer shows which page it came from
5. **Never-say list** — `never_say.txt`, editable per your rules
6. **Gap logging** — every failure becomes a question for the owner (Phase 3 = the queue UI)
7. **Eval suite** — `eval` command with a golden-question file

## Crawler safety (built in)

sitemap-first · hard page cap · depth limit · robots.txt respected · honest
User-Agent · URL normalization · content-hash dedupe · trap detection
(pagination/calendars/sort loops) · per-page timeout · politeness delay ·
charset detection (no `Â£` mojibake) · duplicate-chunk suppression.

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
├── cli.py        crawl / chat / gaps / eval / sites / doctor / embed
├── config.py     all caps, thresholds, .env secrets
├── providers.py  OpenAI + Gemini + mock (with streaming for all three)
├── crawler.py    responsible crawler
├── indexer.py    chunker + batched embeddings
├── store.py      SQLite records + Chroma/NumPy vectors (namespaced by site)
├── rag.py        retrieval threshold + grounded answering (+ stream_answer)
├── gaps.py       the seed of the differentiator
└── evals.py      quality harness
api/
├── server.py     FastAPI: /chat (SSE) · /widget.js · /test · /health
├── security.py   token + origin + rate limit + daily quota
├── static/       widget.js (Shadow DOM web component) + hostile test page
└── __main__.py   python -m api
tests/            API suite + headless-browser widget suite
never_say.txt     owner-editable forbidden answers
evals/golden.example.jsonl
```

## Phase 2 does NOT include (by design)

Dashboard · auth · payments · public hosting/HTTPS · ownership verification ·
multi-tenancy beyond per-site tokens — those are Phase 4+.

# EasyLink AI — Phase 1: Core RAG CLI

Give any website a brain you can talk to.

**Phase 1 proves one thing:** can this AI answer questions about a website
correctly, honestly, and safely? No API, no widget, no dashboard — just the engine.

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
├── cli.py        crawl / chat / gaps / eval / sites / doctor
├── config.py     all caps, thresholds, .env secrets
├── providers.py  OpenAI + mock (no lock-in — Gemini drops in later)
├── crawler.py    responsible crawler
├── indexer.py    chunker + batched embeddings
├── store.py      SQLite records + Chroma/NumPy vectors (namespaced by site)
├── rag.py        retrieval threshold + grounded answering
├── gaps.py       the seed of the differentiator
└── evals.py      quality harness
never_say.txt     owner-editable forbidden answers
evals/golden.example.jsonl
```

## Phase 1 does NOT include (by design)

FastAPI · JS widget · dashboard · auth · hosting · ownership verification ·
multi-tenancy · payments — those are Phase 2+.

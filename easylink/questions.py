"""Question drafting — the AI asks the OWNER precise questions.

The pipeline (all on demand — `easylink questions`):

  1. CONSOLIDATE — the pile of visitor gaps is grouped by meaning:
       embedding recall   -> cosine >= 0.75 pre-clusters (cheap, over-groups)
       LLM precision      -> ONE call regroups by underlying meaning, names a
                             topic per cluster, marks clusters already covered
       store merge        -> duplicate gap rows collapse (counts summed)
  2. DEDUPE PENDING    — drafted questions that near-duplicate an existing
                         pending one (cosine >= 0.85) are marked 'duplicate'
                         so the owner never sees the same ask twice.
  3. DRAFT             — for each uncovered cluster, ONE precise question for
                         the owner (grounded in what the site already has).
  4. --scan            — proactive: review crawled pages, draft what they
                         fail to explain, same dedupe rules.

Why two stages instead of one big prompt: embeddings do the scalable,
deterministic recall (O(n) vector math, works at 10 or 10,000 gaps); the LLM
does what embeddings can't — judge whether two DIFFERENTLY WORDED questions
need the SAME answer. Fail-safe: if the LLM returns garbage, every gap keeps
its own topic and nothing is merged or lost.

The AI only ASKS. It never drafts your answer (that would be hallucination
in your voice — the one thing we never do).
"""

from __future__ import annotations

import json
import re

import numpy as np

from easylink.gaps import (
    _embedding_blob,
    backfill_embeddings,
    cosine_groups,
    rank_gaps,
)
from easylink.store import Database, normalize_question, vec_cosine

# How many ranked gaps feed the regrouping call (all of them for any real
# site in Phase 3; cap keeps one run's prompt bounded).
CONSOLIDATE_INPUT = 30

# Two thresholds, two jobs — the same embedding-recall -> LLM-precision
# split used by gap consolidation, because ONE magic number can't separate
# "same ask, different words" from "genuinely different" across every site:
#
#   measured on real Bookstore data (tests/_cos_probe.py): true twins reach
#   as low as 0.62 ("What is your shipping policy?" vs "shipping options and
#   costs for orders") while distinct pairs were seen at 0.56 — the bands
#   OVERLAP, so the floor only bounds what we bother to ask about; the LLM
#   call decides correctness, never the number.
NEAR_DUPLICATE_SIM = 0.85  # deterministic: at/above this, hide with no judgement call
DEDUPE_RECALL_SIM = 0.60   # recall floor: propose at most 2 partners per question

GEN_SYSTEM = """You help the owner of "{site_name}" ({site_url}) improve their website.
You will be given question(s) a VISITOR asked that the website could NOT answer — they all ask the same underlying thing.
Reply with ONLY a JSON object — no markdown, no code fences, no commentary:
{{"topic": "<3-6 word topic>", "question": "<one precise question to ask the site owner, phrased so they can answer it in one sentence>", "why": "<one short sentence: why visitors need this>"}}

RULES:
- NEVER answer the question yourself. NEVER invent facts about the business.
- The question must ask for something a customer needs that the site does not state.
- Keep the question concrete (policies, areas, fees, timings, limits — not vague).
- If the visitor questions need DIFFERENT answers from each other, ask about the broadest one the site seems to miss."""

CONSOLIDATE_SYSTEM = """You group VISITOR QUESTIONS (asked to "{site_name}") by their UNDERLYING MEANING — questions needing the same answer belong together even if the wording differs.
You get: (1) open gaps with id, ask-count and text; (2) candidate groups from embeddings — a HINT only, you may regroup; (3) questions already drafted and pending for the owner.
Reply with ONLY a JSON object — no markdown, no commentary:
{{"clusters": [{{"topic": "<3-6 word topic>", "gap_ids": [<ids>], "action": "draft"|"skip"}}]}}

RULES:
- Every given id must appear in EXACTLY one cluster. None dropped, none repeated.
- Merge wordings that need the SAME answer (e.g. "do you ship?", "what does shipping cost?", "delivery charges?" are one topic).
- Do NOT merge questions needing DIFFERENT answers (different fee vs different area vs different product).
- action="skip" if a PENDING question already covers this cluster; else "draft".
- NEVER answer any question. NEVER invent facts about the business."""

DEDUPE_SYSTEM = """You compare pairs of questions about "{site_name}" ({site_url}) — either asked by visitors or drafted for the owner to answer.
For each pair, decide whether ONE answer from the owner would satisfy BOTH questions.
Reply with ONLY a JSON object — no markdown, no commentary:
{{"same": [<1-based pair numbers that ask the same thing>]}}

RULES:
- SAME if one clear owner answer covers both — even when wording, grammar, or scope-words differ ("for this book" vs "for orders", "charges" vs "costs", "you" vs "your store").
- SAME example: ("What are the shipping options and costs for orders?", "What are the shipping options and costs for this book?") — one shipping policy answers both.
- DIFFERENT only when the pairs need answers about DIFFERENT SUBJECTS, e.g. shipping vs returns vs warranty.
- DIFFERENT example: ("What are the shipping options and costs?", "What is the return policy?").
- When the core subject matches, prefer "same" — the hidden twin stays on file, nothing is lost.
- NEVER answer the questions yourself. NEVER invent facts about the business."""

SCAN_SYSTEM = """You are reviewing pages of "{site_name}" ({site_url}) as a demanding visitor.
Find IMPORTANT questions a customer would ask that this page does NOT answer
(missing policies, unclear pricing, unexplained terms, absent practical details).
Reply with ONLY a JSON array — no markdown, no commentary. Each item:
[{{"topic": "<3-6 word topic>", "question": "<one question for the site owner>", "why": "<one sentence: what a visitor is left wondering>"}}]

RULES:
- Only real gaps a customer would care about. If the page is complete, reply [].
- NEVER invent facts or draft answers for the owner.
- Maximum 2 items per page, best first."""

_RETRY_SUFFIX = (
    "\n\nYour previous reply was not valid JSON. Reply again with ONLY the "
    "raw JSON, nothing else."
)

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)
_JSON_ARR_RE = re.compile(r"\[.*\]", re.S)


def _parse_json(text: str, expect_array: bool):
    if not text:
        return None
    # strip code fences if the model added them anyway
    text = re.sub(r"```(?:json)?", "", text).strip()
    m = (_JSON_ARR_RE if expect_array else _JSON_OBJ_RE).search(text)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except ValueError:
        return None
    return parsed


def _valid_item(item) -> bool:
    return (
        isinstance(item, dict)
        and str(item.get("question", "")).strip()
        and str(item.get("topic", "")).strip()
    )


def _clean_item(item, fallback_topic: str) -> dict:
    return {
        "topic": str(item.get("topic", "")).strip() or fallback_topic,
        "question": str(item.get("question", "")).strip(),
        "why": str(item.get("why", "")).strip(),
    }


def _fallback_item(visitor_question: str) -> dict:
    """Offline/mock providers (or a stubborn model) still yield a usable
    question — built deterministically from what the visitor actually asked."""
    q = visitor_question.strip().rstrip("?")
    return {
        "topic": (q[:40] + "...") if len(q) > 43 else q,
        "question": f'Visitors asked "{q}" but the site never addresses it — what should the AI say?',
        "why": "Auto-drafted fallback (provider returned no JSON).",
    }


def _llm_json(provider, system: str, user: str, expect_array: bool):
    """Ask for JSON, retry once with a stricter reminder, give up cleanly."""
    for attempt in (0, 1):
        try:
            text = provider.complete(system + (_RETRY_SUFFIX if attempt else ""), user)
        except Exception:
            return None
        parsed = _parse_json(text, expect_array)
        if parsed is not None:
            return parsed
    return None


def _draft(provider, system: str, user: str, expect_array: bool, fallback: list[dict]) -> list[dict]:
    parsed = _llm_json(provider, system, user, expect_array)
    if parsed is None:
        return fallback
    if expect_array:
        if not isinstance(parsed, list):
            return fallback
        return [_clean_item(i, "site gap") for i in parsed if _valid_item(i)]
    if not isinstance(parsed, dict) or not _valid_item(parsed):
        return fallback
    return [_clean_item(parsed, "site gap")]


def _topic_from(question: str) -> str:
    words = normalize_question(question).split()[:5]
    return " ".join(words) or "site gap"


def _context(db: Database, provider, vectors, site_id: str, question: str) -> str:
    """Raw top-k around the gap (no threshold — we WANT to see what the
    site has nearby so the AI drafts around it)."""
    if vectors is None or provider is None:
        return "(no index available)"
    try:
        [emb] = provider.embed([question])
        hits = vectors.query(site_id, emb, 4)
    except Exception:
        return "(index unavailable)"
    if not hits:
        return "(the site has nothing related to this topic)"
    blocks = [f"- {h['url']} | {h['title']}\n  {h['text'][:400]}" for h in hits]
    return "\n".join(blocks)


# --------------------------------------------------------------------------- #
# Consolidation: embedding recall -> LLM precision -> store merge
# --------------------------------------------------------------------------- #
def _validate_clusters(parsed, gap_map: dict[int, dict]) -> list[dict]:
    """Structural guarantees the LLM can't break:
    - every gap id lands in exactly one cluster (duplicates dropped)
    - any id the model dropped gets its own topic (nothing is lost)
    - malformed entries are ignored, not fatal."""
    clusters: list[dict] = []
    seen: set[int] = set()
    raw = parsed.get("clusters") if isinstance(parsed, dict) else None
    for c in (raw if isinstance(raw, list) else []):
        if not isinstance(c, dict):
            continue
        ids = [i for i in c.get("gap_ids", [])
               if isinstance(i, int) and i in gap_map and i not in seen]
        if not ids:
            continue
        seen.update(ids)
        action = "skip" if str(c.get("action", "")).lower() == "skip" else "draft"
        clusters.append({
            "topic": str(c.get("topic", "")).strip() or _topic_from(gap_map[ids[0]]["question"]),
            "action": action,
            "gap_ids": ids,
        })
    for gid, g in gap_map.items():  # dropped ids -> their own topic
        if gid not in seen:
            clusters.append({
                "topic": _topic_from(g["question"]),
                "action": "draft",
                "gap_ids": [gid],
            })
    return clusters


def consolidate_gaps(
    db: Database, provider, site: dict, top_n: int = CONSOLIDATE_INPUT
) -> list[dict]:
    """Group a site's open gaps by underlying meaning and merge duplicates.

    Returns clusters sorted by priority:
      [{"topic", "action", "gap_ids" (survivor first), "keep_id",
        "ask_count", "questions" (member texts)}]

    Side effect: multi-member clusters are MERGED in the store (counts
    summed, questions repointed) — `easylink gaps` shows one row per topic.
    """
    site_id = site["id"]
    backfill_embeddings(db, provider)  # legacy rows join semantic merging
    gaps = rank_gaps(db.list_gaps(site_id, CONSOLIDATE_INPUT, status="open"))
    if not gaps:
        return []
    gap_map = {g["id"]: g for g in gaps}

    if len(gaps) == 1:
        g = gaps[0]
        return [{
            "topic": _topic_from(g["question"]), "action": "draft",
            "gap_ids": [g["id"]], "keep_id": g["id"],
            "ask_count": int(g["count"] or 1), "questions": [g["question"]],
        }]

    # -- stage 1: embedding recall (cheap, over-groups on purpose) --------
    groups: list[list[int]] = cosine_groups(gaps)

    # -- stage 2: ONE LLM call regroups by meaning ------------------------
    user_lines = ["VISITOR QUESTIONS (open gaps, ranked by priority):"]
    for g in gaps:
        user_lines.append(f"id={g['id']} [asked x{int(g['count'] or 1)}] {g['question']}")
    multi = [ids for ids in groups if len(ids) > 1]
    if multi:
        user_lines.append("")
        user_lines.append("CANDIDATE GROUPS from embeddings (a HINT — regroup as you see fit):")
        for n, ids in enumerate(multi, 1):
            user_lines.append(f"G{n}: ids {ids}")
    pending = db.list_questions(site_id, "pending", limit=100)
    if pending:
        user_lines.append("")
        user_lines.append("PENDING QUESTIONS ALREADY DRAFTED (owner sees these):")
        for q in pending:
            user_lines.append(f"- {q['question']}")
    system = CONSOLIDATE_SYSTEM.format(site_name=site["name"], site_url=site["url"])
    parsed = _llm_json(provider, system, "\n".join(user_lines), expect_array=False)
    clusters = _validate_clusters(parsed, gap_map) if parsed else [
        {"topic": _topic_from(g["question"]), "action": "draft", "gap_ids": [g["id"]]}
        for g in gaps
    ]

    # -- stage 3: merge duplicates in the store, one row per topic --------
    rank_pos = {g["id"]: i for i, g in enumerate(gaps)}  # lower = higher priority
    out: list[dict] = []
    for c in clusters:
        ids = sorted(c["gap_ids"], key=lambda i: rank_pos.get(i, 1 << 30))
        keep, drop = ids[0], ids[1:]
        if drop:
            db.merge_gaps(keep, drop)
        out.append({
            "topic": c["topic"],
            "action": c["action"],
            "gap_ids": ids,
            "keep_id": keep,
            "ask_count": sum(int(gap_map[i]["count"] or 1) for i in ids),
            "questions": [gap_map[i]["question"] for i in ids],
        })
    out.sort(key=lambda c: c["ask_count"], reverse=True)
    return out


# --------------------------------------------------------------------------- #
# Drafted-question hygiene: no near-duplicate ever reaches the owner
# --------------------------------------------------------------------------- #
def _question_blobs(db: Database, provider, site_id: str, status: str = "pending") -> list:
    """Embeddings of the site's questions with `status` — backfilling any
    legacy rows first so old questions participate in dedupe too."""
    rows = db.list_questions(site_id, status, limit=500)
    if not rows:
        return []
    missing = [r for r in rows if r["embedding"] is None]
    if missing and provider is not None:
        try:
            vecs = provider.embed([r["question"] for r in missing])
        except Exception:
            vecs = [None] * len(missing)
        for r, vec in zip(missing, vecs):
            if vec is None:
                continue
            blob = np.asarray(vec, dtype=np.float32).tobytes()
            db._exec("UPDATE questions SET embedding=? WHERE id=?", (blob, r["id"]))
    blobs = []
    for r in rows:
        if r["embedding"] is not None:
            blobs.append(r["embedding"])
        else:
            blobs.append(_embedding_blob(provider, r["question"]))
    return blobs


def dedupe_pending(db: Database, provider, site_id: str, site: dict | None = None) -> int:
    """Hide redundant PENDING questions so the owner sees ONE per underlying ask.

    Two stages — the same embedding-recall -> LLM-precision split used by gap
    consolidation, because a single cosine threshold can't separate "same ask,
    different words" from "genuinely different" (real data: twins as low as
    0.62, distinct pairs from 0.56 — the bands OVERLAP, so the floor only
    bounds the cost of asking; the model decides correctness):

      stage 1  deterministic, no model call — exact-normalized twins, plus any
               pair at cosine >= NEAR_DUPLICATE_SIM (0.85) hide instantly.
               Pairs in [DEDUPE_RECALL_SIM, NEAR_DUPLICATE_SIM) become CANDIDATES.
      stage 2  ONE LLM call judges the candidates: same answer needed, or not?

    Newest question wins (the owner's freshest wording survives). Fail-safe: if
    the model call fails or returns garbage, only stage 1 applies — nothing is
    wrongly hidden, worst case a near-duplicate survives. Returns count hidden.
    """
    _question_blobs(db, provider, site_id)  # backfill legacy embeddings first
    rows = sorted(db.list_questions(site_id, "pending", limit=500),
                  key=lambda r: r["id"], reverse=True)  # newest first
    if len(rows) < 2:
        return 0

    blobs = {r["id"]: r["embedding"] for r in rows}
    norms = {r["id"]: normalize_question(r["question"]) for r in rows}
    id_text = {r["id"]: r["question"] for r in rows}

    kept: list[int] = []            # survivor ids, newest first
    hidden: set[int] = set()
    partner: dict[int, int] = {}     # hidden id -> the surviving twin
    candidates: list[tuple[int, int]] = []  # (survivor, candidate) in recall band

    for r in rows:
        qid, blob, norm = r["id"], blobs[r["id"]], norms[r["id"]]
        twin = next((k for k in kept if norm == norms[k]), None)
        if twin is not None:
            hidden.add(qid)                         # exact-normalized twin
            partner[qid] = twin
            continue
        band: list[tuple[float, int]] = []          # (cosine, survivor) at floor
        hide_with: int | None = None
        if blob is not None:
            vec = np.frombuffer(blob, dtype=np.float32)
            for k in kept:
                other = blobs[k]
                if other is None:
                    continue
                cos = vec_cosine(vec, other)
                if cos >= NEAR_DUPLICATE_SIM:
                    hide_with = k
                    break                           # near-identical: no judgement
                if cos >= DEDUPE_RECALL_SIM:
                    band.append((cos, k))
        if hide_with is not None:
            hidden.add(qid)
            partner[qid] = hide_with
            continue
        for _, k in sorted(band, reverse=True)[:2]:  # strongest 2, cost stays O(n)
            candidates.append((k, qid))
        kept.append(qid)

    # stage 2 — one call settles every ambiguous pair at once
    for hid in _confirm_same_ask(provider, site, id_text, candidates):
        hidden.add(hid)
        for s, c in candidates:                    # remember who survives
            if c == hid:
                partner[hid] = s
                break

    # The survivor inherits the hidden twin's visitor-gap link so the cluster
    # still counts as "covered" next run. If the survivor is ALREADY linked
    # to a different gap, those two gaps describe the same ask (that's why
    # their questions deduped) — merge them rather than dropping the link,
    # or the orphaned gap would be re-drafted forever.
    gap_of = {r["id"]: r["gap_id"] for r in rows}
    transfers: list[tuple[int, int]] = []          # (gap id -> survivor question)
    merges: list[tuple[int, int]] = []             # (keep gap, drop gap)
    for hid in hidden:
        surv = partner.get(hid)
        if surv is None:
            continue
        g_hid, g_surv = gap_of.get(hid), gap_of.get(surv)
        if not g_hid or g_hid == g_surv:
            continue                              # nothing to move / already same
        if g_surv:
            merges.append((g_surv, g_hid))
        else:
            transfers.append((g_hid, surv))

    for hid in hidden:
        db.set_question_status(hid, "duplicate")
    for g_hid, surv in transfers:
        db._exec("UPDATE questions SET gap_id=? WHERE id=?", (g_hid, surv))
    for g_surv, g_hid in merges:                   # duplicate questions imply
        db.merge_gaps(g_surv, [g_hid])             # duplicate gaps: unify
    return len(hidden)


def _confirm_same_ask(provider, site, id_text: dict,
                      pairs: list[tuple[int, int]]) -> set[int]:
    """One LLM call decides which recall-band pairs need the SAME owner answer.

    Returns the ids to hide — the older of each confirmed pair (newest wins).
    Any failure returns an empty set: we'd rather show a near-duplicate than
    wrongly hide a real question."""
    if not pairs or provider is None:
        return set()
    name = site["name"] if site else "this website"
    url = site["url"] if site else ""
    lines = []
    for n, (a, b) in enumerate(pairs, 1):
        lines.append(f"{n}. A: {id_text[a]}")
        lines.append(f"   B: {id_text[b]}")
    system = DEDUPE_SYSTEM.format(site_name=name, site_url=url)
    parsed = _llm_json(provider, system, "\n".join(lines), expect_array=False)
    if not isinstance(parsed, dict):
        return set()
    same = parsed.get("same", [])
    if not isinstance(same, list):
        return set()
    out: set[int] = set()
    for n in same:
        if isinstance(n, int) and 1 <= n <= len(pairs):
            a, b = pairs[n - 1]
            out.add(min(a, b))  # lower id = older = the one that loses
    return out


def _add_question_deduped(
    db: Database, provider, site_id: str, item: dict,
    gap_id: int | None, source: str, pending_blobs: list,
) -> int | None:
    """Insert a drafted question unless it reads like a PENDING one
    (embedding cosine >= NEAR_DUPLICATE_SIM). Deterministic only — the
    judgement calls happen in dedupe_pending's LLM stage, not per-insert.
    Returns the id or None (skipped)."""
    blob = _embedding_blob(provider, item["question"])
    if blob is not None and pending_blobs:
        vec = np.frombuffer(blob, dtype=np.float32)
        for other in pending_blobs:
            if other is not None and vec_cosine(vec, other) >= NEAR_DUPLICATE_SIM:
                return None
    qid = db.add_question(site_id, item["question"], item["topic"], item["why"],
                          gap_id=gap_id, source=source, embedding=blob)
    if blob is not None:
        pending_blobs.append(blob)
    return qid


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def generate_questions(
    db: Database,
    provider,
    vectors,
    settings,
    site: dict,
    top_n: int = 5,
    regen: bool = False,
) -> list[dict]:
    """Consolidate the site's gaps, then draft one question per uncovered topic.

    Covered = the cluster links a question that's already pending, OR a
    pending question reads like the cluster's questions (both the LLM's
    judgement and a deterministic embedding check — belt and braces)."""
    site_id = site["id"]
    if regen:
        for q in db.list_questions(site_id, "pending"):
            if q["gap_id"]:
                db.set_question_status(q["id"], "superseded")

    clusters = consolidate_gaps(db, provider, site)
    dedupe_pending(db, provider, site_id, site)

    pending_blobs = _question_blobs(db, provider, site_id)
    pending_gap_ids = {
        q["gap_id"] for q in db.list_questions(site_id, "pending") if q["gap_id"]
    }
    pending_texts = [
        normalize_question(q["question"])
        for q in db.list_questions(site_id, "pending", limit=500)
    ]

    drafted: list[dict] = []
    for c in clusters:
        if len(drafted) >= top_n:
            break
        # covered by an already-pending question? (deterministic gap link,
        # LLM's skip action, or an embedding/normalize twin of a pending one)
        if c["action"] == "skip":
            continue
        if any(i in pending_gap_ids for i in c["gap_ids"]):
            continue
        rep_blob = _embedding_blob(provider, c["questions"][0])
        covered = False
        if rep_blob is not None and pending_blobs:
            vec = np.frombuffer(rep_blob, dtype=np.float32)
            covered = any(
                other is not None and vec_cosine(vec, other) >= NEAR_DUPLICATE_SIM
                for other in pending_blobs
            )
        if covered or normalize_question(c["questions"][0]) in pending_texts:
            continue

        ctx = _context(db, provider, vectors, site_id, c["questions"][0])
        asked = "\n".join(
            f"- {q}" for q in c["questions"][:4]
        )
        user = (
            f"VISITOR QUESTIONS (all ask the same underlying thing; the site failed to answer):\n"
            f"{asked}\n\nTOPIC: {c['topic']}\n\n"
            f"WHAT THE SITE HAS NEAR THIS TOPIC:\n{ctx}"
        )
        items = _draft(
            provider,
            GEN_SYSTEM.format(site_name=site["name"], site_url=site["url"]),
            user,
            expect_array=False,
            fallback=[_fallback_item(c["questions"][0])],
        )
        item = items[0]
        qid = _add_question_deduped(
            db, provider, site_id, item, gap_id=c["keep_id"], source="gap",
            pending_blobs=pending_blobs,
        )
        if qid is None:
            continue  # the model re-drafted something the owner already has
        drafted.append({**item, "id": qid, "gap_count": c["ask_count"],
                        "merged_from": len(c["gap_ids"])})

    # final pass: fresh drafts can still land in the recall band (0.60-0.85)
    # against what was already pending — settle those before the owner looks.
    if drafted:
        dedupe_pending(db, provider, site_id, site)
        hidden_ids = {
            q["id"] for q in db.list_questions(site_id, "duplicate", limit=500)
        }
        drafted = [d for d in drafted if d["id"] not in hidden_ids]
    return drafted


def scan_site(
    db: Database, provider, settings, site: dict, max_pages: int = 10
) -> list[dict]:
    """Proactive mode: review crawled pages, draft what they FAIL to explain."""
    site_id = site["id"]
    rows = db._rows(
        "SELECT page_url, title, text FROM chunks WHERE site_id=? ORDER BY page_url, idx",
        (site_id,),
    )
    pages: dict[str, list] = {}
    for r in rows:
        pages.setdefault(r["page_url"], {"title": r["title"], "parts": []})["parts"].append(
            r["text"]
        )

    # exact-text bookkeeping + embedding-level dedupe vs everything pending
    existing = {
        q["question"].lower().strip()
        for q in db.list_questions(site_id, None, limit=500)
    }
    pending_blobs = _question_blobs(db, provider, site_id)

    drafted: list[dict] = []
    for page_url, data in list(pages.items())[:max_pages]:
        content = "\n".join(data["parts"])[:6000]
        user = (
            f"PAGE: {page_url}\nTITLE: {data['title']}\n\nCONTENT:\n{content}"
        )
        items = _draft(
            provider,
            SCAN_SYSTEM.format(site_name=site["name"], site_url=site["url"]),
            user,
            expect_array=True,
            fallback=[],
        )
        for item in items[:2]:
            if item["question"].lower().strip() in existing:
                continue
            qid = _add_question_deduped(
                db, provider, site_id, item, gap_id=None, source="scan",
                pending_blobs=pending_blobs,
            )
            if qid is None:
                continue  # near-duplicate of something already pending
            existing.add(item["question"].lower().strip())
            drafted.append({**item, "id": qid, "page": page_url})

    # same final pass as generate_questions: scan drafts get one chance to be
    # judged against the recall band before the owner sees them
    if drafted:
        dedupe_pending(db, provider, site_id, site)
        hidden_ids = {
            q["id"] for q in db.list_questions(site_id, "duplicate", limit=500)
        }
        drafted = [d for d in drafted if d["id"] not in hidden_ids]
    return drafted


def show_questions(db: Database, site: dict) -> str:
    site_id = site["id"]
    pending = db.list_questions(site_id, "pending")
    hidden = len(db.list_questions(site_id, "duplicate", limit=500))
    if not pending:
        msg = (
            f"No pending questions for {site['name']}. Generate some:\n"
            "  easylink questions            (from visitor gaps)\n"
            "  easylink questions --scan     (proactive site review)"
        )
        others = db.pending_by_site(exclude_site_id=site_id)
        for r in others:
            msg += (f"\n  -> {r['n']} waiting on '{r['name']}'  ->  "
                    f'easylink answer --site "{r["name"]}"')
        if hidden:
            msg += f"\n({hidden} near-duplicate question(s) hidden as 'duplicate')"
        return msg
    lines = [f"Pending questions for {site['name']} ({len(pending)}):", ""]
    for q in reversed(pending):
        src = "\U0001F464 visitor gap" if q["source"] == "gap" else "\U0001F50D site scan"
        extra = ""
        if q["gap_id"]:
            g = db._rows("SELECT count FROM gaps WHERE id=?", (q["gap_id"],))
            if g:
                extra = f" · asked x{int(g[0]['count'] or 1)}"
        lines.append(f"  #{q['id']}  [{src}{extra}]  {q['topic']}")
        lines.append(f"      Q: {q['question']}")
        if q["why"]:
            lines.append(f"      Why: {q['why']}")
        lines.append("")
    if hidden:
        lines.append(f"({hidden} near-duplicate question(s) hidden — consolidation at work)")
        lines.append("")
    lines.append("Answer one:  easylink answer --id <N>")
    lines.append("Or go interactive:  easylink answer")
    return "\n".join(lines)

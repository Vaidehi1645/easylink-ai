"""Gap log — the seed of EasyLink's differentiator.

Phase 3 turns raw abstentions into a RANKED, deduped question queue:

  normalize  -> exact repeats collapse immediately (count +1)
  embed      -> semantic twins merge at cosine >= 0.90 (instant, per-ask)
  groups     -> cosine >= 0.75 pre-clusters for the AI regrouping pass
                (embedding recall -> LLM precision; see questions.py)
  rank       -> frequency x recency x buying intent
  resolve    -> owner answers -> gap flips to 'resolved' (and can reopen
                if visitors keep hitting the same wall).
"""

from __future__ import annotations

import time

import numpy as np

from easylink.store import Database, normalize_question, vec_cosine

# Conservative on purpose: merging two DIFFERENT questions is worse than
# showing a near-duplicate twice (risk analysis, dedupe section).
DEDUPE_SIMILARITY = 0.90

# Recall threshold for CANDIDATE groups handed to the AI regrouping pass.
# Deliberately lower than DEDUPE_SIMILARITY: embeddings propose, the LLM
# decides — so we can afford to over-group here and let the model split.
GROUP_SIMILARITY = 0.75

# Buying intent = questions worth answering first.
_INTENT_WORDS = {
    "price", "pricing", "cost", "buy", "purchase", "order", "deliver",
    "delivery", "ship", "shipping", "return", "refund", "discount",
    "offer", "offers", "sale", "hours", "open", "contact", "support",
    "warranty", "stock", "available", "availability", "appointment",
    "payment", "pay", "book", "subscribe", "plan", "plans",
}


def normalize(question: str) -> str:
    """Exact dedupe key — shared with the store's migration so legacy rows
    and new asks always agree."""
    return normalize_question(question)


def intent_boost(question: str) -> float:
    """1.0 baseline, up to 1.6 for commercial questions."""
    words = set(normalize(question).split())
    hits = len(words & _INTENT_WORDS)
    return min(1.0 + 0.15 * hits, 1.6)


def recency_factor(last_seen: str | None) -> float:
    """Half-life of 14 days: fresh questions rank above stale ones."""
    if not last_seen:
        return 1.0
    try:
        then = time.mktime(time.strptime(last_seen[:19], "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return 1.0
    days = max(0.0, (time.time() - then) / 86400.0)
    return 0.5 ** (days / 14.0)


def score_gap(row) -> float:
    count = int(row["count"] or 1)
    return round(count * recency_factor(row["last_seen"]) * intent_boost(row["question"]), 1)


def _embedding_blob(provider, text: str) -> bytes | None:
    """Embed for semantic dedupe. Never fatal — a failed embed just means
    this gap falls back to exact-match dedupe only."""
    if provider is None:
        return None
    try:
        [vec] = provider.embed([text])
        arr = np.asarray(vec, dtype=np.float32)
        return arr.tobytes()
    except Exception:
        return None


def record_gap(
    db: Database, site_id: str, question: str, source: str, provider=None
) -> None:
    """source: 'no_match' | 'owner_downvote' | 'manual'

    Dedupe order: exact normalized match -> semantic (embedding) match ->
    insert new. A RESOLVED gap that gets re-asked REOPENS with count +1 —
    that's the loop telling you the first answer wasn't enough.
    """
    question = question.strip()
    if not question:
        return
    norm = normalize(question)

    row = db.find_gap_by_norm(site_id, norm)
    if row is None:
        blob = _embedding_blob(provider, norm)
        if blob is not None:
            row = db.find_gap_by_similarity(site_id, blob, DEDUPE_SIMILARITY)

    if row is not None:
        db.bump_gap(row["id"], reopen=(row["status"] != "open"))
        return

    db.insert_gap(site_id, question, norm, source, _embedding_blob(provider, norm))


def backfill_embeddings(db: Database, provider, limit: int = 200) -> int:
    """Legacy gaps (pre-Phase 3) have no embedding and can never
    semantic-merge. Embed up to `limit` of them per consolidation run —
    batched, so it's one provider call, not one per gap. Returns count."""
    if provider is None:
        return 0
    rows = db._rows(
        "SELECT id, norm FROM gaps WHERE embedding IS NULL LIMIT ?", (limit,)
    )
    if not rows:
        return 0
    try:
        vecs = provider.embed([r["norm"] or "" for r in rows])
    except Exception:
        return 0
    n = 0
    for r, vec in zip(rows, vecs):
        if vec is None:
            continue
        db._exec(
            "UPDATE gaps SET embedding=? WHERE id=?",
            (np.asarray(vec, dtype=np.float32).tobytes(), r["id"]),
        )
        n += 1
    return n


def cosine_groups(rows, threshold: float = GROUP_SIMILARITY) -> list[list[int]]:
    """Greedy single-pass pre-clustering over gap embeddings.

    Walks rows in (ranked) order; each row joins the FIRST group whose
    representative vector is >= threshold similar, else starts its own.
    O(n*k) — fine for the top-N gaps that feed consolidation. Gaps without
    embeddings become singleton groups (exact dedupe already covers them).
    Returns lists of gap ids.
    """
    groups: list[list] = []  # [ids, representative vec]
    for row in rows:
        blob = row["embedding"]
        placed = False
        if blob is not None:
            vec = np.frombuffer(blob, dtype=np.float32)
            for g in groups:
                if g[1] is not None and vec_cosine(vec, g[1]) >= threshold:
                    g[0].append(row["id"])
                    placed = True
                    break
        if not placed:
            groups.append([[row["id"]], np.frombuffer(blob, dtype=np.float32) if blob is not None else None])
    return [g[0] for g in groups]


def rank_gaps(rows) -> list:
    """Newest-first id order -> priority score order (desc)."""
    return sorted(rows, key=score_gap, reverse=True)


def show_gaps(
    db: Database,
    site_id: str | None = None,
    limit: int = 50,
    status: str | None = "open",
    resolved: bool = False,
) -> str:
    if resolved:
        status = "resolved"
    rows = db.list_gaps(site_id, limit, status)
    if not rows:
        if resolved:
            return "No resolved gaps yet — answer a question first: easylink answer"
        return "No open gaps — the AI has answered everything it was asked."

    ranked = rank_gaps(rows)
    stats = db.gap_stats(site_id) if site_id else None

    lines = []
    if status == "open":
        lines.append(f"Open knowledge gaps — ranked by priority ({len(ranked)} shown):")
    else:
        lines.append(f"Resolved knowledge gaps ({len(ranked)} shown):")
    lines.append("")
    for r in ranked:
        marker = "\u2139" if r["source"] == "no_match" else ("\U0001F44E" if r["source"] == "owner_downvote" else "\u270F")
        sc = score_gap(r)
        count = int(r["count"] or 1)
        count_txt = f" x{count}" if count > 1 else ""
        res = "" if (r["status"] or "open") == "open" else "  [resolved]"
        lines.append(
            f"  {marker} score {sc:>4}{count_txt:<6} [{r['last_seen'] or r['created_at']}] {r['question']}{res}"
        )
    if stats:
        lines.append("")
        lines.append(
            f"  stats: {stats['open']} open ({stats['open_asks']} asks) · "
            f"{stats['resolved']} resolved · facts on file: {db.count_facts(site_id)}"
        )
    lines.append("")
    lines.append("Next: easylink questions (draft answers to ask yourself) · easylink answer (teach the AI)")
    return "\n".join(lines)

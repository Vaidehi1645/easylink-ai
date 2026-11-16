"""Owner-verified facts — the self-improvement loop's payoff.

An answer typed by the owner is stored as a FACT with the highest
authority tier in the prompt. If retrieval fails but a fact could answer,
the model is allowed to answer from the fact alone (the owner said it —
that outranks page content and outranks "I don't know").
"""

from __future__ import annotations

import numpy as np

from easylink.store import Database

# Up to this many facts: include ALL (owner Q&A sets are short and precious).
# Beyond it: top-5 by similarity to the visitor's question.
ALL_FACTS_LIMIT = 50
TOP_K_FACTS = 5


def add_fact(db: Database, site_id: str, question: str, answer: str, provider=None) -> int:
    blob = None
    if provider is not None:
        try:
            [vec] = provider.embed([question])
            blob = np.asarray(vec, dtype=np.float32).tobytes()
        except Exception:
            blob = None  # fact still works; similarity ranking just skips it
    return db.add_fact(site_id, question, answer, blob)


def load_facts(db: Database | None, site_id: str, embedding=None) -> list[dict]:
    """Facts for the prompt: all (<=50) or top-k similar. Newest first."""
    if db is None:
        return []
    rows = db.list_facts(site_id)
    if not rows:
        return []
    facts = [{"question": r["question"], "answer": r["answer"]} for r in rows]
    if len(facts) <= ALL_FACTS_LIMIT:
        return facts
    if embedding is None:
        return facts[:TOP_K_FACTS]
    q = np.asarray(embedding, dtype=np.float32)
    scored = []
    for r in rows:
        if r["embedding"] is None:
            scored.append((0.0, r))
            continue
        v = np.frombuffer(r["embedding"], dtype=np.float32)
        if v.shape != q.shape:
            scored.append((0.0, r))
            continue
        denom = float(np.linalg.norm(q) * np.linalg.norm(v)) or 1.0
        scored.append((float(np.dot(q, v)) / denom, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [
        {"question": r["question"], "answer": r["answer"]}
        for _, r in scored[:TOP_K_FACTS]
    ]


def format_facts(facts: list[dict]) -> str:
    if not facts:
        return ""
    lines = []
    for i, f in enumerate(facts, 1):
        lines.append(f"[{i}] Q: {f['question']}\n    A: {f['answer']}")
    return "\n".join(lines)

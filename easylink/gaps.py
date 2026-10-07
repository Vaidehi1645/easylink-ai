"""Gap log — the seed of EasyLink's differentiator.

When the AI can't answer (layer 1 fails) or the owner flags an answer as
wrong, the question lands here. Phase 3 turns this into the ranked,
auto-generated question queue that makes the AI fill its own blind spots.
"""

from __future__ import annotations

from easylink.store import Database


def record_gap(db: Database, site_id: str, question: str, source: str) -> None:
    """source: 'no_match' (AI couldn't retrieve) | 'owner_downvote' | 'manual'"""
    db.add_gap(site_id, question.strip(), source)


def show_gaps(db: Database, site_id: str | None = None, limit: int = 50) -> str:
    rows = db.list_gaps(site_id, limit)
    if not rows:
        return "No gaps yet — the AI has answered everything it was asked."
    lines = [f"Open knowledge gaps ({len(rows)} shown):", ""]
    for r in rows:
        marker = "✗" if r["source"] == "no_match" else "👎"
        lines.append(f"  {marker} [{r['created_at']}] {r['question']}")
    lines.append("")
    lines.append("Next step (Phase 3): answer these in your dashboard and the AI learns.")
    return "\n".join(lines)

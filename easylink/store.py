"""Storage layer.

Two stores:
- SQLite   : sites, pages, chunks, gaps, eval runs (the record of truth)
- Vectors  : ChromaDB when available, otherwise a NumPy/SQLite fallback.
             BOTH are namespaced by site_id — tenant isolation from day one
             (risk register #1: no cross-tenant retrieval, ever).
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from easylink.config import CHROMA_DIR, DB_PATH


def normalize_question(question: str) -> str:
    """Canonical form for exact gap dedupe (shared by store + gaps.py)."""
    q = (question or "").lower().strip()
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _norm_host(netloc: str) -> str:
    """jewelsmars.com / www.Jewelsmars.com:443 -> jewelsmars.com"""
    h = (netloc or "").lower().split("@")[-1].split(":")[0].strip(".")
    return h[4:] if h.startswith("www.") else h


# TLDs dropped when building a readable /demo/<slug> base
_COMMON_TLDS = {
    "com", "net", "org", "io", "ai", "co", "in", "uk", "us", "ca", "de",
    "fr", "au", "jp", "br", "za", "nl", "it", "es", "ch", "se", "no",
    "fi", "pl", "ru", "cn", "kr", "sg", "hk", "nz", "ie", "mx", "app",
    "dev", "shop", "store", "xyz", "info", "biz", "me", "tv", "cc",
    "online", "site", "tech", "space", "fun",
}


def slug_base(url: str) -> str:
    """https://www.jewelsmars.com/some/path -> 'jewelsmars' (demo slug base)."""
    labels = [l for l in _norm_host(urlparse(url).netloc).split(".") if l]
    if len(labels) > 1 and labels[-1] in _COMMON_TLDS:
        labels = labels[:-1]
    base = re.sub(r"[^a-z0-9-]+", "-", "-".join(labels)).strip("-")
    return (base or "site")[:48]


def vec_cosine(a, b) -> float:
    """Cosine similarity between two float32 vectors given as blobs or arrays.

    Shared by every similarity decision in the codebase (gap dedupe, fact
    retrieval, question consolidation) so the math lives in exactly one place.
    """
    if a is None or b is None:
        return 0.0
    va = np.frombuffer(a, dtype=np.float32) if isinstance(a, (bytes, memoryview)) \
        else np.asarray(a, dtype=np.float32)
    vb = np.frombuffer(b, dtype=np.float32) if isinstance(b, (bytes, memoryview)) \
        else np.asarray(b, dtype=np.float32)
    if va.shape != vb.shape or va.size == 0:
        return 0.0
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb)) / denom

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sites (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    url         TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    page_count  INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS pages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id    TEXT NOT NULL,
    url        TEXT NOT NULL,
    title      TEXT DEFAULT '',
    content_hash TEXT NOT NULL,
    chars      INTEGER DEFAULT 0,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id       TEXT PRIMARY KEY,
    site_id  TEXT NOT NULL,
    page_url TEXT NOT NULL,
    title    TEXT DEFAULT '',
    idx      INTEGER NOT NULL,
    text     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gaps (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id    TEXT NOT NULL,
    question   TEXT NOT NULL,
    source     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id      TEXT NOT NULL,
    ran_at       TEXT NOT NULL,
    total        INTEGER,
    retrieved_ok INTEGER,
    answered     INTEGER,
    abstained    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_site ON chunks(site_id);
CREATE INDEX IF NOT EXISTS idx_gaps_site ON gaps(site_id);
CREATE TABLE IF NOT EXISTS usage (
    site_id TEXT NOT NULL,
    day     TEXT NOT NULL,
    count   INTEGER DEFAULT 0,
    PRIMARY KEY (site_id, day)
);
CREATE TABLE IF NOT EXISTS questions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id    TEXT NOT NULL,
    gap_id     INTEGER,
    topic      TEXT DEFAULT '',
    question   TEXT NOT NULL,
    why        TEXT DEFAULT '',
    source     TEXT DEFAULT 'gap',
    status     TEXT DEFAULT 'pending',
    embedding  BLOB,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_questions_site ON questions(site_id);
CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    site_id    TEXT NOT NULL,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    source     TEXT DEFAULT 'owner',
    embedding  BLOB,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_site ON facts(site_id);
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    email       TEXT DEFAULT '',
    name        TEXT DEFAULT '',
    picture     TEXT DEFAULT '',
    created_at  TEXT NOT NULL,
    UNIQUE (provider, provider_id)
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Database:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + lock: the API server uses this connection
        # from multiple request threads (Phase 1 was single-threaded CLI).
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(_SCHEMA)
            self.conn.execute("PRAGMA busy_timeout=5000")
            # migration: widget tokens (Phase 2)
            cols = [r[1] for r in self.conn.execute("PRAGMA table_info(sites)")]
            if "token" not in cols:
                self.conn.execute("ALTER TABLE sites ADD COLUMN token TEXT")
            # migration: shareable demo slugs (Phase 6)
            if "slug" not in cols:
                self.conn.execute("ALTER TABLE sites ADD COLUMN slug TEXT")
            # migration: gap dedupe/ranking (Phase 3)
            gap_cols = [r[1] for r in self.conn.execute("PRAGMA table_info(gaps)")]
            for col, decl in (
                ("count", "INTEGER DEFAULT 1"),
                ("last_seen", "TEXT"),
                ("status", "TEXT DEFAULT 'open'"),
                ("norm", "TEXT"),
                ("embedding", "BLOB"),
            ):
                if col not in gap_cols:
                    self.conn.execute(f"ALTER TABLE gaps ADD COLUMN {col} {decl}")
            # migration: drafted-question embeddings (Phase 3 consolidation)
            q_cols = [r[1] for r in self.conn.execute("PRAGMA table_info(questions)")]
            if "embedding" not in q_cols:
                self.conn.execute("ALTER TABLE questions ADD COLUMN embedding BLOB")
            # backfill legacy rows (pre-Phase 3) so dedupe works on them too
            self.conn.execute(
                "UPDATE gaps SET count=1 WHERE count IS NULL"
            )
            self.conn.execute(
                "UPDATE gaps SET last_seen=created_at WHERE last_seen IS NULL"
            )
            self.conn.execute(
                "UPDATE gaps SET status='open' WHERE status IS NULL"
            )
            # self-heal: every gap's norm must match normalize_question()
            # (legacy rows were backfilled with a weaker normalizer)
            for r in self.conn.execute("SELECT id, question, norm FROM gaps").fetchall():
                n = normalize_question(r["question"])
                if r["norm"] != n:
                    self.conn.execute("UPDATE gaps SET norm=? WHERE id=?", (n, r["id"]))
            # collapse duplicates that predate dedupe (keep newest, sum counts,
            # repoint drafted questions, keep the first available embedding)
            groups: dict[tuple, list[int]] = {}
            for r in self.conn.execute(
                "SELECT id, site_id, norm FROM gaps ORDER BY id"
            ).fetchall():
                groups.setdefault((r["site_id"], r["norm"]), []).append(r["id"])
            for ids in groups.values():
                if len(ids) < 2:
                    continue
                keep, drop = ids[-1], ids[:-1]
                total, last_seen, emb = 0, "", None
                for gid in ids:
                    g = self.conn.execute(
                        "SELECT count, last_seen, embedding FROM gaps WHERE id=?",
                        (gid,),
                    ).fetchone()
                    total += int(g["count"] or 1)
                    last_seen = max(last_seen, g["last_seen"] or "")
                    if emb is None and g["embedding"] is not None:
                        emb = g["embedding"]
                self.conn.execute(
                    "UPDATE gaps SET count=?, last_seen=?, embedding=? WHERE id=?",
                    (total, last_seen, emb, keep),
                )
                marks = ",".join("?" * len(drop))
                self.conn.execute(f"DELETE FROM gaps WHERE id IN ({marks})", tuple(drop))
                self.conn.execute(
                    f"UPDATE questions SET gap_id=? WHERE gap_id IN ({marks})",
                    (keep, *tuple(drop)),
                )
            self.conn.commit()

    def _rows(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    # -- sites ------------------------------------------------------------
    def create_site(self, url: str, name: str = "") -> str:
        site_id = uuid.uuid4().hex[:12]
        self._exec(
            "INSERT INTO sites (id, name, url, created_at) VALUES (?,?,?,?)",
            (site_id, name or url, url, now_iso()),
        )
        return site_id

    def update_site_counts(self, site_id: str, pages: int, chunks: int) -> None:
        self._exec(
            "UPDATE sites SET page_count=?, chunk_count=? WHERE id=?",
            (pages, chunks, site_id),
        )

    def list_sites(self) -> list[sqlite3.Row]:
        return self._rows("SELECT * FROM sites ORDER BY created_at DESC")

    def get_site(self, site_id: str | None = None) -> sqlite3.Row | None:
        """site_id, 'latest', or None -> most recent site.

        A name that was reused (re-crawl with the same --name) resolves to
        the NEWEST row — consistent with the 'latest' default, so you never
        silently land on a stale duplicate."""
        if not site_id or site_id == "latest":
            rows = self._rows(
                "SELECT * FROM sites ORDER BY created_at DESC, rowid DESC LIMIT 1"
            )
        else:
            rows = self._rows(
                "SELECT * FROM sites WHERE id=? OR name=?"
                " ORDER BY created_at DESC, rowid DESC",
                (site_id, site_id),
            )
        return rows[0] if rows else None

    # -- widget tokens (Phase 2) ------------------------------------------
    def get_or_create_token(self, site_id: str) -> str:
        row = self._rows("SELECT token FROM sites WHERE id=?", (site_id,))[0]
        if row["token"]:
            return row["token"]
        import secrets

        token = secrets.token_hex(24)
        self._exec("UPDATE sites SET token=? WHERE id=?", (token, site_id))
        return token

    def resolve_token(self, token: str) -> sqlite3.Row | None:
        if not token or len(token) < 16:
            return None
        rows = self._rows("SELECT * FROM sites WHERE token=?", (token,))
        return rows[0] if rows else None

    # -- demo slugs + host reuse (Phase 6) --------------------------------
    def find_site_by_host(self, host: str) -> sqlite3.Row | None:
        """Newest site crawled from this host that actually has content
        (www-insensitive). Lets /ingest reuse an existing crawl instead of
        paying for a second one — same site, same demo link, no new cost."""
        want = _norm_host(host)
        if not want:
            return None
        for row in self._rows(
            "SELECT * FROM sites ORDER BY created_at DESC, rowid DESC"
        ):
            if int(row["chunk_count"] or 0) > 0 and _norm_host(
                urlparse(row["url"]).netloc
            ) == want:
                return row
        return None

    def page_titles(self, site_id: str, limit: int = 20) -> list[str]:
        """Titles of this site's pages (what a crawl 'learned'), for the
        ingest done event — the visitor sees exactly what the bot knows."""
        rows = self._rows(
            "SELECT title FROM pages WHERE site_id=? ORDER BY rowid LIMIT ?",
            (site_id, limit),
        )
        return [r["title"] for r in rows if r["title"]]

    def get_site_by_slug(self, slug: str) -> sqlite3.Row | None:
        if not slug or len(slug) > 80:
            return None
        rows = self._rows("SELECT * FROM sites WHERE slug=?", (slug,))
        return rows[0] if rows else None

    def ensure_slug(self, site_id: str) -> str:
        """Claim (or return) this site's readable /demo/<slug>.

        Base comes from the domain — jewelsmars.com -> 'jewelsmars' — with a
        numeric suffix on collision ('jewelsmars-2'). Stored once under the
        DB lock, so the link a user shares never changes."""
        with self.lock:
            rows = self._rows("SELECT * FROM sites WHERE id=?", (site_id,))
            if not rows:
                raise ValueError(f"unknown site {site_id}")
            if rows[0]["slug"]:
                return rows[0]["slug"]
            base = slug_base(rows[0]["url"])
            slug, n = base, 1
            while self._rows(
                "SELECT id FROM sites WHERE slug=? AND id!=?", (slug, site_id)
            ):
                n += 1
                slug = f"{base}-{n}"
            self._exec("UPDATE sites SET slug=? WHERE id=?", (slug, site_id))
            return slug

    # -- daily usage quota -------------------------------------------------
    def usage_today(self, site_id: str) -> int:
        rows = self._rows(
            "SELECT count FROM usage WHERE site_id=? AND day=?", (site_id, time.strftime("%Y-%m-%d"))
        )
        return int(rows[0]["count"]) if rows else 0

    def bump_usage(self, site_id: str) -> int:
        day = time.strftime("%Y-%m-%d")
        with self.lock:
            self.conn.execute(
                "INSERT INTO usage (site_id, day, count) VALUES (?,?,1) "
                "ON CONFLICT(site_id, day) DO UPDATE SET count=count+1",
                (site_id, day),
            )
            self.conn.commit()
            rows = self.conn.execute(
                "SELECT count FROM usage WHERE site_id=? AND day=?", (site_id, day)
            ).fetchone()
        return int(rows["count"])

    # -- pages / chunks ---------------------------------------------------
    def add_page(self, site_id: str, url: str, title: str, content_hash: str, chars: int) -> None:
        self._exec(
            "INSERT INTO pages (site_id, url, title, content_hash, chars, fetched_at) "
            "VALUES (?,?,?,?,?,?)",
            (site_id, url, title, content_hash, chars, now_iso()),
        )

    def add_chunks(self, rows: list[tuple]) -> None:
        with self.lock:
            self.conn.executemany(
                "INSERT OR IGNORE INTO chunks (id, site_id, page_url, title, idx, text) "
                "VALUES (?,?,?,?,?,?)",
                rows,
            )
            self.conn.commit()

    # -- gaps (Phase 3: dedupe + count + status) --------------------------
    def find_gap_by_norm(self, site_id: str, norm: str) -> sqlite3.Row | None:
        rows = self._rows(
            "SELECT * FROM gaps WHERE site_id=? AND norm=? ORDER BY status='open' DESC, id DESC",
            (site_id, norm),
        )
        return rows[0] if rows else None

    def find_gap_by_similarity(
        self, site_id: str, vec: np.ndarray, threshold: float = 0.90
    ) -> sqlite3.Row | None:
        """Cosine vs existing embedded gaps (open first). NULLs skipped."""
        rows = self._rows(
            "SELECT * FROM gaps WHERE site_id=? AND embedding IS NOT NULL "
            "ORDER BY status='open' DESC, id DESC",
            (site_id,),
        )
        for row in rows:
            if vec_cosine(row["embedding"], vec) >= threshold:
                return row
        return None

    def bump_gap(self, gap_id: int, reopen: bool = False) -> None:
        """Repeat question: +1 count, refresh last_seen (reopen if resolved)."""
        if reopen:
            self._exec(
                "UPDATE gaps SET count=count+1, last_seen=?, status='open' WHERE id=?",
                (now_iso(), gap_id),
            )
        else:
            self._exec(
                "UPDATE gaps SET count=count+1, last_seen=? WHERE id=?",
                (now_iso(), gap_id),
            )

    def insert_gap(
        self,
        site_id: str,
        question: str,
        norm: str,
        source: str,
        embedding: bytes | None = None,
    ) -> int:
        cur = self._exec(
            "INSERT INTO gaps (site_id, question, norm, source, created_at, count, last_seen, status, embedding) "
            "VALUES (?,?,?,?,?,1,?,'open',?)",
            (site_id, question.strip(), norm, source, now_iso(), now_iso(), embedding),
        )
        return int(cur.lastrowid)

    def resolve_gap(self, gap_id: int) -> None:
        self._exec(
            "UPDATE gaps SET status='resolved' WHERE id=?", (gap_id,)
        )

    def merge_gaps(self, keep_id: int, drop_ids: list[int]) -> None:
        """Collapse duplicate gap rows into one: counts summed, newest
        last_seen wins, the first available embedding is kept, drafted
        questions are repointed at the survivor. Used by consolidation and
        by the startup self-heal (pre-dedupe legacy rows)."""
        if not drop_ids:
            return
        ids = [keep_id, *drop_ids]
        total, last_seen, emb = 0, "", None
        for gid in ids:
            rows = self._rows(
                "SELECT count, last_seen, embedding FROM gaps WHERE id=?", (gid,)
            )
            if not rows:
                continue
            g = rows[0]
            total += int(g["count"] or 1)
            last_seen = max(last_seen, g["last_seen"] or "")
            if emb is None and g["embedding"] is not None:
                emb = g["embedding"]
        self._exec(
            "UPDATE gaps SET count=?, last_seen=?, embedding=? WHERE id=?",
            (total, last_seen, emb, keep_id),
        )
        marks = ",".join("?" * len(drop_ids))
        self._exec(f"DELETE FROM gaps WHERE id IN ({marks})", tuple(drop_ids))
        self._exec(
            f"UPDATE questions SET gap_id=? WHERE gap_id IN ({marks})",
            (keep_id, *tuple(drop_ids)),
        )

    def list_gaps(
        self,
        site_id: str | None = None,
        limit: int = 50,
        status: str | None = None,
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM gaps WHERE 1=1"
        params: list = []
        if site_id:
            sql += " AND site_id=?"
            params.append(site_id)
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return self._rows(sql, tuple(params))

    def gap_stats(self, site_id: str) -> dict:
        rows = self._rows(
            "SELECT status, COUNT(*) AS n, COALESCE(SUM(count),0) AS asks "
            "FROM gaps WHERE site_id=? GROUP BY status",
            (site_id,),
        )
        out = {"open": 0, "resolved": 0, "open_asks": 0, "resolved_asks": 0}
        for r in rows:
            key = r["status"] or "open"
            out[key] = int(r["n"])
            out[f"{key}_asks"] = int(r["asks"])
        return out

    # -- owner-drafted questions ------------------------------------------
    def add_question(
        self,
        site_id: str,
        question: str,
        topic: str,
        why: str,
        gap_id: int | None = None,
        source: str = "gap",
        embedding: bytes | None = None,
    ) -> int:
        cur = self._exec(
            "INSERT INTO questions (site_id, gap_id, topic, question, why, source, status, embedding, created_at) "
            "VALUES (?,?,?,?,?,?,'pending',?,?)",
            (site_id, gap_id, topic, question, why, source, embedding, now_iso()),
        )
        return int(cur.lastrowid)

    def list_questions(
        self, site_id: str, status: str | None = "pending", limit: int = 50
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM questions WHERE site_id=?"
        params: list = [site_id]
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return self._rows(sql, tuple(params))

    def get_question(self, qid: int) -> sqlite3.Row | None:
        rows = self._rows("SELECT * FROM questions WHERE id=?", (qid,))
        return rows[0] if rows else None

    def set_question_status(self, qid: int, status: str) -> None:
        self._exec("UPDATE questions SET status=? WHERE id=?", (status, qid))

    def pending_by_site(self, exclude_site_id: str | None = None) -> list[sqlite3.Row]:
        """(site name, pending count) for every site that HAS pending questions.

        Used by `answer`/`questions` when the DEFAULT site (latest created)
        has nothing waiting — so we can point the owner at the site that does
        instead of a dead-end 'nothing here' message."""
        sql = ("SELECT s.name AS name, COUNT(*) AS n FROM questions q "
               "JOIN sites s ON s.id = q.site_id WHERE q.status='pending'")
        params: list = []
        if exclude_site_id:
            sql += " AND q.site_id != ?"
            params.append(exclude_site_id)
        sql += " GROUP BY q.site_id ORDER BY n DESC"
        return self._rows(sql, tuple(params))

    # -- owner-verified facts ---------------------------------------------
    def add_fact(
        self, site_id: str, question: str, answer: str, embedding: bytes | None = None
    ) -> int:
        cur = self._exec(
            "INSERT INTO facts (site_id, question, answer, embedding, created_at) VALUES (?,?,?,?,?)",
            (site_id, question.strip(), answer.strip(), embedding, now_iso()),
        )
        return int(cur.lastrowid)

    def list_facts(self, site_id: str) -> list[sqlite3.Row]:
        return self._rows(
            "SELECT * FROM facts WHERE site_id=? ORDER BY id DESC", (site_id,)
        )

    def count_facts(self, site_id: str) -> int:
        rows = self._rows(
            "SELECT COUNT(*) AS n FROM facts WHERE site_id=?", (site_id,)
        )
        return int(rows[0]["n"])

    # -- auth: owners + sessions ------------------------------------------
    def upsert_user(
        self, provider: str, provider_id: str, email: str, name: str, picture: str
    ) -> str:
        """Find or create the account for a provider identity; returns user id."""
        with self.lock:
            row = self._exec(
                "SELECT id FROM users WHERE provider=? AND provider_id=?",
                (provider, str(provider_id)),
            ).fetchone()
            if row:
                user_id = str(row["id"])
                self._exec(
                    "UPDATE users SET email=?, name=?, picture=? WHERE id=?",
                    (email or "", name or "", picture or "", user_id),
                )
                return user_id
            user_id = uuid.uuid4().hex[:16]
            self._exec(
                "INSERT INTO users (id, provider, provider_id, email, name, picture, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (user_id, provider, str(provider_id), email or "", name or "",
                 picture or "", now_iso()),
            )
            return user_id

    def create_session(self, token_hash: str, user_id: str, days: int) -> None:
        with self.lock:
            self._exec(
                "INSERT OR REPLACE INTO sessions (token_hash, user_id, created_at, expires_at) "
                "VALUES (?,?,?,?)",
                (token_hash, user_id, now_iso(),
                 time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + days * 86400))),
            )

    def session_user(self, token_hash: str) -> sqlite3.Row | None:
        """The signed-in user for a session token, or None (missing/expired)."""
        rows = self._rows(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash=? AND s.expires_at > ?",
            (token_hash, now_iso()),
        )
        return rows[0] if rows else None

    def delete_session(self, token_hash: str) -> None:
        with self.lock:
            self._exec("DELETE FROM sessions WHERE token_hash=?", (token_hash,))

    # -- evals ------------------------------------------------------------
    def add_eval_run(self, site_id: str, total: int, ok: int, answered: int, abstained: int) -> None:
        self._exec(
            "INSERT INTO eval_runs (site_id, ran_at, total, retrieved_ok, answered, abstained) "
            "VALUES (?,?,?,?,?,?)",
            (site_id, now_iso(), total, ok, answered, abstained),
        )


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
class VectorStore:
    """Interface: everything is namespaced by site_id."""

    backend: str = "?"

    def add(self, site_id, ids, embeddings, documents, metadatas) -> None:
        raise NotImplementedError

    def query(self, site_id: str, embedding: list[float], k: int) -> list[dict]:
        """Returns [{id, score, text, url, title}] sorted by score desc."""
        raise NotImplementedError

    def count(self, site_id: str) -> int:
        raise NotImplementedError

    def delete_site(self, site_id: str) -> None:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    backend = "chroma"

    def __init__(self):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    def _coll(self, site_id: str):
        return self._client.get_or_create_collection(
            name=f"site_{site_id}", metadata={"hnsw:space": "cosine"}
        )

    def add(self, site_id, ids, embeddings, documents, metadatas) -> None:
        self._coll(site_id).add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(self, site_id: str, embedding: list[float], k: int) -> list[dict]:
        coll = self._coll(site_id)
        if coll.count() == 0:
            return []
        try:
            res = coll.query(
                query_embeddings=[embedding], n_results=min(k, coll.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            if "dimension" in str(exc).lower():
                raise RuntimeError(
                    "This site was indexed with a DIFFERENT AI provider than the "
                    "one now in use (vector sizes don't match). Either switch "
                    "EASYLINK_PROVIDER back in .env, or re-crawl this site with "
                    "the current provider."
                ) from exc
            raise
        out = []
        for i in range(len(res["ids"][0])):
            meta = (res["metadatas"][0] or [{}])[i] or {}
            out.append({
                "id": res["ids"][0][i],
                "score": 1.0 - float(res["distances"][0][i]),
                "text": res["documents"][0][i],
                "url": meta.get("url", ""),
                "title": meta.get("title", ""),
            })
        return out

    def count(self, site_id: str) -> int:
        try:
            return self._coll(site_id).count()
        except Exception:
            return 0

    def delete_site(self, site_id: str) -> None:
        try:
            self._client.delete_collection(f"site_{site_id}")
        except Exception:
            pass


class NumpyVectorStore(VectorStore):
    """Fallback backend: vectors as BLOBs in SQLite, brute-force cosine.

    Perfect for Phase 1 scale (hundreds-to-thousands of chunks).
    """
    backend = "numpy"

    def __init__(self, db: Database):
        self.db = db
        with db.lock:
            db.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    id      TEXT PRIMARY KEY,
                    site_id TEXT NOT NULL,
                    dim     INTEGER NOT NULL,
                    vec     BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_vectors_site ON vectors(site_id);
                """
            )
            db.conn.commit()

    def add(self, site_id, ids, embeddings, documents, metadatas) -> None:
        rows = []
        for _id, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            vec = np.asarray(emb, dtype=np.float32)
            norm = np.linalg.norm(vec) or 1.0
            vec = vec / norm
            rows.append((_id, site_id, vec.size, vec.tobytes()))
        with self.db.lock:
            self.db.conn.executemany(
                "INSERT OR IGNORE INTO vectors (id, site_id, dim, vec) VALUES (?,?,?,?)", rows
            )
            self.db.conn.commit()

    def query(self, site_id: str, embedding: list[float], k: int) -> list[dict]:
        rows = self.db._rows(
            "SELECT v.id, v.vec, c.page_url, c.title, c.text FROM vectors v "
            "JOIN chunks c ON c.id = v.id WHERE v.site_id=?",
            (site_id,),
        )
        if not rows:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scored = []
        for row in rows:
            vec = np.frombuffer(row["vec"], dtype=np.float32)
            score = float(np.dot(q, vec))
            scored.append({
                "id": row["id"],
                "score": score,
                "text": row["text"],
                "url": row["page_url"],
                "title": row["title"],
            })
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:k]

    def count(self, site_id: str) -> int:
        cur = self.db._rows(
            "SELECT COUNT(*) AS n FROM vectors WHERE site_id=?", (site_id,)
        )
        return int(cur[0]["n"])

    def delete_site(self, site_id: str) -> None:
        self.db._exec("DELETE FROM vectors WHERE site_id=?", (site_id,))


def get_vector_store(db: Database) -> VectorStore:
    """Prefer ChromaDB; fall back cleanly if it can't start."""
    try:
        return ChromaVectorStore()
    except Exception:
        return NumpyVectorStore(db)

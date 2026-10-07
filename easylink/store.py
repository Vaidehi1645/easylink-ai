"""Storage layer.

Two stores:
- SQLite   : sites, pages, chunks, gaps, eval runs (the record of truth)
- Vectors  : ChromaDB when available, otherwise a NumPy/SQLite fallback.
             BOTH are namespaced by site_id — tenant isolation from day one
             (risk register #1: no cross-tenant retrieval, ever).
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np

from easylink.config import CHROMA_DIR, DB_PATH

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
"""


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Database:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- sites ------------------------------------------------------------
    def create_site(self, url: str, name: str = "") -> str:
        site_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO sites (id, name, url, created_at) VALUES (?,?,?,?)",
            (site_id, name or url, url, now_iso()),
        )
        self.conn.commit()
        return site_id

    def update_site_counts(self, site_id: str, pages: int, chunks: int) -> None:
        self.conn.execute(
            "UPDATE sites SET page_count=?, chunk_count=? WHERE id=?",
            (pages, chunks, site_id),
        )
        self.conn.commit()

    def list_sites(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM sites ORDER BY created_at DESC"
        ).fetchall()

    def get_site(self, site_id: str | None = None) -> sqlite3.Row | None:
        """site_id, 'latest', or None -> most recent site."""
        if not site_id or site_id == "latest":
            return self.conn.execute(
                "SELECT * FROM sites ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return self.conn.execute(
            "SELECT * FROM sites WHERE id=? OR name=?", (site_id, site_id)
        ).fetchone()

    # -- pages / chunks ---------------------------------------------------
    def add_page(self, site_id: str, url: str, title: str, content_hash: str, chars: int) -> None:
        self.conn.execute(
            "INSERT INTO pages (site_id, url, title, content_hash, chars, fetched_at) "
            "VALUES (?,?,?,?,?,?)",
            (site_id, url, title, content_hash, chars, now_iso()),
        )
        self.conn.commit()

    def add_chunks(self, rows: list[tuple]) -> None:
        self.conn.executemany(
            "INSERT OR IGNORE INTO chunks (id, site_id, page_url, title, idx, text) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    # -- gaps -------------------------------------------------------------
    def add_gap(self, site_id: str, question: str, source: str) -> None:
        self.conn.execute(
            "INSERT INTO gaps (site_id, question, source, created_at) VALUES (?,?,?,?)",
            (site_id, question, source, now_iso()),
        )
        self.conn.commit()

    def list_gaps(self, site_id: str | None = None, limit: int = 50) -> list[sqlite3.Row]:
        if site_id:
            q = "SELECT * FROM gaps WHERE site_id=? ORDER BY id DESC LIMIT ?"
            return self.conn.execute(q, (site_id, limit)).fetchall()
        return self.conn.execute(
            "SELECT * FROM gaps ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    # -- evals ------------------------------------------------------------
    def add_eval_run(self, site_id: str, total: int, ok: int, answered: int, abstained: int) -> None:
        self.conn.execute(
            "INSERT INTO eval_runs (site_id, ran_at, total, retrieved_ok, answered, abstained) "
            "VALUES (?,?,?,?,?,?)",
            (site_id, now_iso(), total, ok, answered, abstained),
        )
        self.conn.commit()


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
        self.db.conn.executescript(
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
        self.db.conn.commit()

    def add(self, site_id, ids, embeddings, documents, metadatas) -> None:
        rows = []
        for _id, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            vec = np.asarray(emb, dtype=np.float32)
            norm = np.linalg.norm(vec) or 1.0
            vec = vec / norm
            rows.append((_id, site_id, vec.size, vec.tobytes()))
        self.db.conn.executemany(
            "INSERT OR IGNORE INTO vectors (id, site_id, dim, vec) VALUES (?,?,?,?)", rows
        )
        self.db.conn.commit()

    def query(self, site_id: str, embedding: list[float], k: int) -> list[dict]:
        rows = self.db.conn.execute(
            "SELECT v.id, v.vec, c.page_url, c.title, c.text FROM vectors v "
            "JOIN chunks c ON c.id = v.id WHERE v.site_id=?",
            (site_id,),
        ).fetchall()
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
        cur = self.db.conn.execute(
            "SELECT COUNT(*) FROM vectors WHERE site_id=?", (site_id,)
        )
        return int(cur.fetchone()[0])

    def delete_site(self, site_id: str) -> None:
        self.db.conn.execute("DELETE FROM vectors WHERE site_id=?", (site_id,))
        self.db.conn.commit()


def get_vector_store(db: Database) -> VectorStore:
    """Prefer ChromaDB; fall back cleanly if it can't start."""
    try:
        return ChromaVectorStore()
    except Exception:
        return NumpyVectorStore(db)

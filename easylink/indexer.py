"""Chunking + embedding pipeline.

Chunking strategy matters more to answer quality than the model choice:
- split on headings/paragraphs, never mid-sentence
- ~chunk_chars with overlap so answers spanning a boundary still retrieve
- every chunk carries its source (url + title) for citations
"""

from __future__ import annotations

from dataclasses import dataclass

from easylink.config import Settings
from easylink.crawler import Page
from easylink.providers import BaseProvider
from easylink.store import Database, VectorStore


@dataclass
class Chunk:
    id: str
    site_id: str
    url: str
    title: str
    idx: int
    text: str


def chunk_page(page: Page, site_id: str, page_idx: int, settings: Settings) -> list[Chunk]:
    # Split into atomic pieces on paragraph breaks, keeping headings attached.
    paragraphs = [p.strip() for p in page.text.split("\n") if p.strip()]
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n{para}".strip()
        if len(candidate) <= settings.chunk_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            # Hard-split a single monster paragraph
            while len(para) > settings.chunk_chars:
                pieces.append(para[: settings.chunk_chars])
                para = para[settings.chunk_chars - settings.chunk_overlap :]
            current = para
    if current:
        pieces.append(current)

    # Overlap: prepend the tail of the previous piece
    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        if i > 0 and settings.chunk_overlap > 0:
            tail = pieces[i - 1][-settings.chunk_overlap :]
            cut = tail.find(" ")
            if cut != -1:
                tail = tail[cut + 1 :]
            if tail and tail.lower() not in piece.lower():
                piece = f"...{tail}\n{piece}"
        chunks.append(
            Chunk(
                id=f"{site_id}:{page_idx}:{i}",
                site_id=site_id,
                url=page.url,
                title=page.title,
                idx=i,
                text=piece,
            )
        )
    return chunks


def index_pages(
    site_id: str,
    pages: list[Page],
    provider: BaseProvider,
    db: Database,
    vectors: VectorStore,
    settings: Settings,
    log=print,
) -> tuple[int, int]:
    """Chunk -> embed (batched) -> persist. Returns (page_count, chunk_count)."""
    all_chunks: list[Chunk] = []
    seen_text: set[str] = set()  # near-identical boilerplate across pages
    for pi, page in enumerate(pages):
        db.add_page(site_id, page.url, page.title, page.content_hash, len(page.text))
        for chunk in chunk_page(page, site_id, pi, settings):
            # Skip duplicate chunk text (site-wide banners, repeated notices):
            # identical text indexed once keeps retrieval precise and embeddings cheap.
            key = " ".join(chunk.text.split()).lower()
            if key in seen_text:
                continue
            seen_text.add(key)
            all_chunks.append(chunk)

    if not all_chunks:
        return len(pages), 0

    total = 0
    batch = settings.embed_batch_size
    for i in range(0, len(all_chunks), batch):
        group = all_chunks[i : i + batch]
        embeddings = provider.embed([c.text for c in group])
        db.add_chunks(
            [(c.id, c.site_id, c.url, c.title, c.idx, c.text) for c in group]
        )
        vectors.add(
            site_id=site_id,
            ids=[c.id for c in group],
            embeddings=embeddings,
            documents=[c.text for c in group],
            metadatas=[{"url": c.url, "title": c.title} for c in group],
        )
        total += len(group)
        log(f"  embedded {total}/{len(all_chunks)} chunks")

    return len(pages), len(all_chunks)

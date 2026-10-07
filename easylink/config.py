"""Configuration & secrets handling.

Rules:
- Secrets live only in .env (never in code, never in git).
- Every cap/threshold in the system is configurable here with a safe default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "easylink.db"
CHROMA_DIR = DATA_DIR / "chroma"
NEVER_SAY_PATH = ROOT / "never_say.txt"
ENV_PATH = ROOT / ".env"


def _load_dotenv() -> None:
    """Load .env without failing if python-dotenv is missing (tiny fallback)."""
    if not ENV_PATH.exists():
        return
    try:
        from dotenv import load_dotenv

        # override=True: the .env file is the user's deliberate choice and must
        # win over ambient system env vars (e.g. a stale key set in Windows).
        load_dotenv(ENV_PATH, override=True)
        return
    except ImportError:
        pass
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ[key] = value


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Settings:
    # --- provider ---
    provider: str = "openai"  # openai | gemini | mock
    api_key: str = ""         # OpenAI key (sk-...)
    gemini_key: str = ""      # Google key (AIza...)
    chat_model: str = ""      # "" -> provider default
    embed_model: str = ""     # "" -> provider default
    base_url: str = ""
    temperature: float = 0.1

    # --- retrieval ---
    top_k: int = 6
    min_similarity: float | None = None  # None -> provider default

    # --- crawler caps (risk register: crawl runaway) ---
    max_pages: int = 100
    max_depth: int = 4
    page_timeout: float = 15.0
    politeness_delay: float = 0.25  # seconds between requests to same host
    user_agent: str = "EasyLinkAI-Crawler/1.0 (+https://easylink.ai/bot)"

    # --- chunking ---
    chunk_chars: int = 1600
    chunk_overlap: int = 200

    # --- misc ---
    never_say: list[str] = field(default_factory=list)

    @property
    def embed_batch_size(self) -> int:
        return 64


def load_settings() -> Settings:
    _load_dotenv()

    openai_key = _get("OPENAI_API_KEY")
    gemini_key = _get("GEMINI_API_KEY")

    # A Google key (AIza...) sometimes lives in OPENAI_API_KEY by mistake —
    # detect it so the right provider is chosen automatically.
    if openai_key.startswith("AIza") and not gemini_key:
        gemini_key, openai_key = openai_key, ""

    provider = _get("EASYLINK_PROVIDER", "").lower()
    if not provider:
        if openai_key:
            provider = "openai"
        elif gemini_key:
            provider = "gemini"
        else:
            provider = "mock"

    s = Settings(
        provider=provider,
        api_key=openai_key,
        gemini_key=gemini_key,
        chat_model=_get("EASYLINK_CHAT_MODEL", ""),
        embed_model=_get("EASYLINK_EMBED_MODEL", ""),
        base_url=_get("OPENAI_BASE_URL", ""),
    )

    # Optional overrides
    if v := _get("EASYLINK_MIN_SIMILARITY"):
        s.min_similarity = float(v)
    if v := _get("EASYLINK_TOP_K"):
        s.top_k = int(v)
    if v := _get("EASYLINK_MAX_PAGES"):
        s.max_pages = int(v)
    if v := _get("EASYLINK_TEMPERATURE"):
        s.temperature = float(v)

    # Owner-defined "never say" list (layer 5 of the anti-hallucination stack)
    if NEVER_SAY_PATH.exists():
        s.never_say = [
            line.strip()
            for line in NEVER_SAY_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return s

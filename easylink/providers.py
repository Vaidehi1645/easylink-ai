"""LLM provider abstraction (risk #10: no provider lock-in).

Implementations:
- GeminiProvider : Google Gemini (works today — this is where your key lives)
- OpenAIProvider : gpt-4o-mini / text-embedding-3-small (drop in an sk- key)
- MockProvider   : offline, free, deterministic — proves the pipeline with
                   zero API spend (hashed bag-of-words + extractive answers)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from typing import Any, Iterator

from easylink.config import Settings


class ProviderError(RuntimeError):
    """Raised when a provider cannot be used (missing key, network, etc.)."""


History = list[dict]  # [{"q": ..., "a": ...}] — oldest first


class BaseProvider:
    name: str = "base"
    default_threshold: float = 0.30
    default_chat_model: str = ""
    default_embed_model: str = ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def complete(self, system: str, user: str, history: History | None = None) -> str:
        raise NotImplementedError

    def complete_stream(
        self, system: str, user: str, history: History | None = None
    ) -> "Iterator[str]":
        """Yield answer text incrementally. Default: single-chunk fallback."""
        yield self.complete(system, user, history)


def _check(resp: Any, what: str) -> None:
    if getattr(resp, "status_code", 200) >= 400:
        raise ProviderError(f"{what} failed: HTTP {resp.status_code}")


# --------------------------------------------------------------------------- #
# Gemini (REST — no extra dependency)
# --------------------------------------------------------------------------- #
class GeminiProvider(BaseProvider):
    name = "gemini"
    default_threshold = 0.58          # calibrated: real matches >=0.647, unrelated <=0.524
    default_chat_model = "gemini-3.8-flash"
    default_embed_model = "gemini-embedding-001"
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, settings: Settings):
        self.key = settings.gemini_key or settings.api_key
        if not self.key:
            raise ProviderError(
                "GEMINI_API_KEY not found. Put it in .env (see .env.example), "
                "or set EASYLINK_PROVIDER=mock to test offline for free."
            )
        self.chat_model = settings.chat_model or self.default_chat_model
        self.embed_model = settings.embed_model or self.default_embed_model
        self.temperature = settings.temperature

    def _post(self, path: str, payload: dict, what: str, retries: int = 5) -> dict:
        """POST with backoff — transient 429/503 must never kill a crawl or chat."""
        import time

        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            req = urllib.request.Request(
                f"{self.BASE}/{path}",
                data=json.dumps(payload).encode("utf-8"),
                headers={"x-goog-api-key": self.key, "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "ignore")
                # Prefer the API's human message over the raw JSON blob
                try:
                    msg = json.loads(body)["error"]["message"]
                except Exception:
                    msg = body[:250]
                last_exc = ProviderError(f"{what} failed ({exc.code}): {msg}")
                if exc.code in (429, 500, 503) and attempt < retries:
                    time.sleep(min(2**attempt, 16))  # 1s, 2s, 4s, 8s, 16s
                    continue
                raise last_exc from exc
            except urllib.error.URLError as exc:
                last_exc = ProviderError(f"{what} failed: {exc.reason}")
                if attempt < retries:
                    time.sleep(2**attempt)
                    continue
                raise last_exc from exc
        raise last_exc or ProviderError(f"{what} failed")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch = 100
        for i in range(0, len(texts), batch):
            group = texts[i : i + batch]
            data = self._post(
                f"models/{self.embed_model}:batchEmbedContents",
                {
                    "requests": [
                        {
                            "model": f"models/{self.embed_model}",
                            "content": {"parts": [{"text": t}]},
                            "outputDimensionality": 768,
                        }
                        for t in group
                    ]
                },
                "embeddings",
            )
            vectors.extend(e["values"] for e in data["embeddings"])
        return vectors

    def complete(self, system: str, user: str, history: History | None = None) -> str:
        contents: list[dict] = []
        for turn in history or []:
            contents.append({"role": "user", "parts": [{"text": turn["q"]}]})
            contents.append({"role": "model", "parts": [{"text": turn["a"]}]})
        contents.append({"role": "user", "parts": [{"text": user}]})

        data = self._post(
            f"models/{self.chat_model}:generateContent",
            {
                "contents": contents,
                "systemInstruction": {"parts": [{"text": system}]},
                "generationConfig": {"temperature": self.temperature},
            },
            "chat",
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"unexpected chat response: {exc}") from exc

    def complete_stream(
        self, system: str, user: str, history: History | None = None
    ) -> Iterator[str]:
        """Server-sent-events streaming variant of generateContent."""
        contents: list[dict] = []
        for turn in history or []:
            contents.append({"role": "user", "parts": [{"text": turn["q"]}]})
            contents.append({"role": "model", "parts": [{"text": turn["a"]}]})
        contents.append({"role": "user", "parts": [{"text": user}]})
        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"temperature": self.temperature},
        }

        import time as _time

        last_exc: Exception | None = None
        for attempt in range(6):  # open with backoff; mid-stream errors raise
            req = urllib.request.Request(
                f"{self.BASE}/models/{self.chat_model}:streamGenerateContent?alt=sse",
                data=json.dumps(payload).encode("utf-8"),
                headers={"x-goog-api-key": self.key, "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8", "ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        body = line[5:].strip()
                        if not body or body == "[DONE]":
                            continue
                        try:
                            event = json.loads(body)
                            parts = event["candidates"][0]["content"]["parts"]
                        except (ValueError, KeyError, IndexError):
                            continue
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                yield text
                return
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "ignore")[:250]
                try:
                    detail = json.loads(detail)["error"]["message"]
                except Exception:
                    pass
                last_exc = ProviderError(f"chat failed ({exc.code}): {detail}")
                if exc.code in (429, 500, 503) and attempt < 5:
                    _time.sleep(min(2**attempt, 16))
                    continue
                raise last_exc from exc
            except urllib.error.URLError as exc:
                last_exc = ProviderError(f"chat failed: {exc.reason}")
                if attempt < 5:
                    _time.sleep(min(2**attempt, 16))
                    continue
                raise last_exc from exc
        raise last_exc or ProviderError("chat failed")


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
class OpenAIProvider(BaseProvider):
    name = "openai"
    # Calibrated 2026-10-06: unrelated queries maxed at 0.214 (mean 0.134),
    # in-corpus questions averaged 0.412. We sit LOW (0.23) on purpose:
    # a false pass still gets caught by the grounded-refusal layer, while a
    # false block would wrongly silence a real answer (unrecoverable).
    default_threshold = 0.23
    default_chat_model = "gpt-4o-mini"
    default_embed_model = "text-embedding-3-small"

    def __init__(self, settings: Settings):
        if not settings.api_key or settings.api_key.startswith("AIza"):
            raise ProviderError(
                "No OpenAI key found (OpenAI keys start with 'sk-'). "
                "Put it in .env, or use your Gemini key with EASYLINK_PROVIDER=gemini, "
                "or set EASYLINK_PROVIDER=mock to test offline."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("openai package missing: pip install openai") from exc

        kwargs = {"api_key": settings.api_key}
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self._client = OpenAI(**kwargs)
        self._chat_model = settings.chat_model or self.default_chat_model
        self._embed_model = settings.embed_model or self.default_embed_model
        self._temperature = settings.temperature

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch = 100  # API max for this endpoint
        for i in range(0, len(texts), batch):
            group = texts[i : i + batch]
            resp = self._client.embeddings.create(model=self._embed_model, input=group)
            vectors.extend(d.embedding for d in resp.data)
        return vectors

    def complete(self, system: str, user: str, history: History | None = None) -> str:
        messages: list[dict] = [{"role": "system", "content": system}]
        for turn in history or []:
            messages.append({"role": "user", "content": turn["q"]})
            messages.append({"role": "assistant", "content": turn["a"]})
        messages.append({"role": "user", "content": user})

        resp = self._client.chat.completions.create(
            model=self._chat_model,
            temperature=self._temperature,
            messages=messages,
        )
        return (resp.choices[0].message.content or "").strip()

    def complete_stream(
        self, system: str, user: str, history: History | None = None
    ) -> Iterator[str]:
        messages: list[dict] = [{"role": "system", "content": system}]
        for turn in history or []:
            messages.append({"role": "user", "content": turn["q"]})
            messages.append({"role": "assistant", "content": turn["a"]})
        messages.append({"role": "user", "content": user})

        stream = self._client.chat.completions.create(
            model=self._chat_model,
            temperature=self._temperature,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


# --------------------------------------------------------------------------- #
# Mock (offline / free / deterministic)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "is", "are", "was", "were", "be", "been", "do", "does", "did", "what",
    "which", "who", "whom", "how", "when", "where", "why", "can", "could",
    "would", "should", "will", "shall", "may", "might", "must", "this", "that",
    "these", "those", "it", "its", "as", "by", "from", "with", "about", "you",
    "your", "i", "we", "they", "he", "she", "them", "their", "our", "my",
    "me", "us", "not", "no", "yes", "if", "then", "than", "so", "such",
    "there", "here", "have", "has", "had", "get", "got", "any", "some",
    "tell", "give", "show", "please", "info", "information", "much", "many",
    "more", "most", "all", "into", "out", "up", "down", "over", "under",
}

DIM = 1024


def _hash_embed(text: str) -> list[float]:
    """Deterministic hashed bag-of-words + bigrams.

    Bigrams make phrase matches ("light attic") score sharply higher, which
    keeps unrelated questions from passing the similarity threshold.
    """
    words = _WORD_RE.findall(text.lower())
    vec = [0.0] * DIM

    def bump(token: str) -> None:
        h = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")
        vec[h % DIM] += 1.0

    for word in words:
        bump(word)
    for i in range(len(words) - 1):
        bump(f"{words[i]}_{words[i + 1]}")

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _content_words(text: str) -> set[str]:
    """Content words with a tiny stemmer so 'example' matches 'examples'."""
    out = set()
    for w in _WORD_RE.findall(text.lower()):
        if w in _STOP or len(w) <= 2:
            continue
        if len(w) > 4 and w.endswith("s"):
            w = w[:-1]
        out.add(w)
    return out


class MockProvider(BaseProvider):
    """Free offline provider — lets the whole pipeline run before any API spend."""

    name = "mock"
    default_threshold = 0.15

    def __init__(self, settings: Settings):
        self._ = settings

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embed(t) for t in texts]

    def complete(self, system: str, user: str, history: History | None = None) -> str:
        """Extractive stub: best-overlapping passage from the context.

        Returns "" when it can't answer — the RAG layer treats that as an
        abstention (gap logged), exactly like the real model's refusal.
        """
        if "QUESTION:" not in user:
            return ""
        context_part, _, question = user.partition("QUESTION:")
        question = question.strip()

        q_words = _content_words(question)
        best_score, best_passage = 0, ""
        for sent in re.split(r"(?<=[.!?])\s+|\n+", context_part):
            sent = sent.strip()
            if len(sent) < 15:
                continue
            if re.match(r"^\[\d+\]", sent):  # skip "[1] url | title" headers
                continue
            overlap = len(q_words & _content_words(sent))
            if overlap > best_score:
                best_score, best_passage = overlap, sent

        if best_score < 2 or not best_passage:
            return ""
        return best_passage

    def complete_stream(
        self, system: str, user: str, history: History | None = None
    ) -> Iterator[str]:
        """Mock has nothing to stream — yield the full text in a few pieces
        so the widget's streaming path still gets exercised."""
        text = self.complete(system, user, history)
        if not text:
            yield ""
            return
        step = max(len(text) // 4, 1)
        for i in range(0, len(text), step):
            yield text[i : i + step]


# --------------------------------------------------------------------------- #
def get_provider(settings: Settings) -> BaseProvider:
    if settings.provider == "mock":
        return MockProvider(settings)
    if settings.provider == "gemini":
        return GeminiProvider(settings)
    if settings.provider == "openai":
        return OpenAIProvider(settings)
    raise ProviderError(f"Unknown provider '{settings.provider}' (openai, gemini or mock)")

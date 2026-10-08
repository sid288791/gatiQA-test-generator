"""Semantic similarity via Ollama embeddings (nomic-embed-text).

Used to deduplicate test-generation requests: before calling the LLM we embed
the incoming description and compare it against stored embeddings of previous
inputs. Cosine similarity >= threshold means "same test case" even when the
wording differs — something plain hashing can't do.
"""

from __future__ import annotations

import logging
import math
import re

import httpx

from app.config import Settings
from app.observability import Tracer, cost_details_for, ollama_metadata, ollama_usage_details

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s\-]")


def normalize_text(text: str) -> str:
    """Canonical form for cheap exact-match pre-checks.

    Lowercases, strips punctuation, collapses whitespace — so trivially
    different inputs ("Search for drill!" vs "search for drill") match
    without needing an embedding call.
    """
    return _WS_RE.sub(" ", _PUNCT_RE.sub("", text.lower())).strip()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class Embedder:
    """Async client for Ollama's /api/embed endpoint."""

    def __init__(self, settings: Settings, *, tracer=None):
        self._settings = settings
        self._tracer = tracer if tracer is not None else Tracer(settings)
        base = settings.ollama_base_url.replace("/v1", "").rstrip("/")
        self._url = f"{base}/api/embed"

    async def embed(self, text: str) -> list[float] | None:
        """Return the embedding vector, or None if the endpoint is unavailable."""
        with self._tracer.span("embedding", kind="embedding", metadata={
            "provider": "ollama", "model": self._settings.embedding_model,
        }) as obs:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(
                        self._url,
                        json={"model": self._settings.embedding_model, "input": text},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                usage = ollama_usage_details({**data, "eval_count": 0})
                obs.update(
                    usage_details=usage,
                    cost_details=cost_details_for(
                        usage, input_per_mtok=self._settings.langfuse_embedding_cost_per_mtok,
                        output_per_mtok=0,
                    ),
                    metadata=ollama_metadata(data),
                )
                embeddings = data.get("embeddings")
                if not embeddings:
                    obs.update(level="ERROR", metadata={"error.type": "EmptyEmbedding"})
                    return None
                return embeddings[0]
            except Exception as exc:
                obs.set_error(exc)
                logger.warning("Embedding call failed (cache lookup skipped): %s", exc)
                return None

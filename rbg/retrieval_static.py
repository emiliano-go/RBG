"""CPU-only static code encoder (Model2Vec).

Loads a pulled ``potion-code`` checkpoint with only numpy/tokenizers.  No
torch, no network at query time.  Falls back to ``None`` when the package or
the checkpoint is missing, so the caller keeps the lexical/LSI encoder.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from . import model_registry


class StaticCodeEncoder:
    kind = "static"

    def __init__(self, path: str | None = None, dim: int = 256) -> None:
        self.path = path
        self.dim = int(dim)
        self.model = None
        self._cache: dict[str, np.ndarray] = {}
        self._cache_limit = 200_000
        if path:
            self._load(path)

    def _load(self, path: str) -> None:
        from model2vec import StaticModel  # optional dependency
        self.model = StaticModel.from_pretrained(path)
        self.dim = int(getattr(self.model, "dim", self.dim) or self.dim)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim))
        vecs = np.asarray(self.model.encode(texts), dtype=float)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        self.dim = vecs.shape[1]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def encode(self, text: str) -> np.ndarray:
        return self._encode([text])[0]

    def document_vectors(self, chunks: list) -> np.ndarray:
        if not chunks:
            return np.zeros((0, self.dim))
        keys = [getattr(c, "content_hash", None)
                or hashlib.sha1(c.content.encode()).hexdigest() for c in chunks]
        if len(self._cache) > self._cache_limit:
            self._cache.clear()
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            vecs = self._encode([chunks[i].content for i in missing])
            for i, vec in zip(missing, vecs):
                self._cache[keys[i]] = vec
        return np.vstack([self._cache[k] for k in keys])

    def rank(self, query: str, chunks: list, limit: int = 50) -> list[tuple[int, float]]:
        if not chunks or self.model is None:
            return []
        sims = self.document_vectors(chunks) @ self.encode(query)
        order = np.argsort(-sims)
        return [(int(i), float(sims[i])) for i in order[:limit] if sims[i] > 0]

    def to_state(self) -> dict:
        return {"kind": "static", "model": self.path, "dim": self.dim}

    @classmethod
    def from_state(cls, state: dict, doc_vecs) -> "StaticCodeEncoder":
        return cls(path=state.get("model"), dim=int(state.get("dim", 256)))


def encoder(path: str | None = None):
    """A static encoder for ``RBG_STATIC_MODEL`` (or a pulled default), or None."""
    if path is None:
        path = os.environ.get("RBG_STATIC_MODEL", "")
        if not path:
            path = model_registry.resolve("potion-code") or ""
    if not path:
        return None
    try:
        if os.path.isdir(path):
            return StaticCodeEncoder(path)
        return None
    except Exception:
        return None

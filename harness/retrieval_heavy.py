"""Optional heavier retrievers, kept out of the default install.

The default harness is numpy-only.  A learned sparse retriever (SPLADE-Code)
or a code cross-encoder lifts recall but needs a transformer runtime, so it
lives behind the ``semantic`` extra::

    pip install 'rbg-harness[semantic]'

Only the *reranker* slot is filled here, because it duck-types the one method
``HybridRetriever`` calls (``Reranker.rerank``) and therefore needs no core
change.  The encoder slot has a larger surface (``encode`` / ``document_vectors``
/ ``rank``) and is left for when the reranker's gain is measured to be
insufficient.  Everything degrades gracefully: with no runtime or no configured
model, the linear :class:`~harness.retrieval.Reranker` is used unchanged.
"""

from __future__ import annotations

import hashlib
import os

from .retrieval import Hit, Reranker


def available() -> bool:
    """True when the transformer runtime for the heavy retriever is installed."""
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


# One process may build many retrievers (one per suite task/revision); loading
# the checkpoint is tens of MB from disk, so keep it process-wide.  Keyed by
# (path, device) because a CUDA and a CPU instance are different objects.
_MODELS: dict[tuple[str, str], object] = {}


def _resolve_device(device: str | None = None) -> str:
    """Explicit device > ``RBG_EMBEDDING_DEVICE`` > CUDA when available > CPU."""
    if device:
        return device
    env = os.environ.get("RBG_EMBEDDING_DEVICE", "").strip()
    if env:
        return env
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_model(path: str, device: str):
    key = (path, device)
    model = _MODELS.get(key)
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(path, device=device)
        _MODELS[key] = model
    return model


def encoder(model_path: str | None = None, device: str | None = None):
    """A configured :class:`SentenceTransformerEncoder`, or ``None``.

    Opt-in: returns nothing unless the ``semantic`` extra is installed *and*
    ``RBG_EMBEDDING_MODEL`` (or ``model_path``) names a checkpoint.  Callers
    then keep the dependency-free LSI/hash encoder, so the default install is
    unchanged.  Runs on CUDA when available unless ``RBG_EMBEDDING_DEVICE``
    pins a device.
    """
    if not available():
        return None
    path = model_path or os.environ.get("RBG_EMBEDDING_MODEL", "")
    if not path:
        return None
    try:
        return SentenceTransformerEncoder(path, device=device)
    except Exception:
        return None


class SentenceTransformerEncoder:
    """Dense semantic encoder backed by a sentence-transformers checkpoint.

    Duck-types the encoder slot of :class:`~harness.retrieval.HybridRetriever`
    (``encode`` / ``document_vectors`` / ``rank`` / ``to_state``).  This is the
    real embedding the dependency-free hash/LSA encoders stood in for; it is the
    slot that matters when lexical recall plateaus on vocabulary mismatch.
    """

    def __init__(self, model_path: str | None = None, model=None,
                 batch_size: int = 64, dim: int | None = None,
                 device: str | None = None) -> None:
        self.model_path = model_path
        self.model = model
        self.batch_size = int(os.environ.get("RBG_EMBEDDING_BATCH", batch_size))
        self.dim = dim
        self.device = _resolve_device(device)
        # Embedding cache by content hash: a new revision of the same repo only
        # re-encodes the files that actually changed (SWE-bench scores dozens of
        # commits of one repo).  ponytail: crude size cap; LRU if it matters.
        self._cache: dict[str, object] = {}
        self._cache_limit = 200_000
        if self.model is None and model_path and available():
            self.model = _load_model(model_path, self.device)

    def _encode(self, texts: list[str]):
        import numpy as np
        if not texts:
            return np.zeros((0, self.dim or 1))
        vecs = self.model.encode(texts, batch_size=self.batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True)
        vecs = np.asarray(vecs, dtype=float)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        self.dim = vecs.shape[1]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def encode(self, text: str):
        return self._encode([text])[0]

    def document_vectors(self, chunks: list):
        import numpy as np
        if not chunks:
            return np.zeros((0, self.dim or 1))
        keys = [getattr(c, "content_hash", None)
                or hashlib.sha1(c.content.encode("utf-8")).hexdigest() for c in chunks]
        if len(self._cache) > self._cache_limit:
            self._cache.clear()
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            vecs = self._encode([chunks[i].content for i in missing])
            for i, vec in zip(missing, vecs):
                self._cache[keys[i]] = vec
        return np.vstack([self._cache[k] for k in keys])

    def rank(self, query: str, chunks: list, limit: int = 50) -> list[tuple[int, float]]:
        import numpy as np
        if not chunks or self.model is None:
            return []
        sims = self.document_vectors(chunks) @ self.encode(query)
        order = np.argsort(-sims)
        return [(int(i), float(sims[i])) for i in order[:limit] if sims[i] > 0]

    def to_state(self) -> dict:
        """Model identity + width, so a reload restores the same encoder and the
        persisted document vectors stay dimensionally compatible."""
        return {"kind": "sentence-transformer", "model": self.model_path, "dim": self.dim}

    @classmethod
    def from_state(cls, state: dict, doc_vecs) -> "SentenceTransformerEncoder":
        model_path = state.get("model")
        if not model_path or not available():
            # Persisted by a run that had the extra; this one cannot use it, so
            # fail loudly and let the caller fall back to the lexical encoder.
            raise RuntimeError("semantic encoder unavailable in this environment")
        dim = int(state.get("dim") or (doc_vecs.shape[1] if getattr(doc_vecs, "size", 0) else 0)) or None
        return cls(model_path, dim=dim)



class CrossEncoderReranker:
    """Code cross-encoder reranker over the fused candidates.

    Duck-types :meth:`harness.retrieval.Reranker.rerank`, so it plugs into
    ``HybridRetriever(index, reranker=CrossEncoderReranker())``.  Point
    ``RBG_CROSS_ENCODER`` at a sentence-transformers / sequence-classification
    checkpoint; with no runtime or no model it delegates to the trained linear
    reranker, so a caller never gets a different contract, only a weaker score.
    """

    def __init__(self, model_path: str | None = None, max_len: int = 512,
                 device: str | None = None) -> None:
        self._fallback = Reranker()
        self.model = None
        self.max_len = max_len
        self.device = _resolve_device(device)
        if not available():
            return
        path = model_path or os.environ.get("RBG_CROSS_ENCODER", "")
        if not path:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.model.eval()

    def score(self, query: str, chunk) -> float:
        import torch
        with torch.no_grad():
            enc = self.tok(query, chunk.content, truncation=True,
                           max_length=self.max_len, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            return float(self.model(**enc).logits[0][0])  # ponytail: single logit; task-tuned heads differ

    def rerank(self, query: str, hits: list[Hit], chunks: list) -> list[Hit]:
        if self.model is None:
            return self._fallback.rerank(query, hits, chunks)
        by_key = {f"{c.path}:{c.start_line}": c for c in chunks}
        scored = []
        for h in hits:
            c = by_key.get(f"{h.path}:{h.start_line}")
            scored.append((self.score(query, c) if c else -1e9, h))
        scored.sort(key=lambda kv: -kv[0])
        return [Hit(h.path, h.symbol, h.start_line, h.end_line, s, "cross-encoder",
                    h.reason, h.content_hash, h.content, h.commit_hash)
                for s, h in scored]

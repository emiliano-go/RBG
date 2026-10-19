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

import os

from .retrieval import Hit, Reranker


def available() -> bool:
    """True when the transformer runtime for the heavy retriever is installed."""
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


class CrossEncoderReranker:
    """Code cross-encoder reranker over the fused candidates.

    Duck-types :meth:`harness.retrieval.Reranker.rerank`, so it plugs into
    ``HybridRetriever(index, reranker=CrossEncoderReranker())``.  Point
    ``RBG_CROSS_ENCODER`` at a sentence-transformers / sequence-classification
    checkpoint; with no runtime or no model it delegates to the trained linear
    reranker, so a caller never gets a different contract, only a weaker score.
    """

    def __init__(self, model_path: str | None = None, max_len: int = 512) -> None:
        self._fallback = Reranker()
        self.model = None
        self.max_len = max_len
        if not available():
            return
        path = model_path or os.environ.get("RBG_CROSS_ENCODER", "")
        if not path:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.eval()

    def score(self, query: str, chunk) -> float:
        import torch
        with torch.no_grad():
            enc = self.tok(query, chunk.content, truncation=True,
                           max_length=self.max_len, return_tensors="pt")
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

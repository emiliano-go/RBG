"""Code-aware RAG retrieval: BM25 lexical search, a dense encoder (signed
feature-hashing TF-IDF, or LSA for a genuine semantic space), reciprocal-rank
fusion, RM3-lite query expansion and MMR diversification, plus an optional
non-generative reranker.

The dense encoder is deliberately dependency-free.  The spec leaves the
embedding model open and asks for an encoder to be benchmarked against BM25
before committing (spec 4.1, 5.2); a cheap deterministic baseline is the right
starting point, and :class:`LSIEncoder` adds the synonymy/vocabulary-mismatch
coverage a lexical hash cannot.
"""

from __future__ import annotations

import fnmatch
import hashlib
import math
import os
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .index import Chunk, RepoIndex, chunk_key

SNIPPET_CHARS = 400

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")

# Above this corpus size the LSA projector is more memory/latency than it is
# worth, so HybridRetriever falls back to the lexical hashing encoder.
LSI_MAX_CORPUS = 20000

# Path classes: source files are full weight; docs and tests are down-weighted
# so a query for "architecture" stops surfacing README/llms-full.txt and test
# helpers ahead of the actual implementation.  (Weights are mild by design.)
PATH_WEIGHT = {"source": 1.0, "docs": 0.6, "tests": 0.4}


def _path_class(path: str) -> str:
    """Classify a repo-relative path as source/docs/tests."""
    norm = path.replace(os.sep, "/")
    low = norm.lower()
    base = os.path.basename(low)
    if "/tests/" in "/" + low + "/" or low.startswith("tests/"):
        return "tests"
    if "/test/" in "/" + low + "/" or low.startswith("test/"):
        return "tests"
    if base.startswith("test_") or base.endswith("_test.py"):
        return "tests"
    docs_dirs = {"/docs/", "/doc/", "/documentation/", "/site/", "/wiki/"}
    if any(d in "/" + low + "/" for d in docs_dirs):
        return "docs"
    if base in {"readme.md", "readme.rst", "readme.txt", "readme"}:
        return "docs"
    if os.path.splitext(base)[1] in {".md", ".rst", ".txt"}:
        return "docs"
    if base.startswith("llms-") or base.endswith(".min.js") or base.endswith(".min.css"):
        return "docs"
    return "source"


def _stable_hash(token: str, dim: int) -> tuple[int, float]:
    """Deterministic signed feature hash.

    Python's built-in ``hash`` is salted per process (PYTHONHASHSEED), so the
    old encoder produced *different* rankings on every process start; a silent
    correctness bug for anything that compares or caches retrieval results.  A
    fixed digest makes the encoder reproducible.
    """
    n = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")
    return n % dim, (1.0 if (n >> 63) & 1 else -1.0)


def _features(text: str) -> list[str]:
    """Word tokens plus character trigrams of longer words.

    The subword features let the dense encoder match morphological variants and
    parts of compound identifiers (``line_is_inert`` vs ``inert_line``), which
    plain bag-of-words misses; it is the cheap dependency-free stand-in for the
    semantic embedding the spec leaves open (spec 4.1).
    """
    words = tokenize(text)
    out = list(words)
    for w in words:
        if len(w) >= 5:
            out.extend("~" + w[i:i + 3] for i in range(len(w) - 2))
    return out


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for w in _WORD.findall(text):
        out.append(w.lower())
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", w)
        if len(parts) > 1:
            out.extend(p.lower() for p in parts)
    return out


_QUERY_TERM_CAP = 64


def _clean_query(text: str) -> str:
    """Deduplicate query tokens and cap their count so a long, repetitive query
    does not multiply the lexical/dense scans for terms already covered."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tokenize(text):
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= _QUERY_TERM_CAP:
            break
    return " ".join(out)


@dataclass
class Hit:
    path: str
    symbol: str
    start_line: int
    end_line: int
    score: float
    score_type: str
    reason: str
    content_hash: str
    content: str = ""
    commit_hash: str = ""

    @property
    def chunk_id(self) -> str:
        return chunk_key(self.path, self.start_line, self.end_line)

    def to_dict(self, snippet_chars: int = SNIPPET_CHARS) -> dict:
        """Discovery output is a capped snippet, not the whole chunk; fetch the
        full text with read_chunk(chunk_id)."""
        return {
            "path": self.path, "symbol": self.symbol,
            "start_line": self.start_line, "end_line": self.end_line,
            "score": round(self.score, 6), "score_type": self.score_type,
            "reason": self.reason, "content_hash": self.content_hash,
            "commit_hash": self.commit_hash, "chunk_id": self.chunk_id,
            "snippet": self.content[:snippet_chars],
            "content_chars": len(self.content),
            "truncated": len(self.content) > snippet_chars,
        }


class BM25:
    """Okapi BM25 over a fixed corpus of chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.content) for c in chunks]
        self.tf: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        for doc in self.docs:
            counts: dict[str, int] = {}
            for t in doc:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                self.df[t] = self.df.get(t, 0) + 1
        self.n = len(chunks)
        self.avgdl = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def rank(self, query: str, limit: int = 50) -> list[tuple[int, float]]:
        if not self.n:
            return []
        q = tokenize(query)
        scores = np.zeros(self.n)
        for t in q:
            idf = self._idf(t)
            if idf <= 0:
                continue
            for i, counts in enumerate(self.tf):
                f = counts.get(t, 0)
                if f:
                    dl = len(self.docs[i]) or 1
                    scores[i] += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
        order = np.argsort(-scores)
        return [(int(i), float(scores[i])) for i in order[:limit] if scores[i] > 0]


class FTSBackend:
    """Lexical candidates served by a persisted FTS5 index (native BM25).

    ``search(query, limit)`` returns ``(index, score)`` pairs, best first; an
    empty result falls back to the in-memory scorer so a missing or stale FTS
    index degrades to the previous behaviour rather than returning nothing.
    """

    def __init__(self, chunks: list[Chunk], search) -> None:
        self.chunks = chunks
        self._search = search
        # Building the in-memory scorer tokenizes the whole corpus (~18 s on a
        # 108k-chunk monorepo).  It is only a fallback for a missing/stale FTS
        # index, so construct it on first failure instead of on the read path.
        self._fallback = None

    def rank(self, query: str, limit: int = 50) -> list[tuple[int, float]]:
        try:
            out = self._search(query, int(limit))
        except Exception:
            out = None
        if out:
            return out
        if self._fallback is None:
            self._fallback = BM25(self.chunks)
        return self._fallback.rank(query, limit)


class DenseEncoder:
    """Deterministic signed feature-hashing TF-IDF encoder -> L2-normalised
    dense vectors (word tokens + char trigrams)."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.idf: dict[str, float] = {}
        self._mat: np.ndarray | None = None
        self._mat_chunks: int = -1

    def _matrix(self, chunks: list[Chunk]) -> np.ndarray:
        if self._mat is None or self._mat_chunks != len(chunks):
            self._mat = np.vstack([self.encode(c.content) for c in chunks]) if chunks else np.zeros((0, self.dim))
            self._mat_chunks = len(chunks)
        return self._mat

    def document_vectors(self, chunks: list[Chunk]) -> np.ndarray:
        return self._matrix(chunks)

    def to_state(self) -> dict:
        """Serialisable parameters, excluding the document matrix (stored once,
        per chunk, so a reloaded encoder does not duplicate it)."""
        return {"kind": "dense", "dim": self.dim, "idf": self.idf}

    @classmethod
    def from_state(cls, state: dict, doc_vecs: np.ndarray) -> "DenseEncoder":
        self = cls(dim=int(state["dim"]))
        self.idf = dict(state.get("idf", {}))
        self._mat = np.asarray(doc_vecs, dtype=float)
        self._mat_chunks = len(self._mat)
        return self

    def fit(self, texts: list[str]) -> "DenseEncoder":
        n = len(texts)
        df: dict[str, int] = {}
        for text in texts:
            for t in set(_features(text)):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        return self

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        for t in _features(text):
            i, sign = _stable_hash(t, self.dim)
            vec[i] += sign * self.idf.get(t, 1.0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def rank(self, query: str, chunks: list[Chunk], limit: int = 50) -> list[tuple[int, float]]:
        if not chunks:
            return []
        q = self.encode(query)
        mat = self._matrix(chunks)
        sims = mat @ q
        order = np.argsort(-sims)
        return [(int(i), float(sims[i])) for i in order[:limit] if sims[i] > 0]


class LexicalOnlyEncoder:
    """Placeholder dense leg for corpora too large to fit/encode cheaply.

    Fitting ``DenseEncoder`` over ~100k chunks is a multi-minute pure-Python
    pass that used to run inside the first MCP call and block the stdio loop.
    Above ``LSI_MAX_CORPUS`` the lexical FTS5/BM25 leg is the useful one, so the
    dense leg is deliberately empty instead of paying that cost on every cold
    revision.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def fit(self, texts) -> "LexicalOnlyEncoder":
        return self

    def encode(self, text: str) -> np.ndarray:
        return np.zeros(self.dim)

    def rank(self, query: str, chunks: list[Chunk], limit: int = 50) -> list[tuple[int, float]]:
        return []

    def document_vectors(self, chunks: list[Chunk]) -> np.ndarray:
        return np.zeros((0, self.dim))

    def to_state(self) -> dict:
        return {"kind": "lexical-only", "dim": self.dim}

    @classmethod
    def from_state(cls, state: dict, doc_vecs: np.ndarray) -> "LexicalOnlyEncoder":
        return cls(dim=int(state.get("dim", 512)))


class LSIEncoder:
    """Latent semantic encoder: TF-IDF -> truncated SVD (randomised), giving a
    genuine dense semantic space (synonymy, vocabulary mismatch) rather than a
    lexical hash.

    Dependency-free: the randomised range finder is plain numpy.  Bounded by a
    vocabulary cap and a corpus-size cap so the ``V x k`` query projector stays
    a few tens of MB; callers fall back to :class:`DenseEncoder` above the cap.
    ``ponytail:`` exact SVD would be marginally better; randomised is O(nnz*k)
    and plenty for retrieval, upgrade only if recall plateaus.
    """

    def __init__(self, k: int = 128, vocab_cap: int = 20000, corpus_cap: int = LSI_MAX_CORPUS,
                 min_df: int = 2, oversample: int = 12, seed: int = 0) -> None:
        self.k = k
        self.vocab_cap = vocab_cap
        self.corpus_cap = corpus_cap
        self.min_df = min_df
        self.oversample = oversample
        self.seed = seed
        self.idf: dict[str, float] = {}
        self.vocab: dict[str, int] = {}
        self.proj: np.ndarray | None = None   # (V, k) query projector
        self.doc_vecs: np.ndarray | None = None  # (N, k) L2-normalised doc embeddings

    def _encode_docs(self, docs: list[list[str]]) -> list[tuple[list[int], list[float]]]:
        out = []
        for doc in docs:
            counts = Counter(doc)
            terms, vals = [], []
            for t, c in counts.items():
                j = self.vocab.get(t)
                if j is not None:
                    terms.append(j)
                    vals.append((1.0 + math.log(c)) * self.idf[t])
            out.append((terms, vals))
        return out

    def fit(self, texts: list[str]) -> "LSIEncoder":
        docs = [tokenize(t) for t in texts]
        n = len(docs)
        if n == 0:
            self.proj = np.zeros((0, self.k))
            self.doc_vecs = np.zeros((0, self.k))
            return self
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        keep = [t for t, c in df.items() if c >= self.min_df]
        keep.sort(key=lambda t: -df[t])
        keep = keep[: self.vocab_cap]
        self.vocab = {t: i for i, t in enumerate(keep)}
        self.idf = {t: math.log((1 + n) / (1 + df[t])) + 1 for t in keep}
        V = len(self.vocab)
        # Degenerate corpora (empty vocabulary, or a single document) cannot be
        # projected; keep the width at ``k`` so query/doc vectors stay the same
        # dimension and ``vector_distance_cos`` is never handed a mismatched blob.
        if V == 0 or n < 2:
            self.proj = np.zeros((V, self.k))
            self.doc_vecs = np.zeros((n, self.k))
            return self
        width = max(1, min(self.k + self.oversample, V, n))
        packed = self._encode_docs(docs)
        rng = np.random.default_rng(self.seed)
        omega = rng.standard_normal((n, width))
        Y = np.zeros((V, width))
        for d, (terms, vals) in enumerate(packed):
            if terms:
                Y[terms] += np.outer(vals, omega[d])
        Q, _ = np.linalg.qr(Y)
        Z = np.zeros((width, n))
        for d, (terms, vals) in enumerate(packed):
            if terms:
                Z[:, d] = Q[terms].T @ np.asarray(vals, dtype=float)
        Ub, s, Vt = np.linalg.svd(Z, full_matrices=False)
        k = min(self.k, len(s))
        self.doc_vecs = (s[:k, None] * Vt[:k]).T          # (N, k)
        self.proj = Q @ Ub[:, :k]                          # (V, k)
        self._normalise()
        return self

    def refresh(self, texts: list[str]) -> "LSIEncoder":
        """Re-embed documents against the existing projector without a new SVD.

        Doc vectors are ``X^T proj`` (sparse), so a revision change reuses the
        vocabulary basis and only re-encodes the changed corpus; no QR/SVD.
        Falls back to a full fit when there is no projector to reuse.
        """
        if self.proj is None or self.proj.shape[0] == 0:
            return self.fit(texts)
        self.doc_vecs = np.vstack([self.encode(t) for t in texts]) if texts else np.zeros((0, self.k))
        return self

    def _normalise(self) -> None:
        for mat_name in ("doc_vecs", "proj"):
            mat = getattr(self, mat_name)
            if mat is None or mat.size == 0:
                continue
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            setattr(self, mat_name, mat / norms)

    def document_vectors(self, chunks: list[Chunk]) -> np.ndarray:
        if self.doc_vecs is not None and len(self.doc_vecs) == len(chunks):
            return self.doc_vecs
        return np.vstack([self.encode(c.content) for c in chunks]) if chunks else np.zeros((0, self.k))

    def to_state(self) -> dict:
        """Serialisable parameters, excluding ``doc_vecs`` (persisted per chunk)."""
        return {"kind": "lsi", "k": self.k, "vocab": self.vocab, "idf": self.idf, "proj": self.proj}

    @classmethod
    def from_state(cls, state: dict, doc_vecs: np.ndarray) -> "LSIEncoder":
        self = cls(k=int(state.get("k", 128)))
        self.vocab = dict(state["vocab"])
        self.idf = dict(state["idf"])
        self.proj = np.asarray(state["proj"], dtype=float)
        self.doc_vecs = np.asarray(doc_vecs, dtype=float)
        return self

    def encode(self, text: str) -> np.ndarray:
        if self.proj is None or self.proj.shape[0] == 0:
            return np.zeros(self.k)
        q = np.zeros(self.proj.shape[0])
        for t, c in Counter(tokenize(text)).items():
            j = self.vocab.get(t)
            if j is not None:
                q[j] += (1.0 + math.log(c)) * self.idf[t]
        lat = self.proj.T @ q
        norm = np.linalg.norm(lat)
        return lat / norm if norm else lat

    def rank(self, query: str, chunks: list[Chunk], limit: int = 50) -> list[tuple[int, float]]:
        if not chunks or self.doc_vecs is None or len(self.doc_vecs) != len(chunks):
            return []
        q = self.encode(query)
        if not np.any(q):
            return []
        sims = self.doc_vecs @ q
        order = np.argsort(-sims)
        return [(int(i), float(sims[i])) for i in order[:limit] if sims[i] > 0]


def expand_query(query: str, bm25: "BM25", chunks: list[Chunk], *,
                 fb_docs: int = 8, fb_terms: int = 20, orig_weight: float = 0.6) -> str:
    """RM3-lite pseudo-relevance feedback: append salient terms from the top
    BM25 documents, interpolated with the original query to curb topic drift
    (spec 5.2 query handling).  Purely lexical, so it is deterministic and cheap.
    """
    hits = bm25.rank(query, fb_docs)
    if not hits:
        return query
    q_tokens = tokenize(query)
    fb: Counter = Counter()
    for i, _ in hits:
        fb.update(set(tokenize(chunks[i].content)))
    if not fb:
        return query
    # keep feedback terms that are discriminative (not corpus stop-words)
    total = sum(fb.values())
    ranked = sorted(((t, c / total) for t, c in fb.items() if t not in set(q_tokens)),
                    key=lambda kv: -kv[1])[:fb_terms]
    extra = [t for t, w in ranked if w * (1 - orig_weight) > 0.002]
    return query + (" " + " ".join(extra) if extra else "")


def encoder_from_state(state: dict, doc_vecs: np.ndarray) -> "DenseEncoder | LSIEncoder":
    """Rebuild a persisted encoder without re-fitting (offline reload)."""
    kind = state.get("kind")
    if kind == "lsi":
        return LSIEncoder.from_state(state, doc_vecs)
    if kind == "sentence-transformer":
        from .retrieval_heavy import SentenceTransformerEncoder
        return SentenceTransformerEncoder.from_state(state, doc_vecs)
    if kind == "static":
        from .retrieval_static import StaticCodeEncoder
        return StaticCodeEncoder.from_state(state, doc_vecs)
    return DenseEncoder.from_state(state, doc_vecs)


def heavy_encoder():
    """The optional semantic encoder, or ``None`` (import kept out of core)."""
    try:
        from .retrieval_heavy import encoder as _encoder
        return _encoder()
    except Exception:
        return None


# Fixed width for the experience-memory vector index (rbg/db.py).
EXPERIENCE_DIM = 512


def experience_encoder(dim: int = EXPERIENCE_DIM) -> DenseEncoder:
    """Deterministic, *unfit* dense encoder for experience matching.

    Left unfitted so every idf defaults to 1.0: the signed feature hash needs
    no corpus statistics, so experiences written under one repo's vocabulary
    are directly comparable with a query from another.  ponytail: hashing
    stand-in for a code embedding, swap for SPLADE-Code only if retrieval gains
    plateau.
    """
    return DenseEncoder(dim)


def mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, indices: list[int],
        k: int, lam: float = 0.7) -> list[int]:
    """Maximal Marginal Relevance selection (Carbonell & Goldstein, 1998).

    Re-orders the candidate ``indices`` (given best-first) so the chosen ``k``
    balance query relevance against redundancy; ``lam=1`` is pure relevance.
    ``ponytail:`` greedy O(k*n*d), fine at retrieval-candidate scale.
    """
    if not indices:
        return []
    cand = list(dict.fromkeys(indices))
    rel = {i: float(doc_vecs[i] @ query_vec) if doc_vecs.shape[0] else 0.0 for i in cand}
    selected: list[int] = []
    while cand and len(selected) < k:
        best, best_score = None, -math.inf
        for i in cand:
            if selected:
                red = max(float(doc_vecs[i] @ doc_vecs[j]) for j in selected)
            else:
                red = 0.0
            score = lam * rel[i] - (1 - lam) * red
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
        cand.remove(best)
    return selected


class HybridRetriever:
    """The RAG retriever: parallel lexical (BM25) + dense retrieval fused by
    reciprocal-rank fusion, with optional query expansion, reranking and MMR
    diversification.

    This is the retrieval half of RAG; it is what feeds the single generative
    LLM (spec 5.2).  ``semantic=True`` swaps the lexical hashing encoder for the
    LSA encoder; ``expansion`` adds RM3-lite pseudo-relevance feedback;
    ``diversify`` applies MMR so the context is not three near-duplicate chunks
    from one file.  Every stage is non-generative and dependency-free.
    """

    def __init__(self, index: RepoIndex, dim: int = 512,
                  reranker: "Reranker | None" = None, *, semantic: bool = True,
                  expansion: bool = False, diversify: bool = False,
                  mmr_lambda: float = 0.7, lsi_k: int = 128,
                  encoder: "DenseEncoder | LSIEncoder | None" = None,
                  lexical=None,
                  include: list[str] | None = None,
                  exclude: list[str] | None = None) -> None:
        self.index = index
        self.bm25 = lexical or BM25(index.chunks)
        if encoder is not None:
            self.encoder = encoder
        elif (heavy := heavy_encoder()) is not None:
            self.encoder = heavy
        elif len(index.chunks) <= LSI_MAX_CORPUS:
            if semantic:
                self.encoder = LSIEncoder(k=lsi_k).fit([c.content for c in index.chunks])
            else:
                self.encoder = DenseEncoder(dim).fit([c.content for c in index.chunks])
        else:
            # ponytail: lexical-only above the cap; add a background/offline
            # dense build (or an ANN vector table) if large-repo recall needs it.
            self.encoder = LexicalOnlyEncoder(dim)
        self.reranker = reranker
        self.expansion = expansion
        self.diversify = diversify
        self.mmr_lambda = mmr_lambda
        self._path_weight = [PATH_WEIGHT[_path_class(c.path)] for c in index.chunks]
        self.include = list(include) if include else None
        self.exclude = list(exclude) if exclude else None

    def _diversify(self, query: str, hits: list[Hit], limit: int) -> list[Hit]:
        chunks = self.index.chunks
        pos = {(c.path, c.start_line): i for i, c in enumerate(chunks)}
        indexed, ordered = [], []
        for h in hits:
            i = pos.get((h.path, h.start_line))
            if i is not None:
                indexed.append(i)
                ordered.append(h)
        if len(indexed) < 2:
            return hits
        vecs = self.encoder.document_vectors(chunks)
        qv = self.encoder.encode(query)
        chosen = mmr(qv, vecs, indexed, min(limit, len(indexed)), self.mmr_lambda)
        by_idx = {i: h for i, h in zip(indexed, ordered)}
        return [by_idx[i] for i in chosen]

    def retrieve(self, query: str, limit: int = 20, rrf_k: int = 60,
                  expand: bool | None = None, diversify: bool | None = None,
                  include: list[str] | None = None, exclude: list[str] | None = None) -> list[Hit]:
        expand = self.expansion if expand is None else expand
        diversify = self.diversify if diversify is None else diversify
        eff_query = expand_query(query, self.bm25, self.index.chunks) if expand else query
        eff_query = _clean_query(eff_query)
        inc = include if include is not None else self.include
        exc = exclude if exclude is not None else self.exclude
        pool = limit * 2 if diversify else limit
        raw = pool * 3 * 2
        lexical = self.bm25.rank(eff_query, raw)
        dense = self.encoder.rank(eff_query, self.index.chunks, raw)
        if inc or exc:
            def _keep(i: int) -> bool:
                p = self.index.chunks[i].path
                if inc and not any(fnmatch.fnmatch(p, g) for g in inc):
                    return False
                if exc and any(fnmatch.fnmatch(p, g) for g in exc):
                    return False
                return True
            lexical = [(i, s) for i, s in lexical if _keep(i)]
            dense = [(i, s) for i, s in dense if _keep(i)]
        fused: dict[int, float] = {}
        reasons: dict[int, list[str]] = {}
        for rank, (i, _) in enumerate(lexical):
            fused[i] = fused.get(i, 0.0) + (1.0 / (rrf_k + rank + 1)) * self._path_weight[i]
            reasons.setdefault(i, []).append("bm25" + ("+prf" if expand else ""))
        for rank, (i, _) in enumerate(dense):
            fused[i] = fused.get(i, 0.0) + (1.0 / (rrf_k + rank + 1)) * self._path_weight[i]
            reasons.setdefault(i, []).append("dense")
        # symbol/reference boost: exact identifier hits get a nudge
        q_tokens = set(tokenize(eff_query))
        for i, chunk in enumerate(self.index.chunks):
            if chunk.symbol and chunk.symbol.lower() in q_tokens:
                fused[i] = fused.get(i, 0.0) + (1.0 / (rrf_k + 1)) * self._path_weight[i]
                reasons.setdefault(i, []).append("symbol")
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:pool]
        hits: list[Hit] = []
        for i, score in ranked:
            c = self.index.chunks[i]
            hits.append(Hit(c.path, c.symbol, c.start_line, c.end_line, score,
                            "rrf", "+".join(sorted(reasons.get(i, []))), c.content_hash, c.content))
        if self.reranker is not None:
            hits = self.reranker.rerank(query, hits, self.index.chunks)
        if diversify:
            hits = self._diversify(query, hits, limit)
        for h in hits:  # stamp the snapshot revision so stale chunks are visible
            h.commit_hash = self.index.revision
        return hits[:limit]

    def expand(self, hits: list[Hit], budget: int = 5) -> list[Hit]:
        """Follow same-symbol neighbours around strong hits (spec 5.2 step 6).

        A single symbol → chunks pass replaces the per-caller scan of every
        chunk, so cost is O(chunks) once rather than O(hits x callers x chunks).
        """
        by_symbol: dict[str, list[Chunk]] = {}
        for c in self.index.chunks:
            if c.symbol:
                by_symbol.setdefault(c.symbol, []).append(c)
        have = {(h.path, h.start_line) for h in hits}
        extra: list[Hit] = []
        for h in hits:
            for c in by_symbol.get(h.symbol or "", []):
                if (c.path, c.start_line) not in have:
                    extra.append(Hit(c.path, c.symbol, c.start_line, c.end_line,
                                     0.01, "expand", "same_symbol", c.content_hash, c.content,
                                     self.index.revision))
                    have.add((c.path, c.start_line))
        for h in hits:
            h.commit_hash = h.commit_hash or self.index.revision
        return hits + extra[:budget]


class Reranker:
    """Non-generative reranker: a learned linear model over cheap pairwise
    features (spec P1 cross-encoder stand-in).  Trained from labelled
    query/chunk pairs; falls back to identity when untrained."""

    def __init__(self, weights: np.ndarray | None = None) -> None:
        self.weights = weights

    @staticmethod
    def features(query: str, chunk: Chunk) -> np.ndarray:
        q = set(tokenize(query))
        d = tokenize(chunk.content)
        dset = set(d)
        overlap = len(q & dset)
        cov = overlap / (len(q) or 1)
        sym = 1.0 if chunk.symbol and chunk.symbol.lower() in q else 0.0
        path_hit = 1.0 if any(t in chunk.path.lower() for t in q) else 0.0
        return np.array([cov, math.log1p(overlap), sym, path_hit, min(len(d), 400) / 400.0])

    def score(self, query: str, chunk: Chunk) -> float:
        f = self.features(query, chunk)
        if self.weights is None:
            return float(f[0] + f[2] + f[3])
        return float(f @ self.weights)

    def rerank(self, query: str, hits: list[Hit], chunks: list[Chunk]) -> list[Hit]:
        by_key = {f"{c.path}:{c.start_line}": c for c in chunks}
        scored = []
        for h in hits:
            c = by_key.get(f"{h.path}:{h.start_line}")
            s = self.score(query, c) if c else 0.0
            scored.append((s, h))
        scored.sort(key=lambda kv: -kv[0])
        out = []
        for s, h in scored:
            out.append(Hit(h.path, h.symbol, h.start_line, h.end_line, s,
                           "rerank", h.reason, h.content_hash, h.content))
        return out

    @classmethod
    def train(cls, examples: list[tuple[str, Chunk, float]]) -> "Reranker":
        if not examples:
            return cls()
        X = np.vstack([cls.features(q, c) for q, c, _ in examples])
        y = np.array([lbl for _, _, lbl in examples])
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        return cls(weights=w)

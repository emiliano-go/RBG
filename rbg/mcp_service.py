"""MCP service layer: authorization, caches, indexing, retrieval, learning.

Business logic lives here so the JSON-RPC adapter in :mod:`rbg.mcp_server`
stays thin and the tools can be exercised without a protocol transport.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX; DB busy-timeout still guards writes
    fcntl = None

from . import db, retrieval
from .index import RepoIndex, _tree_hash, _tree_signature
from .retrieval import (
    DenseEncoder,
    FTSBackend,
    HybridRetriever,
    LexicalOnlyEncoder,
    LSIEncoder,
    heavy_encoder,
    tokenize,
)
from .training import PROMOTION_KEY, LearnedModels, train_from_ledger

SERVER_NAME = "rbg-mcp"
SERVER_VERSION = "0.2.0"
DEFAULT_PROTOCOL = "2025-06-18"

_INDEX_CACHE: dict[str, RepoIndex] = {}
_INDEX_SIGNATURE: dict[str, str] = {}
_LEARNED_CACHE: dict[str, LearnedModels] = {}
_PLAN_DRIFT_LOGGED: "OrderedDict[tuple, bool]" = OrderedDict()
_PLAN_DRIFT_LOGGED_MAX = 512
_RETRIEVER_CACHE: dict[str, HybridRetriever] = {}

# Bounded LRU over retrieval payloads, keyed by repo/revision/query/filters.
# Only the result list is cached; latency/timing fields are recomputed per call.
_RESULT_CACHE: "OrderedDict[tuple, list]" = OrderedDict()
_RESULT_CACHE_MAX = 256


def _result_cached(key: tuple, build):
    hit = _RESULT_CACHE.get(key)
    if hit is not None:
        _RESULT_CACHE.move_to_end(key)
        return hit
    val = build()
    _RESULT_CACHE[key] = val
    if len(_RESULT_CACHE) > _RESULT_CACHE_MAX:
        _RESULT_CACHE.popitem(last=False)
    return val

# Tokenized chunk contents, keyed by content hash.  Tokenizing every test chunk
# on each call dominated get_change_impact/rank_tests on large repos (~0.5 s per
# call on a 6.5k-chunk repo); the hash makes a stale entry impossible.
_TOKENS_CACHE: dict[str, frozenset] = {}


def _chunk_tokens(chunk) -> frozenset:
    toks = _TOKENS_CACHE.get(chunk.content_hash)
    if toks is None:
        if len(_TOKENS_CACHE) >= 200_000:  # ponytail: crude cap; LRU if it ever matters
            _TOKENS_CACHE.clear()
        toks = frozenset(tokenize(chunk.content))
        _TOKENS_CACHE[chunk.content_hash] = toks
    return toks


# Advisory read-path budget.  A retrieval result is still returned when the soft
# budget is exceeded, but it is reported as degraded (and the optional reranker
# is skipped) so a caller learns the answer was cheap rather than complete.
_READ_BUDGET_MS = int(os.environ.get("RBG_READ_BUDGET_MS", "1500"))


def _read_budget(timeout_ms: int | None = None) -> float:
    return float(_READ_BUDGET_MS if timeout_ms is None else max(0, int(timeout_ms)))


def _timings(t0: float, t_i: float, t_l: float, t_r: float, t_q: float,
             t_x: float | None = None) -> dict:
    def ms(a: float, b: float) -> float:
        return round((b - a) * 1000, 1)
    out = {"index_ms": ms(t0, t_i), "learned_ms": ms(t_i, t_l),
           "retriever_ms": ms(t_l, t_r), "retrieve_ms": ms(t_r, t_q)}
    if t_x is not None:
        out["expand_ms"] = ms(t_q, t_x)
    out["total_ms"] = ms(t0, t_x if t_x is not None else t_q)
    return out
_STORE_CACHE: dict[str, "db.Store | None"] = {}


def _store(root: str) -> "db.Store":
    """The repo's libSQL store (cached per repo)."""
    if root not in _STORE_CACHE:
        _STORE_CACHE[root] = db.Store(root)
    return _STORE_CACHE[root]


def _allowed_roots() -> list[str]:
    """The authorization allowlist from ``RBG_ALLOWED_ROOTS`` (os.pathsep
    separated).  Empty means nothing is authorized."""
    raw = os.environ.get("RBG_ALLOWED_ROOTS", "")
    return [os.path.realpath(os.path.expanduser(p))
            for p in raw.split(os.pathsep) if p.strip()]


def _authorize_root(root: str) -> str:
    """Resolve a caller-supplied root and require it under the allowlist.

    Deny by default: with no allowlist configured, no repository is authorized,
    so a tool call can only ever touch a path an operator explicitly named.
    """
    resolved = os.path.realpath(os.path.expanduser(root))
    if not os.path.isdir(resolved):
        raise ValueError(f"not a directory: {root}")
    allowed = _allowed_roots()
    if not allowed:
        raise PermissionError(
            "no repository is authorized: set RBG_ALLOWED_ROOTS to an "
            "os.pathsep-separated allowlist of repository roots")
    if not any(a == os.sep or resolved == a or resolved.startswith(a + os.sep)
               for a in allowed):
        raise PermissionError(f"repository not allowed: {root}")
    return resolved


def _evict_dependents(root: str) -> None:
    """Drop per-repo derivatives keyed to a revision that is no longer current."""
    _RETRIEVER_CACHE.pop(root, None)
    _FAST_RETRIEVER_CACHE.pop(root, None)
    _RETRIEVER_BUILD.pop(root, None)
    for key in [k for k in _LEARNED_CACHE if k == root or k.startswith(root + ":")]:
        _LEARNED_CACHE.pop(key, None)
    for key in [k for k in _RESULT_CACHE if k[0] == root]:
        _RESULT_CACHE.pop(key, None)


@contextmanager
def _index_lock(path: str):
    """Cross-process lock so N agents do not all parse and persist the same tree.

    The DB serialises the write, but a 47k-chunk parse is ~5 s of duplicated CPU
    per agent; the first process builds, the rest wait and then read its index.
    """
    if fcntl is None:  # pragma: no cover
        yield
        return
    fd = os.open(path + ".index.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class IndexBuilding(RuntimeError):
    """The index is being built in the background; retry this tool shortly."""

    def __init__(self, root: str, started: float) -> None:
        self.root = root
        self.started = started
        super().__init__("index build in progress")

    def payload(self) -> dict:
        elapsed_ms = round((time.monotonic() - self.started) * 1000, 1)
        return {
            "root": self.root,
            "status": "building",
            "degraded": True,
            "index_build_ms": elapsed_ms,
            "message": "index build in progress; retry this tool in a few seconds",
        }


_BUILD_STATE: dict[str, dict] = {}
_BUILD_LOCK = threading.Lock()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def _start_index_build(root: str, signature: str) -> None:
    with _BUILD_LOCK:
        state = _BUILD_STATE.get(root)
        if state and state["signature"] == signature and state["status"] != "error":
            return
        _BUILD_STATE[root] = {"signature": signature, "status": "building",
                              "started": time.monotonic(), "error": None}
        started = _BUILD_STATE[root]["started"]

    def _worker() -> None:
        try:
            # Own Store/connection: the main loop may serve other tools on the
            # cached connection while this thread writes the index.
            _index_sync(root, signature, isolated=True)
        except Exception as exc:  # noqa: BLE001 - surfaced on the next call
            with _BUILD_LOCK:
                _BUILD_STATE[root] = {"signature": signature, "status": "error",
                                      "started": started,
                                      "error": f"{type(exc).__name__}: {exc}"}
            return
        with _BUILD_LOCK:
            _BUILD_STATE[root] = {"signature": signature, "status": "ready",
                                  "started": started, "error": None}

    threading.Thread(target=_worker, name=f"index-build:{root}", daemon=True).start()


def _index(root: str) -> RepoIndex:
    root = _authorize_root(root)
    signature = _tree_signature(root)
    cached = _INDEX_CACHE.get(root)
    if cached is not None and _INDEX_SIGNATURE.get(root) == signature:
        return cached
    # Async is the default.  A cold parse of a large repo (plus its tree hash)
    # can otherwise exceed the client's whole request timeout; the caller gets
    # a retryable "building" result while the parse runs in the background.
    explicit_async = "RBG_INDEX_ASYNC" in os.environ
    if _env_flag("RBG_INDEX_ASYNC", "1"):
        _start_index_build(root, signature)
        # Small repos finish in milliseconds; wait briefly so the common case
        # is still a single call.  An explicitly configured async caller gets
        # the old immediate "building" behaviour.
        grace = 0.0 if explicit_async else max(
            0, int(os.environ.get("RBG_INDEX_GRACE_MS", "1000"))) / 1000.0
        deadline = time.monotonic() + grace
        while True:
            with _BUILD_LOCK:
                state = dict(_BUILD_STATE.get(root) or {})
            if state.get("signature") != signature:
                _start_index_build(root, signature)  # tree changed again
                raise IndexBuilding(root, time.monotonic())
            if state.get("status") == "error":
                raise RuntimeError(state.get("error") or "index build failed")
            if state.get("status") == "ready" and root in _INDEX_CACHE:
                return _INDEX_CACHE[root]
            if time.monotonic() >= deadline:
                raise IndexBuilding(root, state.get("started", time.monotonic()))
            time.sleep(0.02)
    return _index_sync(root, signature, cached)


def _index_sync(root: str, signature: str, cached: RepoIndex | None = None,
                isolated: bool = False) -> RepoIndex:
    store = db.Store(root) if isolated else _store(root)
    try:
        # Cheap freshness fast path: a warm DB index carries the metadata
        # signature it was built from, so the whole-tree content hash (123 MB
        # read here) is only paid when files actually changed.
        try:
            if store.get_meta("tree_signature") == signature:
                idx = store.load_index(None)
            else:
                idx = None
        except Exception:
            idx = None
        if idx is None:
            fingerprint = _tree_hash(root)
            try:
                idx = store.load_index(fingerprint)  # cache hit: no re-parse
            except Exception:
                idx = None
            if idx is None:
                with _index_lock(store.path):
                    try:  # another process may have built it while we waited
                        idx = store.load_index(fingerprint)
                    except Exception:
                        idx = None
                    if idx is None:
                        idx = RepoIndex.build(root, revision=fingerprint)
                        store.save_index(idx)
        try:
            if store.get_meta("tree_signature") != signature:
                store.set_meta("tree_signature", signature)
        except Exception:
            pass
    finally:
        if isolated:
            store.close()
    if cached is not None and cached.revision != idx.revision:
        _evict_dependents(root)  # stale index/retriever/models must not be served
    _INDEX_CACHE[root] = idx
    _INDEX_SIGNATURE[root] = signature
    return idx


def _warm_lsi(store, idx: RepoIndex):
    """Re-embed a changed corpus with the previously persisted LSI projector
    instead of refitting the SVD.  Returns None when there is nothing reusable
    (no prior encoder, or a non-LSI encoder)."""
    try:
        prev = store.load_prev_encoder()
    except Exception:
        return None
    if not prev or prev[1].get("kind") != "lsi":
        return None
    if len(idx.chunks) > retrieval.LSI_MAX_CORPUS:
        return None
    enc = LSIEncoder.from_state(prev[1], [])
    return enc.refresh([c.content for c in idx.chunks])


_FAST_RETRIEVER_CACHE: dict[str, HybridRetriever] = {}
_RETRIEVER_BUILD: dict[str, dict] = {}
_RETRIEVER_BUILD_LOCK = threading.Lock()


def _fast_retriever(idx: RepoIndex) -> HybridRetriever:
    """A lexical-only retriever that is cheap to build on the read path.

    Used while the semantic retriever is warming: FTS/BM25 answers immediately
    instead of making the client wait for an SVD/embedding pass.
    """
    retr = _FAST_RETRIEVER_CACHE.get(idx.root)
    if retr is None or retr.index.revision != idx.revision:
        lexical = FTSBackend(idx.chunks, _store(idx.root).fts_search)
        retr = HybridRetriever(idx, encoder=LexicalOnlyEncoder(), lexical=lexical)
        _FAST_RETRIEVER_CACHE[idx.root] = retr
    return retr


def _start_retriever_build(idx: RepoIndex) -> None:
    with _RETRIEVER_BUILD_LOCK:
        state = _RETRIEVER_BUILD.get(idx.root)
        if state and state["revision"] == idx.revision and state["status"] != "error":
            return
        _RETRIEVER_BUILD[idx.root] = {"revision": idx.revision,
                                      "status": "building", "error": None}

    def _worker() -> None:
        try:
            # Fit only: no DB access from this thread.  A second libSQL
            # connection writing while the main connection reads pansics the
            # binding; the encoder cache is swapped in memory instead.
            heavy = heavy_encoder()
            if heavy is not None:
                encoder = heavy
            elif len(idx.chunks) <= retrieval.LSI_MAX_CORPUS:
                encoder = LSIEncoder(k=128).fit([c.content for c in idx.chunks])
            else:
                encoder = LexicalOnlyEncoder()
            lexical = FTSBackend(idx.chunks, _store(idx.root).fts_search)
            retr = HybridRetriever(idx, encoder=encoder, lexical=lexical)
            status, error = "ready", None
        except Exception as exc:  # noqa: BLE001 - retryable on the next call
            status, error = "error", f"{type(exc).__name__}: {exc}"
            retr = None
        if retr is not None:
            _RETRIEVER_CACHE[idx.root] = retr
        with _RETRIEVER_BUILD_LOCK:
            _RETRIEVER_BUILD[idx.root] = {"revision": idx.revision,
                                          "status": status, "error": error}

    threading.Thread(target=_worker, name=f"retriever-build:{idx.root}",
                     daemon=True).start()


def _retriever(root: str) -> HybridRetriever:
    """Full retriever when warm, lexical-only while it builds in the background."""
    idx = _index(root)
    retr = _RETRIEVER_CACHE.get(idx.root)
    if retr is not None:
        return retr
    # A persisted encoder is a cheap main-thread read and skips the fit.
    try:
        encoder = _store(idx.root).load_encoder(idx.revision, len(idx.chunks))
    except Exception:
        encoder = None
    if encoder is not None:
        lexical = FTSBackend(idx.chunks, _store(idx.root).fts_search)
        retr = HybridRetriever(idx, encoder=encoder, lexical=lexical)
        _RETRIEVER_CACHE[idx.root] = retr
        return retr
    _start_retriever_build(idx)
    return _fast_retriever(idx)


def _build_retriever(idx: RepoIndex, store=None) -> HybridRetriever:
    store = store or _store(idx.root)
    if idx.root not in _RETRIEVER_CACHE:
        lexical = FTSBackend(idx.chunks, store.fts_search)
        # An opt-in semantic encoder wins over the persisted lexical one, and is
        # persisted itself so vector_search queries vectors from the same model.
        heavy = heavy_encoder()
        if heavy is not None:
            retr = HybridRetriever(idx, encoder=heavy, lexical=lexical)
            store.save_encoder(idx.revision, retr.encoder.to_state(),
                               retr.encoder.document_vectors(idx.chunks))
        else:
            encoder = None
            try:
                encoder = store.load_encoder(idx.revision, len(idx.chunks))
            except Exception:
                encoder = None
            warmed = False
            if encoder is None:
                encoder = _warm_lsi(store, idx)
                warmed = encoder is not None
            if encoder is None and len(idx.chunks) <= retrieval.LSI_MAX_CORPUS:
                # Reuse a persisted dense encoder's idf across revisions: the
                # document matrix is rebuilt lazily, but the O(corpus) fit is
                # skipped.  (Above the cap HybridRetriever stays lexical-only.)
                try:
                    prev = store.load_prev_encoder()
                except Exception:
                    prev = None
                if prev and prev[1].get("kind") == "dense":
                    enc = DenseEncoder(dim=int(prev[1].get("dim", 512)))
                    enc.idf = dict(prev[1].get("idf", {}))
                    encoder = enc
                    warmed = True
            if encoder is not None:
                retr = HybridRetriever(idx, encoder=encoder, lexical=lexical)
            else:
                retr = HybridRetriever(idx, lexical=lexical)
            if (encoder is None or warmed) and not isinstance(retr.encoder, LexicalOnlyEncoder):
                store.save_encoder(idx.revision, retr.encoder.to_state(),
                                   retr.encoder.document_vectors(idx.chunks))
        _RETRIEVER_CACHE[idx.root] = retr
    return _RETRIEVER_CACHE[idx.root]


def _apply_validation(models: LearnedModels | None, store, revision: str) -> LearnedModels | None:
    """Restore the promotion verdict only for the artifact it was earned on."""
    if models is None:
        return None
    record = store.get_meta(PROMOTION_KEY) or {}
    models.validated = bool(
        record.get("promoted")
        and record.get("revision") == revision
        and int(record.get("ledger_n", -1)) == store.ledger_count())
    return models


def _learned(root: str, retrain: bool = False) -> LearnedModels:
    """Trained models for a repo.

    The read path never trains: it loads a model persisted for the current
    revision *and* ledger size (a cheap ``COUNT(*)``), or returns an empty set.
    Training happens only through the explicit ``train_models`` tool, so no
    retrieval or impact query ever pays to refit.
    """
    if retrain:
        idx = _index_sync(root, _tree_signature(root))
    else:
        try:
            idx = _index(root)
        except IndexBuilding:
            # A model-dependent read degrades to "no model yet" rather than
            # blocking the client on the same cold index build.
            return LearnedModels()
    store = _store(idx.root)
    key = f"{idx.root}:{idx.revision}:{store.ledger_count()}"
    if retrain:
        models = _apply_validation(
            train_from_ledger(idx.root, index=idx, store=store), store, idx.revision)
        for k in [k for k in _LEARNED_CACHE if k.startswith(idx.root + ":")]:
            _LEARNED_CACHE.pop(k, None)
        _LEARNED_CACHE[key] = models
        return models
    if key in _LEARNED_CACHE:
        return _LEARNED_CACHE[key]
    models = None
    try:
        got = store.load_model_at("learned", idx.revision)
        if got is not None and got[2].get("ledger_n") == store.ledger_count():
            models = _apply_validation(got[0], store, idx.revision)
    except Exception:
        models = None
    _LEARNED_CACHE[key] = models or LearnedModels()
    return _LEARNED_CACHE[key]


def _index_fresh(idx: RepoIndex) -> bool:
    """Whether the cached index still matches the on-disk tree (spec 5.2: never
    present stale content as current)."""
    try:
        return _tree_signature(idx.root) == _INDEX_SIGNATURE.get(idx.root)
    except OSError:
        return False


def _index_summary(root: str) -> dict:
    idx = _index(root)
    return {"root": idx.root, "revision": idx.revision, "files": idx.files,
            "chunks": len(idx.chunks), "symbols": [s.name for s in idx.symbols]}


